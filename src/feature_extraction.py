import argparse
from argparse import Namespace
import pandas as pd
import numpy as np
import torch
import os
import time
from datasets import Dataset
from torch.utils.data import DataLoader
import json
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from typing import List, Any, Dict
from src.data import load_dataset
from src.utils.workspace import get_workdir

OUTPUT_DIR = f"{get_workdir()}/feature_extraction/"


def load_data_split(tokenizer: AutoTokenizer,
                    dataset: pd.DataFrame,
                    batch_size: int = 64) -> DataLoader:
    # Read CSV file
    # df = dataset_prep(pd.read_csv(dataset_path))
    # # Create Dataset object
    ds = Dataset.from_pandas(dataset)
    # Define tokenizer function

    def remove_additional_columns(ds: Dataset):
        columns = ds.column_names
        to_remove = [col for col in columns if col != "text"]
        return ds.remove_columns(to_remove)

    ds = remove_additional_columns(ds)

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=120, truncation=True, padding="max_length")
    # Tokenize values in Dataset object
    ds = ds.map(tokenize_function, batched=True, batch_size=64, num_proc=4, remove_columns=["text"])
    # Remove original text from dataset

    # Create DataCollator object
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length")
    # Create DataLoader to be returned
    return DataLoader(dataset=ds,
                      batch_size=batch_size,
                      collate_fn=data_collator)


def get_cls_token_features(outputs: BaseModelOutput) -> np.ndarray:
    last_hidden_states = outputs[0].to("cpu")
    # extract [CLS] token (position 0) hidden representation from last (output) layer
    return last_hidden_states[:, 0, :].cpu().numpy()


def get_layer_features(outputs: BaseModelOutput, layers: List[int], aggregation_method: str = "avg") -> np.ndarray:
    assert aggregation_method == "avg" or aggregation_method == "concatenate", "to aggregate hidden representation from multiple layers, you must select the aggregation method: 'avg' or 'concatenate'"
    # hidden states from all layers
    all_hidden_states = torch.stack(outputs.hidden_states, dim=0)
    # select layers to extract features from
    selected_hidden_states = all_hidden_states[layers]
    # reorder dimensions for ease of usage, result will be:
    # s sentences * t tokens per sentence * l layers per token * u hidden units per layer
    hidden_states = selected_hidden_states.permute(1, 2, 0, 3)
    embeddings = None
    for i in range(len(hidden_states)):
        hidden_token_embeddings_for_sentence = hidden_states[i]
        # calculate hidden representation for each layer using vector averaging. Will average token vectors on each layer
        sentence_embeddings = torch.mean(hidden_token_embeddings_for_sentence, dim=0)
        # calculate final hidden representation of the sentence using vector averaging
        if aggregation_method == "avg":
            sentence_embeddings = torch.mean(sentence_embeddings, dim=0)
        # obtains final hidden representation of the sentence by concatenating each averaged hidden layer
        else:
            sentence_embeddings = torch.cat(tuple(sentence_embeddings), dim=0)
        sentence_embeddings = sentence_embeddings.cpu().numpy()
        if embeddings is None:
            embeddings = np.array([sentence_embeddings])
        else:
            embeddings = np.concatenate((embeddings, [sentence_embeddings]))
    return embeddings

# Mean Pooling - Take attention mask into account for correct averaging
def get_sentence_features(outputs: BaseModelOutput, attention_mask: torch.Tensor) -> np.ndarray:
    token_embeddings = outputs[0].cpu()  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().cpu()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return embeddings.cpu().numpy()


def extract_features(
        model: AutoModel,
        data_loader: DataLoader,
        extraction_method: str,
        layers: List[int],
        aggregation_method: str = None) -> np.ndarray:
    if extraction_method == "cls":
        assert len(layers) == 0, "'layers' argument is not supported if CLS token extraction is enabled, please choose one of the two"
        assert aggregation_method is None, "'agg_method' argument is not supported if CLS token extraction is enabled, please choose one of the two"
    elif extraction_method == "layers":
        assert len(layers) > 0, "layers extraction method needs 'layers' argument to be passed"
        assert aggregation_method is not None, "layers extraction method needs 'agg_method' argument to be passed, use either 'avg' or 'concatenate'"
    elif extraction_method == "sentence":
        assert len(layers) == 0, "'layers' argument is not supported for sentence embedding models"
        assert aggregation_method is None, "'agg_method' argument is not supported for sentence embedding models"

    features = None
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            print(f'Batch no. {idx+1}')
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            if extraction_method == "cls":
                batch_features = get_cls_token_features(outputs)
            elif extraction_method == "layers":
                batch_features = get_layer_features(outputs, layers, aggregation_method)
            else:
                batch_features = get_sentence_features(outputs, batch["attention_mask"])
            if features is None:
                features = batch_features
            else:
                features = np.concatenate((features, batch_features))
        return np.array(features)


def save_ft_files(ft_data: Dict[str, Any], ft_path: str):
    ft_json = ft_data
    ft_array_path = os.path.dirname(ft_path) + os.path.sep + os.path.basename(ft_path).split('.')[0] + "_array.json"
    ft_array = ft_json["features"]
    ft_json["features"] = ft_array_path
    with open(ft_path, "w") as f:
        json.dump(ft_json, f, indent=4)

    pd.Series(list(ft_array)).to_json(ft_array_path, orient="records")


def feature_extraction(args: Namespace):
    train, dev, test = load_dataset(args.dataset)
    print("Dataset Lengths", len(train), len(dev), len(test))
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    device_map = {"": "cpu"}
    if torch.cuda.is_available():
        device_map = {"": 0}
    print(f"Using device: {device_map}")
    model = AutoModel.from_pretrained(args.model, device_map=device_map, output_hidden_states=True)
    train_dl = load_data_split(tokenizer=tokenizer, dataset=train)
    dev_dl = load_data_split(tokenizer=tokenizer, dataset=dev)
    test_dl = load_data_split(tokenizer=tokenizer, dataset=test)
    train_ft, test_ft, dev_ft = map(
        lambda dl: extract_features(
            model, dl, args.extraction_method, args.layers, args.agg_method), [
            train_dl, test_dl, dev_dl])
    train_ft_json, test_ft_json, dev_ft_json = map(lambda ft: {
        "model": args.model,
        "embeddings_type": "word" if args.extraction_method != "sentence" else "sentence",
        "extraction_method": args.extraction_method,
        "layers": ", ".join([str(arg) for arg in args.layers]) if args.layers is not None else "-",
        "layers_aggregation_method": args.agg_method if args.agg_method is not None else "-",
        "dataset": args.dataset,
        "features": ft,
    }, [train_ft, test_ft, dev_ft])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ts = int(time.time())
    ft_prefix = f"{ts}_{args.model.replace('/', '-')}_" if args.output_dir == "" else ""
    output_dir = args.output_dir if args.output_dir.endswith("/") else args.output_dir + "/"
    train_ft_path = f"{OUTPUT_DIR}{output_dir}{ft_prefix}train_features.json"
    print("train_ft_path:", train_ft_path)
    save_ft_files(train_ft_json, train_ft_path)
    print(f"Saved feature-extraction file to {train_ft_path}")

    test_ft_path = f"{OUTPUT_DIR}{output_dir}{ft_prefix}test_features.json"
    save_ft_files(test_ft_json, test_ft_path)
    print(f"Saved feature-extraction file to {test_ft_path}")

    dev_ft_path = f"{OUTPUT_DIR}{output_dir}{ft_prefix}dev_features.json"
    save_ft_files(dev_ft_json, dev_ft_path)
    print(f"Saved feature-extraction file to {dev_ft_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("feature_extraction", description="feature-extraction with pretrained models")
    parser.add_argument("--output_dir", type=str, default="", help="directory for the feature files")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "ptc2019",
            "semeval2024",
            "semeval2024_augmented_smears",
            "semeval_augmented",
            "semeval_internal",
            "semeval2024_test_set_unlabeled",
            "semeval2024_test_set_no_concat"],
        help="corpus for feature-extraction",
        required=True)
    parser.add_argument(
        "--model",
        type=str,
        help="name or path to fine-tuned model",
        required=True)
    parser.add_argument(
        "--extraction_method",
        type=str,
        choices=[
            "cls",
            "layers", "sentence"],
        help="extraction method, 'cls', 'layers' or 'sentence'",
        required=True)
    parser.add_argument(
        "--layers",
        nargs='+',
        help="list of layers to extract, only supported if extraction_method=layers",
        default=[])
    parser.add_argument(
        "--agg_method",
        type=str,
        choices=[
            "avg",
            "concatenate"],
        help="aggregation method to consolidate layer features, only supported if extraction_method=layer",
        default=None)
    args = parser.parse_args()
    args.layers = [int(arg) for arg in args.layers]
    print("Arguments:", args)
    feature_extraction(args)
