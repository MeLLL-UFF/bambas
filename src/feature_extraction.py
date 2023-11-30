import argparse
from argparse import Namespace
import pandas as pd
import numpy as np
import torch
import os
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import json
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from typing import List
from src.data import load_dataset
from src.utils.workspace import get_workdir

OUTPUT_DIR = f"{get_workdir()}/feature_extraction"

def load_data_split(tokenizer: AutoTokenizer,
                    dataset: pd.DataFrame,
                    batch_size: int = 64) -> DataLoader:
    # Read CSV file
    # df = dataset_prep(pd.read_csv(dataset_path))
    # # Create Dataset object
    ds = Dataset.from_pandas(dataset)
    # Define tokenizer function

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="longest")
    # Tokenize values in Dataset object
    ds = ds.map(tokenize_function, batched=True)
    # Remove original text from dataset
    ds = ds.remove_columns("text")
    ds = ds.remove_columns("__index_level_0__")
    ds = ds.remove_columns("label")
    # Discover max_length for dataset

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
    all_hidden_states = torch.stack(outputs[1], dim=0)
    # select layers to extract features from
    selected_hidden_states = all_hidden_states[layers]
    # reorder dimensions for ease of usage, result will be: 
    # s sentences * t tokens per sentence * l layers per token * u hidden units per layer
    hidden_states = selected_hidden_states.permute(1, 2, 0, 3)
    embeddings = np.array([])
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
        embeddings = np.concatenate((embeddings, [sentence_embeddings.cpu().numpy()]))
    return embeddings


def extract_features(
        model: AutoModel,
        data_loader: DataLoader,
        extraction_method: str,
        layers: List[int] = None,
        aggregation_method: str = None) -> np.ndarray:
    if extraction_method == "cls":
        assert layers is None, "'layers' argument is not supported if CLS token extraction is enabled, please choose one of the two"
        assert aggregation_method is None, "'agg_method' argument is not supported if CLS token extraction is enabled, please choose one of the two"
    elif extraction_method == "layers":
        assert layers is not None, "layers extraction method needs 'layers' argument to be passed"
        assert aggregation_method is not None, "layers extraction method needs 'agg_method' argument to be passed, use either 'avg' or 'concatenate'"

    features = np.array([])
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            print(f'Batch no. {idx+1}')
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            if extraction_method == "cls":
                batch_features = get_cls_token_features(outputs)
            else:
                batch_features = get_layer_features(outputs, layers, aggregation_method)
            features = np.concatenate((features, batch_features))
        return features

def feature_extraction(args: Namespace):
    train, test, dev = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model).to(device)
    train_dl = load_data_split(tokenizer=tokenizer, dataset=train)
    test_dl = load_data_split(tokenizer=tokenizer, dataset=test)
    dev_dl = load_data_split(tokenizer=tokenizer, dataset=dev)
    train_ft, test_ft, dev_ft = map(lambda dl: extract_features(model, dl, args.extraction_method, args.layers, args.agg_method), [train_dl, test_dl, dev_dl])
    train_ft_json, test_ft_json, dev_ft_json = map(lambda x: {
        "model": args.model,
        "extraction_method": args.extraction_method,
        "layers": ", ".join(args.layers) if args.layers is not None else "",
        "layers_aggregation_method": args.agg_method if args.agg_method is not None else "",
        "dataset": args.dataset,
        "features": x,
    }, [train_ft, test_ft, dev_ft])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_ft_path = f"{OUTPUT_DIR}/train_features.json"
    with open(train_ft_path, "w") as f:
        json.dump(train_ft_json, f)
    print(f"Saved feature-extraction file to {train_ft_path}")
    test_ft_path = f"{OUTPUT_DIR}/test_features.json"
    with open(test_ft_path, "w") as f:
        json.dump(test_ft_json, f)
    print(f"Saved feature-extraction file to {test_ft_path}")
    dev_ft_path = f"{OUTPUT_DIR}/dev_features.json"
    with open(dev_ft_path, "w") as f:
        json.dump(dev_ft_json, f)
    print(f"Saved feature-extraction file to {dev_ft_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("feature_extraction", description="feature-extraction with pretrained models")
    parser.add_argument("--dataset", type=str, choices=["ptc2019", "semeval2024"], help="corpus for feature-extraction", required=True)
    parser.add_argument("--model", type=str, help="name or path to fine-tuned model", required=True)
    parser.add_argument("--extraction_method", type=str, choices=["cls", "layers"], help="extraction method, 'cls' or 'layers'", required=True)
    parser.add_argument("--layers", type=List[int], help="list of layers to extract, only supported if extraction_method=layers")
    parser.add_argument("--agg_method", type=str, choices=["avg", "concatenate"], help="aggregation method to consolidate layer features, only supported if extraction_method=layer")
    args = parser.parse_args()
    print("Arguments:", args)
    feature_extraction(args)
