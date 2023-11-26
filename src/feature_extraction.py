import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from typing import List


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
    all_hidden_states = torch.stack(outputs[1].to("cpu"), dim=0).to("cpu")
    # select layers to extract features from
    selected_hidden_states = all_hidden_states[layers]
    # reorder dimensions for ease of usage, result will be: s sentences * t
    # tokens per sentence * l layers per token * u hidden units per layer
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
        embeddings = np.concatenate((embeddings, sentence_embeddings))
    return embeddings


def extract_features(
        model: AutoModel,
        data_loader: DataLoader,
        use_cls: bool = True,
        layers_to_extract: List[int] = None,
        layer_aggregation_method: str = None) -> np.ndarray:
    if use_cls:
        assert layers_to_extract is None, "layers_to_extract argument is not supported if CLS token extraction is enabled, please choose one of the two"
        assert layer_aggregation_method is None, "layer_aggregation_method is not supported if CLS token extraction is enabled, please choose one of the two"
    if layers_to_extract is not None:
        assert use_cls is False, "layers_to_extract argument is not supported if CLS token extraction is enabled, please choose one of the two"
        assert layer_aggregation_method is not None, "layers_to_extract argument needs layer_aggregation_method to be passed, use either 'avg' or 'concatenate'"

    features = torch.tensor([]).to("cpu")
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            print(f'Running inference with batch no. {idx+1}')
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            if use_cls:
                batch_features = get_cls_token_features(outputs)
            else:
                batch_features = get_layer_features(outputs, layers_to_extract, layer_aggregation_method)
            features = torch.cat((features, batch_features))
        return features


if __name__ == "__main__":
    DATASET_DIR = "../dataset/semeval2024/"

    def load_ptc() -> List[pd.DataFrame]:
        train = pd.read_csv(DATASET_DIR + "train.csv", sep=";").dropna(subset=["text", "label"])[["text", "label"]]
        train = train.drop_duplicates(subset=["text"])
        test = pd.read_csv(DATASET_DIR + "test.csv", sep=";").dropna(subset=["text", "label"])[["text", "label"]]
        test = test.drop_duplicates(subset=["text"])
        return train, test

    train, test = load_ptc()

    dataloader = load_data_split(tokenizer=AutoTokenizer.from_pretrained("xlm-roberta-base"),
                                 dataset=train)

    print("dataloader is done")
    for idx, batch in enumerate(dataloader):
        print(f"batch {idx}:")
        print(batch["input_ids"].shape)
        print(batch["attention_mask"].shape)
