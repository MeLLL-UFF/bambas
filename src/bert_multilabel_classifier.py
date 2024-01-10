import argparse
from argparse import Namespace
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, AdamW
from src.data import load_dataset
from src.utils.workspace import get_workdir
from typing import List, Any, Dict
from sklearn.preprocessing import MultiLabelBinarizer

# OUTPUT_DIR = f"{get_workdir()}/feature_extraction"

def load_data_split(tokenizer: AutoTokenizer,
                    dataset: pd.DataFrame,
                    batch_size: int = 64) -> DataLoader:

    # Transform labels into multi-hot encoding
    labels = dataset["labels"].to_list()
    labels = np.unique(np.array([item for row in labels for item in row]))
    labels.sort()
    
    mlb = MultiLabelBinarizer(classes=labels)
    
    multihot_labels = mlb.fit_transform(dataset["labels"])
    for idx, label_list in enumerate(dataset["labels"]):
        dataset["labels"][idx] = multihot_labels[idx]

    # print(dataset["labels"].head())

    # Create Dataset object
    ds = Dataset.from_pandas(dataset)
    
    def remove_additional_columns(ds: Dataset):
        columns = ds.column_names
        # TODO: Fix to_remove filtering
        to_remove = [col for col in columns if col != "text" and col != "labels"]
        print("Columns to remove: ", to_remove)
        return ds.remove_columns(to_remove)

    ds = remove_additional_columns(ds)

    # Define tokenizer function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="longest")
    
    # Tokenize values in Dataset object
    # Remove original text from dataset
    ds = ds.map(tokenize_function, batched=True, batch_size=64, num_proc=4, remove_columns=["text"])

    # Create DataCollator object
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length")
    # Create DataLoader to be returned
    return DataLoader(dataset=ds,
                      batch_size=batch_size,
                      collate_fn=data_collator)

def classify(args: Namespace):
    train, dev, test = load_dataset(args.dataset)
    print("Dataset Lengths", len(train), len(dev), len(test))
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    device_map = {"": "cpu"}
    if torch.cuda.is_available():
        device_map = {"": 0}
    print(f"Using device: {device_map}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, 
        device_map=device_map, 
        num_labels=20,
        problem_type="multi_label_classification")
    
    train_dl = load_data_split(tokenizer=tokenizer, dataset=train)
    dev_dl = load_data_split(tokenizer=tokenizer, dataset=dev)
    test_dl = load_data_split(tokenizer=tokenizer, dataset=test)

    # Train Model
    # model_train(model, train_dl, test_dl)
    extract_preds(model, test_dl)

def model_train(model, train_dl: DataLoader, eval_dl: DataLoader, epochs: int):
    optimizer = AdamW(model.parameters())
    model.train()
    for epoch in range(epochs):
        # For every epoch run the training 
        for batch in train_dl:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            # loss = outputs.loss
            optimizer.step()
            optimizer.zero_grad()
        # End of epoch evaluate
    # End of training evaluate
    print("Finished Training Successfuly")

def extract_preds(model, eval_dl: DataLoader):
    for batch in eval_dl:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            logits.save()
            # predictions = torch.argmax(logits, dim=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("bert_multilabel_classifier", description="feature-extraction with pretrained models")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "ptc2019",
            "semeval2024"],
        help="corpus for feature-extraction",
        required=True)
    parser.add_argument(
        "--model", 
        type=str, 
        help="name or path to fine-tuned model", 
        required=True)
    args = parser.parse_args()
    # args.layers = [int(arg) for arg in args.layers]
    # print("Arguments:", args)
    classify(args)
    
    train, dev, test = load_dataset(args.dataset)
    print("Dataset Lengths", len(train), len(dev), len(test))
    load_data_split(AutoTokenizer.from_pretrained("xlm-roberta-base"), train)
