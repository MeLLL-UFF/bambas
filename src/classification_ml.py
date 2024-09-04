import argparse
from argparse import Namespace
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from typing import List, Dict, Any, Tuple
from src.data import load_dataset
from src.utils.workspace import get_workdir
from src.confusion_matrix import *
from functools import reduce
from copy import deepcopy

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
# from transformers import AdamW, Adafactor
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from evaluate import load

OUTPUT_DIR = f"{get_workdir()}/classification"
GOLD_DIR = f"{get_workdir()}/dataset/semeval2024/subtask1"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
MODEL = "jhu-clsp/bernice"

def load_data_split(tokenizer: AutoTokenizer,
                    dataset: pd.DataFrame,
                    batch_size: int = 64) -> DataLoader:
    # Create Dataset object
    ds = Dataset.from_pandas(dataset)
    
    def remove_additional_columns(ds: Dataset):
        columns = ds.column_names
        to_remove = [col for col in columns if ((col != "text") and (col !="labels"))]
        print("removing columns:", to_remove)
        return ds.remove_columns(to_remove)

    ds = remove_additional_columns(ds)

    print("dataset")
    print(ds)

    # Define tokenizer function
    def tokenize_function(examples):
        try:
            return tokenizer(examples["text"], max_length=120, truncation=True, padding="max_length")
        except:
            print(examples["text"])
    # Tokenize values in Dataset object
    ds = ds.map(tokenize_function, batched=True, batch_size=64, num_proc=4, remove_columns=["text"])
    # Remove original text from dataset

    # Create DataCollator object
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length")
    # Create DataLoader to be returned
    return DataLoader(dataset=ds,
                      batch_size=batch_size,
                      collate_fn=data_collator)

def train(net, trainloader, epochs, lr):
    optimizer = AdamW(net.parameters(), lr=lr)  # Adafactor(net.parameters(), warmup_init=True) 
    training_loss = 0
    net.train()
    for i in range(epochs):
        for batch in trainloader:
            # TODO : Separate input and label
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            training_loss += outputs.loss.item()
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # training_loss /= len(trainloader.dataset)

        print(f"Epoch {i+1} | Loss = {training_loss}")
        training_loss = 0

def pred(net, valloader):
    net.eval()
    preds = []
    with torch.no_grad():
        for batch in valloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                outputs = net(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            preds.extend(predictions.tolist())

    return preds

def save_predictions(test_df: pd.DataFrame, predictions: List[List[str]],
                     kind: str, timestamp: int) -> Tuple[str, List[Dict[str, Any]]]:
    predictions_json = []
    for idx, row in test_df.iterrows():
        predictions_json.append({
            "id": row["id"],
            "labels": predictions[idx],
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # submission format has txt extension but json format
    predictions_json_path = f"{OUTPUT_DIR}/{timestamp}_{kind}_predictions.json.txt"
    with open(predictions_json_path, "w") as f:
        json.dump(predictions_json, f, indent=4)

    return predictions_json_path, predictions_json

def binary_classify(args: Namespace):
   
    print("Loading dataset files")
    train_ds, dev_ds, test_ds = load_dataset(args.dataset)
    print("Dataset Lengths", len(train_ds), len(dev_ds), len(test_ds))

    
    # Transform dataset into binary
    def binarize(labelset):
        labelset = str(labelset)
        if args.dataset == "semeval2015" or args.dataset == "semeval2016":
            if labelset=="negative": return 0
            else: return 1
        else:
            if labelset == "[]": return 0
            else: return 1

    train_labels = [binarize(labelset) for labelset in train_ds["labels"]]
    dev_labels = [binarize(labelset) for labelset in dev_ds["labels"]]
    labels = [[0, 1]]

    train_ds["labels"] = train_labels
    dev_ds["labels"] = dev_labels

    # Loading dataset into DataLoader objects
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    train_dl = load_data_split(tokenizer=tokenizer, dataset=train_ds)
    dev_dl = load_data_split(tokenizer=tokenizer, dataset=dev_ds)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.to(DEVICE)
    train(net=model, trainloader=train_dl, epochs=args.epochs, lr=args.lr)
    dev_preds = pred(net=model, valloader=dev_dl)
    ts = int(time.time())

    # Debug prediction print
    print("\nPREDS:\n")
    print(dev_preds)
    print("\n\n")

    # Create the Binary Relevance Confusion Matrix
    cf_mtx = str(confusion_matrix(np.array(dev_labels), np.array(dev_preds))).replace("\n","")
    prec, rec, f1, _ = precision_recall_fscore_support(dev_labels, dev_preds)
    print(f"\nValidation set:\n\tPrecision: {prec}\n\tRecall: {rec}\n\tF1: {f1}\n")

    results_csv_path = f"{OUTPUT_DIR}/results.csv"
    print(f"Saving validation set results to {results_csv_path}")
    if os.path.exists(results_csv_path):
        results = pd.read_csv(results_csv_path, index_col=0)
    else:
        dir = os.path.sep.join(results_csv_path.split(os.path.sep)[:-1])
        print(f"Creating dir {dir}")
        os.makedirs(dir, exist_ok=True)
        results = pd.DataFrame(
            columns=[
                "FE Model",
                "FE Method",
                "FE Layers",
                "FE Layers Agg Method",
                "FE Dataset",
                "Classifier",
                "F1",
                "Precision",
                "Recall",
                "Confusion Matrix",
                "Timestamp"])

    # TODO : Create a dummy train_ft with fields "model" and "dataset" 
    results.loc[len(results.index) + 1] = [MODEL, 
                                           "fine-tuning",
                                           "na",
                                           "na",
                                           args.dataset,
                                           "val_set",
                                           "see model",
                                           f1,
                                           prec,
                                           rec,
                                           cf_mtx,
                                           ts]
    results.to_csv(results_csv_path)

    print(f"Finished successfully. results saved to results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("classification", description="classification with feedforward model")
    parser.add_argument("--classifier", type=str,
                        default="MLP")
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
            "semeval2024_test_set_no_concat",
            "semeval2024_all",
            "paraphrase",
            "paraphrase4",
            "outsiders",
            "semeval2015",
            "semeval2015_paraphrased",
            "semeval2015_2",
            "semeval2015_paraphrased2",
            "semeval2015_paraphrased2_1to4_selected",
            "semeval2015_paraphrased2_1to4_selected2",
            "semeval2016",
            "semeval2016_paraphrased",
            "semeval2016_paraphrased_1to4"],
        help="corpus for masked-language model pretraining task",
        required=True)
    parser.add_argument("--epochs", type=int, default=50, help="epochs for training language model")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate training language model")
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducibility")

    args = parser.parse_args()
    print("Arguments:", args)
    binary_classify(args)