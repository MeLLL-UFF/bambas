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

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

OUTPUT_DIR = f"{get_workdir()}/classification"
GOLD_DIR = f"{get_workdir()}/dataset/semeval2024/subtask1"

# TODO : Create compute_metrics for metric computation at each epoch

def compute_metrics(eval_pred):
    preds, labels = eval_pred

def multihot_parse(labelset_string:str):
    labelset = []
    for char in labelset_string:
        if char=="0":
            labelset.append(0)
        elif char=="1":
            labelset.append(1)
        else: continue
    return labelset

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
    train, dev, test = load_dataset(args.dataset)
    print("Dataset Lengths", len(train), len(dev), len(test))

    # Transform dataset into binary
    def binarize(labelset):
        labelset = str(labelset)
        if args.dataset == "semeval2015" or args.dataset == "semeval2016":
            if labelset=="negative": return 0
            else: return 1
        else:
            if labelset == "[]": return 0
            else: return 1

    train_labels = [binarize(labelset) for labelset in train["labels"]]
    dev_labels = [binarize(labelset) for labelset in dev["labels"]]
    labels = [[0, 1]]

    print(f"Labels: {labels[0]}")
    
    # Initializing Classifier
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/bernice")

    data_collator = DataCollatorWithPadding(

    )
    
    model = AutoModelForSequenceClassification.from_pretrained("jhu-clsp/bernice")

    training_args = TrainingArguments(
        save_strategy="no",
        learning_rate=2e-5,
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        push_to_hub=False,
        
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    dev_preds = NotImplemented()
    ts = int(time.time())

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
    results.loc[len(results.index) + 1] = [train_ft_info["model"], 
                                           "fine-tuning",
                                           "na",
                                           "na",
                                           train_ft_info["dataset"],
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
    parser.add_argument("--train_features", type=str, help="path to extracted features file (JSON)", required=True)
    parser.add_argument("--test_features", type=str, help="path to extracted features file (JSON)", required=True)
    parser.add_argument(
        "--dev_features",
        type=str,
        help="path to extracted features file (JSON). Currently not used",
        required=True)
    parser.add_argument("--max_iter", type=int, default=400, help="max iterations for ff classifier")
    parser.add_argument("--alpha", type=float, default=0.0001, help="weight of the L2 regularitation term")
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducibility")
    parser.add_argument(
        "--oversampling",
        type=str,
        default=None,
        help="if oversampling methods should be used (available only with binary relevance classifiers)")
    parser.add_argument(
        "--sampling_strategy",
        type=float,
        default=None,
        help="define the sampling strategy for oversamplers (available only with binary relevance classifiers)")
    parser.add_argument(
        "--concat_train_dev",
        action="store_true",
        help="wheter to concatenate train+dev(validation) datasets for training")

    args = parser.parse_args()
    print("Arguments:", args)
    binary_classify(args)