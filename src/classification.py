import argparse
from argparse import Namespace
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from typing import List, Dict, Any, Tuple
from src.data import load_dataset
from src.utils.workspace import get_workdir
from src.subtask_1_2a import evaluate_h
from functools import reduce

OUTPUT_DIR = f"{get_workdir()}/classification"
GOLD_PATH = f"{get_workdir()}/dataset/semeval2024/subtask1/validation.json"

def save_predictions(test_df: pd.DataFrame, predictions: List[List[str]]) -> Tuple[str, List[Dict[str, Any]]]:
    predictions_json = []
    for idx, row in test_df.iterrows():
        predictions_json.append({
            "id": row["id"],
            # Removes "None" predictions, as samples with empty labels must provide an empty prediction
            "labels": list(filter(lambda x: x != "None", predictions[idx]))
        })
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions_json_path = f"{OUTPUT_DIR}/predictions.json"
    with open(predictions_json_path, "w") as f:
        json.dump(predictions_json, f)
    
    return predictions_json_path, predictions_json


def load_features_info(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_features_array(path: str) -> np.ndarray:
    with open(path, "r") as f:
        df = pd.DataFrame().from_records(json.load(f))
        return np.array(df)

def classify(args: Namespace):
    print("Loading features info files")
    train_ft_info, test_ft_info, _ = map(load_features_info, [args.train_features, args.test_features, args.dev_features])

    print("Loading dataset files")
    train, test, _ = load_dataset(args.dataset)

    def fill_none_samples(dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset.fillna("None")
    
    train, test, _ = map(fill_none_samples, [train, test, _])
    print(train.head())
    all_labels = pd.concat([train["labels"], test["labels"]])

    labels = [list(set(reduce(lambda x, y: x+y, all_labels.to_numpy().tolist())))]
    print(f"Labels: {labels}")
    print(f"No. of labels in dataset (includes non-labeled samples): {len(labels[0])}")

    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit(labels).transform(train["labels"].to_numpy())

    print("Loading features array files")
    train_ft, test_ft = map(load_features_array, [train_ft_info["features"], test_ft_info["features"]])

    # TODO: we are not using the validation set correctly. As such, only the train and test splits are used throughout this code.
    # We must implement manual validation set evaluation
    ff = MLPClassifier(
        random_state=1,
        max_iter=400,
        alpha=0.0001,
        shuffle=True,
        early_stopping=True,
        verbose=True
    ).fit(train_ft, train_labels)

    test_predicted_labels_binarized = ff.predict(test_ft)
    
    test_predicted_labels = mlb.inverse_transform(test_predicted_labels_binarized)
    pred_path, _ = save_predictions(test, test_predicted_labels)

    prec, rec, f1 = evaluate_h(pred_path, GOLD_PATH)
    print(f"\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}\n")

    results_csv_path = f"{OUTPUT_DIR}/results.csv"
    if os.path.exists(results_csv_path):
        results = pd.read_csv(results_csv_path)
    else:
        dir = os.path.sep.join(results_csv_path.split(os.path.sep)[:-1])
        print(f"Creating dir {dir}")
        os.makedirs(dir, exist_ok=True)
        results = pd.DataFrame(columns=["FE Model", "FE Method", "FE Layers", "FE Layers Agg Method", "FT Dataset", "Test Dataset", "Classifier", "F1", "Precision", "Recall", "Timestamp"])
    results.loc[len(results.index)] = [train_ft_info["model"], train_ft_info["extraction_method"], train_ft_info["layers"], train_ft_info["layers_aggregation_method"], train_ft_info["dataset"], "test_set", "MLP", f1, prec, rec, int(time.time())]
    results.to_csv(results_csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("classification", description="classification with feedforward model")
    parser.add_argument("--dataset", type=str, choices=["ptc2019", "semeval2024"], help="corpus for masked-language model pretraining task", required=True)
    parser.add_argument("--train_features", type=str, help="path to extracted features file (JSON)", required=True)
    parser.add_argument("--test_features", type=str, help="path to extracted features file (JSON)", required=True)
    parser.add_argument("--dev_features", type=str, help="path to extracted features file (JSON). Currently not used", required=True)
    parser.add_argument("--max_iter", type=int, default=400, help="max iterations for ff classifier")
    parser.add_argument("--alpha", type=float, default=0.0001, help="weight of the L2 regularitation term")
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducibility")
    args = parser.parse_args()
    print("Arguments:", args)
    classify(args)