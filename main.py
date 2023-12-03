import torch
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from transformers import AutoModel, AutoTokenizer
from src.feature_extraction import load_data_split, extract_features
from src.metrics import compute_scores
from typing import List

from config.config import get_config

BATCH_SIZE = 32
OUTPUT_CSV_PATH = "./results_2411.csv"
DATASET_DIR = "./dataset/"
DEVICE = torch.device("cuda:0")  # torch.device("cuda:1" if torch.cuda.is_available else "cpu")


def load_semeval2024() -> List[pd.DataFrame]:
    train = pd.read_csv(DATASET_DIR + "semeval2024/train.csv", sep=";").dropna(subset=["text"])[["text", "label"]]
    train = train.drop_duplicates(subset=["text"])
    test = pd.read_csv(DATASET_DIR + "semeval2024/validation.csv", sep=";").dropna(subset=["text"])[["text", "label"]]
    test = test.drop_duplicates(subset=["text"])
    return train, test


def load_ptc() -> List[pd.DataFrame]:
    train = pd.read_csv(DATASET_DIR + "ptc_adjust/ptc_preproc_train.csv",
                        sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    train = train.drop_duplicates(subset=["text"])
    test = pd.read_csv(DATASET_DIR + "ptc_adjust/ptc_preproc_test.csv",
                       sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    test = test.drop_duplicates(subset=["text"])
    return train, test

def feature_extraction_with_pretrained_model(model_name, train_dataset, test_dataset):
    
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)

    def load_data_split_for_tokenizer(dataset):
        return load_data_split(tokenizer, dataset, BATCH_SIZE)

    train_loader, test_loader = map(load_data_split_for_tokenizer, [train_dataset, test_dataset])

    def extract_features_for_loader(data_loader):
        return extract_features(model, data_loader)

    def extract_layer_features_for_loader(data_loader):
        return extract_features(model, data_loader, use_cls=False, layers_to_extract=[
                                4, 5, 6, 7], layer_aggregation_method="concatenate")

    train_features, test_features = map(extract_features_for_loader, [train_loader, test_loader])
    # train_features, test_features = map(extract_layer_features_for_loader, [train_loader, test_loader])
    train_labels = train_dataset["label"].fillna("None").str.split(",").to_numpy()
    test_labels = test_dataset["label"].fillna("None").str.split(",").to_numpy()

    del tokenizer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    dataset = pd.concat([train_dataset, test_dataset])
    dataset = dataset.fillna("None")
    dataset = dataset["label"].str.split(",")

    dataset

    labels_set = []
    for labels in dataset:
        labels_set.extend(labels)
    labels_set = list(set(labels_set))
    if "None" in labels_set:
        labels_set.remove("None")

    mlb = MultiLabelBinarizer(classes=labels_set)
    train_labels_binarized = mlb.fit_transform(train_labels)
    test_labels_binarized = mlb.fit_transform(test_labels)

    ff = MLPClassifier(
        random_state=1,
        max_iter=400,
        alpha=0.0001,
        shuffle=True,
        early_stopping=True,
        verbose=True
    ).fit(train_features, train_labels_binarized)

    test_predicted_labels_binarized = ff.predict(test_features)

    micro_f1, acc, prec, rec, cf_mtx = compute_scores(test_labels_binarized, test_predicted_labels_binarized)
    return micro_f1, acc, prec, rec, cf_mtx


def main():
    train, test = load_semeval2024()
    df = pd.DataFrame()
    results = []
    for model_name in ["xlm-roberta-base"]:
        micro_f1, acc, prec, rec, cf_mtx = feature_extraction_with_pretrained_model(model_name, train, test)
        results.append(dict(
            micro_f1=micro_f1,
            acc=acc,
            prec=prec,
            rec=rec,
            cf_mtx=cf_mtx,
            model_name=model_name,
        ))
    df = pd.DataFrame.from_records(results)
    df.to_csv(OUTPUT_CSV_PATH)
    return None


if __name__ == "__main__":
    main()
