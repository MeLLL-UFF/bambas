import torch
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from feature_extraction.feature_extraction import load_data_split, extract_features
from feature_extraction.metrics import compute_scores

BATCH_SIZE = 64
OUTPUT_CSV_PATH = "./results.csv"
PTC_DATASET_DIR = "./dataset/"

def load_ptc() -> List[pd.DataFrame]:
    train = pd.read_csv(PTC_DATASET_DIR+"ptc_preproc_train.csv", sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    train = train.drop_duplicates(subset=["text"])
    test = pd.read_csv(PTC_DATASET_DIR+"ptc_preproc_test.csv", sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    test = test.drop_duplicates(subset=["text"])
    return train, test

def feature_extraction_with_pretrained_model(model_name, train_dataset, test_dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)
    model = AutoModel.from_pretrained(model_name)

    def load_data_split_for_tokenizer(dataset):
        return load_data_split(tokenizer, dataset, BATCH_SIZE)

    train_loader, test_loader = map(load_data_split_for_tokenizer, [train_dataset, test_dataset])

    def extract_features_for_loader(data_loader):
        return extract_features(model, data_loader)

    train_features, test_features = map(extract_features_for_loader, [train_loader, test_loader])    
    train_labels, test_labels = train_dataset["label"].str.split(",").to_numpy(), test_dataset["label"].str.split(",").to_numpy()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    labels_with_duplicates = np.hstack(np.concatenate((train_labels, test_labels), axis=None))
    labels = [list(set(labels_with_duplicates))]

    mlb = MultiLabelBinarizer()
    train_labels_binarized = mlb.fit(labels).transform(train_labels)
    test_labels_binarized = mlb.transform(test_labels)

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
    train, test = load_ptc()
    df = pd.DataFrame()
    results = []
    for model_name in ["xlm-roberta-base"]:
        micro_f1, acc, prec, rec, cf_mtx = feature_extraction_with_pretrained_model(model_name, train, test)
        results.append(dict(
            micro_f1 = micro_f1,
            acc = acc,
            prec = prec,
            rec = rec,
            cf_mtx = cf_mtx,
            model_name = model_name,
        ))
    df = pd.DataFrame.from_records(results)
    df.to_csv(OUTPUT_CSV_PATH)
    return None

if __name__ == "main":
    main()
