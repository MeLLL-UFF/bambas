import torch
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from transformers import AutoModel, AutoTokenizer
from feature_extraction.feature_extraction import load_data_split, extract_features
from feature_extraction.metrics import compute_scores
from typing import List

from config.config import get_config

# BATCH_SIZE = 32
# OUTPUT_CSV_PATH = "./results_2411.csv"
DATASET_DIR = "./dataset/"
# DEVICE = torch.device("cuda:0")#torch.device("cuda:1" if torch.cuda.is_available else "cpu")

def load_semeval2024() -> List[pd.DataFrame]:
    train = pd.read_csv(DATASET_DIR+"semeval2024/train.csv", sep=";").dropna(subset=["text"])[["text", "label"]]
    train = train.drop_duplicates(subset=["text"])
    test = pd.read_csv(DATASET_DIR+"semeval2024/validation.csv", sep=";").dropna(subset=["text"])[["text", "label"]]
    test = test.drop_duplicates(subset=["text"])
    return train, test

def load_ptc() -> List[pd.DataFrame]:
    train = pd.read_csv(DATASET_DIR+"ptc_adjust/ptc_preproc_train.csv", sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    train = train.drop_duplicates(subset=["text"])
    test = pd.read_csv(DATASET_DIR+"ptc_adjust/ptc_preproc_test.csv", sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    test = test.drop_duplicates(subset=["text"])
    return train, test

def feature_extraction_with_pretrained_model(model_name, train_dataset, test_dataset, device, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)
    model = AutoModel.from_pretrained(model_name).to(device)

    def load_data_split_for_tokenizer(dataset):
        return load_data_split(tokenizer, dataset, batch_size)

    train_loader, test_loader = map(load_data_split_for_tokenizer, [train_dataset, test_dataset])

    def extract_features_for_loader(data_loader):
        return extract_features(model, data_loader)

    train_features, test_features = map(extract_features_for_loader, [train_loader, test_loader])    
    train_labels = train_dataset["label"].fillna("None").str.split(",").to_numpy()
    test_labels = test_dataset["label"].fillna("None").str.split(",").to_numpy()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    labels_with_duplicates = np.hstack(np.concatenate((train_labels, test_labels), axis=None))
    labels = [list(set(labels_with_duplicates))]

    dataset = pd.concat([train_dataset, test_dataset])
    dataset = dataset.fillna("None")
    dataset = dataset["label"].str.split(",")

    dataset

    labels_set = []
    for labels in dataset: labels_set.extend(labels) 
    labels_set = list(set(labels_set))
    if "None" in labels_set: labels_set.remove("None")

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
    
    config = get_config("config/config.cfg")

    train, test = None, None
    if config["feature_extraction"]["dataset"] == "semeval2024":
        train,test = load_semeval2024()
    elif config["feature_extraction"]["dataset"] == "ptc":
        train,test = load_ptc()
    else:
        raise ValueError("Unsupported dataset in config value {0}".format(config["feature_extraction"]["dataset"]))

    df = pd.DataFrame()
    results = []
    # This will later turn into a list
    for model_name in [config["feature_extraction"]["language_model"]]:
        micro_f1, acc, prec, rec, cf_mtx = \
            feature_extraction_with_pretrained_model(model_name, 
                                                     train, test, 
                                                     config["main"]["device"], 
                                                     config["main"]["batch_size"])
        results.append(dict(
            micro_f1 = micro_f1,
            acc = acc,
            prec = prec,
            rec = rec,
            cf_mtx = cf_mtx,
            model_name = model_name,
        ))
    df = pd.DataFrame.from_records(results)
    df.to_csv(config["main"]["output_csv_path"])
    return None

if __name__ == "__main__":
    main()
