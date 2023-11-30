import json
import pandas as pd
from functools import reduce

def main():
    with open("./dataset/semeval2024/subtask1/train.json", "r") as f:
        train_original = pd.DataFrame().from_records(json.load(f))
    with open("./dataset/semeval2024/subtask1/dev_unlabeled.json", "r") as f:
        dev_unlabeled_original = pd.DataFrame().from_records(json.load(f))
    with open("./dataset/semeval2024/subtask1/validation.json", "r") as f:
        validation_original = pd.DataFrame().from_records(json.load(f))
    
    print(train_original.head())
    # print(train_original["text"].astype("string").head())
    print(dev_unlabeled_original.head())
    print(validation_original.head())

    labels = list(set(reduce(lambda x, y: x+y, train_original["labels"].to_list())))
    print(labels, len(labels))

    # train = pd.read_csv("./dataset/ptc_adjust/ptc_preproc_train.csv",
    #                     sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    # train = train.drop_duplicates(subset=["text"])
    # test = pd.read_csv("./dataset/ptc_adjust/ptc_preproc_test.csv",
    #                    sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    # test = test.drop_duplicates(subset=["text"])

    # print(train.head())
    # print(test.head())

if __name__ == "__main__":
    main()