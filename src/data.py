import json
import pandas as pd
from typing import List
from src.utils.workspace import get_workdir

DATASET_DIR = f"{get_workdir()}/dataset"


def _load_semeval2024() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2024/subtask1/train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    return train, validation, dev_unlabeled

def _load_semeval_augmented() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval_augmented/train_aug_ptc_reductio.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    return train, validation, dev_unlabeled

def _load_semeval_internal() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval_internal/train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    return train, validation, dev_unlabeled


def _load_ptc2019() -> List[pd.DataFrame]:
    train = pd.read_csv(f"{DATASET_DIR}/ptc_adjust/ptc_preproc_train.csv",
                        sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    train = train.drop_duplicates(subset=["text"])
    test = pd.read_csv(f"{DATASET_DIR}/ptc_adjust/ptc_preproc_test.csv",
                       sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    test = test.drop_duplicates(subset=["text"])
    dev = pd.read_csv(f"{DATASET_DIR}/ptc_adjust/ptc_preproc_dev.csv",
                      sep=";").dropna(subset=["text", "label"])[["text", "label"]]
    dev = dev.drop_duplicates(subset=["text"])

    def rename_label_column(df: pd.DataFrame) -> pd.DataFrame:
        df["labels"] = df["label"].str.split(",")
        df = df.drop(columns=["label"])
        return df

    train, test, dev = map(rename_label_column, [train, test, dev])
    return train, test, dev


def load_dataset(dataset: str) -> List[pd.DataFrame]:
    """Load a given dataset, returning splits

    Parameters:
    -----------
    dataset : str
              name of dataset, currently only 'ptc2019' or 'semeval2024' are supported

    Returns:
    --------
    List[pd.DataFrame]
              train, test and dev splits. They include the columns `text` (str) and `labels` (List[str])
    """
    if dataset == "semeval2024":
        return _load_semeval2024()
    elif dataset == "ptc2019":
        return _load_ptc2019()
    elif dataset == "semeval_augmented":
        return _load_semeval_augmented()
    elif dataset == "semeval_internal":
        return _load_semeval_internal()
    raise Exception(f"{dataset} is not available")
