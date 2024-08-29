import json
import pandas as pd
from typing import List
from src.utils.workspace import get_workdir
from sklearn.model_selection import train_test_split

DATASET_DIR = f"{get_workdir()}/dataset"

def _load_semeval2024_augmented_smears() -> List[pd.DataFrame]:
    # with open(f"{DATASET_DIR}/augmented/augmented_chatgpt_2201.json", "r") as f:
    with open(f"{DATASET_DIR}/semeval2024/subtask1/train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    # with open(f"{DATASET_DIR}/augmented/augmented_chatgpt_2201.json", "r") as g:
    with open(f"{DATASET_DIR}/augmented/augmented_chatgpt_2601_br_Smears.json", "r") as g:
        train_augmented = pd.DataFrame().from_records(json.load(g))
        train = pd.concat([train, train_augmented])
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    return train, validation, dev_unlabeled


def _load_semeval2024_test() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2024/subtask1/train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/test_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    # with open(f"{DATASET_DIR}/semeval2024/subtask1/test_unlabeled.json", "r") as f:
    #     test_unlabeled = pd.DataFrame().from_records(json.load(f))
    return train, validation, dev_unlabeled  # , test_unlabeled


def _load_semeval2024() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2024/subtask1/train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    
    return train, validation, dev_unlabeled


def _load_semeval2024_dev_labeled() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2024/subtask1/train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_subtask1_en.json", "r") as f:
        dev_labeled = pd.DataFrame().from_records(json.load(f))
    return train, validation, dev_labeled


def _load_semeval2024_all() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2024/subtask1/train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_subtask1_en.json", "r") as f:
        dev_labeled = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/test_unlabeled.json", "r") as f:
        test_unlabeled = pd.DataFrame().from_records(json.load(f))
    return train, validation, dev_labeled, test_unlabeled


def _load_semeval2024_test_unlabeled() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2024/subtask1/train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
        train = pd.concat([train, validation])
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_subtask1_en.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/test_unlabeled.json", "r") as f:
        test_unlabeled = pd.DataFrame().from_records(json.load(f))
    return train, validation, test_unlabeled


def _load_semeval_augmented() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval_augmented/train_aug_ptc_reductio.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/test_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    # with open(f"{DATASET_DIR}/semeval2024/subtask1/test_unlabeled.json", "r") as f:
    #     test_unlabeled = pd.DataFrame().from_records(json.load(f))
    return train, validation, dev_unlabeled  # , test_unlabeled


def _load_semeval_internal() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval_internal/train_internal.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/test_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    # with open(f"{DATASET_DIR}/semeval2024/subtask1/test_unlabeled.json", "r") as f:
    #     test_unlabeled = pd.DataFrame().from_records(json.load(f))
    return train, validation, dev_unlabeled  # , test_unlabeled


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

def _load_paraphrase() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/paraphrasing/semeval_binarized_paraphrased.json", "r") as f:
    # with open(f"{DATASET_DIR}/paraphrasing/semeval_binarized_paraphrased.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    print("check to see if data was loaded successfuly: ")
    print(train.columns)
    print(validation.columns)
    return train, validation, dev_unlabeled

def _load_paraphrase_outliers() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/paraphrasing/semeval_outliers.json", "r") as f:
    # with open(f"{DATASET_DIR}/paraphrasing/semeval_binarized_paraphrased.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    print("check to see if data was loaded successfuly: ")
    print(train.columns)
    print(validation.columns)
    return train, validation, dev_unlabeled

def _load_paraphrase_outsiders() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/paraphrasing/semeval2024_paraphrased_outsiders.json", "r") as f:
    # with open(f"{DATASET_DIR}/paraphrasing/semeval_binarized_paraphrased.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    print("check to see if data was loaded successfuly: ")
    print(train.columns)
    print(validation.columns)
    return train, validation, dev_unlabeled

def _load_paraphrase_4() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/paraphrasing/semeval_binarized_paraphrased_4.json", "r") as f:
    # with open(f"{DATASET_DIR}/paraphrasing/semeval_binarized_paraphrased.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/validation.json", "r") as f:
        validation = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2024/subtask1/dev_unlabeled.json", "r") as f:
        dev_unlabeled = pd.DataFrame().from_records(json.load(f))
    print("check to see if data was loaded successfuly: ")
    print(train.columns)
    print(validation.columns)
    return train, validation, dev_unlabeled

def _load_positive_examples() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2024/subtask1/train.json", "r") as f:
        examples = pd.DataFrame().from_records(json.load(f))

    def binarize(labelset):
        labelset=str(labelset)
        if labelset=="[]":return 0
        else: return 1
    examples["labels"]=[binarize(labelset) for labelset in examples["labels"]]
    positive = examples[examples["labels"]==1]
    return positive, positive, positive

def _load_paraphrase_dict() -> List[pd.DataFrame]:
    paraphrase = pd.read_csv(f"/home/arthur/Documents/Trab/NLP/bambas/dataset/paraphrase_csvs/negative_paraphrasing_4f.csv")
    paraphrase["labels"] = [[] for _ in range(len(paraphrase))]
    train = paraphrase[["paraphrase","labels"]].rename({"paraphrase":"text"}, axis=1)
    test = paraphrase[["original_text","labels"]].rename({"original_text":"text"}, axis=1)
    return train, test, test

def _load_semeval2015() -> List[pd.DataFrame]:
    dataset = pd.read_csv(f"{DATASET_DIR}/semeval2015/SemEval15.csv")
    dataset = dataset.rename(columns={"tweet":"text", "class":"labels"})
    train, test = train_test_split(dataset, test_size=0.3, random_state=1, stratify=dataset["labels"])
    print(train.columns)
    train[train["labels"]=="positive"]="[positive]"
    train[train["labels"]=="negative"]="[]"
    test[test["labels"]=="positive"]="[positive]"
    test[test["labels"]=="negative"]="[]"
    print("Positive examples: ", len(train[train["labels"]=="[positive]"]))
    print("Negative examples: ", len(train[train["labels"]=="[]"]))
    return train, test, test

def _load_semeval2015_2() -> List[pd.DataFrame]:
    dataset = pd.read_csv(f"{DATASET_DIR}/semeval2015/SemEval15.csv")
    dataset = dataset.rename(columns={"tweet":"text", "class":"labels"})
    train, test = train_test_split(dataset, test_size=0.3, random_state=2, stratify=dataset["labels"])
    print(train.columns)
    train[train["labels"]=="positive"]="[positive]"
    train[train["labels"]=="negative"]="[]"
    test[test["labels"]=="positive"]="[positive]"
    test[test["labels"]=="negative"]="[]"
    print("Positive examples: ", len(train[train["labels"]=="[positive]"]))
    print("Negative examples: ", len(train[train["labels"]=="[]"]))
    
    return train, test, test

def _load_semeval2015_paraphrased() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2015_paraphrased/semeval2015_paraphrased_1to1_train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2015_paraphrased/semeval2015_paraphrased_1to1_test.json", "r") as f:
        test = pd.DataFrame().from_records(json.load(f))
    return train, test, test

def _load_semeval2015_paraphrased2() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2015_paraphrased/semeval2015_paraphrased2_train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2015_paraphrased/semeval2015_paraphrased2_test.json", "r") as f:
        test = pd.DataFrame().from_records(json.load(f))
    return train, test, test

def _load_semeval2015_paraphrased2_1to4_selected() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2015_paraphrased/semeval2015_paraphrased2_1to4_selected_train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2015_paraphrased/semeval2015_paraphrased2_1to4_selected_test.json", "r") as f:
        test = pd.DataFrame().from_records(json.load(f))
    return train, test, test

def _load_semeval2015_paraphrased2_1to4_selected2() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2015_paraphrased/semeval2015_paraphrased2_1to4_selected2_train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2015_paraphrased/semeval2015_paraphrased2_1to4_selected2_test.json", "r") as f:
        test = pd.DataFrame().from_records(json.load(f))
    return train, test, test

def _load_semeval2016() -> List[pd.DataFrame]:
    dataset = pd.read_csv(f"{DATASET_DIR}/semeval2016/SemEval16.csv")
    dataset = dataset.rename(columns={"tweet":"text", "class":"labels"})
    train, test = train_test_split(dataset, test_size=0.3, random_state=1, stratify=dataset["labels"])
    print(train.columns)
    return train, test, test

def _load_semeval2016_paraphrased() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2016_paraphrased/semeval2016_paraphrased_train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2016_paraphrased/semeval2016_paraphrased_test.json", "r") as f:
        test = pd.DataFrame().from_records(json.load(f))
    return train, test, test

def _load_semeval2016_paraphrased_1to4() -> List[pd.DataFrame]:
    with open(f"{DATASET_DIR}/semeval2016_paraphrased/semeval2016_paraphrased_1to4_train.json", "r") as f:
        train = pd.DataFrame().from_records(json.load(f))
    with open(f"{DATASET_DIR}/semeval2016_paraphrased/semeval2016_paraphrased_1to4_test.json", "r") as f:
        test = pd.DataFrame().from_records(json.load(f))
    return train, test, test


def load_dataset(dataset: str) -> List[pd.DataFrame]:
    """Load a given dataset, returning splits

    Parameters:
    -----------
    dataset : str
              name of dataset

    Returns:
    --------
    List[pd.DataFrame]
              train, test and dev splits. They include the columns `text` (str) and `labels` (List[str])
    """
    if dataset == "semeval2024":
        return _load_semeval2024()
    elif dataset == "semeval2024_test_set_no_concat":
        return _load_semeval2024_test()
    elif dataset == "semeval2024_dev_set_labeled":
        return _load_semeval2024_dev_labeled()
    elif dataset == "semeval2024_test_set_unlabeled":
        return _load_semeval2024_test_unlabeled()
    elif dataset == "semeval2024_augmented_smears":
        return _load_semeval2024_augmented_smears()
    elif dataset == "semeval2024_all":
        return _load_semeval2024_all()
    elif dataset == "ptc2019":
        return _load_ptc2019()
    elif dataset == "semeval_augmented":
        return _load_semeval_augmented()
    elif dataset == "paraphrase":
        return _load_paraphrase()
    elif dataset == "paraphrase4":
        return _load_paraphrase_4()
    elif dataset == "semeval_internal":
        return _load_semeval_internal()
    elif dataset == "paraphrase_dict":
        return _load_paraphrase_dict()
    elif dataset == "positive":
        return _load_positive_examples()
    elif dataset == "outliers":
        return _load_paraphrase_outliers()
    elif dataset == "outsiders":
        return _load_paraphrase_outsiders()
    elif dataset == "semeval2015":
        return _load_semeval2015()
    elif dataset == "semeval2015_2":
        return _load_semeval2015_2()
    elif dataset == "semeval2015_paraphrased":
        return _load_semeval2015_paraphrased()
    elif dataset == "semeval2015_paraphrased2":
        return _load_semeval2015_paraphrased2()
    elif dataset == "semeval2015_paraphrased2_1to4_selected":
        return _load_semeval2015_paraphrased2_1to4_selected()
    elif dataset == "semeval2015_paraphrased2_1to4_selected2":
        return _load_semeval2015_paraphrased2_1to4_selected2()
    elif dataset == "semeval2016":
        return _load_semeval2016()
    elif dataset == "semeval2016_paraphrased":
        return _load_semeval2016_paraphrased()
    elif dataset == "semeval2016_paraphrased_1to4":
        return _load_semeval2016_paraphrased_1to4()
    raise Exception(f"{dataset} is not available")
