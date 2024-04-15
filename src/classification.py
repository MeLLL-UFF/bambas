#%%
import argparse
from argparse import Namespace
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from src.utils.br import BinaryRelevance
from src.utils.br import BinaryRelevance, add_internals, evaluate_per_label
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from typing import List, Dict, Any, Tuple
from src.data import load_dataset
from src.utils.workspace import get_workdir
from src.confusion_matrix import *
from src.subtask_1_2a import evaluate_h
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from functools import reduce
from src.subtask_1_2a import get_dag, get_dag_labels, get_dag_parents, get_leaf_parents
from imblearn.over_sampling import RandomOverSampler, SMOTE
from copy import deepcopy
from src.mlsmote import MLSMOTE

OUTPUT_DIR = f"{get_workdir()}/classification"
GOLD_DIR = f"{get_workdir()}/dataset/semeval2024/subtask1"


def append_dag_parents(leaves: List[str]) -> List[str]:
    labels = deepcopy(leaves)
    for leaf in leaves:
        parents = get_leaf_parents(leaf)
        labels = list(set(labels.extend(parents)))
    return labels


def remove_non_leaf_nodes(df: pd.DataFrame) -> pd.DataFrame:
    parents = get_dag_parents()

    def remove_parents(labels: List[str]) -> List[str]:
        return list(filter(lambda x: x not in parents, labels))

    df["labels"] = df["labels"].apply(remove_parents)
    return df


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


def load_features_info(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_features_array(path: str) -> np.ndarray:
    with open(path, "r") as f:
        df = pd.DataFrame().from_records(json.load(f))
        return np.array(df)


def classify(args: Namespace):
    print("Loading features info files")
    train_ft_info, dev_ft_info, test_ft_info = map(
        load_features_info, [
            args.train_features, args.dev_features, args.test_features])

    print("Loading dataset files")
    train, dev, test = load_dataset(args.dataset)
    print("Dataset Lengths", len(train), len(dev), len(test))

    if args.concat_train_dev:
        print("Concatenating train and dev sets")
        train = pd.concat([train, dev])
        print("Merged dataset length:", len(train))

    all_labels = pd.concat([train["labels"], dev["labels"]])

    if args.classifier == "HiMLP":
        labels = [get_dag_labels()]
    else:
        labels = [list(set(reduce(lambda x, y: x + y, all_labels.to_numpy().tolist())))]
        labels[0].sort()
    print(f"Labels: {labels}")
    print(f"No. of labels in {'DAG' if args.classifier == 'HiMLP' else 'train+dev datasets'}: {len(labels[0])}")

    # Transforming label lists into Multihot encoding
    mlb = MultiLabelBinarizer(classes=labels[0])
    train_labels = mlb.fit(labels).transform(train["labels"].to_numpy())
    if args.classifier != "HiMLP" and args.classifier != "MLP":
        dev_internals = add_internals(deepcopy(dev))
        dev_labels = mlb.fit(labels).transform(dev_internals["labels"].to_numpy())
    else:
        dev_labels = mlb.fit(labels).transform(dev["labels"].to_numpy())

    print("Loading features array files")
    train_ft, test_ft, dev_ft = map(
        load_features_array, [
            train_ft_info["features"], test_ft_info["features"], dev_ft_info["features"]])
    
    #%%
    # Using MLSMOTE to oversample the training data
    train_ft, train_labels = MLSMOTE(pd.DataFrame(train_ft), pd.DataFrame(train_labels), n_sample=1000)
    train_ft, train_labels = train_ft.to_numpy(), train_labels.to_numpy()

    print("Features Lengths", len(train_ft), len(test_ft), len(dev_ft))
    print(train_ft)
    print(type(train_labels))
    #%%
    if args.concat_train_dev:
        print("Concatenating train and dev features arrays")
        train_ft = np.concatenate((train_ft, dev_ft))
        print("Merged features array length:", train_ft.shape)

    # Loading oversamplers
    oversamplers = None
    if args.oversampling == "SMOTE":
        oversamplers = SMOTE(random_state=args.seed, sampling_strategy=args.sampling_strategy)
    elif args.oversampling == "RandomOverSampler":
        oversamplers = RandomOverSampler(random_state=args.seed, sampling_strategy=args.sampling_strategy)
    elif args.oversampling == "Combination":
        oversamplers = {
            "Simplification": None,
            "Bandwagon": None,
            "Ethos": None,
            "Reasoning": None,
            "Pathos": None,
            "Justification": None,
            "Ad Hominem": None,
            "Distraction": None,
            "Logos": None,
            "Flag-waving": SMOTE(sampling_strategy=0.5, random_state=args.seed),
            "Exaggeration/Minimisation": SMOTE(sampling_strategy=0.4, random_state=args.seed),
            "Glittering generalities (Virtue)": RandomOverSampler(sampling_strategy=0.7, random_state=args.seed),
            "Doubt": RandomOverSampler(sampling_strategy=0.5, random_state=args.seed),
            "Causal Oversimplification": RandomOverSampler(sampling_strategy=0.4, random_state=args.seed),
            "Slogans": RandomOverSampler(sampling_strategy=0.6, random_state=args.seed),
            "Appeal to authority": SMOTE(sampling_strategy=0.8, random_state=args.seed),
            "Thought-terminating cliché": RandomOverSampler(sampling_strategy=0.8, random_state=args.seed),
            "Name calling/Labeling": SMOTE(sampling_strategy=0.8, random_state=args.seed),
            "Repetition": RandomOverSampler(sampling_strategy=0.5, random_state=args.seed),
            "Smears": SMOTE(sampling_strategy=0.8, random_state=args.seed),
            "Reductio ad hitlerum": None,  # SMOTE(sampling_strategy=args.sampling_strategy, random_state=args.seed),
            "Misrepresentation of Someone's Position (Straw Man)": None,
            "Appeal to fear/prejudice": SMOTE(sampling_strategy=0.8, random_state=args.seed),
            "Black-and-white Fallacy/Dictatorship": SMOTE(sampling_strategy=0.4, random_state=args.seed),
            "Presenting Irrelevant Data (Red Herring)": RandomOverSampler(sampling_strategy=0.4, random_state=args.seed),
            "Obfuscation, Intentional vagueness, Confusion": None,
            "Loaded Language": SMOTE(sampling_strategy=0.9, random_state=args.seed),
            "Bandwagon": RandomOverSampler(sampling_strategy=0.4, random_state=args.seed),
            "Whataboutism": SMOTE(sampling_strategy=0.9, random_state=args.seed)
        }

    # Initializing Classifier
    if args.classifier == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=(768, ),
            random_state=args.seed,
            alpha=args.alpha,
            shuffle=True,
            early_stopping=True,
            verbose=True
        )
    elif args.classifier == "HiMLP":
        clf = HierarchicalClassifier(
            base_estimator=MLPClassifier(
                hidden_layer_sizes=(768, ),
                random_state=args.seed,
                alpha=args.alpha,
                shuffle=True,
                early_stopping=True,
                verbose=True
            ),
            class_hierarchy=get_dag(),
            # prediction_depth="nmlnp",
            # stopping_criteria=0.5,
            prediction_depth="mlnp",
            # Overcomes problem with dimension validation on sklearn_hierarchical_classification
            feature_extraction="raw",
            mlb=mlb,
            # no labels with prob lower than that will be considered for prediction
            mlb_prediction_threshold=0.35,
        )
    elif args.classifier == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier()  # BinaryRelevance(DecisionTreeClassifier(), labels=labels[0], oversampler=oversamplers)
    elif args.classifier == "ExtraTreeClassifier":
        clf = ExtraTreeClassifier()  # BinaryRelevance(ExtraTreeClassifier(), labels=labels[0], oversampler=oversamplers)
    elif args.classifier == "ExtraTreesClassifier":
        clf = ExtraTreesClassifier()  # BinaryRelevance(ExtraTreesClassifier(), labels=labels[0], oversampler=oversamplers)
    elif args.classifier == "KNeighborsClassifier":
        clf = KNeighborsClassifier()  # BinaryRelevance(KNeighborsClassifier(), labels=labels[0], oversampler=oversamplers)
    elif args.classifier == "MLPClassifier":
        clf = MLPClassifier()
    elif args.classifier == "RadiusNeighborsClassifier":
        # BinaryRelevance(RadiusNeighborsClassifier(), labels=labels[0], oversampler=oversamplers)
        clf = RadiusNeighborsClassifier()
    elif args.classifier == "RandomForestClassifier":
        clf = RandomForestClassifier()  # BinaryRelevance(RandomForestClassifier(), labels=labels[0], oversampler=oversamplers)
    elif args.classifier == "RidgeClassifier":
        clf = RidgeClassifier()  # BinaryRelevance(RidgeClassifier(), labels=labels[0], oversampler=oversamplers)
    elif args.classifier == "RidgeClassifierCV":
        clf = RidgeClassifierCV()
    elif args.classifier == "BRMLP":
        clf = BinaryRelevance(
            classifier=MLPClassifier(random_state=args.seed,
                                     max_iter=400),
            labels=labels[0],
            oversampler=oversamplers
        )
    elif args.classifier == "LogisticRegression":
        clf = BinaryRelevance(
            classifier=LogisticRegression(random_state=args.seed,
                                          max_iter=600,
                                          multi_class="multinomial"),
            labels=labels[0],
            oversampler= None # oversamplers
        )
    elif args.classifier == "GradientBoostingClassifier":
        clf = BinaryRelevance(
            classifier=GradientBoostingClassifier(random_state=args.seed),
            labels=labels[0],
            oversampler=oversamplers
        )
    elif args.classifier == "ClassifierChain":
        from sklearn.multioutput import ClassifierChain
        clf = ClassifierChain(base_estimator=LogisticRegression())
    else:
        raise Exception("Not implemented yet")

    # Check if label order is preserved in the binarizer
    clf = clf.fit(train_ft, train_labels)
    dev_predicted_labels_binarized = clf.predict(dev_ft)
    if args.classifier != "HiMLP" and args.classifier != "MLP":
        evaluate_per_label(dev_labels, dev_predicted_labels_binarized, labels[0])
    if args.classifier != "HiMLP":
        dev_predicted_labels = mlb.inverse_transform(dev_predicted_labels_binarized)
    else:
        dev_predicted_labels = dev_predicted_labels_binarized

    ts = int(time.time())
    pred_path, _ = save_predictions(dev, dev_predicted_labels, "dev", ts)

    # Create the Binary Relevance Confusion Matrix
    cf_mtx = ""
    if args.classifier != "HiMLP" and args.classifier != "MLP":
        cf_mtx = binary_relevance_confusion_matrix(np.array(dev_labels), np.array(dev_predicted_labels_binarized), labels[0])

    gold_file = (
        "dev_subtask1_en.json"
        if args.dataset == "semeval2024_test_unlabeled"
        else "validation.json"
    )

    gold_path = os.path.sep.join([GOLD_DIR, gold_file])

    prec, rec, f1 = evaluate_h(pred_path, gold_path)
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
                "Test Dataset",
                "Classifier",
                "F1",
                "Precision",
                "Recall",
                "Confusion Matrix",
                "Timestamp"])

    results.loc[len(results.index) + 1] = [train_ft_info["model"],
                                           train_ft_info["extraction_method"],
                                           train_ft_info["layers"],
                                           train_ft_info["layers_aggregation_method"],
                                           train_ft_info["dataset"],
                                           "dev_set",
                                           args.classifier,
                                           f1,
                                           prec,
                                           rec,
                                           cf_mtx,
                                           ts]
    results.to_csv(results_csv_path)

    print("\nPredicting for test file")
    test_predicted_labels = clf.predict(test_ft)
    if args.classifier != "HiMLP":
        test_predicted_labels = mlb.inverse_transform(test_predicted_labels)
    pred_path, _ = save_predictions(test, test_predicted_labels, "dev_unlabeled", ts)
    print(f"Finished successfully. dev_unlabeled predictions saved at {pred_path}")


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
            "semeval2024_all"],
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
    classify(args)
