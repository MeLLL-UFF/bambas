import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler, SMOTE
from typing import Union
from src.utils.workspace import get_workdir
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import ClassifierChain
import os

OUTPUT_DIR = f"{get_workdir()}/classification/"

HIERARCHY = {
    "root": ["Logos", "Ethos", "Pathos"],
    "Logos": ["Repetition", "Obfuscation, Intentional vagueness, Confusion", "Reasoning", "Justification"],
    "Justification": [
        "Slogans",
        "Bandwagon",
        "Appeal to authority",
        "Flag-waving",
        "Appeal to fear/prejudice",
    ],
    "Reasoning": [
        "Simplification",
        "Distraction",
    ],
    "Simplification": [
        "Causal Oversimplification",
        "Black-and-white Fallacy/Dictatorship",
        "Thought-terminating cliché",
    ],
    "Distraction": [
        "Misrepresentation of Someone's Position (Straw Man)",
        "Presenting Irrelevant Data (Red Herring)",
        "Whataboutism",
    ],
    "Ethos": [
        "Appeal to authority",
        "Glittering generalities (Virtue)",
        "Bandwagon",
        "Ad Hominem",
        "Transfer",
    ],
    "Ad Hominem": [
        "Doubt",
        "Name calling/Labeling",
        "Smears",
        "Reductio ad hitlerum",
        "Whataboutism",
    ],
    "Pathos": [
        "Exaggeration/Minimisation",
        "Loaded Language",
        "Appeal to (Strong) Emotions",
        "Appeal to fear/prejudice",
        "Flag-waving",
        "Transfer",
    ]
}

def search_parents(label, hier=HIERARCHY, parents=[]):
    parents = []
    if label=="root":return []
    for key in hier:
        value = hier[key]
        if key=="root":continue
        if label in value: 
            parents.append(key)
            parents.extend(search_parents(key, hier, parents))
    return parents

def add_internals(dataset):
    for idx, label_list in enumerate(dataset["labels"]):
        for label in label_list:
            parents = search_parents(label)
            if "Ad Hominem" in parents: label_list.append("Ad Hominem")
            if "Distraction" in parents: label_list.append("Distraction")
            if "Logos" in parents: label_list.append("Logos")
            # label_list.extend(parents)
        dataset["labels"][idx] = label_list
    return dataset

def add_internals_preds(preds):
    for idx, label_list in enumerate(preds):
        for label in label_list:
            parents = search_parents(label)
            # if "Ad Hominem" in parents: label_list.append("Ad Hominem")
            # if "Distraction" in parents: label_list.append("Distraction")
            # if "Logos" in parents: label_list.append("Logos")
            label_list.extend(parents)
        preds[idx] = label_list
    return preds

def evaluate_per_label(y_true:np.ndarray, y_pred:np.ndarray, labels:list):
    num_labels = y_true.shape[1]

    # Create the BR per label results if they havent been created yet
    if not os.path.exists(OUTPUT_DIR+"per_label_results.csv"):
        with open(OUTPUT_DIR+"per_label_results.csv", mode="w") as file: 
            file.write("experiment,label,precision,recall,f1\n")
            print("Creating per_label_results.txt file...")
    with open(OUTPUT_DIR+"per_label_results.csv", mode="r") as file: 
        last_line = file.readlines()[-1]
        last_experiment_id = last_line.split(",")[0]
        if last_experiment_id == "experiment":experiment_id = 1
        else: experiment_id = int(last_experiment_id)+1
    
    # Evaluate prediction for each label
    for idx in range(num_labels):
        preds_for_label = y_pred.transpose()[idx]
        precision, recall, f1, _ = precision_recall_fscore_support(y_true.transpose()[idx], preds_for_label, zero_division=0)
        _, precision = precision
        _, recall = recall
        _, f1 = f1
        with open(OUTPUT_DIR+"per_label_results.csv", mode="a") as file: file.write(f"{experiment_id},\"{labels[idx]}\",{precision},{recall},{f1}\n")

# WIP for injecting oversamplers in ClassifierChain
class ClassifierChainWrapper():
    def __init__(self, classifier, labels, oversampler):
        self.chains = []



class BinaryRelevance():
    def __init__(self, classifier, labels, oversampler:Union[RandomOverSampler,SMOTE,dict , None]=None):
        self.classifier_list = [deepcopy(classifier) for _ in range(len(labels))]
        self.labels = labels

        # If there is a list of oversamplers
        if type(oversampler) == dict: self.oversamplers = oversampler
        else: self.oversamplers = {label:oversampler for label in labels}
    
    def fit(self, X, y):
        num_labels = len(self.labels)
        for idx in range(num_labels):
            x_, y_ = X, y
            y_ = y.transpose()[idx]
            print(f"Training binary classifier for label: {self.labels[idx]} ({idx+1}/{num_labels})")

            if self.oversamplers == None or self.oversamplers[self.labels[idx]] == None: pass
            else: x_, y_ = self.oversamplers[self.labels[idx]].fit_resample(x_,y_)

            self.classifier_list[idx] = self.classifier_list[idx].fit(X=x_, y=y_)
        return self

    

    def predict(self, X, Y_true:np.ndarray=np.ndarray([])):
        num_labels = len(self.labels)
        preds = []

        # Create the BR per label results if they havent been created yet
        if Y_true.shape != ():
            if not os.path.exists(OUTPUT_DIR+"per_label_results.csv"):
                with open(OUTPUT_DIR+"per_label_results.csv", mode="w") as file: 
                    file.write("experiment,label,precision,recall,f1\n")
                    print("Creating per_label_results.txt file...")
            with open(OUTPUT_DIR+"per_label_results.csv", mode="r") as file: 
                lines = len(file.readlines())
                lines -= 1
                experiment_id = int(lines / num_labels) +1
        # Predict for each label
        for idx in range(num_labels):
            preds_for_label = self.classifier_list[idx].predict(X)
            preds.append(preds_for_label)
            if Y_true.shape == ():
                continue
            precision, recall, f1, _ = precision_recall_fscore_support(Y_true.transpose()[idx], preds_for_label)
            _, precision = precision
            _, recall = recall
            _, f1 = f1
            # print("Confusion Matrix for", self.labels[idx])
            # print(confusion_matrix(Y_true.transpose()[idx], preds_for_label))
            # print("Precison, Recall, F1")
            # print(f"{precision},{recall},{f1}")
            with open(OUTPUT_DIR+"per_label_results.csv", mode="a") as file: file.write(f"{experiment_id},\"{self.labels[idx]}\",{precision},{recall},{f1}\n")

        return np.array(preds).transpose()

if __name__ == "__main__":
    labels = ["A", "B", "C"]
    br = BinaryRelevance(classifier=LogisticRegression(), labels=labels)
    x = np.array([[12,9, 0, 700],[1,-1,-50, 0.1], [12,12,64, 15555]])
    y = np.array([[1,0,1],
                  [1,1,0],
                  [0,1,1]])

    br.fit(x, y)
    preds = br.predict(x)
    evaluate_per_label(preds, y, labels)
