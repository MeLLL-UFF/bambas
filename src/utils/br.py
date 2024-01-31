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
                experiment_id = int(lines / 23) +1
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
            with open(OUTPUT_DIR+"per_label_results.csv", mode="a") as file: file.write(f"{experiment_id},{self.labels[idx]},{precision},{recall},{f1}\n")

        return np.array(preds).transpose()

if __name__ == "__main__":
    br = BinaryRelevance(classifier=LogisticRegression(), labels=["A","B","C"])
    x = np.array([[12,9, 0, 700],[1,-1,-50, 0.1], [12,12,64, 15555]])
    y = np.array([[1,0,1],
                  [1,1,0],
                  [0,1,1]])

    br.fit(x, y)
    br.predict(x, y)
