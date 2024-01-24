import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler, SMOTE
from typing import Union
from src.utils.workspace import get_workdir
import os

OUTPUT_DIR = f"{get_workdir()}/classification/"

class BinaryRelevance():
    def __init__(self, classifier, labels, oversampler:Union[RandomOverSampler,SMOTE,None]=None):
        self.classifier_list = [deepcopy(classifier) for _ in range(len(labels))]
        print(self.classifier_list)
        self.labels = labels
        self.oversampler=oversampler
    
    def fit(self, X, y):

        num_labels = len(self.labels)
        for idx in range(num_labels):
            X_, y_ = X, y
            y_ = y.transpose()[idx]
            print("training with:", idx, "element")

            if self.oversampler == None: pass
            else: X_, y_ = self.oversampler.fit_resample(X_,y_)

            self.classifier_list[idx] = self.classifier_list[idx].fit(X=X_, y=y_)
        return self

    def predict(self, X, Y_true:np.ndarray=np.ndarray([])):
        num_labels = len(self.labels)
        preds = []
        if Y_true.shape != ():
            # Create the BR per label results if they havent been created yet
            if not os.path.exists(OUTPUT_DIR+"per_label_results.csv"):
                with open(OUTPUT_DIR+"per_label_results.csv", mode="w") as file: 
                    file.write("experiment,label,precision,recall,f1\n")
                    print("Creating per_label_results.txt file...")
            with open(OUTPUT_DIR+"per_label_results.csv", mode="r") as file: 
                lines = len(file.readlines())
                lines -= 1
                experiment_id = int(lines / 20) +1
        for idx in range(num_labels):
            preds_for_label = self.classifier_list[idx].predict(X)
            preds.append(preds_for_label)
            if Y_true.shape == ():
                continue
            
            precision, recall, f1, _ = precision_recall_fscore_support(Y_true.transpose()[idx], preds_for_label, average="micro")
            with open(OUTPUT_DIR+"per_label_results.csv", mode="a") as file: file.write(f"{experiment_id},{self.labels[idx]},{precision},{recall},{f1}\n")

        return np.array(preds).transpose()

if __name__ == "__main__":
    br = BinaryRelevance(classifier=LogisticRegression(), labels=["A","B","C"])
    x = np.array([[12,9, 0, 700],[1,-1,-50, 0.1], [12,12,64, 15555]])
    y = np.array([[1,0,1],
                  [1,1,0],
                  [0,1,1]])

    br.fit(x, y)
    print(br.predict(x), y)
