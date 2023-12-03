import numpy as np
from sklearn.linear_model import LogisticRegression

class BinaryRelevance():
    def __init__(self, classifier, labels):
        self.classifier_list = [classifier]*len(labels)
        self.labels = labels
    
    def fit(self, X, y):
        num_labels = len(self.labels)
        for idx in range(num_labels):
            # print("training with:", y.transpose()[idx])
            self.classifier_list[idx] = self.classifier_list[idx].fit(X=X, y=y.transpose()[idx])
        return self

    def predict(self, X):
        num_labels = len(self.labels)
        preds = []
        for idx in range(num_labels):
            preds_for_label = self.classifier_list[idx].predict(X)
            # print("Preds for label: ", preds_for_label)
            preds.append(preds_for_label)
        
        return np.array(preds).transpose()

if __name__ == "__main__":
    br = BinaryRelevance(classifier=LogisticRegression(), labels=["A","B","C"])
    x = np.array([[12,9],[9,11], [12,12]])
    y = np.array([[1,0,0],
                  [1,0,0],
                  [0,1,1]])

    br.fit(x, y)
    print(br.predict(x))
