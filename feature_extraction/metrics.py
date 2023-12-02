import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, multilabel_confusion_matrix, confusion_matrix

def multi_cf_mtx(labels_set, y_true, y_preds):
    cf_mtx_list = ""
    for idx, label in enumerate(labels_set):
        cf_mtx = confusion_matrix(y_true.transpose()[idx], y_preds.transpose()[idx])
        cf_mtx = cf_mtx.flatten()        
        cf_mtx_list += f"{label};{cf_mtx[0]};{cf_mtx[1]};{cf_mtx[2]};{cf_mtx[3]},"
    return cf_mtx_list

def compute_scores(labels_set, test_labels_binarized, test_predicted_labels_binarized):
    micro_f1 = f1_score(test_labels_binarized, test_predicted_labels_binarized, average="micro")
    acc = accuracy_score(test_labels_binarized, test_predicted_labels_binarized)
    prec = precision_score(test_labels_binarized, test_predicted_labels_binarized, average="micro")
    rec = recall_score(test_labels_binarized, test_predicted_labels_binarized, average="micro")
    cf_mtx = multi_cf_mtx(labels_set, test_labels_binarized, test_predicted_labels_binarized)
    return micro_f1, acc, prec, rec, cf_mtx

if __name__ == "__main__":
    labels_set = ["A", "B", "C"]
    
    y_true = np.array([[1,0,0],[0,1,1],[1,0,0]])
    y_preds = np.array([[0,0,0],[0,1,0],[1,0,0]])

    res = multi_cf_mtx(labels_set, y_true, y_preds)
    print(res)
