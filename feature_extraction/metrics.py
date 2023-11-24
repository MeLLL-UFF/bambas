from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, multilabel_confusion_matrix

def compute_scores(test_labels_binarized, test_predicted_labels_binarized):
    micro_f1 = f1_score(test_labels_binarized, test_predicted_labels_binarized, average="micro")
    acc = accuracy_score(test_labels_binarized, test_predicted_labels_binarized)
    prec = precision_score(test_labels_binarized, test_predicted_labels_binarized, average="micro")
    rec = recall_score(test_labels_binarized, test_predicted_labels_binarized, average="micro")
    cf_mtx = multilabel_confusion_matrix(test_labels_binarized, test_predicted_labels_binarized)
    return micro_f1, acc, prec, rec, cf_mtx
