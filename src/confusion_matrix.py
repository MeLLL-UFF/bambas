import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def binary_relevance_confusion_matrix(Y_true: np.ndarray, Y_pred: np.ndarray, labels: list = None) -> str:
    res = []
    # if labels == None: TODO: Create placeholder labels
    for label, label_true, label_preds in zip(labels, Y_true.transpose(), Y_pred.transpose()):
        cf_mtx = confusion_matrix(label_true, label_preds)
        res.append((label, cf_mtx))
    return br_cf_mtx_to_str(res)


def br_cf_mtx_to_str(br_cf_mtx: list) -> str:
    str_res = ""
    for label, cf_mtx in br_cf_mtx:
        str_res += f"{label}\t"
        for n1, n2 in cf_mtx:
            str_res += f"{n1}\t{n2}\t"
    return str_res


def br_cf_mtx_to_display(br_cf_mtx: str) -> None:
    br_cf_mtx = br_cf_mtx.split("\t")[:-1]
    for idx in range(0, len(br_cf_mtx), 5):
        label, cf_mtx = br_cf_mtx[idx], np.array(br_cf_mtx[idx + 1:idx + 5]).astype(int)
        cf_mtx = np.array([
            [cf_mtx[0], cf_mtx[1]],
            [cf_mtx[2], cf_mtx[3]]
        ])
        display = ConfusionMatrixDisplay(confusion_matrix=cf_mtx)
        display.plot()
        plt.show()
        print(label, cf_mtx)


if __name__ == "__main__":
    Y_true = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [1, 1, 1]
    ])
    Y_pred = np.array([
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])
    labels = ["A", "B", "C"]

    br_cf_mtx = binary_relevance_confusion_matrix(Y_true, Y_pred, labels)
    print(br_cf_mtx)
