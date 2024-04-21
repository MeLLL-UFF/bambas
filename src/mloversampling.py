import pandas as pd
import numpy as np

def get_majority_label(y:np.ndarray)->int:
    pass

def irlbl(y:np.ndarray)->dict:
    """
    Calculates IRLbl(lambda) as proposed in tarekegn21

    y : Multi-hot encoded labels

    """
    # Count Label occurences and search for most frequent label
    label_occurrence = {}
    most_freq_label_idx = 0
    
    for idx, binary_label in enumerate(y.transpose()):
        curr_label_occurrence = np.count_nonzero(binary_label)
        label_occurrence[idx] = curr_label_occurrence
        if curr_label_occurrence > label_occurrence[most_freq_label_idx]:
            most_freq_label_idx = idx

    # Calculate IRLbl for every label
    label_irlbl = {}

    for idx, binary_label in enumerate(y.transpose()):
        curr_label_irlbl = label_occurrence[most_freq_label_idx] / label_occurrence[idx]
        label_irlbl[idx] = curr_label_irlbl
    
    return label_irlbl

def get_minority_labels(y)->list:
    irbl = irlbl(y)
    mean_irbl = np.mean(list(irbl))
    irbl_items = np.array(list(irbl.values()))
    print(mean_irbl, irbl_items)
    return irbl_items[irbl_items > mean_irbl]

def ml_ros(X, y)->tuple:
    get_minority_labels


if __name__ == "__main__":
    
    test_y = np.array([
        [1,0,0],
        [1,1,0],
        [1,1,0],
        [1,1,0],
        [0,0,1],
        [0,0,1]])

    print(get_minority_labels(test_y))