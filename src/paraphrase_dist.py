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

def load_features_info(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def load_features_array(path: str) -> np.ndarray:
    with open(path, "r") as f:
        df = pd.DataFrame().from_records(json.load(f))
        return np.array(df)
    
# TODO: Calculate distance between original and paraphrasing