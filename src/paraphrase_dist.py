import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

ORIGINAL_FTS_PATH = "/home/arthur/Documents/Trab/NLP/bambas/feature_extraction_paraphrasing/paraphrasing_4f/test_features_array.json"
PARAPHRASE_FTS_PATH = "/home/arthur/Documents/Trab/NLP/bambas/feature_extraction_paraphrasing/paraphrasing_4f/train_features_array.json"

def load_features_info(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def load_features_array(path: str) -> np.ndarray:
    with open(path, "r") as f:
        df = pd.DataFrame().from_records(json.load(f))
        return np.array(df)
    
def euclidian_dist(a:np.ndarray, b:np.ndarray):
    return np.linalg.norm(a-b)

if __name__ == "__main__":
    original_fts = load_features_array(ORIGINAL_FTS_PATH)
    paraphrase_fts = load_features_array(PARAPHRASE_FTS_PATH)

    smote = SMOTE(sampling_strategy={0:len(original_fts)*2, 1:len(original_fts)})
    dummy_labels = [0]*len(original_fts)
    dummy_labels.extend([1]*len(original_fts))
    smote_fts,smote_lbls = smote.fit_resample(np.concatenate([original_fts, original_fts]), dummy_labels)
    smote_fts = smote_fts[:len(original_fts)]

    # Reduce dimensionality of features
    pca = PCA(copy=True, n_components=2)
    original_reduced = pd.DataFrame(pca.fit_transform(X=original_fts), columns=["X", "Y"])
    paraphrase_reduced = pd.DataFrame(pca.fit_transform(X=paraphrase_fts), columns=["X", "Y"])
    smote_reduced = pd.DataFrame(pca.fit_transform(X=smote_fts), columns=["X", "Y"])
    
    original_reduced["src"] = ["original"]*len(original_reduced)
    paraphrase_reduced["src"] = ["paraphrase"]*len(paraphrase_reduced)
    smote_reduced["src"] = ["smote"]*len(smote_reduced)

    data = pd.concat([
        original_reduced,
        paraphrase_reduced, 
        smote_reduced,
    ]) 

    # Create Scatterplot
    scatterplot = sns.scatterplot(data=data, x="X", y="Y", hue="src")
    import matplotlib.pyplot as plt
    plt.show()
    exit()

    # Calculate Distances
    # distances = [euclidian_dist(original, paraphrase) for original,paraphrase in zip(original_fts, paraphrase_fts)]
    # distances = cosine_similarity(original_fts, paraphrase_fts)
    # print("Distances between original and paraphrases")
    # print(distances)
    
    # distance_matrix =  []
    # fts_concat = np.concatenate([original_fts, paraphrase_fts])
    
    # for home_ft in fts_concat:
    #     matrix_line = []
    #     for out_ft in fts_concat:
    #         matrix_line.append(euclidian_dist(home_ft, out_ft))
    #     distance_matrix.append(matrix_line)
    
    # print(distance_matrix)

    # heatmap_axis = [f"original#{i+1}" for i in range(len(original_fts))]
    # heatmap_axis.extend([f"smote#{i+1}" for i in range(len(original_fts))])
    # heatmap_axis.extend([f"paraphrase#{i+1}" for i in range(len(original_fts))])

    # print(heatmap_axis)

    heatmap = sns.heatmap(distance_matrix, annot=False)

    plt.show()

