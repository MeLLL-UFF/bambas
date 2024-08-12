import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

ORIGINAL_FTS_PATH = "/home/arthur/Documents/Trab/NLP/bambas/feature_extraction_paraphrasing/paraphrasing_1f/test_features_array.json"
POSITIVE_FTS_PATH = "/home/arthur/Documents/Trab/NLP/bambas/feature_extraction/positive/test_features_array.json"
PARAPHRASE_FTS_PATH = "/home/arthur/Documents/Trab/NLP/bambas/feature_extraction_paraphrasing/paraphrasing_1f/train_features_array.json"
OUTLIER_FTS_PATH = "/home/arthur/Documents/Trab/NLP/bambas/feature_extraction/outliers/train_features_array.json"
DATASET_PATH = "/home/arthur/Documents/Trab/NLP/bambas/dataset/paraphrase_csvs/negative_paraphrasing.csv"

def load_features_info(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def load_features_array(path: str) -> np.ndarray:
    with open(path, "r") as f:
        df = pd.DataFrame().from_records(json.load(f))
        return np.array(df)
    
def euclidian_dist(a:np.ndarray, b:np.ndarray):
    return np.linalg.norm(a-b)

def get_quadrant(dataset:pd.DataFrame, filter:pd.DataFrame):
    upper = dataset[filter["Y"] > 10]
    under = dataset[filter["Y"] < 0]
    return pd.concat([upper, under])

if __name__ == "__main__":

    dataset = pd.read_csv(DATASET_PATH).drop(columns=["Unnamed: 0"])

    negative_fts = load_features_array(ORIGINAL_FTS_PATH)
    positive_fts = load_features_array(POSITIVE_FTS_PATH)
    paraphrase_fts = load_features_array(PARAPHRASE_FTS_PATH)
    outlier_fts = load_features_array(OUTLIER_FTS_PATH)

    smote = SMOTE(sampling_strategy={0:len(negative_fts)*2, 1:len(negative_fts)}, random_state=1)
    dummy_labels = [0]*len(negative_fts)
    dummy_labels.extend([1]*len(negative_fts))
    smote_fts,smote_lbls = smote.fit_resample(np.concatenate([negative_fts, negative_fts]), dummy_labels)
    smote_fts = smote_fts[len(negative_fts)*2:]

    # for i in range(smote_fts.shape[0]):
    #     print(any(np.array_equal(smote_fts[i],j) for j in original_fts),i)

    # Reduce dimensionality of features and convert into pandas
    pca = PCA(copy=True, n_components=2)
    negative_reduced = pd.DataFrame(pca.fit_transform(X=negative_fts), columns=["X", "Y"])
    positive_reduced = pd.DataFrame(pca.fit_transform(X=positive_fts), columns=["X", "Y"])
    paraphrase_reduced = pd.DataFrame(pca.fit_transform(X=paraphrase_fts), columns=["X", "Y"])
    smote_reduced = pd.DataFrame(pca.fit_transform(X=smote_fts), columns=["X", "Y"])
    outlier_reduced = pd.DataFrame(pca.fit_transform(X=outlier_fts), columns=["X", "Y"])
    
    # Concatenate all reduced feature sets
    negative_reduced["src"] = ["original"]*len(negative_reduced)
    positive_reduced["src"] = ["positive"]*len(positive_reduced)
    paraphrase_reduced["src"] = ["paraphrase"]*len(paraphrase_reduced)
    smote_reduced["src"] = ["smote"]*len(smote_reduced)
    outlier_reduced["src"] = ["outlier"]*len(outlier_reduced)
    
    # outlier_paraphrase_pairs = get_quadrant(dataset, paraphrase_reduced)
    # outlier_paraphrase_pairs.to_csv("plots/outsider_paraphrase_pairs.csv")
    # print(len(outlier_paraphrase_pairs))
    # exit()

    # Concatenate data for Scatterplot
    data = pd.concat([
        # positive_reduced,
        # negative_reduced,
        # smote_reduced,
        paraphrase_reduced, 
        # outlier_reduced
    ])


    data = get_quadrant(data, data)

    print("len of paraphrase_reduced: ", len(data))
    # print("len of outlier_reduced: ", len(outlier_reduced))
    # exit()

    # Create Scatterplot
    plt.figure(1)
    scatterplot = sns.scatterplot(
        data=data, 
        x="X", 
        y="Y", 
        hue="src",
        palette=sns.color_palette("dark:salmon")
    )
    #, palette=sns.color_palette("dark:blue_r"))
    plt.xlim((-22,22));plt.ylim((-7,22))
    # plt.show()
    plt.savefig("./plots/PCA_outsiders.png")
    exit()

    # Comparison between distances with reduced features
    dist_csv = negative_reduced.drop(columns=["src"]).rename(columns={"X":"X_original", "Y":"Y_original"})

    dist_csv["X_paraphrase"] = paraphrase_reduced["X"]
    dist_csv["Y_paraphrase"] = paraphrase_reduced["Y"]
    
    dist_csv["X_smote"] = smote_reduced["X"]
    dist_csv["Y_smote"] = smote_reduced["Y"]
    
    dist_csv.to_csv("plots/dist.csv")

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

    # heatmap = sns.heatmap(distance_matrix, annot=False)

    plt.show()

