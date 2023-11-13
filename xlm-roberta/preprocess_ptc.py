import pandas as pd
import os

def get_category(labels, categories):
    c = []
    for label in labels:
        for k, v in categories.items():
            if label in v:
                c.append(k)
    return '_'.join(c)


def main():
    # Based on https://propaganda.math.unipd.it/semeval2023task3/
    categories = dict(
        Justification = [
            "Appeal_to_Authority",
            "Appeal_to_fear-prejudice",
            "Flag-Waving"
        ],
        Simplification = ["Causal_Oversimplification"],
        Distraction = [
            "Red_Herring",
            "Straw_Men",
            "Whataboutism"
        ],
        Call = ["Slogans"],
        Manipulative_wording = [
            "Loaded_Language",
            "Exaggeration_Minimisation",
            "Obfuscation_Intentional_Vagueness_Confusion",
            "Repetition"
        ],
        Attack_on_reputation = [
            "Name_Calling_Labeling",
            "Doubt"
        ]
    )

    dataset_folder = "../dataset/ptc_preproc"

    files = ["ptc_preproc_dev.csv", "ptc_preproc_test.csv", "ptc_preproc_train.csv"]

    os.makedirs("./dataset")

    for file in files:
        df = pd.read_csv(dataset_folder+"/"+file)
        df["label"] = df.apply(lambda row: ",".join(row[row == 1].index.to_list()), axis=1)
        new_df = pd.DataFrame()
        new_df["text"] = df["text"]
        new_df["label"] = df["label"]
        new_df["category"] = df.apply(lambda row: get_category(row["label"].split(","), categories), axis=1)
        new_df.to_csv("./dataset/"+file, sep=";")

if __name__ == "__main__":
    main()
