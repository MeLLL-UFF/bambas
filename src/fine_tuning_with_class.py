import argparse
import torch
import time
import os
import json
import pandas as pd
import numpy as np
from argparse import Namespace
from datasets import Dataset
from transformers import EvalPrediction, AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from src.data import load_dataset
from src.utils.workspace import get_workdir
from huggingface_hub import interpreter_login
from src.subtask_1_2a import evaluate_h, hf1_score, hprec_score, hrec_score
from src.subtask_1_2a import get_dag_labels
from src.classification import save_predictions
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

OUTPUT_DIR = f"{get_workdir()}/fine_tuning_with_class"


def fine_tune(args: Namespace):
    model = args.model
    dataset = args.dataset
    fine_tuned_name = args.fine_tuned_name
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    save_model = args.save_model
    push_model = args.push_model_to_hf_hub
    save_strategy = args.save_strategy

    train_df, dev_df, test_df = load_dataset(dataset)
    print("Concatenating train and dev sets")
    train_df = pd.concat([train_df, dev_df])
    print("Merged dataset length:", len(train_df))

    labels = [get_dag_labels()]
    mlb = MultiLabelBinarizer(classes=labels[0])
    train_labels = np.asarray(mlb.fit(labels).transform(train_df["labels"].to_numpy()), dtype=float).tolist()
    train_df["labels"] = train_labels
    dev_labels = np.asarray(mlb.transform(dev_df["labels"].to_numpy()), dtype=float).tolist()
    dev_df["labels"] = dev_labels

    # id2label = {mlb.transform([[label]])[0][0]:label for label in labels[0]}
    # print(id2label)
    # label2id = {mlb.inverse_transform([[id]])[0][0]:id for id in id2label.keys()}
    # print(label2id)

    train, dev = map(Dataset.from_pandas, [train_df, dev_df])

    # def remove_additional_columns(ds: Dataset):
    #     columns = ds.column_names
    #     to_remove = [col for col in columns if col != "text" and col != "labels"]
    #     return ds.remove_columns(to_remove)

    # train, dev = map(remove_additional_columns, [train, dev])

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    device_map = {"": "cpu"}
    if torch.cuda.is_available():
        device_map = {"": 0}
    print(f"Using device: {device_map}")

    def tokenize(data):
        encoding = tokenizer(data["text"], padding=True, truncation=True)
        encoding["labels"] = data["labels"]
        return encoding

    train = train.map(tokenize, batched=True, batch_size=batch_size, num_proc=4, remove_columns=train.column_names)
    train.set_format("torch")
    dev = dev.map(tokenize, batched=True, batch_size=batch_size, num_proc=4, remove_columns=dev.column_names)
    dev.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        model,
        device_map=device_map,
        num_labels=len(labels[0]),
        problem_type="multi_label_classification",
        # id2label=id2label,
        # label2id=label2id,
    )

    args = TrainingArguments(
        fine_tuned_name,
        evaluation_strategy="epoch",
        learning_rate=lr,
        weight_decay=weight_decay,
        push_to_hub=push_model,
        # TODO: proper configuration
        no_cuda=False,
        report_to=["none"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy=save_strategy,
        metric_for_best_model="hier_f1",
        load_best_model_at_end=True,
    )

    def logits_to_predictions(logits, threshold: float = 0.5):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= threshold)] = 1
        return predictions

    def compute_metrics(p: EvalPrediction):
        print("p: ", p)
        print("p.predictions: ", p.predictions)
        print("p.label_ids: ", p.label_ids)
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = logits_to_predictions(preds)

        gold = mlb.inverse_transform(np.asarray(p.label_ids, dtype=int))
        pred = mlb.inverse_transform(predictions)

        return {
            "hier_f1": hf1_score(gold, pred), 
            "hier_precision": hprec_score(gold, pred), 
            "hier_recall": hrec_score(gold, pred),
        }
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train,
        eval_dataset=dev,
        compute_metrics=compute_metrics,
    )
    if push_model:
        interpreter_login(write_permission=True, new_session=False)

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"(dev_set) Hierarchical F1: {eval_results['hier_f1']:.2f}")
    print(f"(dev_set) Evaluation results: {json.dumps(eval_results, indent=4)}")

    if save_model:
        output_dir = f"{OUTPUT_DIR}/{fine_tuned_name}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving fine-tuned model to {output_dir}")
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)

    if push_model:
        print(f"Uploading fine-tuned model ({fine_tuned_name}) to HuggingFace Hub")
        trainer.push_to_hub()
    

    # Prediction
    print("Predicting for test file. Tokenizing test df")
    ds = Dataset.from_pandas(test_df)
    ds = ds.map(tokenize, batched=True, batch_size=batch_size, num_proc=4, remove_columns=dev.column_names)
    # Remove original text from dataset

    # Create DataCollator object
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length")
    # Create DataLoader to be returned
    dl = DataLoader(dataset=ds,
                    batch_size=batch_size,
                    collate_fn=data_collator)
    test_predicted_labels = None
    with torch.no_grad():
        for idx, batch in enumerate(dl):
            print(f'Batch no. {idx+1}. Transferring to device')
            encoding = {k:v.to(trainer.model.device) for k, v in batch.items()}
            print("Prediction pass")
            outputs = trainer.model(**encoding)
            print("Retrieving logits")
            logits = outputs.logits

            print("Parsing logits")
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits.cpu())
            predictions = np.zeros(probs.shape)
            predictions[np.where(probs >= 0.5)] = 1
            
            print("Transforming to predictions")
            pred = mlb.inverse_transform(predictions)
            if test_predicted_labels is None:
                test_predicted_labels = pred
            else:
                test_predicted_labels = np.concatenate((test_predicted_labels, pred))

    ts = int(time.time())
    pred_path, _ = save_predictions(test_df, test_predicted_labels, "dev_unlabeled", ts)
    print(f"Finished successfully. dev_unlabeled predictions saved at {pred_path}")

    # results_csv_path = f"{OUTPUT_DIR}/results.csv"
    # print(f"Saving validation set results to {results_csv_path}")
    # if os.path.exists(results_csv_path):
    #     results = pd.read_csv(results_csv_path, index_col=0)
    # else:
    #     dir = os.path.sep.join(results_csv_path.split(os.path.sep)[:-1])
    #     print(f"Creating dir {dir}")
    #     os.makedirs(dir, exist_ok=True)
    #     results = pd.DataFrame(
    #         columns=[
    #             "FE Model",
    #             "FE Method",
    #             "FE Layers",
    #             "FE Layers Agg Method",
    #             "FE Dataset",
    #             "Test Dataset",
    #             "Classifier",
    #             "F1",
    #             "Precision",
    #             "Recall",
    #             "Timestamp"])
    
    # results.loc[len(results.index) + 1] = [train_ft_info["model"],
    #                                        train_ft_info["extraction_method"],
    #                                        train_ft_info["layers"],
    #                                        train_ft_info["layers_aggregation_method"],
    #                                        train_ft_info["dataset"],
    #                                        "dev_set",
    #                                        args.classifier,
    #                                        f1,
    #                                        prec,
    #                                        rec,
    #                                        ts]
    # results.to_csv(results_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("fine_tuning",
                                     description="fine-tuning with masked-language model (MLM) objective for language models")
    parser.add_argument("--model", type=str, choices=["xlm-roberta-base"], help="model to fine-tune", required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ptc2019", "semeval2024"],
        help="corpus for masked-language model pretraining task",
        required=True)
    parser.add_argument("--fine_tuned_name", type=str, help="fine-tuned model name", required=True)
    parser.add_argument("--batch_size", type=int, help="batch size for pretraining", default=32)
    parser.add_argument("--lr", type=float, default=2e-4, help="training learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="training weight decay")
    parser.add_argument("--save_model", action="store_true", help="wheter to save adjusted model locally")
    parser.add_argument(
        "--push_model_to_hf_hub",
        action="store_true",
        help="wheter to upload adjusted model to HuggingFace hub. If True, you will be prompted for your authentication token")
    parser.add_argument("--save_strategy", type=str, help="save strategy for trainer, can be no, epoch or steps", default="epoch")
    args = parser.parse_args()
    print("Arguments:", args)
    fine_tune(args)
