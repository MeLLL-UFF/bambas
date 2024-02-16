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
from src.confusion_matrix import binary_relevance_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from typing import Tuple, List
from optuna import trial

OUTPUT_DIR = f"{get_workdir()}/fine_tuning_with_class"
GOLD_DIR = f"{get_workdir()}/dataset/semeval2024/subtask1"

def fine_tune(args: Namespace):
    model = args.model
    dataset = args.dataset
    fine_tuned_name = args.fine_tuned_name
    batch_size = args.batch_size
    max_length = args.max_length
    lr = args.lr
    weight_decay = args.weight_decay
    save_model = args.save_model
    push_model = args.push_model_to_hf_hub
    concat_train_dev = args.concat_train_dev
    hypsearch = args.hypsearch
    hpsearch_min = args.hpsearch_min
    hpsearch_max = args.hpsearch_max
    n_trials = args.n_trials
    save_strategy = args.save_strategy
    epochs = args.epochs

    train_df, dev_df, test_df = load_dataset(dataset)
    
    if concat_train_dev:
        print("Concatenating train and dev sets")
        train_df = pd.concat([train_df, dev_df])
        print("Merged dataset length:", len(train_df))

    labels = [get_dag_labels()]
    labels[0].sort()
    mlb = MultiLabelBinarizer(classes=labels[0])
    train_labels = np.asarray(mlb.fit(labels).transform(train_df["labels"].to_numpy()), dtype=float).tolist()
    train_df["labels"] = train_labels
    dev_labels = np.asarray(mlb.transform(dev_df["labels"].to_numpy()), dtype=float).tolist()
    dev_df["labels"] = dev_labels
    test_labels = None
    if dataset == "semeval2024_dev_labeled":
        test_labels = np.asarray(mlb.transform(test_df["labels"].to_numpy()), dtype=float).tolist()
        test_df["labels"] = test_labels

    train, dev = map(Dataset.from_pandas, [train_df, dev_df])

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, model_max_length=max_length)
    device_map = {"": "cpu"}
    if torch.cuda.is_available():
        device_map = {"": 0}
    print(f"Using device: {device_map}")

    def tokenize(data):
        encoding = tokenizer(data["text"], max_length=max_length, padding=True, truncation=True)
        encoding["labels"] = data["labels"]
        return encoding

    train = train.map(tokenize, batched=True, batch_size=batch_size, num_proc=4, remove_columns=train.column_names)
    train.set_format("torch")
    dev = dev.map(tokenize, batched=True, batch_size=batch_size, num_proc=4, remove_columns=dev.column_names)
    dev.set_format("torch")

    def model_template():
        return AutoModelForSequenceClassification.from_pretrained(
            model,
            device_map=device_map,
            num_labels=len(labels[0]),
            problem_type="multi_label_classification",
        )
    
    def model_init(trial):
        return model_template()

    training_args = TrainingArguments(
        fine_tuned_name,
        evaluation_strategy=save_strategy,
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
        num_train_epochs=epochs,
    )

    def logits_to_predictions(logits, threshold: float = 0.5):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= threshold)] = 1
        return predictions

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = logits_to_predictions(preds)

        all_zeros = not np.any(predictions)
        if all_zeros:
            print("\nNo labels were predicted at all. Every metric is zero\n")
            return {
                "hier_f1": 0.,
                "hier_precision": 0.,
                "hier_recall": 0.,
            }

        gold = mlb.inverse_transform(np.asarray(p.label_ids, dtype=int))
        pred = mlb.inverse_transform(predictions)

        return {
            "hier_f1": hf1_score(gold, pred), 
            "hier_precision": hprec_score(gold, pred), 
            "hier_recall": hrec_score(gold, pred),
        }
    
    trainer = Trainer(
        model=model_template() if not hypsearch else None,
        model_init=None if not hypsearch else model_init,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        compute_metrics=compute_metrics,
    )
    if push_model:
        interpreter_login(write_permission=True, new_session=False)

    if hypsearch:
        print(f"Making hyperparameter search")
        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", hpsearch_min, hpsearch_max, log=True),
            }
        best_trial = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=n_trials,
        )
        print(f"Trial results: {json.dumps(best_trial, indent=4)}")
    else:
    
        trainer.train()

        eval_results = trainer.evaluate()
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
    
        def predict(ds: Dataset) -> Tuple[List[str], np.ndarray]:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
            dl = DataLoader(dataset=ds,
                            batch_size=batch_size,
                            collate_fn=data_collator)
            test_predicted_labels = None
            raw_predictions = None

            trainer.model.eval()

            with torch.no_grad():
                for idx, batch in enumerate(dl):
                    print(f'Batch no. {idx+1}. Transferring to device')
                    encoding = {k:v.to(trainer.model.device) for k, v in batch.items()}
                    outputs = trainer.model(**encoding)
                    logits = outputs.logits

                    sigmoid = torch.nn.Sigmoid()
                    probs = sigmoid(logits.cpu())
                    predictions = np.zeros(probs.shape)
                    predictions[np.where(probs >= 0.5)] = 1

                    if raw_predictions is None:
                        raw_predictions = predictions
                    else:
                        raw_predictions = np.concatenate((raw_predictions, predictions))
                    
                    pred = list(mlb.inverse_transform(predictions))
                    if test_predicted_labels is None:
                        test_predicted_labels = pred
                    else:
                        test_predicted_labels.extend(pred)
            return test_predicted_labels, np.asarray(raw_predictions, dtype=int)

        print("Prediction on test_set")

        def tokenize_test_df(data):
            encoding = tokenizer(data["text"], max_length=max_length, padding=True, truncation=True)
            return encoding
        
        test = Dataset.from_pandas(test_df)
        test = test.map(tokenize_test_df, batched=True, batch_size=batch_size, num_proc=4, remove_columns=test.column_names)
        test_predicted_labels, test_raw_predictions = predict(test)

        ts = int(time.time())
        pred_path, _ = save_predictions(test_df, test_predicted_labels, "dev_labeled", ts)

        gold_file = (
            "dev_subtask1_en.json"
            if dataset == "semeval2024_dev_labeled"
            else "validation.json"
        )

        if dataset == "semeval2024_test_unlabeled":
            print(f"Predicting for task test set. No evaluation can be made locally. Saved predictions file to {pred_path}\n")
            return

        gold_path = os.path.sep.join([GOLD_DIR, gold_file])
        prec, rec, f1 = evaluate_h(pred_path, gold_path)
        print(f"\nTest set:\n\tPrecision: {prec}\n\tRecall: {rec}\n\tF1: {f1}\n")

        results_csv_path = f"{OUTPUT_DIR}/ft_with_class_results.csv"
        print(f"Saving test set results to {results_csv_path}")
        if os.path.exists(results_csv_path):
            results = pd.read_csv(results_csv_path, index_col=0)
            idx = len(results.index)
        else:
            dir = os.path.sep.join(results_csv_path.split(os.path.sep)[:-1])
            print(f"Creating dir {dir}")
            os.makedirs(dir, exist_ok=True)
            results = pd.DataFrame(
                columns=[
                    "Model",
                    "Dataset",
                    "Test Dataset",
                    "F1",
                    "Precision",
                    "Recall",
                    "Confusion Matrix"
                    "Timestamp"])
            idx = 0
        
        results.loc[idx] = [args.model,
                            args.dataset,
                            "test_set",
                            f1,
                            prec,
                            rec,
                            "",
                            ts]
        results.to_csv(results_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("fine_tuning",
                                     description="fine-tuning a language model with a final multilabel classification layer")
    parser.add_argument("--model", type=str, choices=["xlm-roberta-base", "jhu-clsp/bernice"], help="model to fine-tune", required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ptc2019", "semeval2024", "semeval2024_augmented", "semeval2024_dev_labeled", "semeval2024_test_unlabeled"],
        help="corpus for masked-language model pretraining task",
        required=True)
    parser.add_argument("--fine_tuned_name", type=str, help="fine-tuned model name", required=True)
    parser.add_argument("--batch_size", type=int, help="batch size for pretraining", default=32)
    parser.add_argument("--max_length", type=int, help="max sentence length for truncation", default=128)
    parser.add_argument("--lr", type=float, default=3.9e-5, help="training learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="training weight decay")
    parser.add_argument("--save_model", action="store_true", help="wheter to save adjusted model locally")
    parser.add_argument("--concat_train_dev",  action="store_true", help="wheter to concatenate train+dev(validation) datasets for training")
    parser.add_argument("--hypsearch", action="store_true", help="wheter to use hyperparameter search")
    parser.add_argument("--hpsearch_min", type=float, default=3.9e-5, help="training learning rate")
    parser.add_argument("--hpsearch_max", type=float, default=3.9e-5, help="training learning rate")
    parser.add_argument("--n_trials", type=int, default=10, help="number of trials for hyperparameter search")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs to train")
    parser.add_argument(
        "--push_model_to_hf_hub",
        action="store_true",
        help="wheter to upload adjusted model to HuggingFace hub. If True, you will be prompted for your authentication token")
    parser.add_argument("--save_strategy", type=str, help="save strategy for trainer, can be no, epoch or steps", default="epoch")
    args = parser.parse_args()
    print("Arguments:", args)
    fine_tune(args)
