import argparse
import torch
import math
import os
import json
import pandas as pd
from argparse import Namespace
from datasets import Dataset, concatenate_datasets
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from src.data import load_dataset
from src.utils.workspace import get_workdir
from huggingface_hub import interpreter_login

OUTPUT_DIR = f"{get_workdir()}/fine_tuning"


def fine_tune(args: Namespace):
    model = args.model
    dataset = args.dataset
    fine_tuned_name = args.fine_tuned_name
    # batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    mlm_prob = args.mlm_probability
    save_model = args.save_model
    push_model = args.push_model_to_hf_hub

    train, dev, test = map(Dataset.from_pandas, load_dataset(dataset))
    if dataset != "semeval2024":
        # in this case, we can use the test for evaluation and combine train+dev for training
        train = concatenate_datasets([train, dev])
        dev = test

    def remove_additional_columns(ds: Dataset):
        columns = ds.column_names
        to_remove = [col for col in columns if col != "text"]
        return ds.remove_columns(to_remove)

    train, dev = map(remove_additional_columns, [train, dev])

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    device_map = {"": "cpu"}
    if torch.cuda.is_available():
        device_map = {"": 0}
    print(f"Using device: {device_map}")

    def tokenize(data):
        return tokenizer(data["text"], padding=True, truncation=True)

    train = train.map(tokenize, batched=True, batch_size=64, num_proc=4, remove_columns=["text"])
    dev = dev.map(tokenize, batched=True, batch_size=64, num_proc=4, remove_columns=["text"])

    model = AutoModelForMaskedLM.from_pretrained(model, device_map=device_map)

    training_args = TrainingArguments(
        fine_tuned_name,
        evaluation_strategy="epoch",
        learning_rate=lr,
        weight_decay=weight_decay,
        push_to_hub=push_model,
        # TODO: proper configuration
        no_cuda=False,
        report_to=["none"]
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=data_collator,
    )
    if push_model:
        interpreter_login(write_permission=True, new_session=False)

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"(dev_set) Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
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
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser("fine_tuning",
                                     description="fine-tuning with masked-language model (MLM) objective for language models")
    parser.add_argument("--model", type=str, choices=["xlm-roberta-base"], help="model to fine-tune", required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ptc2019"],
        help="corpus for masked-language model pretraining task",
        required=True)
    parser.add_argument("--fine_tuned_name", type=str, help="fine-tuned model name", required=True)
    # parser.add_argument("--batch_size", type=int, help="batch size for pretraining")
    parser.add_argument("--lr", type=float, default=2e-4, help="training learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="training weight decay")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="probability that a sentence token will be replaced with the [MASK] token")
    parser.add_argument("--save_model", action="store_true", help="wheter to save adjusted model locally")
    parser.add_argument(
        "--push_model_to_hf_hub",
        action="store_true",
        help="wheter to upload adjusted model to HuggingFace hub. If True, you will be prompted for your authentication token")
    args = parser.parse_args()
    print("Arguments:", args)
    fine_tune(args)
