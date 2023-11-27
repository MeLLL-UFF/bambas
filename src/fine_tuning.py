import argparse
import math
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from src.data import load_dataset

def fine_tune(model: str, dataset: str, fine_tuned_name: str, batch_size: int, lr: float, weight_decay: float, mlm_prob: float, push_model: bool):
    train, _, dev = load_dataset(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    def tokenize(data):
        return tokenizer(data["text"])
    
    train = train.map(tokenize, batched=True, num_proc=4, remove_columsn=["text"])
    dev = dev.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])

    model = AutoModelForMaskedLM.from_pretrained(model)

    training_args = TrainingArguments(
        f"{fine_tuned_name}",
        evaluation_strategy = "epoch",
        learning_rate = lr,
        weight_decay = weight_decay,
        # TODO: add support
        # push_to_hub = push_model
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=data_collator,
    )
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"(dev) Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    #trainer.push_to_hub()
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser("fine_tuning", description="fine-tuning with masked-language model objective for language models")
    parser.add_argument("--model", type=str, choices=["xlm-roberta-base"], help="model to fine-tune")
    parser.add_argument("--dataset", type=str, choices=["ptc2019"], help="corpus for masked-language model pretraining task")
    parser.add_argument("--fine_tuned_name", type=str, help="fine-tuned model name")
    parser.add_argument("--batch_size", type=int, help="batch size for pretraining")
    parser.add_argument("--lr", type=float, default=2e-4, help="training learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="training weight decay")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="probability that a sentence token will be replaced with the [MASK] token")
    parser.add_argument("--push-model-to-hf-hub", action="store_true", help="wheter to upload adjusted model to HuggingFace hub. If True, you will be prompted for your authentication token")
    args = parser.parse_args()
    print("Arguments:", args)