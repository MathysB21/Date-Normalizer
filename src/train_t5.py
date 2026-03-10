# This trains T5-small on a CSV with columns: input, target
# src/train_t5.py
import pandas as pd
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import os
import torch
import random
import numpy as np

MODEL_NAME = "t5-small"
OUTPUT_DIR = os.path.join("models", "t5-date-normalizer")
MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 16

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    if "input" not in df.columns or "target" not in df.columns:
        raise ValueError("CSV must have columns: input,target")
    df["input"] = "normalize date: " + df["input"].astype(str)
    return Dataset.from_pandas(df)

def main():
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_ds = load_dataset("data/train.csv")
    val_ds = load_dataset("data/val.csv")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input"],
            truncation=True,
            padding="max_length",
            max_length=MAX_INPUT_LENGTH,
        )
        labels = tokenizer(
            text_target=batch["target"],
            truncation=True,
            padding="max_length",
            max_length=MAX_TARGET_LENGTH,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_fp16 = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=64,  # adjust if OOM: try 32 or 16
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    if torch.cuda.is_available():
        model.to("cuda")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()