from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models" / "distilbert"
OUTPUT_DIR = MODEL_DIR / "checkpoints_fp32"
FINAL_MODEL_DIR = MODEL_DIR / "final_fp32"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 77


class BankingIntentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.texts = self.data["text"].astype(str).tolist()
        self.labels = self.data["label"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
        )

        encoded["labels"] = self.labels[index]
        return encoded


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("INTENTFLOW: DISTILBERT FINE-TUNING")
    print("=" * 60)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    # Load the saved intent-to-ID mapping.
    mapping_path = MODEL_DIR / "label_mapping.json"

    with mapping_path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    label_to_id = mapping["label_to_id"]
    id_to_label = {
        int(label_id): intent
        for label_id, intent in mapping["id_to_label"].items()
    }

    if len(label_to_id) != NUM_LABELS:
        raise ValueError(
            f"Expected {NUM_LABELS} labels, found {len(label_to_id)}"
        )

    # Load tokenizer and datasets.
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Preparing datasets...")
    train_dataset = BankingIntentDataset(
        DATA_DIR / "train_labeled.csv",
        tokenizer,
        MAX_LENGTH,
    )

    validation_dataset = BankingIntentDataset(
        DATA_DIR / "validation_labeled.csv",
        tokenizer,
        MAX_LENGTH,
    )

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

    # Dynamic padding reduces unnecessary computation.
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )

    # Load pretrained DistilBERT with a new 77-class head.
    print("\nLoading pretrained DistilBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # GPU-conscious training configuration.
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,

        fp16=False,
        gradient_checkpointing=True,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,

        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,

        optim="adamw_torch",
        seed=42,
        data_seed=42,

        dataloader_num_workers=0,
        dataloader_pin_memory=device.type == "cuda",

        report_to="none",
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    print("Epochs: 3")
    print("Batch size per GPU: 4")
    print("Gradient accumulation steps: 2")
    print("Effective batch size: 8")
    print("Maximum sequence length:", MAX_LENGTH)
    print("Mixed precision:", device.type == "cuda")

    train_result = trainer.train()

    print("\nTraining completed.")

    print("\nEvaluating the best checkpoint on validation data...")
    validation_metrics = trainer.evaluate()

    # Save the best model, tokenizer, and evaluation results.
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    results = {
        "training_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "baseline_accuracy": 0.8774,
        "baseline_macro_f1": 0.8787,
    }

    metrics_path = REPORTS_DIR / "distilbert_fp32_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, default=float)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    for name, value in validation_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")

    print("\nBaseline accuracy: 0.8774")
    print("Baseline macro F1: 0.8787")

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak GPU memory allocated: {peak_memory:.2f} GB")

    print("\nBest model saved to:", FINAL_MODEL_DIR)
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()