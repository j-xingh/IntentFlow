from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "train_labeled.csv"
MODEL_NAME = "distilbert/distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 8


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
        text = self.texts[index]
        label = self.labels[index]

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading training dataset...")
    dataset = BankingIntentDataset(
        DATA_PATH,
        tokenizer,
        max_length=MAX_LENGTH,
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("\nSingle example:")
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Attention mask shape:", sample["attention_mask"].shape)
    print("Label:", sample["labels"].item())
    print("Non-padding tokens:", sample["attention_mask"].sum().item())

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    batch = next(iter(loader))

    print("\nBatch of examples:")
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Attention mask shape:", batch["attention_mask"].shape)
    print("Labels shape:", batch["labels"].shape)
    print("Label dtype:", batch["labels"].dtype)
    print("Minimum label:", batch["labels"].min().item())
    print("Maximum label:", batch["labels"].max().item())

    assert batch["input_ids"].shape == (BATCH_SIZE, MAX_LENGTH)
    assert batch["attention_mask"].shape == (BATCH_SIZE, MAX_LENGTH)
    assert batch["labels"].shape == (BATCH_SIZE,)
    assert batch["labels"].dtype == torch.long
    assert batch["labels"].min().item() >= 0
    assert batch["labels"].max().item() < 77

    print("\nAll dataset checks passed.")


if __name__ == "__main__":
    main()