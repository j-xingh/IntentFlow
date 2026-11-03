from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "validation_labeled.csv"

BASELINE_PATH = (
    ROOT_DIR / "models" / "baseline" / "intent_classifier.joblib"
)

MODEL_PATH = ROOT_DIR / "models" / "distilbert" / "final_fp32"

MAPPING_PATH = (
    ROOT_DIR / "models" / "distilbert" / "label_mapping.json"
)

OUTPUT_PATH = ROOT_DIR / "reports" / "virtual_card_error_analysis.csv"

BATCH_SIZE = 8
MAX_LENGTH = 128

TARGET_INTENT = "virtual_card_not_working"


class IntentDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe["text"].astype(str).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=MAX_LENGTH,
        )


def main():
    validation = pd.read_csv(DATA_PATH)

    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    id_to_label = {
        int(label_id): intent
        for label_id, intent in mapping["id_to_label"].items()
    }

    label_to_id = {
        intent: int(label_id)
        for intent, label_id in mapping["label_to_id"].items()
    }

    if TARGET_INTENT not in label_to_id:
        raise ValueError(f"Unknown intent: {TARGET_INTENT}")

    target_id = label_to_id[TARGET_INTENT]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Baseline predictions.
    baseline = joblib.load(BASELINE_PATH)

    baseline_intents = baseline.predict(
        validation["text"].astype(str).tolist()
    )

    # Load the fine-tuned transformer.
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    )

    model.to(device)
    model.eval()

    dataset = IntentDataset(validation, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    predictions = []
    confidence_scores = []

    with torch.inference_mode():
        for batch in loader:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
            }

            logits = model(**inputs).logits

            probabilities = torch.softmax(logits, dim=-1)

            confidences, predicted_ids = probabilities.max(dim=-1)

            predictions.extend(predicted_ids.cpu().tolist())
            confidence_scores.extend(confidences.cpu().tolist())

    # Construct a row-by-row comparison.
    results = validation[["text", "label"]].copy()

    results["true_intent"] = results["label"].map(id_to_label)
    results["baseline_intent"] = baseline_intents

    results["distilbert_intent"] = [
        id_to_label[prediction]
        for prediction in predictions
    ]

    results["distilbert_confidence"] = confidence_scores

    # Inspect all true examples of the target intent,
    # including examples the transformer classified correctly.
    target_examples = results[
        results["true_intent"] == TARGET_INTENT
    ].copy()

    target_examples["baseline_correct"] = (
        target_examples["baseline_intent"] == TARGET_INTENT
    )

    target_examples["distilbert_correct"] = (
        target_examples["distilbert_intent"] == TARGET_INTENT
    )

    target_examples = target_examples.sort_values(
        "distilbert_confidence",
        ascending=False,
    )

    print(f"\nTarget intent: {TARGET_INTENT}")
    print("Actual validation examples:", len(target_examples))

    print(
        "Baseline correct:",
        int(target_examples["baseline_correct"].sum()),
    )

    print(
        "DistilBERT correct:",
        int(target_examples["distilbert_correct"].sum()),
    )

    print("\nEXAMPLES AND PREDICTIONS")
    print("=" * 100)

    print(
        target_examples[
            [
                "text",
                "baseline_intent",
                "distilbert_intent",
                "distilbert_confidence",
                "distilbert_correct",
            ]
        ].to_string(
            index=False,
            max_colwidth=100,
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    target_examples.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved detailed results to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()