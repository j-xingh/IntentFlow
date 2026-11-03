from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
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

MODEL_PATHS = {
    "distilbert_fp16": ROOT_DIR / "models" / "distilbert" / "final",
    "distilbert_fp32": ROOT_DIR / "models" / "distilbert" / "final_fp32",
}

LABEL_MAPPING_PATH = (
    ROOT_DIR / "models" / "distilbert" / "label_mapping.json"
)

REPORTS_DIR = ROOT_DIR / "reports"

BATCH_SIZE = 8
MAX_LENGTH = 128


class ValidationDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=MAX_LENGTH,
        )

        encoded["labels"] = self.labels[index]

        return encoded


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def predict_with_distilbert(model_path, validation, device):
    print(f"\nEvaluating: {model_path.name}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )

    model.to(device)
    model.eval()

    dataset = ValidationDataset(validation, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    predictions = []

    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
            }

            logits = model(**inputs).logits
            batch_predictions = logits.argmax(dim=-1)

            predictions.extend(batch_predictions.cpu().tolist())

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.array(predictions)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(DATA_PATH)

    y_true = validation["label"].to_numpy()
    texts = validation["text"].astype(str).tolist()

    with LABEL_MAPPING_PATH.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    id_to_label = {
        int(label_id): intent
        for label_id, intent in mapping["id_to_label"].items()
    }

    label_to_id = mapping["label_to_id"]

    labels = list(range(len(id_to_label)))
    target_names = [id_to_label[i] for i in labels]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Validation examples:", len(validation))
    print("Number of intents:", len(labels))

    # 1. Evaluate the traditional ML baseline.

    print("\nEvaluating TF-IDF + Logistic Regression...")

    baseline = joblib.load(BASELINE_PATH)

    baseline_intents = baseline.predict(texts)

    baseline_predictions = np.array([
        label_to_id[intent]
        for intent in baseline_intents
    ])

    # 2. Evaluate both fine-tuned DistilBERT models.

    predictions = {
        "baseline": baseline_predictions,
    }

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_path}"
            )

        predictions[model_name] = predict_with_distilbert(
            model_path,
            validation,
            device,
        )

    # 3. Compare overall validation metrics.

    results = {
        model_name: calculate_metrics(y_true, model_predictions)
        for model_name, model_predictions in predictions.items()
    }

    print("\n" + "=" * 78)
    print("THREE-MODEL VALIDATION COMPARISON")
    print("=" * 78)

    print(
        f"{'Metric':<18}"
        f"{'Baseline':>15}"
        f"{'DistilBERT FP16':>20}"
        f"{'DistilBERT FP32':>20}"
    )

    print("-" * 78)

    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        print(
            f"{metric:<18}"
            f"{results['baseline'][metric]:>15.4f}"
            f"{results['distilbert_fp16'][metric]:>20.4f}"
            f"{results['distilbert_fp32'][metric]:>20.4f}"
        )

    # 4. Generate per-intent classification reports.

    reports = {}

    for model_name, model_predictions in predictions.items():
        reports[model_name] = classification_report(
            y_true,
            model_predictions,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

    comparison_rows = []

    for intent in target_names:
        baseline_f1 = reports["baseline"][intent]["f1-score"]
        fp16_f1 = reports["distilbert_fp16"][intent]["f1-score"]
        fp32_f1 = reports["distilbert_fp32"][intent]["f1-score"]

        comparison_rows.append({
            "intent": intent,

            "baseline_precision":
                reports["baseline"][intent]["precision"],
            "baseline_recall":
                reports["baseline"][intent]["recall"],
            "baseline_f1": baseline_f1,

            "distilbert_fp16_precision":
                reports["distilbert_fp16"][intent]["precision"],
            "distilbert_fp16_recall":
                reports["distilbert_fp16"][intent]["recall"],
            "distilbert_fp16_f1": fp16_f1,
            "fp16_f1_difference_vs_baseline":
                fp16_f1 - baseline_f1,

            "distilbert_fp32_precision":
                reports["distilbert_fp32"][intent]["precision"],
            "distilbert_fp32_recall":
                reports["distilbert_fp32"][intent]["recall"],
            "distilbert_fp32_f1": fp32_f1,
            "fp32_f1_difference_vs_baseline":
                fp32_f1 - baseline_f1,
        })

    comparison = pd.DataFrame(comparison_rows)

    comparison_path = (
        REPORTS_DIR / "three_model_intent_comparison.csv"
    )

    comparison.to_csv(comparison_path, index=False)

    metrics_path = (
        REPORTS_DIR / "three_model_comparison_metrics.json"
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    # 5. Display the intents with the largest FP32 improvements
    # and regressions compared with the baseline.

    comparison = comparison.sort_values(
        "fp32_f1_difference_vs_baseline",
        ascending=False,
    )

    display_columns = [
        "intent",
        "baseline_f1",
        "distilbert_fp16_f1",
        "distilbert_fp32_f1",
        "fp32_f1_difference_vs_baseline",
    ]

    print("\nTOP 10 INTENTS: FP32 IMPROVEMENT VS BASELINE")

    print(
        comparison[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\nTOP 10 INTENTS: FP32 REGRESSION VS BASELINE")

    print(
        comparison[display_columns]
        .tail(10)
        .sort_values("fp32_f1_difference_vs_baseline")
        .to_string(index=False)
    )

    print("\nComparison complete.")

    print("Overall metrics:", metrics_path)
    print("Per-intent comparison:", comparison_path)


if __name__ == "__main__":
    main()