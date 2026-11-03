from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/baseline")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("INTENTFLOW: TF-IDF + LOGISTIC REGRESSION")
    print("=" * 60)

    # Load training and validation data.
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_validation = validation_df["text"]
    y_validation = validation_df["category"]

    print(f"\nTraining examples: {len(X_train):,}")
    print(f"Validation examples: {len(X_validation):,}")
    print(f"Number of intents: {y_train.nunique()}")

    # Create the complete ML pipeline.
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=5.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining baseline model...")

    model.fit(X_train, y_train)

    print("Training complete.")

    # Evaluate on validation data.
    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)
    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
    )
    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
    )

    print("\nVALIDATION RESULTS")
    print("-" * 40)
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")
    print(f"Weighted F1:{weighted_f1:.4f}")

    # Detailed per-intent metrics.
    report = classification_report(
        y_validation,
        predictions,
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT")
    print(report)

    # Save model and results.
    model_path = MODEL_DIR / "intent_classifier.joblib"
    joblib.dump(model, model_path)

    report_path = REPORT_DIR / "baseline_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    metrics_path = REPORT_DIR / "baseline_metrics.txt"
    metrics_path.write_text(
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n",
        encoding="utf-8",
    )

    # Save confusion matrix for further analysis.
    labels = sorted(y_train.unique())
    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=labels,
    )

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(REPORT_DIR / "baseline_confusion_matrix.csv")

    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print("\nBASELINE TRAINING COMPLETE.")


if __name__ == "__main__":
    main()