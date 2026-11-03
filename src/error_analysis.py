from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/baseline/intent_classifier.joblib")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("INTENTFLOW: BASELINE ERROR ANALYSIS")
    print("=" * 60)

    # Load validation data and trained model.
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")
    model = joblib.load(MODEL_PATH)

    texts = validation_df["text"]
    actual = validation_df["category"]
    predicted = model.predict(texts)

    # Collect incorrect predictions.
    errors = validation_df.copy()
    errors["predicted_category"] = predicted

    errors = errors[
        errors["category"] != errors["predicted_category"]
    ].copy()

    errors.rename(
        columns={"category": "actual_category"},
        inplace=True,
    )

    print(f"\nTotal validation examples: {len(validation_df):,}")
    print(f"Incorrect predictions: {len(errors):,}")
    print(
        f"Error rate: "
        f"{len(errors) / len(validation_df):.2%}"
    )

    # Save all misclassified examples.
    errors_path = REPORT_DIR / "baseline_errors.csv"
    errors.to_csv(errors_path, index=False)

    print(f"\nSaved misclassified examples: {errors_path}")

    # Find the most frequent confusion pairs.
    labels = sorted(actual.unique())

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels,
    )

    confusion_pairs = []

    for actual_index, actual_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            if actual_index == predicted_index:
                continue

            count = matrix[actual_index, predicted_index]

            if count > 0:
                confusion_pairs.append(
                    {
                        "actual_intent": actual_label,
                        "predicted_intent": predicted_label,
                        "count": int(count),
                    }
                )

    confusion_df = pd.DataFrame(confusion_pairs)

    confusion_df.sort_values(
        "count",
        ascending=False,
        inplace=True,
    )

    confusion_path = REPORT_DIR / "baseline_confusion_pairs.csv"
    confusion_df.to_csv(confusion_path, index=False)

    print("\nTOP 15 CONFUSION PAIRS")
    print("-" * 60)

    print(
        confusion_df.head(15).to_string(index=False)
    )

    # Display a sample of actual mistakes.
    print("\nEXAMPLES OF MISCLASSIFIED QUERIES")
    print("-" * 60)

    for _, row in errors.head(15).iterrows():
        print(f"\nQuery: {row['text']}")
        print(f"Actual:    {row['actual_category']}")
        print(f"Predicted: {row['predicted_category']}")

    print("\nERROR ANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()