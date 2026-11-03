from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VALIDATION_SIZE = 0.15


def load_and_validate(filename: str) -> pd.DataFrame:
    """Load a dataset and validate its structure."""
    filepath = RAW_DIR / filename

    df = pd.read_csv(filepath)

    required_columns = {"text", "category"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{filename} must contain columns: {required_columns}"
        )

    if df[["text", "category"]].isna().any().any():
        raise ValueError(f"{filename} contains missing values.")

    if df["text"].str.strip().eq("").any():
        raise ValueError(f"{filename} contains empty queries.")

    if df["category"].nunique() != 77:
        raise ValueError(
            f"{filename} should contain 77 intent categories."
        )

    if df.duplicated().any():
        print(
            f"Warning: {filename} contains "
            f"{df.duplicated().sum()} duplicate rows."
        )

    return df


def main():
    print("Loading verified BANKING77 datasets...")

    full_train_df = load_and_validate("train.csv")
    test_df = load_and_validate("test.csv")

    print(f"\nOfficial training examples: {len(full_train_df):,}")
    print(f"Official test examples: {len(test_df):,}")
    print(f"Intent categories: {full_train_df['category'].nunique()}")

    # Split only the official training data.
    train_df, validation_df = train_test_split(
        full_train_df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=full_train_df["category"],
    )

    datasets = {
        "train.csv": train_df,
        "validation.csv": validation_df,
        "test.csv": test_df,
    }

    print("\nSaving processed datasets...")

    for filename, df in datasets.items():
        output_path = PROCESSED_DIR / filename
        df.to_csv(output_path, index=False)

        print(
            f"{filename}: {len(df):,} rows | "
            f"{df['category'].nunique()} categories | "
            f"Saved to {output_path}"
        )

    print("\nChecking intent coverage...")

    train_categories = set(train_df["category"])
    validation_categories = set(validation_df["category"])
    test_categories = set(test_df["category"])

    assert train_categories == validation_categories
    assert train_categories == test_categories

    print("All 77 intents are present in every split.")

    print("\nSUCCESS: Data preparation complete.")


if __name__ == "__main__":
    main()