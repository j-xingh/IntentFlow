from pathlib import Path
import json

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models" / "distilbert"

SPLITS = ["train", "validation", "test"]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataframes = {}

    for split in SPLITS:
        path = DATA_DIR / f"{split}.csv"
        df = pd.read_csv(path)

        if not {"text", "category"}.issubset(df.columns):
            raise ValueError(
                f"{path} must contain 'text' and 'category' columns. "
                f"Found: {list(df.columns)}"
            )

        if df[["text", "category"]].isna().any().any():
            raise ValueError(f"Missing values found in {path}")

        dataframes[split] = df
        print(f"{split}: {len(df):,} examples")

    # Fit the encoder on all intent names, not on individual splits.
    all_intents = sorted(
        set().union(
            *(set(df["category"].unique()) for df in dataframes.values())
        )
    )

    if len(all_intents) != 77:
        raise ValueError(
            f"Expected 77 unique intents, found {len(all_intents)}"
        )

    label_to_id = {
        intent: idx
        for idx, intent in enumerate(all_intents)
    }

    id_to_label = {
        idx: intent
        for intent, idx in label_to_id.items()
    }

    for split, df in dataframes.items():
        df = df.copy()
        df["label"] = df["category"].map(label_to_id).astype(int)

        output_path = DATA_DIR / f"{split}_labeled.csv"
        df.to_csv(output_path, index=False)

        dataframes[split] = df
        print(f"Saved: {output_path.relative_to(ROOT_DIR)}")

    mapping_path = MODEL_DIR / "label_mapping.json"

    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": {
                    str(idx): intent
                    for idx, intent in id_to_label.items()
                },
            },
            f,
            indent=2,
        )

    print(f"\nSaved label mapping: {mapping_path.relative_to(ROOT_DIR)}")
    print(f"Number of intents: {len(all_intents)}")
    print(f"Label IDs: 0 to {len(all_intents) - 1}")

    print("\nSample mappings:")
    for intent in all_intents[:10]:
        print(f"  {label_to_id[intent]:2d} -> {intent}")

    print("\nAll splits prepared successfully.")


if __name__ == "__main__":
    main()