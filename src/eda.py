from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("reports/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load training data only for exploration.
df = pd.read_csv(DATA_DIR / "train.csv")

print("=" * 60)
print("INTENTFLOW: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. Dataset overview
print("\n1. DATASET OVERVIEW")
print(f"Number of queries: {len(df):,}")
print(f"Number of intent classes: {df['category'].nunique()}")
print(f"Missing values:\n{df.isna().sum()}")

# 2. Intent distribution
intent_counts = df["category"].value_counts()

print("\n2. INTENT DISTRIBUTION")
print(f"Most frequent intent: {intent_counts.index[0]}")
print(f"Most frequent count: {intent_counts.iloc[0]}")
print(f"Least frequent intent: {intent_counts.index[-1]}")
print(f"Least frequent count: {intent_counts.iloc[-1]}")
print(f"Mean examples per intent: {intent_counts.mean():.2f}")

# Visualize the 15 most frequent intents.
plt.figure(figsize=(12, 7))
intent_counts.head(15).sort_values().plot(
    kind="barh",
    edgecolor="black",
)
plt.title("Top 15 Most Frequent Banking Intents")
plt.xlabel("Number of queries")
plt.ylabel("Intent")
plt.tight_layout()

chart_path = OUTPUT_DIR / "top_intents.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print(f"\nSaved: {chart_path}")

# 3. Query length
df["word_count"] = df["text"].str.split().str.len()

print("\n3. QUERY LENGTH")
print(df["word_count"].describe().round(2))

plt.figure(figsize=(10, 6))
plt.hist(df["word_count"], bins=30, edgecolor="black")
plt.title("Distribution of Customer Query Lengths")
plt.xlabel("Words per query")
plt.ylabel("Number of queries")
plt.tight_layout()

length_chart_path = OUTPUT_DIR / "query_length_distribution.png"
plt.savefig(length_chart_path, dpi=150)
plt.close()

print(f"Saved: {length_chart_path}")

# 4. Duplicate queries
duplicate_count = df["text"].duplicated().sum()

print("\n4. DATA QUALITY")
print(f"Duplicate query texts: {duplicate_count}")

# 5. Example queries
print("\n5. EXAMPLE CUSTOMER QUERIES")

for intent in [
    "cash_withdrawal",
    "cash_withdrawal_not_recognised",
    "cash_withdrawal_wrong_exchange_rate",
]:
    examples = df.loc[df["category"] == intent, "text"].head(3)

    if not examples.empty:
        print(f"\nIntent: {intent}")

        for index, example in enumerate(examples, start=1):
            print(f"  {index}. {example}")

print("\nEDA COMPLETE.")