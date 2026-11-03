from transformers import AutoTokenizer

MODEL_NAME = "distilbert/distilbert-base-uncased"


def main():
    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    examples = [
        "My card was declined",
        "I need help verifying my identity",
        "How long will my bank transfer take?",
    ]

    encoded = tokenizer(
        examples,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    print("\nOriginal queries:")
    for query in examples:
        print(f"  {query}")

    print("\nTokenized output:")
    print("Input IDs shape:", encoded["input_ids"].shape)
    print("Attention mask shape:", encoded["attention_mask"].shape)

    print("\nFirst query's tokens:")
    print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))

    print("\nFirst query's token IDs:")
    print(encoded["input_ids"][0].tolist())

    print("\nFirst query's attention mask:")
    print(encoded["attention_mask"][0].tolist())

    print("\nTokenizer test completed successfully.")


if __name__ == "__main__":
    main()