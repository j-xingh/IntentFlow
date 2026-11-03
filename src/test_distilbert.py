from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "distilbert/distilbert-base-uncased"
NUM_LABELS = 77


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading pretrained DistilBERT classification model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    model.to(device)
    model.eval()

    # A small synthetic batch to test the model's forward pass.
    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(2, 32),
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    print("\nForward-pass results:")
    print("Input shape:", input_ids.shape)
    print("Logits shape:", outputs.logits.shape)
    print("Model device:", next(model.parameters()).device)

    assert outputs.logits.shape == (2, NUM_LABELS)

    print(
        "\nGPU memory allocated:",
        f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB"
        if device.type == "cuda"
        else "Not applicable",
    )

    print("\nDistilBERT forward-pass test passed.")


if __name__ == "__main__":
    main()