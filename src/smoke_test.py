from pathlib import Path

import torch
import torch.nn as nn


# --------------------------------------------------
# 1. Select the available compute device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"PyTorch version: {torch.__version__}")
print(f"Compute device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# 2. Create a tiny neural network
# --------------------------------------------------

class TinyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.network(x)


model = TinyNeuralNetwork().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)


# --------------------------------------------------
# 3. Create a small synthetic dataset
# --------------------------------------------------

torch.manual_seed(42)

X = torch.randn(256, 10, device=device)
y = (X[:, 0] + X[:, 1] > 0).long()

print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")


# --------------------------------------------------
# 4. Train the model
# --------------------------------------------------

model.train()

for step in range(1, 11):
    optimizer.zero_grad()

    predictions = model(X)

    loss = criterion(predictions, y)

    loss.backward()

    optimizer.step()

    print(f"Step {step:02d} | Loss: {loss.item():.4f}")


# --------------------------------------------------
# 5. Save the training checkpoint
# --------------------------------------------------

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

checkpoint_path = checkpoint_dir / "smoke_checkpoint.pt"

checkpoint = {
    "step": 10,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}

torch.save(checkpoint, checkpoint_path)

print(f"\nCheckpoint saved: {checkpoint_path}")


# --------------------------------------------------
# 6. Restore the checkpoint
# --------------------------------------------------

restored_model = TinyNeuralNetwork().to(device)

restored_optimizer = torch.optim.Adam(
    restored_model.parameters(),
    lr=0.001,
)

saved_checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

restored_model.load_state_dict(
    saved_checkpoint["model_state_dict"]
)

restored_optimizer.load_state_dict(
    saved_checkpoint["optimizer_state_dict"]
)

restored_step = saved_checkpoint["step"]

print(f"Checkpoint restored from step: {restored_step}")


# --------------------------------------------------
# 7. Verify that the restored model works
# --------------------------------------------------

restored_model.eval()

with torch.no_grad():
    restored_predictions = restored_model(X)

print(f"Restored output shape: {restored_predictions.shape}")

assert restored_predictions.shape == (256, 2)

print("\nSUCCESS: GPU training and checkpoint recovery work!")