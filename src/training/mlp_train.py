# train_final.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from src.models.mlp_improved import MLPEmotionImproved

# Load data
X = np.load("data/X_mlp_new.npy")
y = np.load("data/y_mlp_new.npy")
print(f"Data shape: {X.shape}, labels: {y.shape}")

# If you have a separate test set, load it and use only training part here.
# We'll assume we use all data for final training (no test split).
# But ideally you should split into train+val+test earlier; here we just train on all data.

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
# Use 80/20 split for training/validation (or use all for training if you trust CV)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(
    train_dataset, batch_size=best_params["batch_size"], shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"])

# Load best hyperparameters (from previous step)
# We'll manually set them here; alternatively load from file.
best_params = {
    "n_layers": 4,  # example, replace with actual best
    "hidden_0": 256,
    "hidden_1": 192,
    "hidden_2": 128,
    "hidden_3": 96,
    "dropout": 0.3,
    "activation": "swish",
    "use_batch_norm": True,
    "lr": 0.0005,
    "weight_decay": 1e-4,
    "batch_size": 128,
}

# Build hidden_dims list
hidden_dims = [best_params[f"hidden_{i}"] for i in range(best_params["n_layers"])]

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPEmotionImproved(
    input_dim=X.shape[1],
    hidden_dims=hidden_dims,
    num_classes=len(np.unique(y)),
    dropout=best_params["dropout"],
    activation=best_params["activation"],
    use_batch_norm=best_params["use_batch_norm"],
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# Training loop with early stopping
num_epochs = 100
best_val_acc = 0.0
patience = 0
early_stop_patience = 15
train_losses, val_losses, val_accs = [], [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
    val_acc = correct / total
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch + 1:3d}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.4f}, LR {optimizer.param_groups[0]['lr']:.6f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = 0
        torch.save(model.state_dict(), "src/models/mlp_model_improved.pt")
        print(f"  Model saved (val acc {val_acc:.4f})")
    else:
        patience += 1
        if patience >= early_stop_patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_accs) + 1), val_accs, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("final_training_curves.png")
plt.show()

print(f"Best validation accuracy: {best_val_acc:.4f}")
print("Final model saved as src/models/mlp_model_improved.pt")
