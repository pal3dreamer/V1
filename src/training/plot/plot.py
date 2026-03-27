import re
import matplotlib.pyplot as plt

# Path to your log file
log_file = "/home/het/Documents/V1/src/training/plot/training_log.txt"

# Lists to store metrics
epochs = []
train_losses = []
val_losses = []
val_accs = []

# Regular expression to match a log line
pattern = r"Epoch (\d+): Train Loss ([\d\.]+), Val Loss ([\d\.]+), Val Acc ([\d\.]+), LR [\d\.e\-]+"

with open(log_file, "r") as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            val_acc = float(match.group(4))
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

# Create the plot
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss", color="blue")
plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accs, label="Validation Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("final_training_curves.png", dpi=150)
plt.show()
