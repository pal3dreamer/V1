# tune_mlp_cv.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.pruners import MedianPruner
from src.models.mlp import MLPEmotionImproved
import warnings

warnings.filterwarnings("ignore")

# Load data
X = np.load("data/X_mlp.npy")
y = np.load("data/y_mlp.npy")
print(f"Data shape: {X.shape}, labels: {y.shape}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_fold(train_loader, val_loader, model, params, max_epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_acc = 0.0
    patience = 0
    early_stop_patience = 7

    for epoch in range(max_epochs):
        # Training
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
        avg_loss = total_loss / len(train_loader)

        # Validation
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
        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                break

    return best_acc


def objective(trial):
    # Print trial start
    print(f"\n{'=' * 50}")
    print(f"Trial {trial.number} started")

    # Hyperparameters to tune
    n_layers = trial.suggest_int("n_layers", 2, 5)
    hidden_dims = []
    for i in range(n_layers):
        hidden_dims.append(trial.suggest_int(f"hidden_{i}", 64, 512, step=32))
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    activation = trial.suggest_categorical(
        "activation", ["relu", "leaky_relu", "swish"]
    )
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Print hyperparameters for this trial
    print(f"Hyperparameters:")
    print(f"  n_layers: {n_layers}")
    print(f"  hidden_dims: {hidden_dims}")
    print(f"  dropout: {dropout:.4f}")
    print(f"  activation: {activation}")
    print(f"  use_batch_norm: {use_batch_norm}")
    print(f"  lr: {lr:.6f}")
    print(f"  weight_decay: {weight_decay:.6f}")
    print(f"  batch_size: {batch_size}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = MLPEmotionImproved(
            input_dim=X.shape[1],
            hidden_dims=hidden_dims,
            num_classes=len(np.unique(y)),
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        ).to(device)

        fold_acc = train_fold(
            train_loader,
            val_loader,
            model,
            {"lr": lr, "weight_decay": weight_decay},
            max_epochs=30,
        )
        fold_accs.append(fold_acc)
        print(f"  Fold {fold + 1}: accuracy = {fold_acc:.4f}")

    mean_acc = np.mean(fold_accs)
    print(f"Trial {trial.number} mean accuracy: {mean_acc:.4f}")
    print(f"{'=' * 50}\n")

    return mean_acc


# Create study and optimize
study = optuna.create_study(
    direction="maximize", pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Print best results
print("\n" + "=" * 50)
print("Best trial:")
best_trial = study.best_trial
print(f"  Accuracy: {best_trial.value:.4f}")
print("  Params:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
print("=" * 50)

# Save the study for later analysis
import pickle

with open("optuna_study_cv.pkl", "wb") as f:
    pickle.dump(study, f)

# Optionally save best hyperparameters to a file
best_params = best_trial.params
np.save("best_hyperparams.npy", best_params)
