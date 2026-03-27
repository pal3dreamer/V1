# src/models/mlp_improved.py
import torch
import torch.nn as nn


class MLPEmotionImproved(nn.Module):
    def __init__(
        self,
        input_dim=140,
        hidden_dims=[256, 128, 64],
        num_classes=7,
        dropout=0.3,
        activation="relu",
        use_batch_norm=True,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hdim))
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == "swish":
                layers.append(nn.SiLU(inplace=True))
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
