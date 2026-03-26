import torch
import torch.nn as nn


class MLPEmotion(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=7, dropout=0.4):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
