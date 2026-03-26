import torch
import pickle
import numpy as np
from src.models.mlp import MLPEmotion


class MLPEmotionClassifier:
    def __init__(self, model_path, scaler_path, input_dim, num_classes=7):
        self.device = torch.device("cpu")
        self.model = MLPEmotion(
            input_dim=84,
            hidden_dims=[128, 64],  # <-- match training
            num_classes=num_classes,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, feature_vector):
        """
        feature_vector: numpy array of shape (input_dim,)
        returns: probabilities numpy array of shape (num_classes,)
        """
        # Normalise using the same scaler
        feat_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(feat_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        return probs
