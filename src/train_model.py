import os
import json
import numpy as np
import librosa

from src.audio_features import extract_features
from config.audio_config import DATASET_PATH, EMOTION_MAP

# 🔧 CHANGE THIS if your dataset is elsewhere
DATASET_PATH = DATASET_PATH

# map dataset folder → final labels
EMOTION_MAP = EMOTION_MAP

# initialize storage
data = {}
for v in EMOTION_MAP.values():
    data[v] = []


# 🔥 STEP 1 — TRAVERSE DATASET (ROBUST)
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if not file.lower().endswith(".wav"):
            continue

        filepath = os.path.join(root, file)

        # get emotion from parent folder
        emotion_folder = os.path.basename(root).lower()

        if emotion_folder not in EMOTION_MAP:
            continue

        label = EMOTION_MAP[emotion_folder]

        print("Processing:", filepath)

        try:
            audio, sr = librosa.load(filepath, sr=16000)
        except Exception as e:
            print("Error loading:", filepath, e)
            continue

        # 🔥 split into chunks (IMPORTANT)
        CHUNK_SIZE = 16000  # 1 second

        for i in range(0, len(audio), CHUNK_SIZE):
            chunk = audio[i : i + CHUNK_SIZE]

            if len(chunk) < CHUNK_SIZE:
                continue

            try:
                features = extract_features(chunk)
                data[label].append(features)
            except Exception as e:
                print("Feature error:", filepath, e)


# 🧠 STEP 2 — COMPUTE μ AND σ
def compute_params(samples):
    samples = np.array(samples)

    if len(samples.shape) != 2:
        raise ValueError(f"Invalid samples shape: {samples.shape}")

    return {
        "rms_mu": float(np.mean(samples[:, 0])),
        "rms_sigma": float(np.std(samples[:, 0])),
        "pitch_mean_mu": float(np.mean(samples[:, 1])),
        "pitch_mean_sigma": float(np.std(samples[:, 1])),
        "pitch_std_mu": float(np.mean(samples[:, 2])),
        "pitch_std_sigma": float(np.std(samples[:, 2])),
        "zcr_mu": float(np.mean(samples[:, 3])),
        "zcr_sigma": float(np.std(samples[:, 3])),
    }


# 🔥 STEP 3 — BUILD MODEL
model = {}

print("\n--- SUMMARY ---")
for emotion, samples in data.items():
    print(f"{emotion}: {len(samples)} samples")

    if len(samples) < 10:
        print(f"Skipping {emotion} (not enough data)")
        continue

    model[emotion] = compute_params(samples)


# 💾 STEP 4 — SAVE MODEL
os.makedirs("src/models", exist_ok=True)

with open("src/models/audio_model.json", "w") as f:
    json.dump(model, f, indent=2)

print("Model saved to src/models/audio_model.json")
