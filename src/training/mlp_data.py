import os
import numpy as np
import librosa
from src.audio_features import extract_features
from sklearn.preprocessing import StandardScaler
import pickle
from config.audio_config import DATASET_PATH, EMOTION_MAP

DATASET_PATH = DATASET_PATH
EMOTION_MAP = EMOTION_MAP


EMOTION_TO_IDX = {v: i for i, v in enumerate(EMOTION_MAP.values())}


def main():
    all_features = []
    all_labels = []

    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue

            filepath = os.path.join(root, file)

            # emotion detection
            parts = root.lower().split(os.sep)

            emotion_folder = None
            for p in parts:
                if p in EMOTION_MAP:
                    emotion_folder = p
                    break

            if emotion_folder is None:
                continue

            label = EMOTION_MAP[emotion_folder]
            label_id = EMOTION_TO_IDX[label]

            print("Processing:", filepath)

            try:
                audio, sr = librosa.load(filepath, sr=16000)
            except:
                continue

            chunk_size = sr

            for start in range(0, len(audio) - chunk_size + 1, chunk_size):
                chunk = audio[start : start + chunk_size]

                feats = extract_features(chunk, sr=sr)

                all_features.append(feats)
                all_labels.append(label_id)

    # ✅ AFTER LOOP
    X = np.array(all_features)
    y = np.array(all_labels)

    print("\nTotal samples:", len(X))

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save for later use

    os.makedirs("data", exist_ok=True)
    np.save("data/X_mlp.npy", X_scaled)
    np.save("data/y_mlp.npy", y)
    with open("data/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved {len(X)} samples, features shape {X.shape[1]}")


if __name__ == "__main__":
    main()
