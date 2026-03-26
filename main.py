import threading
import numpy as np
import queue
import webrtcvad
from collections import deque, Counter
from src.capture import audio_queue, start_capture
from src.audio_features import extract_features
from src.classifier.mlp_classifier import MLPEmotionClassifier
import pygame

pygame.mixer.init()

# Must match the order used during training
EMOTION_NAMES = [
    "happiness",
    "sadness",
    "fear",
    "neutral",
    "disgust",
    "surprise",
    "sarcasm",
]

# Load classifier
classifier = MLPEmotionClassifier(
    model_path="src/models/mlp_model.pt",
    scaler_path="data/scaler.pkl",
    input_dim=84,
    num_classes=7,
)


# Voice Activity Detection
def is_speech_chunk(
    audio_chunk_float, sr=16000, vad_aggressiveness=2, speech_ratio=0.5
):
    vad = webrtcvad.Vad(vad_aggressiveness)
    audio_int16 = (audio_chunk_float * 32767).astype(np.int16)
    frame_duration = 30  # ms
    frame_size = int(sr * frame_duration / 1000)  # 480 samples for 16kHz
    num_frames = len(audio_int16) // frame_size
    if num_frames == 0:
        return False
    speech_frames = 0
    for i in range(num_frames):
        frame = audio_int16[i * frame_size : (i + 1) * frame_size]
        if vad.is_speech(frame.tobytes(), sr):
            speech_frames += 1
    return (speech_frames / num_frames) >= speech_ratio


# Smoothing parameters
HISTORY_LEN = 5  # number of chunks to consider for majority vote
# CONFIDENCE_THRESH = 0.4  # ignore predictions with confidence below this

# State variables
prediction_history = deque(maxlen=HISTORY_LEN)
current_emotion = None  # the last stable emotion we've output


# Placeholder audio playback function (replace with actual playback logic)
def play_emotion_audio(emotion_name):
    """Play the appropriate audio file for the given emotion."""
    # Example using pygame:
    pygame.mixer.music.load(f"sounds/{emotion_name}.mp3")
    pygame.mixer.music.play()
    print(f"🎵 Playing audio for: {emotion_name}")


# Start capture thread
capture_thread = threading.Thread(target=start_capture, daemon=True)
capture_thread.start()

print("Listening... (press Ctrl+C to stop)")

try:
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Skip silence
        if not is_speech_chunk(audio_chunk):
            continue

        # Extract features and predict
        features = extract_features(audio_chunk)
        probs = classifier.predict(features)
        emotion_id = np.argmax(probs)
        confidence = probs[emotion_id]

        # Skip low‑confidence predictions
        #       if confidence < CONFIDENCE_THRESH:
        #           continue

        # Add to history
        prediction_history.append(emotion_id)

        # If history is full, determine the most common emotion
        if len(prediction_history) == HISTORY_LEN:
            # Count occurrences
            counter = Counter(prediction_history)
            # In case of tie, take the most recent? Or just the one with highest count
            most_common_id = counter.most_common(1)[0][0]

            # If this differs from the current stable emotion, trigger change
            if most_common_id != current_emotion:
                current_emotion = most_common_id
                emotion_name = EMOTION_NAMES[current_emotion]
                print(
                    f"Emotion changed to: {emotion_name} (confidence: {confidence:.2f})"
                )
                play_emotion_audio(emotion_name)

except KeyboardInterrupt:
    print("\nStopped.")
