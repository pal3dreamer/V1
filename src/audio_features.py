import librosa
import numpy as np
from config.audio_config import MAX_PITCH, MIN_PITCH


def extract_features(audio_chunk, sr=16000):
    rms = float(np.sqrt(np.mean(audio_chunk**2)))
    pitches = librosa.yin(audio_chunk, fmin=MIN_PITCH, fmax=MAX_PITCH, sr=sr)
    pitches = pitches[pitches > 0]
    pitch_mean = float(np.mean(pitches)) if len(pitches) else 0
    # std deviation
    pitch_std = float(np.std(pitches)) if len(pitches) else 0

    # Zero crossing rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_chunk)))
    return rms, pitch_mean, pitch_std, zcr
