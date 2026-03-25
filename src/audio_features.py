import librosa
import numpy as np
from config.audio_config import MAX_PITCH, MIN_PITCH


def extract_features(audio_chunk, sr=16000, n_mfcc=13):
    # normalize audio
    audio_chunk = audio_chunk / (np.max(np.abs(audio_chunk)) + 1e-8)

    # RMS energy
    rms = float(np.sqrt(np.mean(audio_chunk**2)))

    # pitch
    pitches = librosa.yin(audio_chunk, fmin=MIN_PITCH, fmax=MAX_PITCH, sr=sr)
    pitches = pitches[pitches > 0]

    if len(pitches):
        pitch_mean = float(np.mean(pitches))
        pitch_std = float(np.std(pitches))
    else:
        pitch_mean = MIN_PITCH
        pitch_std = 0.0

    # ZCR
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_chunk)[0]))

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)

    feature_vector = np.concatenate([mfcc_mean, [pitch_mean, pitch_std, zcr, rms]])
    return feature_vector
