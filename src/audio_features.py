import librosa
import numpy as np
from config.audio_config import MAX_PITCH, MIN_PITCH


def extract_features(audio_chunk, sr=16000, n_mfcc=13):
    audio_chunk = audio_chunk / (np.max(np.abs(audio_chunk)) + 1e-8)

    # RMS
    rms = float(np.sqrt(np.mean(audio_chunk**2)))

    # Pitch
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

    # MFCC
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=n_mfcc)

    # Delta
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    # Stats
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    delta_mean = np.mean(mfcc_delta, axis=1)
    delta_std = np.std(mfcc_delta, axis=1)

    delta2_mean = np.mean(mfcc_delta2, axis=1)
    delta2_std = np.std(mfcc_delta2, axis=1)

    # Spectral features
    spectral_centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=audio_chunk, sr=sr))
    )
    spectral_bandwidth = float(
        np.mean(librosa.feature.spectral_bandwidth(y=audio_chunk, sr=sr))
    )

    feature_vector = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            delta_mean,
            delta_std,
            delta2_mean,
            delta2_std,
            [pitch_mean, pitch_std, zcr, rms, spectral_centroid, spectral_bandwidth],
        ]
    )

    return feature_vector
