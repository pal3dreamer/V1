import librosa
import numpy as np
from config.audio_config import MAX_PITCH, MIN_PITCH


def extract_features(
    audio_chunk,
    sr=16000,
    n_mfcc=13,
    add_spectral_contrast=True,
    add_rolloff=True,
    add_chroma=True,
    add_tonnetz=True,
    add_hpss=True,
    add_jitter_shimmer=True,
    min_fft_size=256,  # minimum FFT size to attempt
):
    """
    Extract features from an audio chunk, handling short signals gracefully.
    """
    # Ensure audio is a numpy array and at least 2 samples long
    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
    if len(audio_chunk) < 2:
        # Return zero feature vector of appropriate size
        return np.zeros(138)  # adjust based on feature set

    # For very short signals, pad to a reasonable length (e.g., 256)
    # Use n_fft = min(512, len(audio_chunk))? We'll compute features with smaller windows.
    # For librosa functions that accept n_fft, we'll set it to the minimum of len(audio) and default.
    # We'll use the actual audio length for FFT-related functions, but pad if too short.

    # Determine effective FFT size (must be <= len(audio))
    # We'll use a default n_fft=512, but cap it to the audio length (or pad if audio is very short)
    target_fft = 512
    if len(audio_chunk) < target_fft:
        # Pad audio to target_fft (or to nearest power of two)
        pad_len = target_fft - len(audio_chunk)
        audio_padded = np.pad(audio_chunk, (0, pad_len), mode="constant")
    else:
        audio_padded = audio_chunk
        target_fft = min(target_fft, len(audio_chunk))

    # Normalize
    audio_padded = audio_padded / (np.max(np.abs(audio_padded)) + 1e-8)

    # ---------- 1. RMS ----------
    rms = float(np.sqrt(np.mean(audio_padded**2)))

    # ---------- 2. Pitch (YIN) ----------
    # YIN works even on short signals, but may fail; use try-except
    try:
        pitches = librosa.yin(audio_padded, fmin=MIN_PITCH, fmax=MAX_PITCH, sr=sr)
        pitches = pitches[pitches > 0]
        if len(pitches):
            pitch_mean = float(np.mean(pitches))
            pitch_std = float(np.std(pitches))
        else:
            pitch_mean = MIN_PITCH
            pitch_std = 0.0
    except Exception:
        pitch_mean = MIN_PITCH
        pitch_std = 0.0

    # ---------- 3. Zero Crossing Rate ----------
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_padded)[0]))

    # ---------- 4. Spectral Centroid & Bandwidth ----------
    # Use the padded audio for spectral features
    spec_cent = librosa.feature.spectral_centroid(
        y=audio_padded, sr=sr, n_fft=target_fft
    )[0]
    spec_cent_mean = float(np.mean(spec_cent))
    spec_bw = librosa.feature.spectral_bandwidth(
        y=audio_padded, sr=sr, n_fft=target_fft
    )[0]
    spec_bw_mean = float(np.mean(spec_bw))

    # ---------- 5. MFCC + deltas ----------
    mfccs = librosa.feature.mfcc(y=audio_padded, sr=sr, n_mfcc=n_mfcc, n_fft=target_fft)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    delta_mean = np.mean(mfcc_delta, axis=1)
    delta_std = np.std(mfcc_delta, axis=1)
    delta2_mean = np.mean(mfcc_delta2, axis=1)
    delta2_std = np.std(mfcc_delta2, axis=1)

    # Start with mandatory features
    feature_list = [
        mfcc_mean,
        mfcc_std,
        delta_mean,
        delta_std,
        delta2_mean,
        delta2_std,
        [pitch_mean, pitch_std, zcr, rms, spec_cent_mean, spec_bw_mean],
    ]

    # ---------- 6. Spectral Contrast ----------
    if add_spectral_contrast:
        try:
            contrast = librosa.feature.spectral_contrast(
                y=audio_padded, sr=sr, n_bands=6, n_fft=target_fft
            )
            contrast_mean = np.mean(contrast, axis=1)
            contrast_std = np.std(contrast, axis=1)
            feature_list.append(contrast_mean)
            feature_list.append(contrast_std)
        except Exception:
            feature_list.append(np.zeros(6))
            feature_list.append(np.zeros(6))

    # ---------- 7. Spectral Rolloff ----------
    if add_rolloff:
        try:
            rolloff = librosa.feature.spectral_rolloff(
                y=audio_padded, sr=sr, n_fft=target_fft
            )[0]
            rolloff_mean = np.mean(rolloff)
            feature_list.append([rolloff_mean])
        except Exception:
            feature_list.append([0.0])

    # ---------- 8. Chroma Features ----------
    if add_chroma:
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio_padded, sr=sr, n_fft=target_fft
            )
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            feature_list.append(chroma_mean)
            feature_list.append(chroma_std)
        except Exception:
            feature_list.append(np.zeros(12))
            feature_list.append(np.zeros(12))

    # ---------- 9. Tonnetz ----------
    if add_tonnetz:
        try:
            tonnetz = librosa.feature.tonnetz(y=audio_padded, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_std = np.std(tonnetz, axis=1)
            feature_list.append(tonnetz_mean)
            feature_list.append(tonnetz_std)
        except Exception:
            feature_list.append(np.zeros(6))
            feature_list.append(np.zeros(6))

    # ---------- 10. Harmonic-Percussive Separation ----------
    if add_hpss:
        try:
            harmonic, percussive = librosa.effects.hpss(audio_padded)
            rms_harmonic = float(np.sqrt(np.mean(harmonic**2)))
            rms_percussive = float(np.sqrt(np.mean(percussive**2)))
            harmonic_ratio = rms_harmonic / (rms_harmonic + rms_percussive + 1e-8)
            feature_list.append([rms_harmonic, rms_percussive, harmonic_ratio])
        except Exception:
            feature_list.append([0.0, 0.0, 0.0])

    # ---------- 11. Jitter & Shimmer ----------
    if add_jitter_shimmer:
        try:
            # Use pyin with the original (unpadded) audio for pitch analysis
            f0, voiced_flag, _ = librosa.pyin(
                audio_chunk, fmin=MIN_PITCH, fmax=MAX_PITCH, sr=sr, fill_na=np.nan
            )
            voiced_frames = np.where(voiced_flag)[0]
            if len(voiced_frames) > 1:
                # Jitter
                f0_voiced = f0[voiced_frames]
                pitch_periods = 1.0 / f0_voiced
                pitch_period_diffs = np.abs(np.diff(pitch_periods))
                jitter = np.mean(pitch_period_diffs) / np.mean(pitch_periods)

                # Shimmer: use RMS of frames aligned with pyin
                hop_len_pyin = 512
                rms_pyin = librosa.feature.rms(
                    y=audio_chunk,
                    frame_length=2048,
                    hop_length=hop_len_pyin,
                    center=True,
                )[0]
                if len(voiced_frames) > 1:
                    rms_voiced = rms_pyin[voiced_frames]
                    rms_diffs = np.abs(np.diff(rms_voiced))
                    shimmer = np.mean(rms_diffs) / (np.mean(rms_voiced) + 1e-8)
                else:
                    shimmer = 0.0
                feature_list.append([jitter, shimmer])
            else:
                feature_list.append([0.0, 0.0])
        except Exception:
            feature_list.append([0.0, 0.0])

    # Concatenate
    feature_vector = np.concatenate(feature_list)
    return feature_vector
