# Audio Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-green?style=flat)]()
[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey?style=flat&logo=linux)]()
[![Librosa](https://img.shields.io/badge/Librosa-Audio_Analysis-orange?style=flat)](https://librosa.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013274?style=flat&logo=numpy)](https://numpy.org/)

Real-time audio emotion recognition using Gaussian statistical modeling. Captures microphone input, extracts audio features (RMS, pitch, ZCR), and predicts emotional states.

## Features

- Real-time microphone audio capture
- Audio feature extraction (RMS, pitch, zero-crossing rate)
- Gaussian statistical modeling for emotion classification
- Background audio processing with threading
- Multi-emotion support

## Installation

```bash
git clone https://github.com/yourusername/V1.git
cd V1

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install librosa numpy sounddevice
```

## Usage

### Run the Emotion Recognizer

```bash
python main.py
```

The system will start listening through your microphone and continuously output predicted emotions with confidence scores.

### Train a Custom Model

```bash
python train_model.py
```

This will:
1. Traverse your dataset directory
2. Extract audio features from WAV files
3. Compute mean and standard deviation for each emotion
4. Save the trained model to `src/models/audio_model.json`

## Project Structure

```
V1/
├── README.md              # This file
├── LICENSE                # GNU Affero General Public License
├── main.py                # Entry point - runs real-time recognition
├── train_model.py         # Model training script
├── capture.py             # Microphone audio capture
├── audio_features.py      # Feature extraction (RMS, pitch, ZCR)
├── gaussian_model.py      # Gaussian classifier implementation
├── transcriber.py         # Audio transcription utilities
├── audio_player.py        # Audio playback utilities
├── config/
│   └── audio_config.py    # Audio settings and emotion mapping
├── src/
│   ├── models/
│   │   └── audio_model.json   # Trained model parameters
│   └── __init__.py
└── venv/                  # Virtual environment (optional)
```

## Configuration

Edit `config/audio_config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `BLOCK_SIZE` | 1600 | Audio chunk size |
| `CHUNK_SECONDS` | 0.5 | Recording duration per chunk |
| `MIN_PITCH` | 80 | Minimum pitch frequency (Hz) |
| `MAX_PITCH` | 250 | Maximum pitch frequency (Hz) |
| `DATASET_PATH` | - | Path to training dataset |
| `INPUT_DEVICE` | None | Microphone device index |

## Model Training

1. Organize your dataset in folders named by emotion:
   ```
   dataset/
   ├── happy/
   │   ├── audio1.wav
   │   └── audio2.wav
   ├── sad/
   │   ├── audio1.wav
   │   └── audio2.wav
   └── ...
   ```

2. Update `config/audio_config.py`:
   - Set `DATASET_PATH` to your dataset location
   - Customize `EMOTION_MAP` if needed

3. Run training:
   ```bash
   python train_model.py
   ```

The model computes statistical parameters (mean, std) for each emotion's audio features.

## Supported Emotions

| Emotion | Description |
|---------|-------------|
| Happiness | Happy/positive emotional state |
| Sadness | Sad/negative emotional state |
| Fear | Fearful emotional state |
| Neutral | Baseline emotional state |
| Disgust | Disgusted emotional state |
| Surprise | Surprised emotional state |
| Sarcasm | Sarcastic vocal tone |

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
