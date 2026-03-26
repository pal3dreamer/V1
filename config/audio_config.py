# Audio input settings

SAMPLE_RATE = 16000  # Hz
BLOCK_SIZE = 1600  # 100ms chunks

CHANNELS = 1  # mono mic
DTYPE = "float32"  # audio format

# Device (None = default mic)
INPUT_DEVICE = None
OUTPUT_DEVICE = None
CHUNK_SECONDS = 2

# PITCH RANGES
MIN_PITCH = 80
MAX_PITCH = 250

DATASET_PATH = "/home/het/Downloads/audio-emotions/Emotions"

EMOTION_MAP = EMOTION_MAP = {
    "angry": "anger",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "suprised": "surprise",
}
