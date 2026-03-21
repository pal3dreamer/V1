# Audio input settings

SAMPLE_RATE = 16000  # Hz
BLOCK_SIZE = 1600  # 100ms chunks

CHANNELS = 1  # mono mic
DTYPE = "float32"  # audio format

# Device (None = default mic)
INPUT_DEVICE = None
OUTPUT_DEVICE = None
CHUNK_SECONDS = 0.5

# PITCH RANGES
MIN_PITCH = 80
MAX_PITCH = 250
