import sounddevice as sd
from config.audio_config import SAMPLE_RATE, BLOCK_SIZE, CHANNELS, DTYPE, CHUNK_SECONDS
import queue
import numpy as np

audio_queue = queue.Queue()  # shared queue — imported by transcriber too


def start_capture():
    buffer = []
    samples_needed = SAMPLE_RATE * CHUNK_SECONDS

    def callback(indata, frames, time, status):
        # indata is a numpy array of shape (frames, 1)
        # this function is called automatically by sounddevice
        buffer.append(indata.copy())

        total = sum(len(b) for b in buffer)
        if total >= samples_needed:
            # concatenate all buffered chunks into one array
            chunk = np.concatenate(buffer, axis=0).flatten()
            audio_queue.put(chunk)
            buffer.clear()

    # start listening — this runs forever in background
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=callback
    ):
        print("Mic is live.")
        # keep thread alive forever
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    import threading

    t = threading.Thread(target=start_capture)
    t.start()

    import time

    while True:
        print(f"Queue size:{audio_queue.qsize()}")
        time.sleep(1)
