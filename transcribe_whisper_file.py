"""One-shot transcription of a local audio file with a quantized Whisper model
(faster-whisper / CTranslate2, INT8 on CPU).

Unlike transcribe_mic.py, this is NOT real-time streaming: it's meant for short,
stateless invocations (e.g. a voice command clip to forward to a chat), one audio
file per run, no live capture, no session state between calls.

Usage:
    python transcribe_whisper_file.py path/to/audio.wav

Logs one row per run to whisper_benchmark_results.csv (kept separate from the
Nemotron streaming benchmark's benchmark_results.csv). stdout carries only the
transcribed text; status/diagnostic messages go to stderr.
"""

import csv
import os
import statistics
import sys
import threading
import time

import psutil
from faster_whisper import WhisperModel

MODEL_SIZE = "base"  # tiny, base, small, medium, large-v3 -- bigger = more accurate, slower, more RAM
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # quantized weights (CTranslate2), lowest CPU/RAM footprint
LANGUAGE = None  # None = auto-detect from the audio; or force e.g. "pt", "en"
BEAM_SIZE = 1  # greedy decoding -- faster, fine for short voice-command-style audio

RESOURCE_SAMPLE_INTERVAL_S = 0.2  # finer than the streaming benchmark: these runs are short
CSV_PATH = os.path.join(os.path.dirname(__file__), "whisper_benchmark_results.csv")
CSV_FIELDS = [
    "timestamp",
    "audio_file",
    "model_size",
    "device",
    "compute_type",
    "audio_duration_s",
    "processing_time_s",
    "speedup_factor",
    "cpu_avg_pct",
    "cpu_peak_pct",
    "ram_avg_mb",
    "ram_peak_mb",
]


def log(message: str) -> None:
    print(message, file=sys.stderr)


class ResourceMonitor:
    """Samples CPU%/RAM of this process only, at a fine interval since file
    transcription is short-lived (unlike the continuous streaming benchmark)."""

    def __init__(self, interval_s: float = RESOURCE_SAMPLE_INTERVAL_S):
        self._interval_s = interval_s
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)  # prime the internal counter
        self._stop_event = threading.Event()
        self.cpu_samples: list[float] = []
        self.ram_samples_mb: list[float] = []

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self.cpu_samples.append(self._process.cpu_percent(interval=None))
            self.ram_samples_mb.append(self._process.memory_info().rss / (1024 * 1024))

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        self._stop_event.set()
        self.cpu_samples.append(self._process.cpu_percent(interval=None))
        self.ram_samples_mb.append(self._process.memory_info().rss / (1024 * 1024))


def write_csv_row(row: dict) -> None:
    is_new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    if len(sys.argv) != 2:
        log(f"Usage: python {os.path.basename(__file__)} <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.isfile(audio_path):
        log(f"File not found: {audio_path}")
        sys.exit(1)

    log(f"Loading Whisper '{MODEL_SIZE}' (compute_type={COMPUTE_TYPE}, device={DEVICE})...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    resources = ResourceMonitor()
    resources.start()
    start_time = time.monotonic()

    segments, info = model.transcribe(audio_path, language=LANGUAGE, beam_size=BEAM_SIZE)
    for segment in segments:
        print(segment.text, end="", flush=True)

    processing_time_s = time.monotonic() - start_time
    resources.stop()
    print()

    audio_duration_s = info.duration
    speedup_factor = audio_duration_s / processing_time_s if processing_time_s > 0 else 0

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "audio_file": os.path.basename(audio_path),
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "audio_duration_s": round(audio_duration_s, 2),
        "processing_time_s": round(processing_time_s, 2),
        "speedup_factor": round(speedup_factor, 2),
        "cpu_avg_pct": round(statistics.fmean(resources.cpu_samples), 1) if resources.cpu_samples else 0,
        "cpu_peak_pct": round(max(resources.cpu_samples), 1) if resources.cpu_samples else 0,
        "ram_avg_mb": round(statistics.fmean(resources.ram_samples_mb), 1) if resources.ram_samples_mb else 0,
        "ram_peak_mb": round(max(resources.ram_samples_mb), 1) if resources.ram_samples_mb else 0,
    }
    write_csv_row(row)

    log("\n--- Test summary ---")
    for key, value in row.items():
        log(f"{key}: {value}")
    log(f"\nAppended to {CSV_PATH}")


if __name__ == "__main__":
    main()
