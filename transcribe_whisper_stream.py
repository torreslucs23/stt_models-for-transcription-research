"""Pseudo real-time transcription with a quantized Whisper (faster-whisper / CTranslate2, INT8 CPU).

Whisper has no native streaming/cache-aware architecture -- there is no separate "streaming"
checkpoint like nemotron-3.5-asr-streaming. This script fakes "live" transcription with a
silence-triggered chunking strategy (simple energy-based VAD, no ML model for it): audio is
buffered while speech is detected; once a pause is found (or a max buffer duration is hit as
a safety cap for continuous speech), the whole buffered utterance is transcribed in ONE batch
call and the buffer resets for the next utterance.

This is fundamentally different from transcribe_mic.py's Nemotron pipeline, which processes
small overlapping chunks incrementally with cache reuse. Here, each utterance is an
independent, self-contained batch job -- there's no cache/context carried between utterances.

Captures system/screen audio the same way as transcribe_mic.py (PipeWire `pw-record`
targeting the default sink's monitor), so results are directly comparable to the Nemotron
streaming benchmark. Logs one summary row per run to whisper_streaming_benchmark_results.csv.

stdout carries only the transcribed text; status/diagnostics go to stderr.
"""

import csv
import os
import statistics
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

import numpy as np
import psutil
from faster_whisper import WhisperModel

MODEL_SIZE = "large-v3-turbo"  # v3 encoder, distilled 4-layer decoder -- fast, low resource use
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LANGUAGE = None  # None = auto-detect -- forcing "pt" leaked Portuguese words into English audio
BEAM_SIZE = 1

SAMPLING_RATE = 16000
BYTES_PER_SAMPLE = 4  # float32
READ_CHUNK_SAMPLES = 1024  # ~64ms per chunk at 16kHz -- VAD resolution

# --- Silence-triggered chunking (VAD-lite: energy threshold, no ML) ---
SILENCE_RMS_THRESHOLD = 0.01  # below this RMS = "silence" -- tune to your noise floor
SILENCE_DURATION_S = 0.6  # pause length that closes an utterance and triggers transcription
MAX_UTTERANCE_S = 15.0  # force-flush safety cap even without silence (bounds worst-case latency)
MIN_UTTERANCE_S = 0.3  # ignore too-short blips (coughs, clicks, noise spikes)

RESOURCE_SAMPLE_INTERVAL_S = 1.0
CSV_PATH = os.path.join(os.path.dirname(__file__), "whisper_streaming_benchmark_results.csv")
CSV_FIELDS = [
    "timestamp",
    "model_size",
    "device",
    "compute_type",
    "test_duration_s",
    "audio_processed_s",
    "realtime_factor",
    "num_utterances",
    "latency_avg_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_max_ms",
    "cpu_avg_pct",
    "cpu_peak_pct",
    "ram_avg_mb",
    "ram_peak_mb",
]


def log(message: str) -> None:
    print(message, file=sys.stderr)


class ResourceMonitor:
    """Samples CPU%/RAM of this process only (not the pw-record capture subprocess)."""

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


def read_exact(stream, n_bytes: int) -> bytes:
    chunks = []
    remaining = n_bytes
    while remaining > 0:
        data = stream.read(remaining)
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def get_default_sink_id() -> str:
    """`pw-record --target @DEFAULT_SINK@` silently falls back to the default
    microphone instead of the sink monitor, so resolve the numeric id ourselves."""
    output = subprocess.check_output(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"], text=True)
    return output.split()[1].rstrip(",")  # "id 51, type ..." -> "51"


def start_system_audio_capture(sampling_rate: int, audio_queue: Queue) -> subprocess.Popen:
    """Launches `pw-record`, targeting the default sink's monitor, and streams
    raw float32 PCM samples into audio_queue from a background thread."""
    sink_id = get_default_sink_id()
    log(f"Capturing monitor of sink id {sink_id}")
    proc = subprocess.Popen(
        [
            "pw-record",
            "--target", sink_id,
            "--rate", str(sampling_rate),
            "--channels", "1",
            "--format", "f32",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    def reader():
        chunk_bytes = READ_CHUNK_SAMPLES * BYTES_PER_SAMPLE
        while True:
            data = read_exact(proc.stdout, chunk_bytes)
            if not data:
                log("[pw-record] stream ended unexpectedly")
                break
            audio_queue.put(np.frombuffer(data, dtype=np.float32))

    threading.Thread(target=reader, daemon=True).start()
    return proc


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * p), len(ordered) - 1)
    return ordered[idx]


def write_csv_row(row: dict) -> None:
    is_new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    log(f"Loading Whisper '{MODEL_SIZE}' (compute_type={COMPUTE_TYPE}, device={DEVICE})...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    audio_queue: Queue[np.ndarray] = Queue()
    capture_proc = start_system_audio_capture(SAMPLING_RATE, audio_queue)
    resources = ResourceMonitor()

    latencies_ms: list[float] = []
    num_utterances = 0
    total_audio_processed_s = 0.0

    utterance_buffer = np.zeros(0, dtype=np.float32)
    silence_duration_s = 0.0
    speech_end_time = None

    def flush_utterance() -> None:
        nonlocal utterance_buffer, num_utterances, total_audio_processed_s, speech_end_time
        duration_s = utterance_buffer.shape[0] / SAMPLING_RATE
        if duration_s < MIN_UTTERANCE_S:
            utterance_buffer = np.zeros(0, dtype=np.float32)
            return

        segments, _info = model.transcribe(utterance_buffer, language=LANGUAGE, beam_size=BEAM_SIZE)
        text = "".join(segment.text for segment in segments)

        latency_ms = (time.monotonic() - speech_end_time) * 1000 if speech_end_time else 0.0
        latencies_ms.append(latency_ms)
        num_utterances += 1
        total_audio_processed_s += duration_s
        print(text, end=" ", flush=True)
        utterance_buffer = np.zeros(0, dtype=np.float32)

    log("Listening for speech (silence-triggered chunking, Ctrl+C to stop)...")
    resources.start()
    test_started_at = time.monotonic()

    try:
        while True:
            try:
                chunk = audio_queue.get(timeout=0.5)
            except Empty:
                continue

            chunk_duration_s = chunk.shape[0] / SAMPLING_RATE
            rms = float(np.sqrt(np.mean(chunk**2)))

            if rms >= SILENCE_RMS_THRESHOLD:
                utterance_buffer = np.concatenate([utterance_buffer, chunk])
                silence_duration_s = 0.0
                speech_end_time = None
            elif utterance_buffer.shape[0] > 0:
                silence_duration_s += chunk_duration_s
                if speech_end_time is None:
                    speech_end_time = time.monotonic()
                if silence_duration_s >= SILENCE_DURATION_S:
                    flush_utterance()
                    silence_duration_s = 0.0

            if utterance_buffer.shape[0] / SAMPLING_RATE >= MAX_UTTERANCE_S:
                speech_end_time = time.monotonic()
                flush_utterance()
                silence_duration_s = 0.0
    except KeyboardInterrupt:
        pass
    finally:
        if utterance_buffer.shape[0] / SAMPLING_RATE >= MIN_UTTERANCE_S:
            speech_end_time = time.monotonic()
            flush_utterance()
        test_duration_s = time.monotonic() - test_started_at
        resources.stop()
        capture_proc.terminate()
        print()

    realtime_factor = total_audio_processed_s / test_duration_s if test_duration_s > 0 else 0

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "test_duration_s": round(test_duration_s, 2),
        "audio_processed_s": round(total_audio_processed_s, 2),
        "realtime_factor": round(realtime_factor, 3),
        "num_utterances": num_utterances,
        "latency_avg_ms": round(statistics.fmean(latencies_ms), 1) if latencies_ms else 0,
        "latency_p50_ms": round(percentile(latencies_ms, 0.50), 1),
        "latency_p95_ms": round(percentile(latencies_ms, 0.95), 1),
        "latency_max_ms": round(max(latencies_ms), 1) if latencies_ms else 0,
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
