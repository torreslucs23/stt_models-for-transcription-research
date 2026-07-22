"""POC: real-time transcription of system/screen audio with NVIDIA Nemotron 3.5 ASR (streaming).

Captures whatever is playing on the machine (loopback) instead of the microphone, by
recording the default output sink's monitor via PipeWire's `pw-record` CLI (Linux/PipeWire only).

Also benchmarks the model itself: latency (audio-in -> text-out) and CPU/RAM usage of
this Python process. Resource usage of the pw-record capture subprocess is intentionally
NOT measured -- pw-record is only a local stand-in for audio input here; in production the
model would receive raw bytes over a websocket instead, so only the model process matters
for hosting cost. Each run appends one summary row to benchmark_results.csv.

stdout carries ONLY the transcribed text, so it can be piped/redirected cleanly.
All status/diagnostic messages go to stderr.
"""

import csv
import os
import statistics
import subprocess
import sys
import threading
import time
from queue import Queue

import numpy as np
import psutil
import torch
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
LANGUAGE = "auto"  # or a fixed locale like "pt-BR" if you know the audio's language
DEVICE = "cpu"  # switch to "cuda" if you have an NVIDIA GPU
QUANTIZE = True  # dynamic INT8 quantization of Linear/LSTM weights -- CPU only

BYTES_PER_SAMPLE = 4  # float32
READ_CHUNK_SAMPLES = 1024
RESOURCE_SAMPLE_INTERVAL_S = 1.0
CSV_PATH = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
CSV_FIELDS = [
    "timestamp",
    "model_id",
    "device",
    "quantized",
    "test_duration_s",
    "audio_processed_s",
    "realtime_factor",
    "num_audio_chunks",
    "num_text_events",
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


class LiveAudioBuffer:
    """Accumulates incoming audio and lets us re-read overlapping windows,
    which is what the Nemotron ASR processor expects between consecutive chunks."""

    def __init__(self, audio_queue: Queue):
        self._queue = audio_queue
        self._data = np.zeros(0, dtype=np.float32)

    def __getitem__(self, key):
        return self._data[key]

    def wait_until(self, n_samples: int) -> None:
        while self._data.shape[0] < n_samples:
            self._data = np.concatenate([self._data, self._queue.get()])


class ResourceMonitor:
    """Samples CPU% / RAM of this process only (not the pw-record subprocess)."""

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


class LatencyTracker:
    """Rough audio-in -> text-out latency: time between the most recently fed
    audio chunk and the next piece of text the model emits for it."""

    def __init__(self):
        self._last_chunk_time = time.monotonic()
        self.samples_ms: list[float] = []
        self.num_chunks = 0

    def mark_chunk_fed(self) -> None:
        self._last_chunk_time = time.monotonic()
        self.num_chunks += 1

    def mark_text_received(self) -> None:
        self.samples_ms.append((time.monotonic() - self._last_chunk_time) * 1000)


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
    log(f"Loading {MODEL_ID} on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID, device_map=DEVICE)

    if QUANTIZE:
        if DEVICE != "cpu":
            raise RuntimeError("Dynamic INT8 quantization only targets CPU (fbgemm/qnnpack).")
        log("Applying dynamic INT8 quantization to Linear/LSTM weights...")
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
        )

    processor.set_num_lookahead_tokens(6)
    log(f"Streaming latency (model-reported): {processor.streaming_latency_ms} ms")

    sampling_rate = processor.feature_extractor.sampling_rate
    hop_length = processor.feature_extractor.hop_length
    n_fft = processor.feature_extractor.n_fft

    audio_queue: Queue[np.ndarray] = Queue()
    audio_history = LiveAudioBuffer(audio_queue)

    capture_proc = start_system_audio_capture(sampling_rate, audio_queue)

    latency = LatencyTracker()
    resources = ResourceMonitor()

    audio_history.wait_until(processor.num_samples_first_audio_chunk)
    first_chunk_inputs = processor(
        audio_history[: processor.num_samples_first_audio_chunk],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
        language=LANGUAGE,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)
    latency.mark_chunk_fed()

    def input_features_generator():
        yield first_chunk_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]

        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        start_idx = mel_frame_idx * hop_length - n_fft // 2

        while True:
            end_idx = start_idx + processor.num_samples_per_audio_chunk
            audio_history.wait_until(end_idx)
            inputs = processor(
                audio_history[start_idx:end_idx],
                sampling_rate=sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=LANGUAGE,
                return_tensors="pt",
            ).to(model.device, dtype=model.dtype)
            latency.mark_chunk_fed()
            yield inputs.input_features

            mel_frame_idx += processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    generate_kwargs = {
        **first_chunk_inputs,
        "input_features": input_features_generator(),
        "streamer": streamer,
    }
    generation_thread = threading.Thread(target=model.generate, kwargs=generate_kwargs, daemon=True)

    log("Listening to system audio (Ctrl+C to stop)...")
    test_started_at = time.monotonic()
    resources.start()
    generation_thread.start()

    num_text_events = 0
    try:
        for text_chunk in streamer:
            latency.mark_text_received()
            num_text_events += 1
            if os.environ.get("DEBUG_STREAM"):
                log(f"[debug] chunk#{num_text_events} = {text_chunk!r}")
            print(text_chunk, end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        test_duration_s = time.monotonic() - test_started_at
        resources.stop()
        capture_proc.terminate()
        print()

    audio_processed_s = latency.num_chunks * processor.num_samples_per_audio_chunk / sampling_rate
    # Live audio arrives at exactly 1x real time, so if the model can't keep up, audio_processed_s
    # falls behind test_duration_s. realtime_factor < 1.0 means transcription is lagging further
    # and further behind live audio; >= 1.0 means the model keeps up with real time.
    realtime_factor = audio_processed_s / test_duration_s if test_duration_s > 0 else 0

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_id": MODEL_ID,
        "device": DEVICE,
        "quantized": QUANTIZE,
        "test_duration_s": round(test_duration_s, 2),
        "audio_processed_s": round(audio_processed_s, 2),
        "realtime_factor": round(realtime_factor, 3),
        "num_audio_chunks": latency.num_chunks,
        "num_text_events": num_text_events,
        "latency_avg_ms": round(statistics.fmean(latency.samples_ms), 1) if latency.samples_ms else 0,
        "latency_p50_ms": round(percentile(latency.samples_ms, 0.50), 1),
        "latency_p95_ms": round(percentile(latency.samples_ms, 0.95), 1),
        "latency_max_ms": round(max(latency.samples_ms), 1) if latency.samples_ms else 0,
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
