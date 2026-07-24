# STT CPU Benchmark: NVIDIA Nemotron ASR vs OpenAI Whisper

A small proof-of-concept comparing two **open** speech-to-text models for **CPU-only**
transcription: one built natively for real-time streaming, and one adapted to it. The
goal is to answer a practical question — *if you can't rely on a GPU, what does
real-time-ish transcription actually cost, and which model is worth it?*

This is a benchmarking/exploration project, not a production system. Numbers are
collected automatically into CSV files as each script runs, so results are measured,
not estimated.

## The two models

- **[NVIDIA Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)** (`nemotron-3.5-asr-streaming-0.6b`) — a 600M-parameter Cache-Aware
  FastConformer-RNNT. This is a genuinely **streaming-native** architecture: it processes
  small audio chunks (80ms–1.12s, configurable) incrementally, reusing cached encoder
  state instead of reprocessing the whole utterance. It's the closest thing here to a
  "real real-time" model.
- **[OpenAI Whisper](https://github.com/openai/whisper)** (`large-v3-turbo`, via
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / CTranslate2) — a
  batch-oriented encoder-decoder Transformer. It has **no native streaming support**: it
  expects a full audio window at once. To approximate real-time behavior, this repo
  layers a simple silence-triggered chunking strategy (energy-based VAD, no ML) on top
  of it — buffer while speech is detected, transcribe the whole utterance once a pause
  is found. It's included both as a real-time comparison point and as a fast, low-latency
  option for one-shot, non-streaming use cases (e.g. transcribing a short voice command).

Both models run **quantized to INT8** on CPU to keep resource usage as low as possible —
dynamic quantization (`torch.ao.quantization`) for Nemotron, native CTranslate2
quantization for Whisper.

## What's in this repo

| Script | What it does |
|---|---|
| `transcribe_mic.py` | Real-time transcription with Nemotron, capturing system/loopback audio (Linux/PipeWire). Logs latency, CPU, RAM, and real-time-factor per run to `benchmark_results.csv`. |
| `transcribe_whisper_stream.py` | Same idea, but with Whisper + silence-triggered chunking, to compare against Nemotron's native streaming. Logs to `whisper_streaming_benchmark_results.csv`. |
| `transcribe_whisper_file.py` | One-shot, stateless transcription of a single audio file with Whisper — no live capture, meant for short "voice command"-style audio. Logs to `whisper_benchmark_results.csv`. |
| `score_transcription.py` | Compares a transcription against a ground-truth reference text and computes WER/CER (word/character error rate), logging to `accuracy_results.csv`. |
| `references/` | Ground-truth texts used to score transcription accuracy. |

Each transcription script prints **only** the transcribed text to stdout (so it's
pipeable/redirectable); all status, timing, and diagnostic output goes to stderr.

## Setup

Requires Python 3.12+ and, for the real-time scripts, a **Linux system running
PipeWire** (uses `pw-record` / `wpctl` to capture system audio — this part won't work
on macOS/Windows or on PulseAudio-only setups without adaptation).

System dependencies (Debian/Ubuntu):

```bash
sudo apt install -y libportaudio2 ffmpeg
```

Install Python dependencies, using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

or plain `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Models are downloaded automatically from Hugging Face on first run (Nemotron ~2.4GB,
Whisper large-v3-turbo ~1.6GB) and cached locally afterwards.

## Usage

```bash
# Real-time streaming, Nemotron (Ctrl+C to stop and print the summary)
python transcribe_mic.py

# Real-time-ish streaming, Whisper + silence-triggered chunking
python transcribe_whisper_stream.py

# One-shot transcription of a file, Whisper (voice-command style)
python transcribe_whisper_file.py path/to/audio.wav

# Score a transcription against a reference text
python score_transcription.py \
  --engine nemotron --audio-file sample.wav --language pt --run 1 \
  --reference @references/some_reference.txt \
  --hypothesis "text the model produced"
```

## Methodology notes / known limitations

This is an early-stage POC, not a rigorous benchmark — worth keeping in mind when
reading the CSVs:

- Small sample size: only a handful of audio clips, mostly run once per condition.
- Some reference texts were manually corrected against apparent errors in scraped
  captions; a couple of ambiguous spots were resolved using a model's own output as the
  best available guess, which can slightly bias that specific comparison.
- CPU/latency numbers were collected on a personal laptop with normal background load,
  not an isolated/dedicated benchmarking machine.
- Word Error Rate is sensitive to formatting mismatches that aren't real transcription
  errors — e.g. spelled-out vs. digit numbers ("thirty three" vs "33") count as errors
  even when semantically correct.

## License

No license file yet — add one before relying on this repo being reusable by others.
