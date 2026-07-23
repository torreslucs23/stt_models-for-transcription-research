"""Score a transcription against a ground-truth reference text (WER/CER), and log
the result to accuracy_results.csv.

Does NOT run any model itself -- it's the comparison step that runs AFTER you've
already produced a transcription with transcribe_mic.py, transcribe_whisper_file.py,
or transcribe_whisper_stream.py. Feed it the reference text (what was actually said)
and the hypothesis text (what the model transcribed).

Both --reference/--hypothesis accept either inline text or, prefixed with "@", a path
to a text file (e.g. @references/audio1_pt.txt).

Usage:
    python score_transcription.py \\
        --engine nemotron --audio-file audio1_pt.wav --language pt --run 1 \\
        --reference @references/audio1_pt.txt \\
        --hypothesis "texto que saiu da transcricao"

Normalization before scoring: lowercase, punctuation stripped, whitespace collapsed.
Accents are kept (e.g. "voce" vs "voce" still counts as an error in Portuguese).
Known caveat: spelled-out vs digit numbers ("thirty three" vs "33") will count as
errors even when semantically correct -- call this out manually when it happens.
"""

import argparse
import csv
import os
import time

import jiwer

CSV_PATH = os.path.join(os.path.dirname(__file__), "accuracy_results.csv")
CSV_FIELDS = [
    "timestamp",
    "engine",
    "audio_file",
    "language",
    "run",
    "reference_text",
    "hypothesis_text",
    "wer",
    "cer",
    "ref_word_count",
    "substitutions",
    "deletions",
    "insertions",
    "hits",
]

NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def resolve_text(value: str) -> str:
    if value.startswith("@"):
        path = value[1:]
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return value.strip()


def write_csv_row(row: dict) -> None:
    is_new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True, choices=["nemotron", "whisper-file", "whisper-stream"])
    parser.add_argument("--audio-file", required=True, help="Filename of the audio tested (just for the record)")
    parser.add_argument("--language", required=True, choices=["pt", "en"])
    parser.add_argument("--run", required=True, type=int, help="Run number (e.g. 1 or 2, for repeated tests)")
    parser.add_argument("--reference", required=True, help='Ground-truth text, or "@path/to/file.txt"')
    parser.add_argument("--hypothesis", required=True, help='Model output text, or "@path/to/file.txt"')
    args = parser.parse_args()

    reference_text = resolve_text(args.reference)
    hypothesis_text = resolve_text(args.hypothesis)

    wer = jiwer.wer(
        reference_text, hypothesis_text,
        reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE,
    )
    cer = jiwer.cer(reference_text, hypothesis_text)
    alignment = jiwer.process_words(
        reference_text, hypothesis_text,
        reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE,
    )

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": args.engine,
        "audio_file": args.audio_file,
        "language": args.language,
        "run": args.run,
        "reference_text": reference_text,
        "hypothesis_text": hypothesis_text,
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "ref_word_count": len(reference_text.split()),
        "substitutions": alignment.substitutions,
        "deletions": alignment.deletions,
        "insertions": alignment.insertions,
        "hits": alignment.hits,
    }
    write_csv_row(row)

    print("--- Accuracy summary ---")
    for key, value in row.items():
        if key not in ("reference_text", "hypothesis_text"):
            print(f"{key}: {value}")
    print(f"\nAppended to {CSV_PATH}")


if __name__ == "__main__":
    main()
