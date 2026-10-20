"""Compare overall WER with aligned civic-term occurrence accuracy on public audio."""

import json
import re
from math import gcd
from pathlib import Path
import jiwer
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

REFERENCE = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
TERMS = ["fellow americans", "country"]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def scores(reference: str, hypothesis: str, terms: list[str]) -> dict:
    reference = normalize(reference)
    hypothesis = normalize(hypothesis)
    alignment = jiwer.process_words(reference, hypothesis)
    equal = set()
    for chunk in alignment.alignments[0]:
        if chunk.type == "equal":
            equal.update(range(chunk.ref_start_idx, chunk.ref_end_idx))
    words = reference.split()
    correct = 0
    total = 0
    for term in terms:
        phrase = normalize(term).split()
        for i in range(len(words) - len(phrase) + 1):
            if words[i : i + len(phrase)] == phrase:
                total += 1
                correct += all(j in equal for j in range(i, i + len(phrase)))
    return {
        "wer": alignment.wer,
        "term_occurrences": total,
        "term_correct": correct,
        "term_accuracy": correct / total if total else None,
    }


def benchmark() -> dict:
    audio, rate = sf.read(".cache/jfk.flac", dtype="float32")
    audio = audio.mean(axis=1) if audio.ndim == 2 else audio
    factor = gcd(rate, 16000)
    audio = resample_poly(audio, 16000 // factor, rate // factor).astype("float32")
    rng = np.random.default_rng(42)
    variants = {
        "clean": audio,
        "noise": (audio + rng.normal(0, 0.02, len(audio))).astype("float32"),
        "quiet": audio * 0.25,
    }
    rows = []
    for name in ["tiny", "base"]:
        model = WhisperModel(
            ".cache/whisper-" + name, device="cpu", compute_type="int8", cpu_threads=2
        )
        for variant, values in variants.items():
            text = " ".join(s.text for s in model.transcribe(values, language="en", beam_size=1)[0])
            rows.append(
                {
                    "model": name,
                    "variant": variant,
                    "hypothesis": text,
                    **scores(REFERENCE, text, TERMS),
                }
            )
    Path("evidence/comparison.json").write_text(json.dumps(rows, indent=2))
    Path("evidence/report.md").write_text(
        "| Model | Variant | WER | Civic-term accuracy |\n|---|---|---|---|\n"
        + "".join(
            f"| {r['model']} | {r['variant']} | {r['wer']} | {r['term_accuracy']} |\n" for r in rows
        )
    )
    return {
        "models_compared": 2,
        "audio_variants_per_model": len(variants),
        "term_occurrences_evaluated": sum(r["term_occurrences"] for r in rows),
        "rows": rows,
        "source": "https://github.com/openai/whisper/blob/main/tests/jfk.flac",
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
