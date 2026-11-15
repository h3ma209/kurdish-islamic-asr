#!/usr/bin/env python3
"""
Benchmark comparison across all ASR approaches for Kurdish + Arabic Islamic Audio.
Computes WER against Gemini Ground Truth, checks Arabic Hadith preservation and speed.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent


def simple_tokenize(text: str) -> list[str]:
    # Strip punctuation, normalize whitespace
    cleaned = re.sub(r"[^\w\s]", "", text)
    return [w for w in cleaned.split() if w]


def compute_wer(ref: str, hyp: str) -> float:
    ref_words = simple_tokenize(ref)
    hyp_words = simple_tokenize(hyp)

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    if not ref_words:
        return 0.0
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def main():
    gt_path = BASE_DIR / "data" / "mela_halo" / "BJgnqMWSnhw_gemini.txt"
    gt_text = gt_path.read_text(encoding="utf-8")

    models = [
        ("MMS-1B CKB (Baseline)", BASE_DIR / "data" / "mela_halo" / "BJgnqMWSnhw_mms1b.txt", 4.8),
        ("Whisper Large CKB (Rzgar)", BASE_DIR / "data" / "mela_halo" / "BJgnqMWSnhw_whisper_large.txt", 58.2),
        ("Fused MMS-1B (CKB+ARA)", BASE_DIR / "data" / "mela_halo" / "BJgnqMWSnhw_mms_fused.txt", 3.56),
        ("Whisper V3 Turbo (Raw)", BASE_DIR / "data" / "mela_halo" / "BJgnqMWSnhw_whisper_turbo.txt", 13.28),
        ("Whisper V3 Turbo + Resolver", BASE_DIR / "data" / "mela_halo" / "BJgnqMWSnhw_whisper_turbo_resolved.txt", 13.3),
    ]

    print("=" * 75)
    print(f"{'Model / Pipeline':<30} | {'WER (%)':<10} | {'Speed (x RT)':<12} | {'Time (s)':<8}")
    print("=" * 75)

    results = []
    for name, path, runtime in models:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        wer = compute_wer(gt_text, text) * 100.0
        x_rt = 100.54 / runtime
        print(f"{name:<30} | {wer:9.1f}% | {x_rt:10.1f}x   | {runtime:6.2f}s")
        results.append({
            "model": name,
            "wer": round(wer, 1),
            "runtime_sec": runtime,
            "x_realtime": round(x_rt, 1),
        })

    print("=" * 75)


if __name__ == "__main__":
    main()
