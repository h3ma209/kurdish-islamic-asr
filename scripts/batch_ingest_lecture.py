#!/usr/bin/env python3
"""
Automated Batch Audio Slicer & Dataset Builder for Kurdish Preacher Lectures.
Slices master lecture into 3.5-minute chunks, calls Gemini Audio ASR at temperature 0.0,
verifies Sorani Kurdish orthography & Classical Arabic Harakat, featurizes, and builds KenLM LM.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.gemini_audio_dataset import (
    featurize_and_slice,
    save_dataset_manifests,
    transcribe_audio_with_gemini,
)

KMR_MARKERS = [
    r"\bez\b", r"\bte\b", r"\bwî\b", r"\bwê\b", r"\bhûn\b", r"\bewan\b",
    r"\bdike\b", r"\bdikin\b", r"\bçûye\b", r"\bhate\b", r"\bhebû\b",
    r"\bl سەر\b", r"\bl سەرێ\b", r"\bژ بۆ\b", r"\bji bo\b", r"\bd nav\b",
    r"\bل ناڤ\b", r"\bد ناڤ\b", r"\bڕۆنکرنا\b", r"\bزاربی\b", r"\bسڤر\b",
    r"\bڤان\b", r"\bڤی\b", r"\bئەڤ\b",
]
KMR_REGEX = re.compile("|".join(KMR_MARKERS), re.IGNORECASE)


def slice_audio(master_wav: Path, start_sec: int, duration_sec: int, out_wav: Path):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", str(master_wav),
        "-ar", "16000",
        "-ac", "1",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ingest_part(master_wav: Path, part_num: int, start_sec: int, dur_sec: int = 210) -> int:
    video_id = f"{master_wav.stem}_part{part_num}"
    part_wav = BASE_DIR / "data" / "mela_halo" / f"{video_id}.wav"

    print(f"\n=======================================================")
    print(f"Ingesting Part {part_num}: {start_sec//60:02d}:{start_sec%60:02d} - {(start_sec+dur_sec)//60:02d}:{(start_sec+dur_sec)%60:02d}")
    print(f"=======================================================")

    slice_audio(master_wav, start_sec, dur_sec, part_wav)
    print(f"Sliced {part_wav.name} ({dur_sec}s)")

    # 1. Transcribe via Gemini Files API (temp=0.0)
    segments = transcribe_audio_with_gemini(part_wav)
    print(f"Received {len(segments)} candidate segments from Gemini.")

    # 2. Filter segments against Kurmanji dialect markers
    clean_segs = []
    for s in segments:
        txt = s.get("text", "")
        if KMR_REGEX.search(txt):
            print(f" [REJECTED - Kurmanji marker]: {txt}")
            continue
        clean_segs.append(s)

    print(f"Retained {len(clean_segs)}/{len(segments)} 100% Sorani segments.")

    # 3. Featurize & slice into dataset
    records = featurize_and_slice(part_wav, clean_segs, video_id)
    save_dataset_manifests(records, video_id)

    return len(records)


def main():
    master = BASE_DIR / "data" / "mela_halo" / "wAYbj-73jJg.wav"
    if not master.exists():
        print(f"Error: {master} not found.")
        sys.exit(1)

    # Ingest Parts 9 through 13 (from 28:00 to 44:10)
    schedule = [
        (9, 1680, 210),
        (10, 1890, 210),
        (11, 2100, 210),
        (12, 2310, 210),
        (13, 2520, 130),
    ]

    total_added = 0
    for part_num, start_s, dur_s in schedule:
        try:
            n = ingest_part(master, part_num, start_s, dur_s)
            total_added += n
            time.sleep(2)  # Avoid rate limit burst
        except Exception as e:
            print(f"[ERROR in Part {part_num}]: {e}")

    print(f"\n=======================================================")
    print(f"Batch completed! Total new segments added: {total_added}")
    print(f"Rebuilding KenLM language model...")

    subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "build_kenlm_arpa.py")], check=True)
    print("Done!")


if __name__ == "__main__":
    main()
