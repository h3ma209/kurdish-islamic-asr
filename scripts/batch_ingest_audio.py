#!/usr/bin/env python3
"""
Automated Batch Ingest & Dataset Expansion for Mela Halo Lectures.
Slices master WAVs into 3.5-minute segments, generates verbatim Sorani + Arabic Harakat labels
using Gemini 3.5 Flash Lite, extracts 80-band Mel-spectrograms, updates dataset.jsonl/metadata.csv,
and rebuilds the KenLM 3-gram language model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON_EXE = BASE_DIR / ".venv-tts" / "bin" / "python"
DATA_DIR = BASE_DIR / "data" / "mela_halo"


def slice_video(master_wav: Path, chunk_sec: int = 210) -> list[Path]:
    import soundfile as sf
    info = sf.info(str(master_wav))
    dur = info.duration
    parts = []

    video_id = master_wav.stem
    num_parts = int(dur // chunk_sec) + (1 if dur % chunk_sec > 30 else 0)
    print(f"\nSlicing {master_wav.name} ({dur/60:.1f} mins) into {num_parts} parts of {chunk_sec}s...")

    for i in range(num_parts):
        start_s = i * chunk_sec
        part_name = f"{video_id}_part{i+1}.wav"
        part_path = DATA_DIR / part_name
        parts.append(part_path)

        if part_path.exists():
            print(f"  Part {i+1}/{num_parts}: {part_name} already exists, skipping ffmpeg.")
            continue

        cmd = [
            "ffmpeg", "-y", "-i", str(master_wav),
            "-ss", str(start_s), "-t", str(chunk_sec),
            "-ar", "16000", "-ac", "1",
            str(part_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            print(f"  [ERROR] ffmpeg failed for part {i+1}")
        else:
            print(f"  Part {i+1}/{num_parts}: created {part_name}")

    return parts


def ingest_part(part_wav: Path):
    cmd = [str(PYTHON_EXE), str(BASE_DIR / "scripts" / "gemini_audio_dataset.py"), str(part_wav)]
    print(f"\n--> Running gemini_audio_dataset on {part_wav.name}...")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[WARN] Failed to process {part_wav.name} (exit code {res.returncode})")
        return False
    return True


def rebuild_kenlm():
    print("\nRebuilding KenLM 3-gram language model from updated dataset...")
    cmd = [str(PYTHON_EXE), str(BASE_DIR / "scripts" / "build_kenlm_arpa.py")]
    res = subprocess.run(cmd)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Batch Ingest Audio Lectures")
    parser.add_argument("videos", nargs="+", help="Video stem or path (e.g. 7wSI2LeTozs, wctC6ZrqChY)")
    parser.add_argument("--chunk-sec", type=int, default=210, help="Chunk length in seconds (default: 210s = 3.5m)")
    args = parser.parse_args()

    for vid in args.videos:
        p = Path(vid)
        if not p.is_absolute() and not p.exists():
            p = DATA_DIR / f"{vid}.wav"
        if not p.exists():
            print(f"[ERROR] Audio file not found: {p}")
            continue

        parts = slice_video(p, chunk_sec=args.chunk_sec)
        for part in parts:
            timeline_json = DATA_DIR / f"{part.stem}_gemini_timeline.json"
            if timeline_json.exists():
                print(f"Skipping {part.name} (already processed: {timeline_json.name})")
                continue
            ingest_part(part)

    rebuild_kenlm()
    print("\nAll batch ingestion complete!")


if __name__ == "__main__":
    main()
