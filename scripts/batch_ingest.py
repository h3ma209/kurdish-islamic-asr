#!/usr/bin/env python3
"""
Batch ingestion runner for 40 minutes of Mela Halo sermon (6 x 400s parts).
Preserves full Arabic Harakat, slices 16kHz WAVs, computes log-mels, updates dataset manifests.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.gemini_audio_dataset import (
    transcribe_audio_with_gemini,
    featurize_and_slice,
    save_dataset_manifests,
)

def process_audio_file(audio_path: Path):
    video_id = audio_path.stem
    segments = transcribe_audio_with_gemini(audio_path)
    records = featurize_and_slice(audio_path, segments, video_id)
    save_dataset_manifests(records, video_id)
    return records

def main():
    parts = [
        BASE_DIR / "data" / "mela_halo" / f"QfeOkeIahwY_part{i}.mp3"
        for i in range(10, 16)
    ]

    print("=" * 65)
    print(f"Starting Batch Ingestion of {len(parts)} Remaining Parts...")
    print("=" * 65)

    total_added = 0
    t0 = time.time()

    for idx, p in enumerate(parts, 1):
        if not p.exists():
            print(f"Skipping {p.name} (file not found)")
            continue

        print(f"\n>>> [{idx}/{len(parts)}] Ingesting {p.name} (400s, {p.stat().st_size / 1024 / 1024:.2f} MB)...")
        max_retries = 5
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                records = process_audio_file(p)
                print(f"[OK] {p.name} ingested successfully: {len(records)} segments added!")
                total_added += len(records)
                success = True
                break
            except Exception as e:
                print(f"[WARN] Attempt {attempt} failed for {p.name}: {e}. Retrying in {attempt * 8}s...")
                time.sleep(attempt * 8)

        if not success:
            print(f"[ERROR] Failed to ingest {p.name} after {max_retries} attempts.")
            continue

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"Batch Ingestion Complete in {elapsed:.2f}s!")
    print(f"Total new segments created: {total_added}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
