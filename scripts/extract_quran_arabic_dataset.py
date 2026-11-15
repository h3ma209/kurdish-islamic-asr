#!/usr/bin/env python3
"""
Extract verified Arabic Quran audio slices from data/aligned/surah_*.json + data/videos/surah_*.mp4.
Merges shared_block verses, handles spoken Isti'adha prefixes, and exports 16kHz mono WAVs + quran_arabic.jsonl.
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
ALIGNED_DIR = BASE_DIR / "data" / "aligned"
VIDEOS_DIR = BASE_DIR / "data" / "videos"
OUT_WAV_DIR = BASE_DIR / "data" / "dataset" / "quran_wavs"
OUT_JSONL = BASE_DIR / "data" / "dataset" / "quran_arabic.jsonl"

ISTIADHA = "أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ"
BASMALA = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"


def clean_uthmani(text: str) -> str:
    return text.replace("\ufeff", "").strip()


def main():
    OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
    aligned_files = sorted(glob.glob(str(ALIGNED_DIR / "surah_*.json")))
    print(f"Found {len(aligned_files)} aligned surah JSON files.")

    entries = []
    skipped_dur = 0
    isti_added = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / "surah_16k.wav"

        for fpath in aligned_files:
            data = json.load(open(fpath, "r", encoding="utf-8"))
            surah_num = int(data["surah_number"])
            vid_name = Path(data.get("video_file") or f"surah_{surah_num:03d}.mp4").name
            vid_path = VIDEOS_DIR / vid_name
            if not vid_path.exists():
                print(f"  [SKIP] Missing video: {vid_path}")
                continue

            # Group verses sharing the same arabic_start..arabic_end block
            blocks: dict[tuple[float, float], list[dict]] = defaultdict(list)
            for v in data.get("verses", []):
                st = float(v.get("arabic_start") or 0.0)
                en = float(v.get("arabic_end") or 0.0)
                if en > st + 0.5 and v.get("arabic_text"):
                    blocks[(round(st, 3), round(en, 3))].append(v)

            if not blocks:
                continue

            # Decode entire surah video once to 16kHz mono wav
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(vid_path), "-ar", "16000", "-ac", "1", "-loglevel", "error", str(tmp_wav)],
                check=True,
            )
            audio, sr = sf.read(str(tmp_wav), dtype="float32")
            total_samples = len(audio)

            for (st, en), vs in sorted(blocks.items()):
                dur = en - st
                if dur < 0.8 or dur > 35.0:
                    skipped_dur += 1
                    continue

                vs.sort(key=lambda x: int(x["number_in_surah"]))
                ar_text = " ".join(clean_uthmani(v["arabic_text"]) for v in vs)

                # Prepend Isti'adha if spoken in audio (detected from whisper_arabic)
                wa = " ".join((v.get("whisper_arabic") or "") for v in vs)
                if ("عوذ" in wa or "عوض" in wa or "أعوذ" in wa) and "أعوذ" not in ar_text and "عوذ" not in ar_text:
                    ar_text = f"{ISTIADHA} {ar_text}"
                    isti_added += 1

                # Prepend Basmala if spoken before verse 1 (not Surah 1 & not already in text)
                if surah_num not in (1, 9) and vs[0]["number_in_surah"] == 1:
                    if ("بسم الله" in wa or "بسم الل" in wa) and "بِسْمِ" not in ar_text:
                        ar_text = f"{BASMALA} {ar_text}"

                ar_text = re.sub(r"\s+", " ", ar_text).strip()
                s0 = max(0, int(st * sr))
                s1 = min(total_samples, int(en * sr))
                if s1 - s0 < int(0.8 * sr):
                    skipped_dur += 1
                    continue

                v_start = vs[0]["number_in_surah"]
                v_end = vs[-1]["number_in_surah"]
                fname = f"s{surah_num:03d}_v{v_start:03d}_{v_end:03d}.wav"
                rel_path = f"quran_wavs/{fname}"
                sf.write(str(OUT_WAV_DIR / fname), audio[s0:s1], sr)

                entries.append({
                    "audio_file": rel_path,
                    "text": ar_text,
                    "duration": round((s1 - s0) / sr, 3),
                    "lang": "ar",
                    "surah": surah_num,
                })

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    total_min = sum(e["duration"] for e in entries) / 60.0
    print(f"Extracted {len(entries)} Arabic Quran clips ({total_min:.1f} min) -> {OUT_JSONL}")
    print(f"Skipped duration (<0.8s or >35s): {skipped_dur} | Isti'adha prepended: {isti_added}")


if __name__ == "__main__":
    main()
