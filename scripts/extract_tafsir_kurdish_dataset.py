#!/usr/bin/env python3
"""
Extract double-verified Sorani Kurdish tafsir slices from data/aligned/surah_*.json + data/videos/surah_*.mp4.
Deduplicates shared tafsir blocks, runs 120-tok MMS-1B + KenLM on each slice, and keeps only slices
where MMS output matches book kurdish_text (full or paren-stripped) >= 85% phonetically.
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import torch
from rapidfuzz import fuzz

from scripts.infer_custom_mms import IslamicKurdishTranscriber
from scripts.scripture_rescorer import fold_phonetic

ALIGNED_DIR = BASE_DIR / "data" / "aligned"
VIDEOS_DIR = BASE_DIR / "data" / "videos"
OUT_WAV_DIR = BASE_DIR / "data" / "dataset" / "tafsir_wavs"
OUT_JSONL = BASE_DIR / "data" / "dataset" / "tafsir_kurdish.jsonl"


def clean_book_text(text: str) -> str:
    t = text.replace("\ufeff", "").replace("\u200c", "").strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main(min_match: float = 85.0):
    OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
    aligned_files = sorted(glob.glob(str(ALIGNED_DIR / "surah_*.json")))
    print(f"Found {len(aligned_files)} aligned surah JSONs.")

    transcriber = IslamicKurdishTranscriber(use_corrector=False, use_arabic_whisper=False)
    entries = []
    scanned = 0
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / "surah_16k.wav"

        for fpath in aligned_files:
            data = json.load(open(fpath, "r", encoding="utf-8"))
            surah_num = int(data["surah_number"])
            vid_name = Path(data.get("video_file") or f"surah_{surah_num:03d}.mp4").name
            vid_path = VIDEOS_DIR / vid_name
            if not vid_path.exists():
                continue

            blocks: dict[tuple[float, float], list[dict]] = defaultdict(list)
            for v in data.get("verses", []):
                st = float(v.get("tafsir_start") or 0.0)
                en = float(v.get("tafsir_end") or 0.0)
                if en > st + 0.8 and v.get("kurdish_text"):
                    blocks[(round(st, 3), round(en, 3))].append(v)

            if not blocks:
                continue

            subprocess.run(
                ["ffmpeg", "-y", "-i", str(vid_path), "-ar", "16000", "-ac", "1", "-loglevel", "error", str(tmp_wav)],
                check=True,
            )
            audio, sr = sf.read(str(tmp_wav), dtype="float32")
            total_samples = len(audio)

            for (st, en), vs in sorted(blocks.items()):
                dur = en - st
                if dur < 0.8 or dur > 28.0:
                    continue
                s0, s1 = max(0, int(st * sr)), min(total_samples, int(en * sr))
                if s1 - s0 < int(0.8 * sr):
                    continue
                chunk = audio[s0:s1]
                scanned += 1

                vs.sort(key=lambda x: int(x["number_in_surah"]))
                kt_full = clean_book_text(" ".join(v["kurdish_text"] for v in vs))
                kt_noparen = clean_book_text(re.sub(r"\([^)]*\)", "", kt_full))

                inp = transcriber.processor(chunk, sampling_rate=16000, return_tensors="pt").to(transcriber.device)
                with torch.no_grad():
                    logits = transcriber.model(**inp).logits[0]
                    probs = torch.softmax(logits, dim=-1)
                    ctc_conf = float(probs.max(dim=-1).values.mean().item())
                    logits_np = logits.cpu().numpy()

                beam_txt = transcriber.decoder.decode(logits_np, beam_width=15).strip()
                rescored_txt = transcriber.rescorer.rescore(beam_txt)

                f_mms = fold_phonetic(rescored_txt)
                score_full = fuzz.ratio(f_mms, fold_phonetic(kt_full))
                score_nop = fuzz.ratio(f_mms, fold_phonetic(kt_noparen)) if kt_noparen else 0.0

                best_score = max(score_full, score_nop)
                if best_score >= min_match and ctc_conf >= 0.80 and len(rescored_txt.split()) >= 2:
                    # Pick the book text variant that matched best (stripping parens if noparen won)
                    chosen_text = kt_noparen if score_nop > score_full else kt_full
                    chosen_text = re.sub(r"[()]", "", chosen_text).strip()
                    v0, v1 = vs[0]["number_in_surah"], vs[-1]["number_in_surah"]
                    fname = f"t_s{surah_num:03d}_v{v0:03d}_{v1:03d}_{int(st * 10)}.wav"
                    sf.write(str(OUT_WAV_DIR / fname), chunk, sr)
                    entries.append({
                        "audio_file": f"tafsir_wavs/{fname}",
                        "text": chosen_text,
                        "duration": round((s1 - s0) / sr, 3),
                        "lang": "ckb",
                        "match_score": round(best_score, 1),
                        "ctc_conf": round(ctc_conf, 3),
                    })

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    tot_min = sum(e["duration"] for e in entries) / 60.0
    print(f"\nScanned {scanned} tafsir blocks in {time.time() - t0:.1f}s")
    print(f"Kept {len(entries)} double-verified Sorani tafsir clips ({tot_min:.1f} min) -> {OUT_JSONL}")


if __name__ == "__main__":
    main()
