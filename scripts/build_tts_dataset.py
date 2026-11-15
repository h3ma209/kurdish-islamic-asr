#!/usr/bin/env python3
"""
Build High-Purity 44.1kHz Kurdish Tafsir Voice Dataset from our 38 aligned Surahs.

Upgrades over v1:
1. Extracts directly at native 44,100 Hz (`-ar 44100`) with a 65Hz high-pass filter
   so full 22.05kHz studio crispness is preserved for `ArkttsCodec`.
2. Trims leading & trailing dead air to a clean 70ms margin and normalizes peak to -1 dBFS.
3. Recursively splits long multi-sentence/multi-verse tafsir blocks (`15s - 45s`) ONLY when
   an actual acoustic silence valley (`>= 240ms`) aligns within ±18% of the character-length
   ratio of a sentence/clause boundary.
4. Strictly filters every candidate clip to the clean Sorani speech-rate window
   (`8.2 <= chars_per_sec <= 16.8`, duration `2.2s - 15.0s`), eliminating all outliers.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from zero_shot_clone import normalize_ckb

ALIGNED_DIR = BASE_DIR / "data" / "aligned"
VIDEOS_DIR = BASE_DIR / "data" / "videos"
DATASET_DIR = BASE_DIR / "data" / "tts_dataset"
WAVS_DIR = DATASET_DIR / "wavs"
CODES_DIR = DATASET_DIR / "codes"

TARGET_SR = 44100
MIN_DUR = 2.2
MAX_DUR = 14.8
MIN_CPS = 8.2
MAX_CPS = 16.8


def extract_raw_44k(video_path: Path, start: float, end: float) -> np.ndarray | None:
    duration = round(end - start, 3)
    if duration < 1.8:
        return None
    cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-af",
        "highpass=f=65",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-f",
        "f32le",
        "-loglevel",
        "error",
        "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or len(res.stdout) < TARGET_SR * 4:
        return None
    wav = np.frombuffer(res.stdout, dtype=np.float32).copy()
    return wav


def trim_and_normalize(wav: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Trim leading/trailing silence using 20ms frame RMS and normalize peak to 0.89."""
    if len(wav) < sr:
        return wav
    hop = int(sr * 0.02)  # 20ms
    n_frames = len(wav) // hop
    if n_frames < 5:
        return wav
    frames = wav[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    peak_rms = np.percentile(rms, 90)
    thresh = max(0.006, peak_rms * 0.07)
    active = np.where(rms > thresh)[0]
    if len(active) < 3:
        return wav
    pad_samples = int(sr * 0.075)  # 75ms natural breath margin
    s_idx = max(0, int(active[0]) * hop - pad_samples)
    e_idx = min(len(wav), int(active[-1] + 1) * hop + pad_samples)
    trimmed = wav[s_idx:e_idx].copy()

    # Gentle 15ms fade-in/out to prevent clicks
    fade_len = min(int(sr * 0.015), len(trimmed) // 8)
    if fade_len > 0:
        trimmed[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        trimmed[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

    peak = float(np.max(np.abs(trimmed))) if len(trimmed) else 0.0
    if peak > 1e-4:
        trimmed = trimmed * (0.89 / peak)
    return trimmed


def find_silence_split(wav: np.ndarray, expected_ratio: float, sr: int = TARGET_SR) -> int | None:
    """
    Find the quietest >=220ms pause valley within ±16% of `expected_ratio * len(wav)`.
    Returns sample index if a clear pause exists, else None.
    """
    total = len(wav)
    hop = int(sr * 0.02)  # 20ms
    n_frames = total // hop
    if n_frames < 40:
        return None

    frames = wav[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    # Smooth over 220ms (11 frames) to locate true breath pauses
    kernel = np.ones(11, dtype=np.float32) / 11.0
    smooth_rms = np.convolve(rms, kernel, mode="same")

    center_f = int(expected_ratio * n_frames)
    tol_f = max(8, int(0.16 * n_frames))
    lo = max(15, center_f - tol_f)
    hi = min(n_frames - 15, center_f + tol_f)
    if hi <= lo:
        return None

    best_f = lo + int(np.argmin(smooth_rms[lo:hi]))
    speech_ref = float(np.percentile(rms, 80))
    # Require the valley to be a genuine pause (< 18% of normal speech RMS)
    if smooth_rms[best_f] > max(0.015, speech_ref * 0.18):
        return None
    return best_f * hop


def split_audio_and_text(wav: np.ndarray, text: str, sr: int = TARGET_SR, depth: int = 0) -> list[tuple[np.ndarray, str]]:
    """Recursively split long clips at punctuation boundaries that match acoustic pauses."""
    wav = trim_and_normalize(wav, sr)
    dur = len(wav) / sr
    clean_text = " ".join(text.split())
    if len(clean_text) < 12 or dur < MIN_DUR:
        return []

    cps = len(clean_text) / dur
    if MIN_DUR <= dur <= MAX_DUR and MIN_CPS <= cps <= MAX_CPS:
        return [(wav, clean_text)]

    if dur <= MAX_DUR or depth >= 3:
        return []

    # Candidate split points at major punctuation (. ؟ ! ؛ ،)
    candidates = [m.end() for m in re.finditer(r"[.؟!؛،:]+\s+", clean_text)]
    if not candidates:
        candidates = [m.start() for m in re.finditer(r"\s+", clean_text)]
    if not candidates:
        return []

    # Sort punctuation splits by closeness to the midpoint
    mid = len(clean_text) * 0.5
    candidates.sort(key=lambda idx: abs(idx - mid))

    for split_char in candidates[:6]:
        left_text = clean_text[:split_char].strip()
        right_text = clean_text[split_char:].strip()
        if len(left_text) < 15 or len(right_text) < 15:
            continue
        ratio = len(left_text) / len(clean_text)
        if not (0.22 <= ratio <= 0.78):
            continue
        split_sample = find_silence_split(wav, ratio, sr)
        if split_sample is not None:
            left_res = split_audio_and_text(wav[:split_sample], left_text, sr, depth + 1)
            right_res = split_audio_and_text(wav[split_sample:], right_text, sr, depth + 1)
            if left_res and right_res:
                return left_res + right_res

    return []


def process_block(job: dict) -> list[dict]:
    wav = extract_raw_44k(job["video_path"], job["start"], job["end"])
    if wav is None:
        return []
    segments = split_audio_and_text(wav, job["text"], TARGET_SR)
    out_records = []
    for idx, (seg_wav, seg_text) in enumerate(segments):
        dur = round(len(seg_wav) / TARGET_SR, 2)
        cps = round(len(seg_text) / dur, 2)
        if not (MIN_DUR <= dur <= MAX_DUR and MIN_CPS <= cps <= MAX_CPS):
            continue
        clip_id = job["id"] if len(segments) == 1 else f"{job['id']}_p{idx+1}"
        out_wav = WAVS_DIR / f"{clip_id}.wav"
        sf.write(str(out_wav), seg_wav, TARGET_SR, subtype="PCM_16")
        out_records.append({
            "id": clip_id,
            "surah": job["surah"],
            "ayah_start": job["ayah_start"],
            "ayah_end": job["ayah_end"],
            "audio_filepath": str(out_wav),
            "duration": dur,
            "chars_per_sec": cps,
            "text": seg_text,
        })
    return out_records


def build_dataset() -> list[dict]:
    # Clean old 24kHz WAVs and stale codes so everything is freshly encoded at 44.1kHz
    if WAVS_DIR.exists():
        for f in WAVS_DIR.glob("*.wav"):
            f.unlink()
    if CODES_DIR.exists():
        for f in CODES_DIR.glob("*.npy"):
            f.unlink()
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    CODES_DIR.mkdir(parents=True, exist_ok=True)

    aligned_files = sorted(ALIGNED_DIR.glob("surah_*.json"))
    jobs = []

    for af in aligned_files:
        data = json.loads(af.read_text(encoding="utf-8"))
        surah_num = data.get("surah_number")
        video_path = VIDEOS_DIR / f"surah_{surah_num:03d}.mp4"
        if not video_path.exists():
            continue

        blocks: dict[tuple[float, float], list[dict]] = {}
        for v in data.get("verses", []):
            ts = v.get("tafsir_start")
            te = v.get("tafsir_end")
            if ts is None or te is None or te <= ts:
                continue
            key = (round(float(ts), 2), round(float(te), 2))
            blocks.setdefault(key, []).append(v)

        for (ts, te), verses in sorted(blocks.items()):
            dur = round(te - ts, 2)
            if dur < MIN_DUR or dur > 45.0:
                continue
            raw_texts = [v.get("kurdish_text", "").strip() for v in verses if v.get("kurdish_text", "").strip()]
            if not raw_texts:
                continue
            combined_kurdish = " . ".join(raw_texts)
            norm_text = normalize_ckb(combined_kurdish)
            norm_text = norm_text.replace("(", " ").replace(")", " ")
            norm_text = " ".join(norm_text.split())
            if len(norm_text) < 12:
                continue

            first_ayah = verses[0]["number_in_surah"]
            last_ayah = verses[-1]["number_in_surah"]
            clip_id = f"s{surah_num:03d}_a{first_ayah:03d}_{last_ayah:03d}"
            jobs.append({
                "id": clip_id,
                "surah": surah_num,
                "ayah_start": first_ayah,
                "ayah_end": last_ayah,
                "video_path": video_path,
                "start": ts,
                "end": te,
                "duration": dur,
                "text": norm_text,
            })

    print(f"Processing {len(jobs)} candidate tafsir blocks at 44.1kHz with acoustic silence splitting...")
    records = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_block, j) for j in jobs]
        for fut in as_completed(futs):
            records.extend(fut.result())

    records.sort(key=lambda r: r["id"])
    meta_path = DATASET_DIR / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_min = sum(r["duration"] for r in records) / 60.0
    avg_cps = np.mean([r["chars_per_sec"] for r in records]) if records else 0.0
    print(f"✅ High-Purity 44.1kHz Dataset ready: {len(records)} clips ({total_min:.1f} mins, avg {avg_cps:.1f} chars/s) -> {meta_path}")
    return records


if __name__ == "__main__":
    build_dataset()
