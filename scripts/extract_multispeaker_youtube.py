#!/usr/bin/env python3
"""
Confidence-filtered multi-speaker Kurdish+Arabic dataset extractor.
Segments full lecture WAVs at silence boundaries (3s-16s), transcribes with 120-tok MMS-1B + KenLM
+ ScriptureRescorer v2, and keeps only high-CTC-confidence clips (>= 0.85) for training.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from scripts.infer_custom_mms import IslamicKurdishTranscriber

SCRATCH_DIR = Path("/Users/hema/.gemini/antigravity/brain/978c4367-5bab-4e40-acb8-efcce301614d/scratch")
OUT_WAV_DIR = BASE_DIR / "data" / "dataset" / "multispeaker_wavs"
OUT_JSONL = BASE_DIR / "data" / "dataset" / "multispeaker_kurdish.jsonl"

SOURCES = [
    ("nuraddin", SCRATCH_DIR / "yt_GeJJS9MfEFI.wav", 0.0),
    ("ranjdar", SCRATCH_DIR / "yt_tH_MArbjlcQ.wav", 5.0),
]


def split_at_silences(wav: np.ndarray, sr: int = 16000, min_dur: float = 3.0, max_dur: float = 16.0) -> list[tuple[int, int]]:
    frame_len = int(0.05 * sr)
    rms = np.array([np.sqrt(np.mean(wav[i : i + frame_len] ** 2)) for i in range(0, len(wav) - frame_len, frame_len)])
    thresh = max(0.004, float(np.percentile(rms, 15)) * 1.8)
    is_sil = rms < thresh

    # Find midpoint of >= 0.25s silence runs as valid split points
    split_samples = [0]
    i = 0
    while i < len(is_sil):
        if is_sil[i]:
            j = i
            while j < len(is_sil) and is_sil[j]:
                j += 1
            if (j - i) >= 5:
                mid = ((i + j) // 2) * frame_len
                if mid - split_samples[-1] >= int(min_dur * sr):
                    split_samples.append(mid)
            i = j
        else:
            i += 1
    if len(wav) - split_samples[-1] >= int(1.5 * sr):
        split_samples.append(len(wav))

    chunks = []
    cur_start = split_samples[0]
    for s in split_samples[1:]:
        dur = (s - cur_start) / sr
        if dur > max_dur:
            # Hard-cap long speech runs into <= max_dur slices
            pos = cur_start
            while pos + int(max_dur * sr) < s:
                chunks.append((pos, pos + int(max_dur * sr)))
                pos += int(max_dur * sr)
            if s - pos >= int(min_dur * sr):
                chunks.append((pos, s))
            cur_start = s
        elif dur >= min_dur:
            chunks.append((cur_start, s))
            cur_start = s
    return chunks


def main(min_conf: float = 0.85):
    OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
    transcriber = IslamicKurdishTranscriber(use_corrector=False, use_arabic_whisper=False)

    entries = []
    t0 = time.time()

    for speaker, wav_path, skip_sec in SOURCES:
        if not wav_path.exists():
            print(f"[SKIP] Missing {wav_path}")
            continue
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = AF.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        if skip_sec > 0:
            wav = wav[int(skip_sec * 16000) :]

        chunks = split_at_silences(wav, sr=16000)
        print(f"Speaker '{speaker}': {len(wav)/16000/60:.1f} min -> {len(chunks)} candidate chunks")
        kept = 0

        for idx, (s0, s1) in enumerate(chunks):
            seg = wav[s0:s1]
            inp = transcriber.processor(seg, sampling_rate=16000, return_tensors="pt").to(transcriber.device)
            with torch.no_grad():
                logits = transcriber.model(**inp).logits[0]
                probs = torch.softmax(logits, dim=-1)
                ctc_conf = float(probs.max(dim=-1).values.mean().item())
                logits_np = logits.cpu().numpy()

            if ctc_conf < min_conf:
                continue

            beams = transcriber.decoder.decode_beams(logits_np, beam_width=15)
            words = [w for w, _ in beams[0][2]]
            if len(words) < 4 or any(len(w) > 18 for w in words):
                continue

            text = transcriber.rescorer.rescore(" ".join(words))
            fname = f"{speaker}_{idx:04d}.wav"
            sf.write(str(OUT_WAV_DIR / fname), seg, 16000)
            entries.append({
                "audio_file": f"multispeaker_wavs/{fname}",
                "text": text,
                "duration": round((s1 - s0) / 16000.0, 3),
                "speaker": speaker,
                "ctc_conf": round(ctc_conf, 3),
            })
            kept += 1

        print(f" -> Kept {kept}/{len(chunks)} high-confidence clips for '{speaker}' (conf >= {min_conf})")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    tot_min = sum(e["duration"] for e in entries) / 60.0
    print(f"\nDone in {time.time() - t0:.1f}s | Total multi-speaker clips: {len(entries)} ({tot_min:.1f} min) -> {OUT_JSONL}")


if __name__ == "__main__":
    main()
