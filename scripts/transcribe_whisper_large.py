#!/usr/bin/env python3
"""
Transcription script for Kurdish Sorani audio using fine-tuned Whisper-Large-V3 (ckb).
Model: rzgar/whisper-large-v3-sorani-kurdish-ckb-v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from zero_shot_clone import normalize_ckb

MODEL_DIR = BASE_DIR / "data" / "models" / "whisper_large_v3_ckb"


def transcribe_file(
    audio_path: Path,
    chunk_sec: float = 24.0,
    stride_sec: float = 2.0,
) -> dict:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Whisper Large V3 CKB on {device}...")
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()

    print(f"Reading audio from {audio_path}...")
    wav, sr = sf.read(str(audio_path))
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)

    t = torch.from_numpy(wav.astype("float32")).unsqueeze(0)
    if sr != 16000:
        wav_16k = AF.resample(t, sr, 16000).squeeze(0).numpy()
    else:
        wav_16k = t.squeeze(0).numpy()

    total_dur = len(wav_16k) / 16000.0
    print(f"Total duration: {total_dur:.2f}s ({total_dur / 60.0:.1f} mins)")

    step_sec = chunk_sec - stride_sec
    step_samples = int(step_sec * 16000)
    chunk_samples = int(chunk_sec * 16000)

    segments = []
    full_texts = []
    curr_idx = 0
    t0 = time.time()

    while curr_idx < len(wav_16k):
        start_s = curr_idx / 16000.0
        end_s = min(total_dur, (curr_idx + chunk_samples) / 16000.0)
        piece = wav_16k[curr_idx : curr_idx + chunk_samples]
        if len(piece) < 16000 * 0.5:
            break

        # Check energy (skip pure dead silence)
        rms = float(np.sqrt(np.mean(piece**2)))
        if rms < 0.003:
            curr_idx += step_samples
            continue

        inp = processor(piece, sampling_rate=16000, return_tensors="pt")
        feats = inp.input_features.to(device)

        with torch.inference_mode():
            pred_ids = model.generate(
                feats,
                max_new_tokens=220,
                num_beams=2,
                no_repeat_ngram_size=4,
                repetition_penalty=1.15,
                early_stopping=True,
            )

        txt = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        if txt:
            norm = normalize_ckb(txt)
            segments.append({
                "start": round(start_s, 2),
                "end": round(end_s, 2),
                "text": txt,
                "normalized": norm,
            })
            full_texts.append(txt)
            print(f"[{start_s:5.1f}s - {end_s:5.1f}s]: {txt}")

        curr_idx += step_samples

    elapsed = time.time() - t0
    full_text = " ".join(full_texts)
    norm_text = normalize_ckb(full_text)

    return {
        "audio_file": str(audio_path),
        "duration_seconds": round(total_dur, 2),
        "elapsed_seconds": round(elapsed, 2),
        "model": "rzgar/whisper-large-v3-sorani-kurdish-ckb-v2",
        "segments": segments,
        "raw_transcript": full_text,
        "normalized_transcript": norm_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Transcribe Kurdish Sorani audio with Whisper Large V3")
    parser.add_argument("audio", type=Path, help="Path to input audio file")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-txt", type=Path, default=None)
    args = parser.parse_args()

    result = transcribe_file(args.audio)
    stem = args.audio.stem
    parent = args.audio.parent

    out_json = args.out_json or parent / f"{stem}_whisper_large_transcript.json"
    out_txt = args.out_txt or parent / f"{stem}_whisper_large_transcript.txt"

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text(f"Whisper Large V3 Kurdish Transcript\n\n{result['normalized_transcript']}\n", encoding="utf-8")
    print(f"\nDone in {result['elapsed_seconds']}s! Output saved to:\n  {out_json}\n  {out_txt}")


if __name__ == "__main__":
    main()
