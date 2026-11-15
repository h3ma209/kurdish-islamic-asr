#!/usr/bin/env python3
"""
Local Offline Whisper-Large-V3-Turbo (Fine-Tuned Decoder) + ScriptureRescorer
for Verbatim Central Kurdish (Sorani) + Classical Arabic Lectures.
Runs 100% locally on Apple Silicon MPS.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from scripts.scripture_rescorer import ScriptureRescorer

MODEL_ID = "openai/whisper-large-v3-turbo"
CKPT_PATH = BASE_DIR / "data" / "models" / "whisper_turbo_islamic_kurdish" / "finetuned_decoder.pt"


def find_silence_split(wav: np.ndarray, sr: int = 16000, target_sec: float = 28.0, search_window: float = 4.0) -> int:
    """Finds the lowest-energy 100ms window around target_sec to split cleanly between words."""
    total = len(wav)
    if total <= int(30.0 * sr):
        return total
    center = int(target_sec * sr)
    lo = max(int((target_sec - search_window) * sr), int(15.0 * sr))
    hi = min(int((target_sec + 1.8) * sr), int(29.8 * sr), total)
    win = int(0.1 * sr)
    best_idx = center
    best_e = float("inf")
    for idx in range(lo, hi - win, int(0.05 * sr)):
        e = float(np.mean(wav[idx : idx + win] ** 2))
        if e < best_e:
            best_e = e
            best_idx = idx + win // 2
    return best_idx


class WhisperKurdishTranscriber:
    def __init__(self, device: str = "mps"):
        self.device = device if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
        t0 = time.time()
        print(f"Loading {MODEL_ID} on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(MODEL_ID)
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float32)

        if CKPT_PATH.exists():
            ckpt = torch.load(CKPT_PATH, map_location="cpu")
            self.model.model.decoder.load_state_dict(ckpt["decoder"])
            self.model.proj_out.load_state_dict(ckpt["proj_out"])
            print(f"Loaded fine-tuned Kurdish+Arabic decoder from {CKPT_PATH.name}!")

        self.model = self.model.to(self.device)
        self.model.eval()
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="ar", task="transcribe", no_timestamps=True
        )
        self.rescorer = ScriptureRescorer()
        print(f"Ready in {time.time() - t0:.2f}s!")

    def transcribe(self, audio_path: str | Path, num_beams: int = 1) -> dict:
        wav, sr = sf.read(str(audio_path))
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = AF.resample(torch.from_numpy(wav.astype("float32")).unsqueeze(0), sr, 16000).squeeze(0).numpy()

        dur = len(wav) / 16000.0
        t0 = time.time()

        chunks = []
        pos = 0
        while pos < len(wav):
            rem = wav[pos:]
            split_len = find_silence_split(rem, sr=16000)
            chunks.append(rem[:split_len])
            pos += split_len

        raw_parts = []
        with torch.no_grad():
            for ch in chunks:
                inp = self.processor(ch, sampling_rate=16000, return_tensors="pt").input_features.to(
                    self.device, dtype=torch.float32
                )
                ids = self.model.generate(
                    inp,
                    forced_decoder_ids=self.forced_decoder_ids,
                    max_new_tokens=256,
                    num_beams=num_beams,
                    no_repeat_ngram_size=4,
                )
                txt = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                if txt:
                    raw_parts.append(txt)

        raw_text = " ".join(raw_parts)
        final_text = self.rescorer.rescore(raw_text)
        elapsed = time.time() - t0

        return {
            "audio": str(audio_path),
            "duration_sec": round(dur, 2),
            "total_time_sec": round(elapsed, 2),
            "speedup_realtime": round(dur / max(elapsed, 0.01), 1),
            "raw_text": raw_text,
            "transcription": final_text,
        }


def main():
    parser = argparse.ArgumentParser(description="Fine-Tuned Whisper-Turbo Kurdish+Arabic Transcriber")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--beams", type=int, default=1)
    args = parser.parse_args()

    t = WhisperKurdishTranscriber()
    res = t.transcribe(args.audio, num_beams=args.beams)
    print(f"\n[{Path(res['audio']).name}] {res['duration_sec']}s in {res['total_time_sec']}s ({res['speedup_realtime']}x)")
    print(res["transcription"])


if __name__ == "__main__":
    main()
