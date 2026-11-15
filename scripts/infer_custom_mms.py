#!/usr/bin/env python3
"""
Meta MMS-1B + KenLM + Scripture Rescorer: Verbatim Kurdish Sorani & Arabic Harakat Transcriber.
Runs 100% locally on Apple Silicon MPS with zero cloud dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as AF
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ForCTC

from scripts.scripture_rescorer import ScriptureRescorer

MODEL_DIR = BASE_DIR / "data" / "models" / "mms_1b_islamic_kurdish"
ARPA_PATH = MODEL_DIR / "kurdish_islamic_3gram.arpa"
HEAD_PATH = MODEL_DIR / "finetuned_ckb_head.pt"
ADAPTER_PATH = MODEL_DIR / "finetuned_ckb_adapter.pt"
VOCAB_PATH = MODEL_DIR / "vocab.json"


class IslamicKurdishTranscriber:
    def __init__(self, device: str = "mps"):
        self.device = device if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
        print(f"Loading Meta MMS-1B backbone on {self.device}...")
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
        self.processor.tokenizer.set_target_lang("ckb")
        self.model.load_adapter("ckb")
        vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8")) if VOCAB_PATH.exists() else self.processor.tokenizer.get_vocab()
        if ADAPTER_PATH.exists():
            ckpt = torch.load(ADAPTER_PATH, map_location="cpu")
            for idx, state in ckpt["adapters"].items():
                self.model.wav2vec2.encoder.layers[int(idx)].adapter_layer.load_state_dict(state)
            head_out = ckpt["lm_head"]["weight"].shape[0]
            if head_out != self.model.lm_head.out_features:
                self.model.lm_head = nn.Linear(
                    self.model.lm_head.in_features,
                    head_out,
                    bias=self.model.lm_head.bias is not None,
                )
            self.model.lm_head.load_state_dict(ckpt["lm_head"])
            print(f"Loaded fine-tuned bilingual adapters ({ckpt['start_layer']}..47) + {head_out}-tok lm_head from {ADAPTER_PATH.name}!")
        elif HEAD_PATH.exists():
            self.model.lm_head.load_state_dict(torch.load(HEAD_PATH, map_location="cpu"))
            print(f"Loaded fine-tuned ckb lm_head from {HEAD_PATH.name}!")
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded in {time.time() - t0:.2f}s!")

        # Initialize PyCTCDecode + KenLM
        inv_vocab = {v: k for k, v in vocab.items()}
        self.inv_vocab = inv_vocab
        labels = []
        for i in range(len(vocab)):
            tok = inv_vocab.get(i, "")
            if i == 0 or tok == "<pad>":
                labels.append("")
            elif tok in ("<s>", "</s>", "<unk>"):
                labels.append(chr(i))
            elif tok == "|":
                labels.append(" ")
            else:
                labels.append(tok)

        words = set()
        in_1g = False
        with open(ARPA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s == "\\1-grams:":
                    in_1g = True
                    continue
                if s.startswith("\\") and in_1g:
                    break
                if in_1g and s:
                    parts = s.split("\t")
                    if len(parts) >= 2 and parts[1] not in ("<s>", "</s>", "<unk>"):
                        words.add(parts[1])

        print(f"Building KenLM Beam Search Decoder with {len(words)} lexicon words...")
        self.decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=str(ARPA_PATH),
            unigrams=sorted(list(words)),
            alpha=0.30,
            beta=0.50,
            unk_score_offset=-10.0,
        )

        print("Initializing Islamic Scripture & Harakat Rescorer...")
        self.rescorer = ScriptureRescorer()
        print(f"Ready! {len(self.rescorer.scriptures)} Quran/Hadith entries indexed.\n")

    def transcribe(self, audio_path: Path, beam_width: int = 15) -> dict:
        wav, sr = sf.read(str(audio_path))
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)

        t = torch.from_numpy(wav.astype("float32")).unsqueeze(0)
        if sr != 16000:
            wav_16k = AF.resample(t, sr, 16000).squeeze(0)
        else:
            wav_16k = t.squeeze(0)

        dur = len(wav_16k) / 16000.0
        t0 = time.time()

        inputs = self.processor(wav_16k, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            logits_np = logits.cpu().numpy()

        t_inf = time.time() - t0

        t0_dec = time.time()
        beams = self.decoder.decode_beams(logits_np, beam_width=beam_width)
        beam_raw = beams[0][0]
        words_out = []
        for word, (st, en) in beams[0][2]:
            # If KenLM glued multiple OOV tokens into a long string, restore acoustic spaces from greedy CTC
            if len(word) > 12:
                span_ids = np.argmax(logits_np[max(0, st - 1):min(len(logits_np), en + 1)], axis=-1)
                # Decode greedy CTC tokens using expanded 120-token inv_vocab
                prev = -1
                chars = []
                for tid in span_ids:
                    if tid != prev:
                        if tid != 0 and tid in self.inv_vocab:
                            tok = self.inv_vocab[tid]
                            if tok not in ("<s>", "</s>", "<unk>", "<pad>"):
                                chars.append(" " if tok == "|" else tok)
                        prev = tid
                greedy_span = "".join(chars).strip()
                words_out.append(greedy_span if " " in greedy_span else word)
            else:
                words_out.append(word)
        selected_raw = " ".join(words_out)
        t_dec = time.time() - t0_dec

        t0_res = time.time()
        final_text = self.rescorer.rescore(selected_raw)
        t_res = time.time() - t0_res

        total_time = t_inf + t_dec + t_res
        speed = dur / total_time if total_time > 0 else 0

        return {
            "audio": str(audio_path),
            "duration_sec": round(dur, 2),
            "inference_time_sec": round(t_inf, 3),
            "beam_decode_sec": round(t_dec, 3),
            "rescore_sec": round(t_res, 3),
            "total_time_sec": round(total_time, 3),
            "speedup_realtime": round(speed, 1),
            "raw_beam_text": beam_raw,
            "transcription": final_text,
        }


def main():
    parser = argparse.ArgumentParser(description="MMS-1B + KenLM + Harakat Kurdish Transcriber")
    parser.add_argument("audio", type=Path, help="Audio file to transcribe (.wav, .mp3, etc.)")
    parser.add_argument("--beam", type=int, default=30, help="Beam width (default: 30)")
    args = parser.parse_args()

    transcriber = IslamicKurdishTranscriber()
    result = transcriber.transcribe(args.audio, beam_width=args.beam)

    print("=" * 70)
    print(f"AUDIO:      {Path(result['audio']).name} ({result['duration_sec']}s)")
    print(f"SPEED:      {result['total_time_sec']}s total ({result['speedup_realtime']}x real-time)")
    print("-" * 70)
    print("FINAL TRANSCRIPTION:")
    print(result["transcription"])
    print("=" * 70)


if __name__ == "__main__":
    main()
