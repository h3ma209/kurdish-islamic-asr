#!/usr/bin/env python3
"""
Test PyCTCDecode Beam Search with Kurdish & Arabic Islamic Lexicon.
"""

from __future__ import annotations

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
import torchaudio.functional as AF
from pyctcdecode import build_ctcdecoder

from scripts.infer_custom_mms import load_custom_mms_model

def extract_lexicon(jsonl_path: Path) -> list[str]:
    words = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Remove punctuation except Harakat
            cleaned = re.sub(r"[«»\"()\[\]{},؛;؟?!—–.:\-_/]", " ", item["text"])
            for w in cleaned.split():
                if len(w) > 1:
                    words.add(w)
    return sorted(list(words))

def main():
    model, adapter, inv_vocab = load_custom_mms_model()
    
    # 1. Labels for pyctcdecode (must match vocab index 0..N)
    vocab_size = len(inv_vocab)
    labels = [inv_vocab[i] for i in range(vocab_size)]
    # pyctcdecode expects blank token as ""
    labels[0] = ""
    # space token
    for i, l in enumerate(labels):
        if l == "|":
            labels[i] = " "

    # 2. Extract rich unigram word list from dataset
    unigrams = extract_lexicon(BASE_DIR / "data" / "dataset" / "dataset.jsonl")
    print(f"Loaded {len(unigrams)} unique vocabulary words into Beam Search Trie.")

    # 3. Build pyctcdecode Beam Search Decoder
    print("Building BeamSearchDecoderCTC...")
    decoder = build_ctcdecoder(
        labels=labels,
        unigrams=unigrams,
        alpha=0.6,
        beta=1.2,
        unk_score_offset=-10.0,
    )
    print("Decoder built successfully!")

    # 4. Transcribe test_1min_unseen.wav
    audio_path = BASE_DIR / "data" / "mela_halo" / "test_1min_unseen.wav"
    wav, sr = sf.read(str(audio_path))
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)

    t = torch.from_numpy(wav.astype("float32")).unsqueeze(0)
    if sr != 16000:
        wav_16k = AF.resample(t, sr, 16000).squeeze(0)
    else:
        wav_16k = t.squeeze(0)

    t0 = time.time()
    with torch.no_grad():
        inputs = wav_16k.unsqueeze(0).to("mps")
        outputs = model.wav2vec2(inputs)
        logits = adapter(outputs.last_hidden_state)[0] # (T, V)
        logits[:, 0] -= 0.8 # Blank discount
        logits_np = logits.cpu().numpy()

    t_inf = time.time() - t0
    
    t0_dec = time.time()
    beam_result = decoder.decode(logits_np, beam_width=25)
    t_dec = time.time() - t0_dec

    dur = len(wav_16k) / 16000.0
    print("\n" + "=" * 65)
    print(f"Audio:          {audio_path.name} ({dur:.1f}s)")
    print(f"Inference Time: {t_inf:.3f}s ({dur/t_inf:.1f}x real-time)")
    print(f"Beam Decoding:  {t_dec:.3f}s")
    print("-" * 65)
    print("PyCTCDecode Beam Search Output:")
    print(beam_result)
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
