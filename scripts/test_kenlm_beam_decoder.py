#!/usr/bin/env python3
"""
Test KenLM 3-gram Beam Search Decoder for Meta MMS-1B Transcriber.
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

def get_kenlm_decoder(inv_vocab: dict[int, str]):
    arpa_path = BASE_DIR / "data" / "models" / "mms_1b_islamic_kurdish" / "kurdish_islamic_3gram.arpa"
    
    # 1. Labels formatting for pyctcdecode
    # Note: Only index 0 (CTC blank) can be "". All other tokens must be distinct single characters.
    vocab_size = len(inv_vocab)
    labels = []
    for i in range(vocab_size):
        tok = inv_vocab.get(i, "")
        if i == 0 or tok == "<pad>":
            labels.append("")
        elif tok in ("<s>", "</s>", "<unk>"):
            labels.append(chr(i))  # Unique dummy ascii char, never in acoustic text
        elif tok == "|":
            labels.append(" ")
        else:
            labels.append(tok)

    # 2. Extract vocabulary words from dataset + Quran
    words = set()
    with open(BASE_DIR / "data" / "dataset" / "dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cleaned = re.sub(r"[«»\"()\[\]{},؛;؟?!—–.:\-_/]", " ", item["text"])
                for w in cleaned.split():
                    if len(w) > 0:
                        words.add(w)

    print(f"Initializing PyCTCDecode Beam Search with {len(words)} lexicon words & KenLM 3-gram...")
    t0 = time.time()
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(arpa_path),
        unigrams=sorted(list(words)),
        alpha=0.7,   # Language model weight
        beta=1.5,    # Word insertion bonus
        unk_score_offset=-10.0,
    )
    print(f"KenLM Beam Decoder built in {time.time() - t0:.2f}s!")
    return decoder

def test_audio(decoder, model, adapter, audio_path: Path):
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
        logits[:, 0] -= 0.5  # slight blank discount
        logits_np = logits.cpu().numpy()

    t_inf = time.time() - t0
    
    t0_dec = time.time()
    beam_output = decoder.decode(logits_np, beam_width=30)
    t_dec = time.time() - t0_dec

    dur = len(wav_16k) / 16000.0
    print("\n" + "=" * 65)
    print(f"Audio:       {audio_path.name} ({dur:.1f}s)")
    print(f"Inference:   {t_inf:.3f}s ({dur/t_inf:.1f}x real-time)")
    print(f"KenLM Decode:{t_dec:.3f}s")
    print("-" * 65)
    print("MMS-1B + KenLM 3-Gram Beam Output:")
    print(beam_output)
    print("=" * 65 + "\n")

def main():
    model, adapter, inv_vocab = load_custom_mms_model()
    decoder = get_kenlm_decoder(inv_vocab)
    
    test_audio(decoder, model, adapter, BASE_DIR / "data" / "mela_halo" / "test_1min_unseen.wav")

if __name__ == "__main__":
    main()
