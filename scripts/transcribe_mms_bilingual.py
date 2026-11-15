#!/usr/bin/env python3
"""
Zero-shot bilingual CTC transcription (Kurdish Sorani + Arabic) using fused MMS-1B heads.
Preserves both Kurdish Sorani consonants/vowels (گ, ڵ, ڕ, پ, ژ, ۆ, ێ, ە)
and Classical Arabic consonants/vowels (ص, ض, ط, ظ, ذ, ث, أ, إ, ة, آ).
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
import torch.nn as nn
import torchaudio.functional as AF
from transformers import AutoProcessor, Wav2Vec2ForCTC

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from zero_shot_clone import normalize_ckb


def build_fused_mms_model(model_name: str = "facebook/mms-1b-all", device: str = "mps"):
    print("Loading base MMS-1B model and processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # 1. Extract CKB adapter & vocab
    processor.tokenizer.set_target_lang("ckb")
    model.load_adapter("ckb")
    vocab_ckb = processor.tokenizer.get_vocab()
    # Sort by index
    ckb_tokens_by_idx = [None] * len(vocab_ckb)
    for tok, idx in vocab_ckb.items():
        ckb_tokens_by_idx[idx] = tok

    w_ckb = model.lm_head.weight.detach().clone()  # (111, 1280)
    b_ckb = model.lm_head.bias.detach().clone()    # (111,)

    # 2. Extract ARA adapter & vocab
    processor.tokenizer.set_target_lang("ara")
    model.load_adapter("ara")
    vocab_ara = processor.tokenizer.get_vocab()
    w_ara = model.lm_head.weight.detach().clone()  # (121, 1280)
    b_ara = model.lm_head.bias.detach().clone()    # (121,)

    # 3. Construct unified vocab (preserve CKB indices 0..110)
    unified_tokens = list(ckb_tokens_by_idx)
    ara_new_tokens = []
    for tok, idx in vocab_ara.items():
        if tok not in vocab_ckb:
            ara_new_tokens.append(tok)

    unified_tokens.extend(ara_new_tokens)
    vocab_union = {tok: idx for idx, tok in enumerate(unified_tokens)}
    inv_vocab = {idx: tok for idx, tok in enumerate(unified_tokens)}
    union_size = len(unified_tokens)

    w_union = torch.zeros((union_size, 1280))
    b_union = torch.zeros((union_size,))

    for tok, idx in vocab_union.items():
        in_ckb = tok in vocab_ckb
        in_ara = tok in vocab_ara
        if in_ckb and in_ara:
            idx_c = vocab_ckb[tok]
            idx_a = vocab_ara[tok]
            w_union[idx] = 0.5 * (w_ckb[idx_c] + w_ara[idx_a])
            b_union[idx] = 0.5 * (b_ckb[idx_c] + b_ara[idx_a])
        elif in_ckb:
            idx_c = vocab_ckb[tok]
            w_union[idx] = w_ckb[idx_c]
            b_union[idx] = b_ckb[idx_c]
        elif in_ara:
            idx_a = vocab_ara[tok]
            w_union[idx] = w_ara[idx_a]
            b_union[idx] = b_ara[idx_a]

    model.lm_head = nn.Linear(1280, union_size)
    model.lm_head.weight.data = w_union
    model.lm_head.bias.data = b_union

    model = model.to(device)
    model.eval()

    return model, processor, inv_vocab, vocab_union


def ctc_decode(pred_ids: list[int], inv_vocab: dict[int, str]) -> str:
    chars = []
    prev = None
    pad_idx = 0  # <pad> is 0
    for p in pred_ids:
        if p != prev:
            if p != pad_idx and p in inv_vocab:
                tok = inv_vocab[p]
                if tok not in ("<pad>", "<s>", "</s>", "<unk>"):
                    chars.append(tok)
        prev = p
    return "".join(chars).replace("|", " ").strip()


def transcribe_file(
    audio_path: Path,
    chunk_sec: float = 20.0,
    stride_sec: float = 1.0,
) -> dict:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, processor, inv_vocab, _ = build_fused_mms_model(device=device)

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

        rms = float(np.sqrt(np.mean(piece**2)))
        if rms < 0.003:
            curr_idx += step_samples
            continue

        inputs = processor(piece, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        txt = ctc_decode(pred_ids, inv_vocab)

        if txt:
            norm = normalize_ckb(txt)
            segments.append({
                "start": round(start_s, 2),
                "end": round(end_s, 2),
                "text": txt,
                "normalized": norm,
            })
            full_texts.append(txt)
            print(f"[{start_s:5.1f}s - {end_s:5.1f}s]: {norm}")

        curr_idx += step_samples

    elapsed = time.time() - t0
    full_text = " ".join(full_texts)
    norm_text = normalize_ckb(full_text)

    return {
        "audio_file": str(audio_path),
        "duration_seconds": round(total_dur, 2),
        "elapsed_seconds": round(elapsed, 2),
        "model": "facebook/mms-1b-all (Fused CKB + ARA)",
        "segments": segments,
        "raw_transcript": full_text,
        "normalized_transcript": norm_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Fused Bilingual MMS-1B Transcription")
    parser.add_argument("audio", type=Path, help="Path to input audio file")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    res = transcribe_file(args.audio)
    out_path = args.output or args.audio.with_name(f"{args.audio.stem}_mms_fused.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    txt_path = args.audio.with_name(f"{args.audio.stem}_mms_fused.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(res["normalized_transcript"] + "\n")

    print(f"\nSaved fused transcript to {out_path} and {txt_path}")
    print(f"Elapsed: {res['elapsed_seconds']}s for {res['duration_seconds']}s audio")


if __name__ == "__main__":
    main()
