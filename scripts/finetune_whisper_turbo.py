#!/usr/bin/env python3
"""
Fine-tune OpenAI Whisper-Large-V3-Turbo 4-Layer Decoder for Verbatim Central Kurdish (Sorani)
+ Classical Arabic Islamic Lectures (with Harakat, ﴿...﴾, «...», and Arabic root letters).
Freezes the 32-layer multilingual encoder (pre-extracting hidden states in fp16 on MPS)
and trains the 4-layer decoder with cosine LR and label smoothing.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "dataset"
OUTPUT_DIR = BASE_DIR / "data" / "models" / "whisper_turbo_islamic_kurdish"
MODEL_ID = "openai/whisper-large-v3-turbo"


def train_whisper_decoder(epochs: int = 6, lr: float = 2e-5, accum_steps: int = 8):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Fine-Tuning {MODEL_ID} 4-layer decoder on {device}...")

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)

    prompt_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe", no_timestamps=True)
    sot_id = model.config.decoder_start_token_id
    prefix_ids = [sot_id] + [tok_id for _, tok_id in prompt_ids]
    eos_id = processor.tokenizer.eos_token_id

    samples = []
    with open(DATASET_DIR / "dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            wav_path = DATASET_DIR / item["audio_file"]
            if wav_path.exists() and 0.8 <= item["duration"] <= 30.0:
                tok_ids = processor.tokenizer(item["text"], add_special_tokens=False).input_ids
                if 0 < len(tok_ids) <= 440:
                    samples.append({
                        "wav_path": str(wav_path),
                        "tok_ids": tok_ids,
                    })

    print(f"Loaded {len(samples)} clean segments (<= 30s) for training.")
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    split = int(len(samples) * 0.92)
    train_samples = [samples[i] for i in indices[:split]]
    val_samples = [samples[i] for i in indices[split:]]

    # Pre-extract encoder hidden states in fp16 (stored in CPU RAM as fp16, ~3.8MB/clip)
    print(f"Pre-extracting encoder hidden states for {len(samples)} clips...")
    t_ext = time.time()
    encoder = model.get_encoder()
    encoder.eval()
    cached_train, cached_val = [], []

    def encode_split(split_samples, label):
        out = []
        with torch.no_grad():
            for idx, s in enumerate(split_samples):
                wav, _ = sf.read(s["wav_path"])
                if len(wav.shape) > 1:
                    wav = wav.mean(axis=1)
                feats = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=torch.float16)
                enc_hidden = encoder(feats).last_hidden_state.squeeze(0).cpu()  # (1500, 1280) fp16
                full_seq = prefix_ids + s["tok_ids"] + [eos_id]
                out.append({
                    "enc_hidden": enc_hidden,
                    "dec_input": torch.tensor(full_seq[:-1], dtype=torch.long),
                    "labels": torch.tensor([-100] * (len(prefix_ids) - 1) + full_seq[len(prefix_ids):], dtype=torch.long),
                })
                if (idx + 1) % 300 == 0 or idx == len(split_samples) - 1:
                    print(f" - [{idx + 1}/{len(split_samples)}] {label} clips encoded...")
        return out

    cached_train = encode_split(train_samples, "train")
    cached_val = encode_split(val_samples, "val")
    print(f"Encoder extraction complete in {time.time() - t_ext:.2f}s! Freeing encoder...")

    del model.model.encoder
    if device == "mps":
        torch.mps.empty_cache()

    # Cast decoder + proj_out to float32 for rock-solid MPS AdamW stability
    decoder = model.model.decoder.float().to(device)
    proj_out = model.proj_out.float().to(device)

    trainable_params = list(decoder.parameters()) + list(proj_out.parameters())
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable 4-layer Whisper Decoder params: {n_params:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = (len(cached_train) // accum_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        decoder.train()
        proj_out.train()
        train_loss = 0.0
        random.shuffle(cached_train)
        optimizer.zero_grad()

        for step, item in enumerate(cached_train, 1):
            enc_h = item["enc_hidden"].unsqueeze(0).float().to(device)
            dec_in = item["dec_input"].unsqueeze(0).to(device)
            labels = item["labels"].unsqueeze(0).to(device)

            dec_out = decoder(input_ids=dec_in, encoder_hidden_states=enc_h).last_hidden_state
            logits = proj_out(dec_out)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            if not torch.isnan(loss) and not torch.isinf(loss):
                (loss / accum_steps).backward()
                train_loss += loss.item()

            if step % accum_steps == 0 or step == len(cached_train):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss /= len(cached_train)

        decoder.eval()
        proj_out.eval()
        val_loss = 0.0
        with torch.no_grad():
            for item in cached_val:
                enc_h = item["enc_hidden"].unsqueeze(0).float().to(device)
                dec_in = item["dec_input"].unsqueeze(0).to(device)
                labels = item["labels"].unsqueeze(0).to(device)
                dec_out = decoder(input_ids=dec_in, encoder_hidden_states=enc_h).last_hidden_state
                logits = proj_out(dec_out)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()

        val_loss /= len(cached_val)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "decoder": decoder.state_dict(),
                    "proj_out": proj_out.state_dict(),
                },
                OUTPUT_DIR / "finetuned_decoder.pt",
            )

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {'[BEST]' if is_best else ''}")

    print(f"\nTraining complete in {time.time() - t0:.2f}s! Best Val Loss: {best_val_loss:.4f}")
    print(f"Saved to: {OUTPUT_DIR / 'finetuned_decoder.pt'}")


if __name__ == "__main__":
    train_whisper_decoder()
