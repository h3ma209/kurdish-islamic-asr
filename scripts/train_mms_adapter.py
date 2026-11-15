#!/usr/bin/env python3
"""
Intensive Meta MMS-1B Custom Adapter Trainer for Kurdish Sorani + Arabic Harakat.
Uses a 2-layer Non-Linear Adapter Head (GELU + LayerNorm) + Acoustic SpecAugment jitter.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "dataset"
OUTPUT_DIR = BASE_DIR / "data" / "models" / "mms_1b_islamic_kurdish"


class IntensiveMMSAdapter(nn.Module):
    def __init__(self, in_features: int = 1280, hidden_dim: int = 512, vocab_size: int = 65, dropout: float = 0.15):
        super().__init__()
        self.proj1 = nn.Linear(in_features, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj2 = nn.Linear(hidden_dim, vocab_size)

        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, 1280)
        h = self.dropout(self.act(self.ln(self.proj1(x))))
        return self.proj2(h)


def build_vocabulary(jsonl_path: Path) -> dict[str, int]:
    chars = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cleaned = item["text"].replace("\n", " ").strip()
            chars.update(cleaned)

    ignored = {
        "«", "»", '"', "(", ")", "[", "]", "{", "}", ",", "،", "؛", ";",
        "؟", "?", "!", "-", "—", "–", ".", ":", "ـ",
        "٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    }
    phonetic_chars = sorted([c for c in chars if c not in ignored and c != " "])

    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,
    }

    for idx, c in enumerate(phonetic_chars, start=5):
        vocab[c] = idx

    return vocab


def text_to_ids(text: str, vocab: dict[str, int]) -> list[int]:
    tokens = []
    for ch in text.strip():
        if ch == " ":
            tok = "|"
        elif ch in vocab:
            tok = ch
        else:
            continue
        tokens.append(vocab[tok])
    return tokens


def pre_extract_embeddings(model, samples: list[dict], device: str) -> list[dict]:
    print(f"Pre-extracting 1280-dim embeddings for {len(samples)} clips through MMS-1B backbone...")
    t0 = time.time()
    extracted = []

    model.wav2vec2.eval()
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            wav, sr = sf.read(sample["wav_path"])
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)

            t_wav = torch.from_numpy(wav.astype("float32")).unsqueeze(0).to(device)
            outputs = model.wav2vec2(t_wav)
            hidden = outputs.last_hidden_state.squeeze(0).cpu()

            extracted.append({
                "hidden": hidden,
                "label_ids": torch.tensor(sample["label_ids"], dtype=torch.long),
            })
            if (idx + 1) % 15 == 0 or idx == len(samples) - 1:
                print(f" - [{idx + 1:02d}/{len(samples):02d}] clips extracted...")

    elapsed = time.time() - t0
    print(f"Extraction complete in {elapsed:.2f}s ({elapsed / len(samples):.3f}s/clip)!")
    return extracted


def apply_acoustic_specaugment(hidden: torch.Tensor) -> torch.Tensor:
    """Applies Gaussian feature noise and random time-masking during training."""
    augmented = hidden.clone()
    # 1. Feature Gaussian noise
    if random.random() < 0.6:
        augmented += torch.randn_like(augmented) * 0.04

    # 2. Random time masking (mask 1 to 3 blocks of 10 frames)
    if random.random() < 0.5 and augmented.shape[0] > 30:
        t_len = augmented.shape[0]
        mask_len = min(15, int(t_len * 0.08))
        start = random.randint(0, t_len - mask_len)
        augmented[start : start + mask_len, :] = 0.0

    return augmented


def train():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")

    jsonl_path = DATASET_DIR / "dataset.jsonl"
    vocab = build_vocabulary(jsonl_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = OUTPUT_DIR / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary created: {len(vocab)} tokens (saved to {vocab_path})")

    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            wav_file = DATASET_DIR / item["audio_file"]
            if wav_file.exists():
                tokens = text_to_ids(item["text"], vocab)
                if len(tokens) > 0:
                    samples.append({
                        "wav_path": str(wav_file),
                        "label_ids": tokens,
                        "text": item["text"],
                    })

    print(f"Total training samples: {len(samples)} clips")

    print("Loading base MMS-1B model...")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device)

    cached_data = pre_extract_embeddings(model, samples, device)
    del model
    if device == "mps":
        torch.mps.empty_cache()

    vocab_size = len(vocab)
    adapter = IntensiveMMSAdapter(in_features=1280, hidden_dim=512, vocab_size=vocab_size, dropout=0.20).to(device)

    total_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Trainable Adapter Parameters: {total_params:,} (2-Layer Non-Linear GELU Head)")

    np.random.seed(42)
    indices = np.random.permutation(len(cached_data))
    split_idx = int(len(cached_data) * 0.9)
    train_data = [cached_data[i] for i in indices[:split_idx]]
    val_data = [cached_data[i] for i in indices[split_idx:]]

    accum_steps = 8
    num_epochs = 35
    learning_rate = 4e-4

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=3e-5)

    print(f"\nStarting Stabilized Training ({num_epochs} Epochs on {len(train_data)} train / {len(val_data)} val, accum={accum_steps})...")
    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0

    t_start = time.time()
    for epoch in range(1, num_epochs + 1):
        adapter.train()
        train_loss = 0.0

        random.shuffle(train_data)
        optimizer.zero_grad()

        for step, item in enumerate(train_data, 1):
            hidden = item["hidden"]
            augmented = apply_acoustic_specaugment(hidden).unsqueeze(0).to(device)
            labels = item["label_ids"].to(device)

            logits = adapter(augmented)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            input_len = torch.tensor([augmented.shape[1]], dtype=torch.long)
            target_len = torch.tensor([len(labels)], dtype=torch.long)

            loss = criterion(log_probs, labels, input_len, target_len)
            if not torch.isnan(loss) and not torch.isinf(loss):
                scaled_loss = loss / accum_steps
                scaled_loss.backward()
                train_loss += loss.item()

            if step % accum_steps == 0 or step == len(train_data):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        train_loss /= len(train_data)

        # Validation (clean, no augmentations)
        adapter.eval()
        val_loss = 0.0
        with torch.no_grad():
            for item in val_data:
                hidden = item["hidden"].unsqueeze(0).to(device)
                labels = item["label_ids"].to(device)
                logits = adapter(hidden)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                input_len = torch.tensor([hidden.shape[1]], dtype=torch.long)
                target_len = torch.tensor([len(labels)], dtype=torch.long)

                loss = criterion(log_probs, labels, input_len, target_len)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()

        val_loss /= len(val_data)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(adapter.state_dict(), OUTPUT_DIR / "adapter_head.pt")
        else:
            patience_counter += 1

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | Train CTC: {train_loss:.4f} | Val CTC: {val_loss:.4f} {'[BEST]' if is_best else ''}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement in {patience} epochs).")
            break

    total_time = time.time() - t_start
    print(f"\nStabilized Training Completed in {total_time:.2f}s! Best Val Loss: {best_val_loss:.4f}")
    print(f"Enhanced Adapter saved to: {OUTPUT_DIR / 'adapter_head.pt'}")


if __name__ == "__main__":
    train()
