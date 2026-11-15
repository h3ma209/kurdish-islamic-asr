#!/usr/bin/env python3
"""
Targeted Fine-Tuner for Meta MMS-1B Central Kurdish (ckb) LM Head.
Warm-starts 100% from Meta's pre-trained ckb lm_head (111 tokens, 142k params),
normalizing rare Arabic-only letters to their ckb phonetic equivalents during CTC
training so zero cold-start tokens exist. KenLM + ScriptureRescorer then restore
authentic Arabic root letters (ض ط ص ذ أ إ اللَّه) and full Harakat at decode time.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as AF
from transformers import AutoProcessor, Wav2Vec2ForCTC

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "dataset"
OUTPUT_DIR = BASE_DIR / "data" / "models" / "mms_1b_islamic_kurdish"

STRIP_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640\uf072\ufeff\u200C-\u200F﴿﴾«»ﷺ]")
EXTRA_ARABIC_TOKENS = ("ذ", "ض", "ظ", "ص", "ط", "ة", "أ", "إ", "آ")
PHONETIC_TWIN_MAP = {
    "ذ": "ز",
    "ض": "ز",
    "ظ": "ز",
    "ص": "س",
    "ط": "ت",
    "ة": "ه",
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
}


def encode_bilingual(text: str, vocab: dict[str, int]) -> list[int]:
    clean = STRIP_RE.sub("", text).replace("ٱ", "ا")
    clean = re.sub(r"\s+", " ", clean).strip()
    ids = []
    for ch in clean:
        tok = "|" if ch == " " else ch
        if tok in vocab:
            ids.append(vocab[tok])
    return ids


def expand_vocab_and_head(
    base_vocab: dict[str, int],
    model: Wav2Vec2ForCTC,
    ckpt_path: Path,
    backup_path: Path,
) -> dict[str, int]:
    vocab = dict(base_vocab)
    next_id = max(vocab.values()) + 1
    for tok in EXTRA_ARABIC_TOKENS:
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1

    old_head = model.lm_head
    in_dim = old_head.in_features
    has_bias = old_head.bias is not None
    new_head = nn.Linear(in_dim, len(vocab), bias=has_bias)

    src_ckpt = backup_path if backup_path.exists() else (ckpt_path if ckpt_path.exists() else None)
    if ckpt_path.exists() and not backup_path.exists():
        tmp = torch.load(ckpt_path, map_location="cpu")
        if tmp["lm_head"]["weight"].shape[0] == len(base_vocab):
            shutil.copy2(ckpt_path, backup_path)
            src_ckpt = backup_path
            print(f"Backed up {len(base_vocab)}-token checkpoint -> {backup_path.name}")

    with torch.no_grad():
        if src_ckpt is not None:
            ckpt = torch.load(src_ckpt, map_location="cpu")
            for idx, state in ckpt["adapters"].items():
                model.wav2vec2.encoder.layers[int(idx)].adapter_layer.load_state_dict(state)
            w = ckpt["lm_head"]["weight"]
            n_copy = min(w.shape[0], len(vocab))
            new_head.weight[:n_copy].copy_(w[:n_copy])
            if has_bias and "bias" in ckpt["lm_head"]:
                b = ckpt["lm_head"]["bias"]
                new_head.bias[:n_copy].copy_(b[:n_copy])
            # Warm-start extra Arabic rows from phonetic twins if expanding from 111
            if w.shape[0] < len(vocab):
                for tok, twin in PHONETIC_TWIN_MAP.items():
                    tid, wid = vocab[tok], vocab[twin]
                    new_head.weight[tid].copy_(new_head.weight[wid])
                    if has_bias:
                        new_head.bias[tid].copy_(new_head.bias[wid])
            print(f"Warm-started adapters + {n_copy}->{len(vocab)} lm_head rows from {src_ckpt.name}!")
        else:
            n_copy = old_head.weight.shape[0]
            new_head.weight[:n_copy].copy_(old_head.weight)
            if has_bias:
                new_head.bias[:n_copy].copy_(old_head.bias)
            for tok, twin in PHONETIC_TWIN_MAP.items():
                tid, wid = vocab[tok], vocab[twin]
                new_head.weight[tid].copy_(new_head.weight[wid])
                if has_bias:
                    new_head.bias[tid].copy_(new_head.bias[wid])

    model.lm_head = new_head
    return vocab


def train_mms_ckb_head(epochs: int = 8, lr: float = 5e-5, accum_steps: int = 8, start_layer: int = 24):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Fine-Tuning Meta MMS-1B bilingual (ckb+ar) top-{48 - start_layer} adapter layers + 120-tok lm_head on {device}...")

    processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", attn_implementation="eager")
    processor.tokenizer.set_target_lang("ckb")
    model.load_adapter("ckb")

    ckpt_path = OUTPUT_DIR / "finetuned_ckb_adapter.pt"
    backup_path = OUTPUT_DIR / "finetuned_ckb_adapter_111.pt.bak"
    vocab = expand_vocab_and_head(processor.tokenizer.get_vocab(), model, ckpt_path, backup_path)
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    enc = model.wav2vec2.encoder
    trainable_params = []
    for i in range(start_layer, len(enc.layers)):
        for p in enc.layers[i].adapter_layer.parameters():
            p.requires_grad = True
            trainable_params.append(p)
    for p in model.lm_head.parameters():
        p.requires_grad = True
        trainable_params.append(p)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    n_params = sum(p.numel() for p in trainable_params)
    print(f"Frozen layers 0..{start_layer - 1} | Trainable adapters ({start_layer}..47) + {len(vocab)}-tok lm_head: {n_params:,} params\n")

    samples = []
    for jname in ("dataset.jsonl", "quran_arabic.jsonl"):
        jpath = DATASET_DIR / jname
        if not jpath.exists():
            continue
        cnt = 0
        with open(jpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                wav_path = DATASET_DIR / item["audio_file"]
                if wav_path.exists() and item["duration"] >= 0.8:
                    labels = encode_bilingual(item["text"], vocab)
                    if len(labels) > 0 and int(item["duration"] * 50) >= len(labels):
                        samples.append({"wav_path": str(wav_path), "labels": labels})
                        cnt += 1
        print(f" - Loaded {cnt} clips from {jname}")

    print(f"Loaded {len(samples)} clean verified clips for training.")
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    split = int(len(samples) * 0.92)
    train_samples = [samples[i] for i in indices[:split]]
    val_samples = [samples[i] for i in indices[split:]]

    print(f"Pre-extracting layer-{start_layer} hidden states for {len(samples)} clips...")
    t_ext = time.time()
    cached_train, cached_val = [], []

    model.wav2vec2.eval()
    with torch.no_grad():
        for idx, s in enumerate(train_samples):
            wav, _ = sf.read(s["wav_path"])
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            wav_t = torch.from_numpy(wav.astype("float32")).unsqueeze(0)
            for spd in (0.9, 1.0, 1.1):
                if spd == 1.0:
                    w_aug = wav_t.squeeze(0).numpy()
                else:
                    # Resample to simulate speed perturbation while keeping 16kHz tensor
                    w_aug = AF.resample(wav_t, int(16000 * spd), 16000).squeeze(0).numpy()
                if int((len(w_aug) / 16000.0) * 50) <= len(s["labels"]):
                    continue
                inp = processor(w_aug, sampling_rate=16000, return_tensors="pt").to(device)
                out = model.wav2vec2(inp.input_values, output_hidden_states=True)
                hidden = out.hidden_states[start_layer].squeeze(0).half().cpu()
                cached_train.append({"hidden": hidden, "labels": torch.tensor(s["labels"], dtype=torch.long)})
            if (idx + 1) % 300 == 0 or idx == len(train_samples) - 1:
                print(f" - [{idx + 1}/{len(train_samples)}] train clips extracted (with 3x speed perturb: {len(cached_train)} samples)...")

        for s in val_samples:
            wav, _ = sf.read(s["wav_path"])
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            inp = processor(wav, sampling_rate=16000, return_tensors="pt").to(device)
            out = model.wav2vec2(inp.input_values, output_hidden_states=True)
            hidden = out.hidden_states[start_layer].squeeze(0).half().cpu()
            cached_val.append({"hidden": hidden, "labels": torch.tensor(s["labels"], dtype=torch.long)})

    print(f"Extraction complete in {time.time() - t_ext:.2f}s! Freeing lower backbone...")
    del model.wav2vec2.feature_extractor
    del model.wav2vec2.feature_projection
    enc.layers = enc.layers[start_layer:]
    if device == "mps":
        torch.mps.empty_cache()

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CTCLoss(blank=vocab["<pad>"], zero_infinity=True)

    best_val_loss = float("inf")
    saved_ckpt_path = OUTPUT_DIR / "finetuned_ckb_adapter.pt"

    def apply_spec_aug(h: torch.Tensor) -> torch.Tensor:
        # h: [1, T, C]
        _, t_len, c_len = h.shape
        h = h.clone()
        if t_len > 20:
            for _ in range(2):
                w = random.randint(1, min(12, t_len // 10))
                t0_m = random.randint(0, t_len - w)
                h[:, t0_m:t0_m + w, :] = 0.0
        for _ in range(2):
            cw = random.randint(1, 64)
            c0_m = random.randint(0, c_len - cw)
            h[:, :, c0_m:c0_m + cw] = 0.0
        return h

    def forward_top(h_in: torch.Tensor) -> torch.Tensor:
        h = h_in
        for layer in enc.layers:
            h = layer(h)[0]
        h = enc.layer_norm(h)
        return model.lm_head(h)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        enc.layers.eval()
        for layer in enc.layers:
            layer.adapter_layer.train()
        model.lm_head.train()
        train_loss = 0.0
        random.shuffle(cached_train)
        optimizer.zero_grad()

        for step, item in enumerate(cached_train, 1):
            hidden = apply_spec_aug(item["hidden"].unsqueeze(0).float()).to(device)
            target = item["labels"].to(device)
            logits = forward_top(hidden)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            input_len = torch.tensor([hidden.shape[1]], dtype=torch.long)
            target_len = torch.tensor([len(target)], dtype=torch.long)

            loss = criterion(log_probs, target, input_len, target_len)
            if not torch.isnan(loss) and not torch.isinf(loss):
                (loss / accum_steps).backward()
                train_loss += loss.item()

            if step % accum_steps == 0 or step == len(cached_train):
                has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in trainable_params)
                if not has_nan:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        train_loss /= len(cached_train)

        enc.layers.eval()
        model.lm_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for item in cached_val:
                hidden = item["hidden"].unsqueeze(0).float().to(device)
                target = item["labels"].to(device)
                logits = forward_top(hidden)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                input_len = torch.tensor([hidden.shape[1]], dtype=torch.long)
                target_len = torch.tensor([len(target)], dtype=torch.long)
                loss = criterion(log_probs, target, input_len, target_len)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()

        val_loss /= len(cached_val)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "start_layer": start_layer,
                    "vocab_size": len(vocab),
                    "adapters": {start_layer + idx: layer.adapter_layer.state_dict() for idx, layer in enumerate(enc.layers)},
                    "lm_head": model.lm_head.state_dict(),
                },
                saved_ckpt_path,
            )

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {'[BEST]' if is_best else ''}")

    print(f"\nTraining completed in {time.time() - t0:.2f}s! Best Val Loss: {best_val_loss:.4f}")
    print(f"Saved fine-tuned adapter + head to: {saved_ckpt_path}")


if __name__ == "__main__":
    train_mms_ckb_head()
