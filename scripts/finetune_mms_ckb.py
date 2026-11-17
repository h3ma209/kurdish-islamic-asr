#!/usr/bin/env python3
"""
Targeted Fine-Tuner for Meta MMS-1B Central Kurdish (ckb) LM Head.
Warm-starts 100% from Meta's pre-trained ckb lm_head (111 tokens, 142k params),
normalizing rare Arabic-only letters to their ckb phonetic equivalents during CTC
training so zero cold-start tokens exist. KenLM + ScriptureRescorer then restore
authentic Arabic root letters (ض ط ص ذ أ إ اللَّه) and full Harakat at decode time.
"""

from __future__ import annotations

import argparse
import contextlib
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
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Wav2Vec2ForCTC

try:
    from transformers.models.wav2vec2.modeling_wav2vec2 import create_bidirectional_mask
except ImportError:
    create_bidirectional_mask = None

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


def _load_wav_mono(wav_path: str) -> torch.Tensor:
    """Load an audio file as a 1D float32 tensor."""
    wav, _ = sf.read(wav_path)
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)
    return torch.from_numpy(wav.astype("float32"))


def _pack_batches_by_budget(items: list, len_fn, max_batch_size: int, max_budget: int) -> list[list]:
    """Group items into batches capped by both count and padded length budget."""
    batches: list[list] = []
    cur: list = []
    cur_max = 0
    for item in items:
        item_len = len_fn(item)
        next_max = max(cur_max, item_len)
        if cur and (len(cur) >= max_batch_size or next_max * (len(cur) + 1) > max_budget):
            batches.append(cur)
            cur = [item]
            cur_max = item_len
        else:
            cur.append(item)
            cur_max = next_max
    if cur:
        batches.append(cur)
    return batches


def _extract_hidden_states(
    samples: list[dict],
    processor: AutoProcessor,
    model: Wav2Vec2ForCTC,
    start_layer: int,
    device: str,
    use_amp: bool,
    max_batch_size: int = 16,
    max_batch_samples: int = 16000 * 80,
    speed_perturb: bool = False,
    label_desc: str = "clips",
) -> list[dict]:
    """Extract layer-`start_layer` hidden states in duration-budgeted batches with AMP."""
    enc = model.wav2vec2.encoder
    all_layers = enc.layers
    orig_ln = enc.layer_norm
    # Only run layers 0..start_layer-1 without final layer_norm to match hidden_states[start_layer]
    enc.layers = all_layers[:start_layer]
    enc.layer_norm = nn.Identity()

    items: list[tuple[np.ndarray, list[int]]] = []
    speeds = (0.9, 1.0, 1.1) if speed_perturb else (1.0,)
    for s in samples:
        wav_t = _load_wav_mono(s["wav_path"])
        wav_dev = wav_t.unsqueeze(0).to(device) if (speed_perturb and device == "cuda") else wav_t.unsqueeze(0)
        for spd in speeds:
            if spd == 1.0:
                w_aug = wav_t.numpy()
            else:
                w_aug = AF.resample(wav_dev, int(16000 * spd), 16000).squeeze(0).cpu().numpy()
            if int((len(w_aug) / 16000.0) * 50) <= len(s["labels"]):
                continue
            items.append((w_aug, s["labels"]))

    # Sort longest-first so large allocations happen upfront without caching fragmentation
    items.sort(key=lambda x: len(x[0]), reverse=True)
    batches = _pack_batches_by_budget(items, lambda x: len(x[0]), max_batch_size, max_batch_samples)
    cached: list[dict] = []
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
    done = 0
    next_log = 640

    with torch.no_grad():
        for batch in batches:
            wavs = [b[0] for b in batch]
            inp = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inp.input_values.to(device)
            attention_mask = inp.attention_mask.to(device) if "attention_mask" in inp else None
            raw_lens = (
                attention_mask.sum(dim=1)
                if attention_mask is not None
                else torch.tensor([len(w) for w in wavs], device=device)
            )
            feat_lens = model.wav2vec2._get_feat_extract_output_lengths(raw_lens).tolist()

            with amp_ctx:
                out = model.wav2vec2(input_values, attention_mask=attention_mask, output_hidden_states=False)

            hidden_batch = out.last_hidden_state.half().cpu()
            for i, flen in enumerate(feat_lens):
                cached.append({
                    "hidden": hidden_batch[i, :int(flen)].contiguous(),
                    "labels": torch.tensor(batch[i][1], dtype=torch.long),
                })

            done += len(batch)
            if done >= next_log or done == len(items):
                print(f" - [{done}/{len(items)}] {label_desc} extracted...")
                next_log += 640

    enc.layers = all_layers
    enc.layer_norm = orig_ln
    return cached


def _apply_spec_aug(h: torch.Tensor) -> torch.Tensor:
    """Apply SpecAugment time and channel masking to [T, C] hidden states."""
    t_len, c_len = h.shape
    h = h.clone()
    if t_len > 20:
        for _ in range(2):
            w = random.randint(1, min(12, t_len // 10))
            t0_m = random.randint(0, t_len - w)
            h[t0_m:t0_m + w, :] = 0.0
    for _ in range(2):
        cw = random.randint(1, 64)
        c0_m = random.randint(0, c_len - cw)
        h[:, c0_m:c0_m + cw] = 0.0
    return h


def _make_length_bucketed_batches(
    data: list[dict], batch_size: int, shuffle: bool, max_batch_frames: int = 4000
) -> list[list[dict]]:
    """Group samples of similar frame lengths into batches capped by count and frame budget."""
    if shuffle:
        # Jitter lengths slightly so batches vary across epochs while staying length-coherent
        ordered = sorted(data, key=lambda x: x["hidden"].shape[0] + random.randint(-16, 16))
    else:
        ordered = sorted(data, key=lambda x: x["hidden"].shape[0])
    batches = _pack_batches_by_budget(ordered, lambda x: x["hidden"].shape[0], batch_size, max_batch_frames)
    if shuffle:
        random.shuffle(batches)
    return batches


def train_mms_ckb_head(
    epochs: int = 8,
    lr: float = 5e-5,
    batch_size: int = 8,
    accum_steps: int = 1,
    start_layer: int = 24,
    use_cache: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device == "cuda"
    attn_impl = "sdpa" if device == "cuda" else "eager"
    print(f"Fine-Tuning Meta MMS-1B bilingual (ckb+ar) top-{48 - start_layer} adapter layers + 120-tok lm_head on {device} (amp={use_amp}, attn={attn_impl}, bs={batch_size})...")

    processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", attn_implementation=attn_impl)
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
    for jname in ("dataset.jsonl", "quran_arabic.jsonl", "tafsir_kurdish.jsonl", "multispeaker_kurdish.jsonl"):
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

    feat_cache_path = OUTPUT_DIR / f"cached_features_L{start_layer}.pt"
    cached_train, cached_val = None, None
    if use_cache and feat_cache_path.exists():
        cache_data = torch.load(feat_cache_path, map_location="cpu")
        if cache_data.get("n_samples") == len(samples):
            cached_train = cache_data["train"]
            cached_val = cache_data["val"]
            print(f"Loaded cached layer-{start_layer} features from {feat_cache_path.name} ({len(cached_train)} train, {len(cached_val)} val).")

    if cached_train is None:
        print(f"Pre-extracting layer-{start_layer} hidden states for {len(samples)} clips (duration-budgeted, max bs=16)...")
        t_ext = time.time()
        model.wav2vec2.eval()
        # Offload unused top layers to CPU during extraction to save ~1.9 GB VRAM
        if device == "cuda":
            for layer in enc.layers[start_layer:]:
                layer.to("cpu")
            torch.cuda.empty_cache()
        cached_train = _extract_hidden_states(
            train_samples, processor, model, start_layer, device, use_amp,
            max_batch_size=16, max_batch_samples=16000 * 80, speed_perturb=True, label_desc="train samples (3x speed perturb)",
        )
        cached_val = _extract_hidden_states(
            val_samples, processor, model, start_layer, device, use_amp,
            max_batch_size=16, max_batch_samples=16000 * 80, speed_perturb=False, label_desc="val samples",
        )
        print(f"Extraction complete in {time.time() - t_ext:.2f}s!")
        if use_cache:
            torch.save({"n_samples": len(samples), "train": cached_train, "val": cached_val}, feat_cache_path)

    print("Freeing lower backbone...")
    del model.wav2vec2.feature_extractor
    del model.wav2vec2.feature_projection
    enc.layers = enc.layers[start_layer:]
    if device == "cuda":
        for layer in enc.layers:
            layer.to(device)
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CTCLoss(blank=vocab["<pad>"], zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()

    best_val_loss = float("inf")
    saved_ckpt_path = OUTPUT_DIR / "finetuned_ckb_adapter.pt"

    def forward_top(h_in: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if create_bidirectional_mask is not None:
            bidir_mask = create_bidirectional_mask(config=enc.config, input_embeds=h_in, attention_mask=attn_mask)
        else:
            bidir_mask = (1.0 - attn_mask[:, None, None, :].to(h_in.dtype)) * torch.finfo(h_in.dtype).min
        h = h_in
        for layer in enc.layers:
            h = layer(h, attention_mask=bidir_mask)[0]
        h = enc.layer_norm(h)
        return model.lm_head(h)

    def run_batch(batch: list[dict], augment: bool) -> torch.Tensor:
        h_list = [_apply_spec_aug(x["hidden"]) if augment else x["hidden"] for x in batch]
        input_lens = torch.tensor([h.shape[0] for h in h_list], dtype=torch.long)
        h_pad = pad_sequence(h_list, batch_first=True)
        if not use_amp:
            h_pad = h_pad.float()
        h_pad = h_pad.to(device, non_blocking=True)

        t_max = h_pad.shape[1]
        attn_mask = (torch.arange(t_max, device=device).unsqueeze(0) < input_lens.to(device).unsqueeze(1))
        targets = torch.cat([x["labels"] for x in batch]).to(device, non_blocking=True)
        target_lens = torch.tensor([len(x["labels"]) for x in batch], dtype=torch.long)

        with amp_ctx:
            logits = forward_top(h_pad, attn_mask)
        log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
        return criterion(log_probs, targets, input_lens, target_lens)

    t0 = time.time()
    val_batches = _make_length_bucketed_batches(cached_val, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        enc.layers.eval()
        for layer in enc.layers:
            layer.adapter_layer.train()
        model.lm_head.train()
        train_loss = 0.0
        train_batches = _make_length_bucketed_batches(cached_train, batch_size=batch_size, shuffle=True)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_batches, 1):
            loss = run_batch(batch, augment=True)
            if not torch.isnan(loss) and not torch.isinf(loss):
                scaler.scale(loss / accum_steps).backward()
                train_loss += loss.item() * len(batch)

            if step % accum_steps == 0 or step == len(train_batches):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in trainable_params)
                    if not has_nan:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                        optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        train_loss /= len(cached_train)

        enc.layers.eval()
        model.lm_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_batches:
                loss = run_batch(batch, augment=False)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item() * len(batch)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    train_mms_ckb_head(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        use_cache=not args.no_cache,
    )
