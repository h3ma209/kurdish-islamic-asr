#!/usr/bin/env python3
"""
High-Purity 44.1kHz DualAR Fine-Tuning for `ArminRahimi/Audio8-TTS-Kurdish-Sorani` (v2).

Key Quality Upgrades over v1:
1. Trained on 339 native 44.1kHz silence-trimmed, speech-rate-filtered Kurdish Tafsir clips (`51.6 mins`).
2. Trains a dedicated `sem_head` parameter (`[4097, 896]`) covering `eos_token_id` + all `4,096` semantic
   audio tokens, and writes the updated semantic vectors back into `model.embeddings.weight[valid_ids]`
   on checkpoint save — adapting the semantic output space without touching the 151,678 text tokens.
3. Unfreezes the top 12 Slow AR transformer blocks (`model.layers[12:]`, `model.norm`) + all Fast AR
   acoustic layers (`fast_project_in`, `fast_embeddings`, `fast_layers`, `fast_norm`, `fast_output`).
4. Increases Fast AR frame coverage from 64 -> 192 frames per clip (3x more acoustic supervision per step)
   with label smoothing (`0.02`) to prevent overconfident token collapse/warble.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torchaudio.functional import resample
from transformers import AutoModel, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from zero_shot_clone import normalize_ckb

AUDIO8_MODEL_ID = "ArminRahimi/Audio8-TTS-Kurdish-Sorani"

DATASET_DIR = BASE_DIR / "data" / "tts_dataset"
CODES_DIR = DATASET_DIR / "codes"
META_JSONL = DATASET_DIR / "metadata.jsonl"
OUTPUT_MODEL_DIR = BASE_DIR / "data" / "models" / "audio8_quran_kurdish"
SAMPLES_DIR = OUTPUT_MODEL_DIR / "samples"

CODEC_SAMPLE_RATE = 44100


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_dataset_codes(model, meta_rows: list[dict]) -> list[dict]:
    """Encode all 44.1kHz WAV clips into [10, T] codec .npy files (cached on disk)."""
    CODES_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(10)
    codec_dev = torch.device("cpu")
    model.load_codec(device=codec_dev, dtype=torch.float32)

    prepared = []
    to_encode = []
    for row in meta_rows:
        npy_path = CODES_DIR / f"{row['id']}.npy"
        if npy_path.exists():
            codes = np.load(str(npy_path))
            prepared.append({**row, "codes_path": str(npy_path), "frames": int(codes.shape[1])})
        else:
            to_encode.append((row, npy_path))

    if to_encode:
        print(f"[Step 1/3] Encoding {len(to_encode)} native 44.1kHz Kurdish Tafsir clips with ArkttsCodec...", flush=True)
        t0 = time.time()
        for idx, (row, npy_path) in enumerate(to_encode, 1):
            wav, sr = sf.read(row["audio_filepath"], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != CODEC_SAMPLE_RATE:
                wav_t = resample(torch.from_numpy(wav), int(sr), CODEC_SAMPLE_RATE)
            else:
                wav_t = torch.from_numpy(wav)

            audio_values = wav_t.unsqueeze(0).unsqueeze(0).to(codec_dev)
            audio_lens = torch.tensor([wav_t.numel()], dtype=torch.long, device=codec_dev)
            with torch.inference_mode():
                codes, code_lens = model.encode_audio(audio_values, audio_lens)
            valid_len = int(code_lens[0].item())
            codes_np = codes[0, :, :valid_len].cpu().numpy().astype(np.int16)
            np.save(str(npy_path), codes_np)
            prepared.append({**row, "codes_path": str(npy_path), "frames": valid_len})
            if idx % 25 == 0 or idx == len(to_encode):
                print(f"   Encoded {idx}/{len(to_encode)} clips ({time.time() - t0:.1f}s)", flush=True)

    model.__dict__["_arktts_codec"] = None
    return prepared


def build_training_sequence(
    processor,
    config,
    text: str,
    target_codes: np.ndarray,
    ref_codes: np.ndarray | None = None,
    ref_text: str | None = None,
    max_seq_len: int = 720,
) -> dict[str, torch.Tensor] | None:
    norm_target = normalize_ckb(text)
    has_ref = ref_codes is not None and ref_text is not None
    prefix_ids, suffix_ids = processor._prompt_segments(
        norm_target,
        normalize_ckb(ref_text) if has_ref else None,
        has_reference=has_ref,
    )

    num_cb = int(config.num_codebooks)        # 10
    sem_begin = int(config.semantic_begin_id)  # 151678
    eos_id = int(config.eos_token_id)          # 151645

    if has_ref:
        ref_t = torch.from_numpy(ref_codes.astype(np.int64))
        ref_len = ref_t.shape[1]
        ref_sem = ref_t[0] + sem_begin
        prompt_sem = torch.cat((prefix_ids, ref_sem, suffix_ids))
        prompt_cols = torch.zeros((num_cb + 1, prompt_sem.numel()), dtype=torch.long)
        prompt_cols[0] = prompt_sem
        prompt_cols[1:, prefix_ids.numel() : prefix_ids.numel() + ref_len] = ref_t
    else:
        prompt_sem = torch.cat((prefix_ids, suffix_ids))
        prompt_cols = torch.zeros((num_cb + 1, prompt_sem.numel()), dtype=torch.long)
        prompt_cols[0] = prompt_sem

    tgt_t = torch.from_numpy(target_codes.astype(np.int64))  # [10, F]
    n_frames = tgt_t.shape[1]

    speech_cols = torch.zeros((num_cb + 1, n_frames), dtype=torch.long)
    speech_cols[0] = tgt_t[0] + sem_begin
    speech_cols[1:] = tgt_t

    eos_col = torch.zeros((num_cb + 1, 1), dtype=torch.long)
    eos_col[0, 0] = eos_id

    full_input = torch.cat((prompt_cols, speech_cols, eos_col), dim=1)  # [11, L]
    if full_input.shape[1] > max_seq_len:
        return None

    return {
        "input_ids": full_input,
        "prompt_len": int(prompt_cols.shape[1]),
        "speech_frames": int(n_frames),
    }


def compute_dualar_loss(
    model,
    example: dict[str, torch.Tensor],
    device: torch.device,
    sem_head: nn.Parameter,
    freeze_slow_layers: int = 12,
    fast_frames_subsample: int = 192,
) -> tuple[torch.Tensor, float, float]:
    input_ids = example["input_ids"].unsqueeze(0).to(device)  # [1, 11, L]
    prompt_len = example["prompt_len"]
    n_frames = example["speech_frames"]
    total_len = input_ids.shape[2]

    attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)
    position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
    rope = model.freqs_cis[position_ids]
    mask = model._causal_mask(
        attention_mask,
        torch.arange(total_len, device=device),
        total_len,
    )

    with torch.no_grad():
        hidden = model._embed(input_ids)
        for layer in model.layers[:freeze_slow_layers]:
            hidden = layer(hidden, rope, mask)

    hidden = hidden.detach().requires_grad_(True)
    for layer in model.layers[freeze_slow_layers:]:
        hidden = layer(hidden, rope, mask)

    # 1. Slow AR semantic + EOS loss over the 4,097 valid speech tokens (with trainable sem_head)
    speech_hidden = hidden[0, prompt_len - 1 : prompt_len + n_frames, :]       # [F + 1, dim]
    norm_speech = model.norm(speech_hidden)                                   # [F + 1, dim]
    slow_pred_logits = F.linear(norm_speech, sem_head)                        # [F + 1, 4097]

    raw_targets = input_ids[0, 0, prompt_len : prompt_len + n_frames + 1]     # [F + 1]
    slow_targets = torch.where(
        raw_targets == model.config.eos_token_id,
        0,
        raw_targets - model.config.semantic_begin_id + 1,
    )
    slow_loss = F.cross_entropy(slow_pred_logits, slow_targets, label_smoothing=0.015)

    # 2. Fast AR 10-codebook acoustic loss (predicts codebooks 1..9 from slow_hidden + codebooks 0..8)
    slow_h_for_frames = norm_speech[:-1] if model.config.norm_fastlayer_input else speech_hidden[:-1]
    target_codebooks = input_ids[0, 1:, prompt_len : prompt_len + n_frames].transpose(0, 1)  # [F, 10]

    if n_frames > fast_frames_subsample:
        idx = torch.randperm(n_frames, device=device)[:fast_frames_subsample]
        slow_h_for_frames = slow_h_for_frames[idx]
        target_codebooks = target_codebooks[idx]

    f_batch = slow_h_for_frames.shape[0]
    num_cb = int(model.config.num_codebooks)  # 10

    h0 = model.fast_project_in(slow_h_for_frames).unsqueeze(1)             # [f_batch, 1, fast_dim]
    c_emb = model.fast_embeddings(target_codebooks[:, :-1])                # [f_batch, 9, fast_dim]
    fast_seq = torch.cat((h0, c_emb), dim=1)                               # [f_batch, 10, fast_dim]

    fast_pos = torch.arange(num_cb, device=device, dtype=torch.long)
    fast_rope = model.fast_freqs_cis[fast_pos]                             # [10, head_dim//2, 2]
    fast_key_mask = torch.ones((f_batch, num_cb), dtype=torch.bool, device=device)
    fast_causal_mask = model._causal_mask(fast_key_mask, fast_pos, num_cb)

    fh = fast_seq
    for flayer in model.fast_layers:
        fh = flayer(fh, fast_rope, fast_causal_mask)
    fast_logits = model.fast_output(model.fast_norm(fh))                   # [f_batch, 10, 4096]

    cb_pred_logits = fast_logits[:, 1:, :].reshape(-1, model.config.codebook_size)
    cb_targets = target_codebooks[:, 1:].reshape(-1)
    fast_loss = F.cross_entropy(cb_pred_logits, cb_targets, label_smoothing=0.01)

    # Weight Fast AR acoustic fidelity 1.25x for sharper vocal timbre & high-frequency clarity
    total_loss = slow_loss + 1.25 * fast_loss
    return total_loss, float(slow_loss.item()), float(fast_loss.item())


def clear_kv_caches(model) -> None:
    for layer in list(model.layers) + list(model.fast_layers):
        attn = getattr(layer, "attention", None)
        if attn is not None:
            attn.kv_cache = None


def generate_sample(model, processor, text: str, out_wav: Path, ref_codes: np.ndarray | None = None, ref_text: str | None = None):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    clear_kv_caches(model)
    device = next(model.parameters()).device

    try:
        norm_text = normalize_ckb(text)
        kwargs = {"text": [norm_text], "return_tensors": "pt"}
        if ref_codes is not None and ref_text is not None:
            kwargs["reference_codes"] = [ref_codes]
            kwargs["reference_text"] = [normalize_ckb(ref_text)]
        inputs = processor(**kwargs)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        gen = torch.Generator(device=device).manual_seed(1234)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=320,
                temperature=0.58,
                top_p=0.86,
                top_k=32,
                do_sample=True,
                return_dict_in_generate=True,
                generator=gen,
            )
            codes_cpu = out.codes.cpu()
            model.load_codec(device=torch.device("cpu"), dtype=torch.float32)
            wavs, lengths = model.decode_audio(codes_cpu)
            wav_np = wavs[0, : int(lengths[0])].float().cpu().numpy()
            sf.write(str(out_wav), wav_np, int(model.config.codec_sample_rate))
            model.__dict__["_arktts_codec"] = None
            print(f"   🔊 Saved sample ({len(wav_np)/44100:.2f}s) -> {out_wav}", flush=True)
    except Exception as exc:
        print(f"   ⚠️ Sample generation warning: {exc}", flush=True)
    finally:
        clear_kv_caches(model)
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Audio8-TTS-Kurdish-Sorani v2 (High-Purity 44.1kHz)")
    parser.add_argument("--epochs", type=int, default=6, help="Number of fine-tuning epochs (default: 6)")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate (default: 2.5e-5)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=720, help="Max sequence length in tokens/frames")
    args = parser.parse_args()

    if not META_JSONL.exists():
        from build_tts_dataset import build_dataset
        build_dataset()

    meta_rows = [json.loads(line) for line in META_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(meta_rows)} high-purity 44.1kHz Kurdish Tafsir dataset clips.", flush=True)

    device = get_device()
    print(f"Using Apple Silicon device: {device}", flush=True)

    processor = AutoProcessor.from_pretrained(AUDIO8_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(AUDIO8_MODEL_ID, trust_remote_code=True, dtype=torch.float32)

    # Step 1: Encode all 44.1kHz dataset clips into RVQ codec `.npy` files
    prepared_rows = encode_dataset_codes(model, meta_rows)

    # Select crystal-clear 5.5s-8.0s anchor reference clip from Surah 1
    ref_candidates = [r for r in prepared_rows if 5.0 <= r["duration"] <= 8.5]
    anchor_ref = next((r for r in ref_candidates if r["id"].startswith("s001_a001")), ref_candidates[0])
    anchor_ref_codes = np.load(anchor_ref["codes_path"])
    anchor_ref_text = anchor_ref["text"]
    print(f"Anchor reference clip: {anchor_ref['id']} ({anchor_ref['duration']}s) -> {anchor_ref_text[:60]}...", flush=True)

    # Step 2: Build training sequences (50% reference-conditioned, 50% unconditional default voice)
    rng = random.Random(42)
    examples = []
    skipped = 0
    for r in prepared_rows:
        codes = np.load(r["codes_path"])
        use_ref = (r["id"] != anchor_ref["id"]) and (rng.random() < 0.50)
        ex = build_training_sequence(
            processor,
            model.config,
            r["text"],
            codes,
            ref_codes=anchor_ref_codes if use_ref else None,
            ref_text=anchor_ref_text if use_ref else None,
            max_seq_len=args.max_seq_len,
        )
        if ex is None and use_ref:
            ex = build_training_sequence(
                processor, model.config, r["text"], codes, max_seq_len=args.max_seq_len
            )
        if ex is not None:
            ex["id"] = r["id"]
            examples.append(ex)
        else:
            skipped += 1

    print(f"[Step 2/3] Built {len(examples)} training sequences (skipped {skipped} > {args.max_seq_len} tokens).", flush=True)

    # Freeze text embeddings + bottom 12 Slow AR layers; train top 12 Slow AR + sem_head + full Fast AR
    clear_kv_caches(model)
    freeze_slow_layers = 12
    model.embeddings.requires_grad_(False)
    model.codebook_embeddings.requires_grad_(False)
    for layer in model.layers[:freeze_slow_layers]:
        layer.requires_grad_(False)

    model.to(device)
    model.train()

    valid_ids = torch.cat(
        [
            torch.tensor([model.config.eos_token_id], device=device, dtype=torch.long),
            torch.arange(
                model.config.semantic_begin_id,
                model.config.semantic_end_id + 1,
                device=device,
                dtype=torch.long,
            ),
        ]
    )
    sem_head = nn.Parameter(model.embeddings.weight[valid_ids].detach().clone())

    trainable_params = [sem_head] + [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable parameters: {sum(p.numel() for p in trainable_params)/1e6:.1f}M "
        f"(top 12 Slow AR + 4097-token sem_head + full Fast AR)",
        flush=True,
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(examples) * args.epochs / args.grad_accum)
    warmup_steps = max(8, int(total_steps * 0.06))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.08, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    status_file = OUTPUT_MODEL_DIR / "training_status.json"

    test_sentence = "بەڕاستی ئێمە خێر و چاکەیەکی زۆر و بێ سنوورمان بە تۆ بەخشیوە ئەی پێغەمبەر."

    print(f"[Step 3/3] Starting v2 fine-tuning for {args.epochs} epochs ({total_steps} optimizer steps)...", flush=True)
    t_start = time.time()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(examples)
        epoch_slow_loss = 0.0
        epoch_fast_loss = 0.0

        for i, ex in enumerate(examples, 1):
            loss, sl, fl = compute_dualar_loss(
                model, ex, device, sem_head, freeze_slow_layers=freeze_slow_layers
            )
            (loss / args.grad_accum).backward()
            del loss
            epoch_slow_loss += sl
            epoch_fast_loss += fl

            if i % args.grad_accum == 0 or i == len(examples):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "mps":
                    torch.mps.empty_cache()
                global_step += 1

                if global_step % 10 == 0 or i == len(examples):
                    elapsed = time.time() - t_start
                    avg_s = epoch_slow_loss / i
                    avg_f = epoch_fast_loss / i
                    eta_sec = (elapsed / max(1, global_step)) * (total_steps - global_step)
                    msg = (
                        f"Epoch {epoch}/{args.epochs} | Clip {i}/{len(examples)} | "
                        f"Step {global_step}/{total_steps} | SlowLoss={avg_s:.3f} FastLoss={avg_f:.3f} | "
                        f"Elapsed={elapsed/60:.1f}m ETA={eta_sec/60:.1f}m"
                    )
                    print(msg, flush=True)
                    status_file.write_text(
                        json.dumps(
                            {
                                "state": "training",
                                "version": "v2_44k_high_purity",
                                "epoch": epoch,
                                "total_epochs": args.epochs,
                                "clip": i,
                                "total_clips": len(examples),
                                "step": global_step,
                                "total_steps": total_steps,
                                "slow_loss": round(avg_s, 4),
                                "fast_loss": round(avg_f, 4),
                                "elapsed_min": round(elapsed / 60, 2),
                                "eta_min": round(eta_sec / 60, 2),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

        # Write updated semantic head rows back into `model.embeddings.weight` before saving/sampling
        with torch.no_grad():
            model.embeddings.weight.data[valid_ids] = sem_head.data

        print(f"\n💾 Saving v2 checkpoint after Epoch {epoch} -> {OUTPUT_MODEL_DIR}", flush=True)
        clear_kv_caches(model)
        model.save_pretrained(str(OUTPUT_MODEL_DIR), safe_serialization=True)
        processor.save_pretrained(str(OUTPUT_MODEL_DIR))
        for py_file in (
            "configuration_arktts.py",
            "modeling_arktts.py",
            "modeling_arktts_codec.py",
            "processing_arktts.py",
        ):
            src_py = hf_hub_download(repo_id=AUDIO8_MODEL_ID, filename=py_file)
            shutil.copy2(src_py, OUTPUT_MODEL_DIR / py_file)
        np.save(str(OUTPUT_MODEL_DIR / "anchor_ref_codes.npy"), anchor_ref_codes)
        (OUTPUT_MODEL_DIR / "anchor_ref_text.txt").write_text(anchor_ref_text, encoding="utf-8")

        sample_wav = SAMPLES_DIR / f"v2_epoch_{epoch}_sample.wav"
        generate_sample(model, processor, test_sentence, sample_wav, anchor_ref_codes, anchor_ref_text)

    total_min = round((time.time() - t_start) / 60, 2)
    status_file.write_text(
        json.dumps(
            {
                "state": "completed",
                "version": "v2_44k_high_purity",
                "epochs": args.epochs,
                "total_steps": global_step,
                "elapsed_min": total_min,
                "model_dir": str(OUTPUT_MODEL_DIR),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n🎉 v2 Fine-tuning complete in {total_min} minutes! Model saved to: {OUTPUT_MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
