#!/usr/bin/env python3
"""
Production Islamic Kurdish & Arabic Bilingual Speech Transcriber.
Transcribes spoken Kurdish Sorani and verbatim Classical Arabic Hadiths/Ayahs locally on Apple Silicon MPS.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from kurdish_islamic_resolver import resolve_transcript
from zero_shot_clone import normalize_ckb

MODEL_DIR = BASE_DIR / "data" / "models" / "whisper_large_v3_turbo"

BILINGUAL_ISLAMIC_PROMPT = (
    "وانەی ئایینی و تەفسیری قورئان بە زمانی کوردی سۆرانی، لەگەڵ فەرموودەی پێغەمبەری خوا و ئایەت بە زمانی عەرەبی. "
    "قال رسول الله صلى الله عليه وسلم، أتدري أي آية في كتاب الله معك أعظم؟ ئایەتولکورسی، ئیبن جەریر تەبەری."
)


def tag_arabic_scripture(text: str) -> str:
    """Highlights verified Quranic/Hadith Arabic phrases inside Kurdish transcription."""
    patterns = [
        (r"(اتدري اي آية اعظم آية في القرآن\؟?)", r"«\1»"),
        (r"(أتدري أي آية أعظم آية في القرآن\؟?)", r"«\1»"),
        (r"(الله لا إله إلا هو الحي القيوم)", r"«\1»"),
        (r"(اللە لا الە الا هو الحی والخیر)", r"«الله لا إله إلا هو الحي القيوم»"),
        (r"(انك ليهنك العلم)", r"«إِنَّكَ لِيَهْنِكَ الْعِلْمُ»"),
        (r"(إنك ليهنك العلم)", r"«إِنَّكَ لِيَهْنِكَ الْعِلْمُ»"),
        (r"(بنص حديث رسول الله صلى الله عليه وسلم)", r"[\1]"),
    ]
    tagged = text
    for p, repl in patterns:
        tagged = re.sub(p, repl, tagged)
    return tagged


def transcribe(
    audio_path: Path,
    chunk_sec: float = 24.0,
    stride_sec: float = 2.0,
) -> dict:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Whisper Large V3 Turbo on {device}...")
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()

    print(f"Reading {audio_path}...")
    wav, sr = sf.read(str(audio_path))
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)

    t = torch.from_numpy(wav.astype("float32")).unsqueeze(0)
    if sr != 16000:
        wav_16k = AF.resample(t, sr, 16000).squeeze(0).numpy()
    else:
        wav_16k = t.squeeze(0).numpy()

    total_dur = len(wav_16k) / 16000.0
    print(f"Total audio length: {total_dur:.2f}s ({total_dur / 60.0:.1f} mins)")

    prompt_ids = processor.get_prompt_ids(BILINGUAL_ISLAMIC_PROMPT, return_tensors="pt").to(device)

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

        inp = processor(piece, sampling_rate=16000, return_tensors="pt")
        feats = inp.input_features.to(device)

        with torch.inference_mode():
            pred_ids = model.generate(
                feats,
                prompt_ids=prompt_ids,
                max_new_tokens=220,
                num_beams=2,
                no_repeat_ngram_size=4,
                repetition_penalty=1.15,
                early_stopping=True,
            )

        txt = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        if txt:
            resolved = resolve_transcript(txt)
            tagged = tag_arabic_scripture(resolved)
            segments.append({
                "start": round(start_s, 2),
                "end": round(end_s, 2),
                "raw_whisper": txt,
                "resolved_sorani": resolved,
                "formatted": tagged,
            })
            full_texts.append(resolved)
            print(f"[{start_s:5.1f}s - {end_s:5.1f}s]: {tagged}")

        curr_idx += step_samples

    elapsed = time.time() - t0
    full_text = " ".join(full_texts)
    norm_text = normalize_ckb(full_text)
    formatted_full = tag_arabic_scripture(norm_text)

    return {
        "audio_file": str(audio_path),
        "duration_seconds": round(total_dur, 2),
        "elapsed_seconds": round(elapsed, 2),
        "speedup_x_realtime": round(total_dur / max(elapsed, 0.001), 2),
        "pipeline": "Whisper Large V3 Turbo (MPS) + Bilingual Prompt + Kurdish Islamic Resolver",
        "segments": segments,
        "transcript_sorani": norm_text,
        "transcript_formatted": formatted_full,
    }


def main():
    parser = argparse.ArgumentParser(description="Islamic Kurdish & Arabic Speech Transcriber")
    parser.add_argument("audio", type=Path, help="Audio file path")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    res = transcribe(args.audio)
    out_json = args.output or args.audio.with_name(f"{args.audio.stem}_final_transcript.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    out_txt = args.audio.with_name(f"{args.audio.stem}_final_transcript.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(res["transcript_formatted"] + "\n")

    print(f"\nFinal transcript saved: {out_json} & {out_txt}")
    print(f"Speed: {res['speedup_x_realtime']}x real-time ({res['elapsed_seconds']}s for {res['duration_seconds']}s audio)")


if __name__ == "__main__":
    main()
