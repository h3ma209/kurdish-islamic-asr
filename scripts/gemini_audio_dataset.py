#!/usr/bin/env python3
"""
Production Gemini Audio Transcriber & ASR Training Dataset Generator.
Transcribes audio with exact timestamps, preserves full Arabic Harakat (Tashkeel)
on Quranic Ayahs, Hadiths, and Islamic formulas, extracts acoustic features,
and writes standard Hugging Face AudioFolder format (metadata.csv + dataset.jsonl).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import torchaudio.transforms as T
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "dataset"
WAVS_DIR = DATASET_DIR / "wavs"
FEATURES_DIR = DATASET_DIR / "features"


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"\'')
                    break

    if not api_key:
        print("\n[ERROR] GEMINI_API_KEY is not set.")
        sys.exit(1)

    return genai.Client(api_key=api_key)


PROMPT = """
You are an expert transcriber of Islamic sermons in Central Kurdish (Sorani) and Classical Arabic.
The speaker speaks Sorani Kurdish and frequently quotes Quranic Ayahs and Hadiths in Classical Arabic.

CRITICAL REQUIREMENTS:
1. COMPLETE HARAKAT / TASHKEEL FOR ARABIC (WITH BRACKETS):
   Every single Arabic word, Quranic verse, Hadith quotation, Islamic formula, and book title MUST be written in authentic Classical Arabic with FULL HARAKAT / TASHKEEL (diacritics: فَتْحَة، ضَمَّة، كَسْرَة، سُكُون، شَدَّة، تَنْوِين).
   - Enclose Quranic Ayahs in ﴿...﴾ brackets (e.g. ﴿وَلَمَّا وَرَدَ مَاءَ مَدْيَنَ وَجَدَ عَلَيْهِ أُمَّةً مِّنَ النَّاسِ يَسْقُونَ﴾).
   - Enclose Hadith quotations in «...» brackets (e.g. «بُنِيَ الْإِسْلَامُ عَلَى خَمْسٍ»).
   - NEVER transliterate Arabic scriptures into Kurdish spelling (e.g. NEVER write 'ولما ورد ماء مدینە' or 'علیها'). Always keep authentic Classical Arabic spelling with all Harakat.
   - Examples:
     - ﴿أَتَدْرِي أَيُّ آيَةٍ مِنْ كِتَابِ اللَّهِ مَعَكَ أَعْظَمُ؟﴾
     - ﴿اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ﴾
     - «وَاللَّهِ لِيَهْنِكَ الْعِلْمُ أَبَا الْمُنْذِرِ»
     - «بِنَصِّ حَدِيثِ رَسُولِ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ»
     - «جَلَّ جَلَالُهُ»
     - «شَيْخُ الْمُفَسِّرِينَ»
   DO NOT omit harakat on any Arabic scripture or quotation.

2. STRICT CENTRAL KURDISH (SORANI / CKB) ONLY:
   The Kurdish speech is 100% Central Kurdish (Sorani / سۆرانی).
   You MUST write exclusively in standard Central Kurdish (Sorani) orthography and vocabulary:
   - Use standard Sorani words: (بەڵام، کاتێک، ڕوونکردنەوە، بۆچی، دەکات، ئەوەیە، دەڵێت، پێشوو، سفر، لێرە، لەسەر).
   - STRICTLY FORBIDDEN: NEVER use Northern Kurdish (Kurmanji / Badini) words, characters, or grammar!
     FORBIDDEN KURMANJI: (ڕۆنکرنا، پېشوا، سڤر، ل سەر، هندە، هاتیە، دێ، بێژم، دکەت، چنکو).
   Every single Kurdish word must be authentic Central Kurdish (Sorani).
   DO NOT transliterate Kurdish speech into pseudo-Arabic. Transcribe spoken Kurdish verbatim in Sorani Kurdish.

3. NUMBERS AS WORDS:
   Spell out all numbers as spoken words in Kurdish (e.g. "سێزدە", "چل", "دوو هەزار و حەوت"). Never output digits (0-9 or ٠-٩).

4. SEGMENTATION:
   Segment speech into natural timestamped phrases (3 to 15 seconds per segment).

5. Return a strict JSON array of objects:
   - "start_s": float
   - "end_s": float
   - "language": string ("ckb" for pure Kurdish, "ara" for pure Arabic quote, "mixed" for code-switching)
   - "text": string (verbatim transcription with full Harakat on all Arabic)
   - "arabic_scripture": list of strings (any exact Ayah or Hadith cited in this segment with full Harakat, else [])

Output ONLY the raw JSON array.
"""


def normalize_preserving_harakat(text: str) -> str:
    """Normalize whitespace and Kurdish characters while strictly preserving all Arabic diacritics."""
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def transcribe_audio_with_gemini(audio_path: Path) -> list[dict]:
    client = get_gemini_client()
    print(f"Uploading {audio_path.name} ({audio_path.stat().st_size / 1024 / 1024:.2f} MB) via Files API...")
    file_ref = client.files.upload(file=str(audio_path))

    model_name = "models/gemini-3.5-flash-lite"
    response = None
    for attempt in range(1, 6):
        print(f"Requesting full Harakat transcription from {model_name} (Attempt {attempt}/5)...")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[file_ref, PROMPT],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                },
            )
            print(f"Successfully received response from {model_name} on attempt {attempt}!")
            break
        except Exception as e:
            print(f"Attempt {attempt} failed ({type(e).__name__}): {e}. Waiting {attempt * 4}s...")
            time.sleep(attempt * 4)

    if not response:
        print("All candidate models failed.")
        sys.exit(1)

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```[a-zA-Z]*\n", "", raw_text)
        raw_text = re.sub(r"\n```$", "", raw_text)

    try:
        segments = json.loads(raw_text)
    except Exception as e:
        print(f"Failed to parse JSON response: {e}")
        print("Raw response:\n", raw_text[:500])
        sys.exit(1)

    return segments


def featurize_and_slice(audio_path: Path, segments: list[dict], video_id: str) -> list[dict]:
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    wav, sr = sf.read(str(audio_path))
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)

    t = torch.from_numpy(wav.astype("float32")).unsqueeze(0)
    if sr != 16000:
        wav_16k = AF.resample(t, sr, 16000).squeeze(0)
    else:
        wav_16k = t.squeeze(0)

    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
    )

    dataset_records = []
    total_audio_s = len(wav_16k) / 16000.0

    print(f"\nProcessing {len(segments)} segments for ASR training dataset (with Harakat)...")
    for idx, seg in enumerate(segments):
        start_s = float(seg["start_s"])
        end_s = float(seg["end_s"])
        if end_s > total_audio_s:
            end_s = total_audio_s
        if end_s <= start_s or (end_s - start_s) < 0.8:
            continue

        dur_est = end_s - start_s
        # Discard clips where audio duration is physically too short for the text length (bad alignment)
        if int(dur_est * 50) < len(seg.get("text", "")):
            continue

        start_samp = int(start_s * 16000)
        end_samp = min(len(wav_16k), int(end_s * 16000))
        clip = wav_16k[start_samp:end_samp]
        if len(clip) < 1600:
            continue

        dur = len(clip) / 16000.0
        clip_name = f"{video_id}_{idx:04d}.wav"
        clip_path = WAVS_DIR / clip_name
        sf.write(str(clip_path), clip.numpy(), 16000)

        # Log-mel Spectrogram
        with torch.no_grad():
            mel = mel_transform(clip.unsqueeze(0)).squeeze(0)
            log_mel = torch.log(torch.clamp(mel, min=1e-5))

        feature_name = f"{video_id}_{idx:04d}_mel.npy"
        feature_path = FEATURES_DIR / feature_name
        np.save(str(feature_path), log_mel.numpy())

        clean_text = normalize_preserving_harakat(seg["text"])
        rms = float(torch.sqrt(torch.mean(clip**2)))

        record = {
            "id": f"{video_id}_{idx:04d}",
            "audio_file": f"wavs/{clip_name}",
            "feature_file": f"features/{feature_name}",
            "start_time": round(start_s, 2),
            "end_time": round(end_s, 2),
            "duration": round(dur, 2),
            "num_samples": len(clip),
            "rms_energy": round(rms, 4),
            "language": seg.get("language", "mixed"),
            "text": clean_text,
            "arabic_scripture": seg.get("arabic_scripture", []),
            "speaker": "مەلا هەڵۆ",
        }
        dataset_records.append(record)
        print(f"[{start_s:5.1f}s - {end_s:5.1f}s] ({seg.get('language')}): {clean_text}")

    return dataset_records


def save_dataset_manifests(records: list[dict], video_id: str):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATASET_DIR / "dataset.jsonl"
    csv_path = DATASET_DIR / "metadata.csv"

    vid_transcript_path = BASE_DIR / "data" / "mela_halo" / f"{video_id}_gemini_timeline.json"
    with open(vid_transcript_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Clean overwrite / rewrite combined dataset with full Harakat
    all_records = []
    # Read existing if different video
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    obj = json.loads(line)
                    if not obj.get("id", "").startswith(video_id):
                        all_records.append(obj)
                except Exception:
                    pass

    all_records.extend(records)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("file_name,sentence,language,duration\n")
        for r in all_records:
            clean_sentence = r["text"].replace('"', '""')
            f.write(f'"{r["audio_file"]}","{clean_sentence}","{r["language"]}",{r["duration"]}\n')

    print(f"\n[OK] Video saved with full Harakat to {vid_transcript_path}")
    print(f"[OK] Training dataset updated: {jsonl_path} ({len(all_records)} total segments)")


def main():
    parser = argparse.ArgumentParser(description="Gemini Audio Dataset Pipeline with Harakat")
    parser.add_argument("audio", type=Path, help="Audio file path (.wav or .mp3)")
    args = parser.parse_args()

    video_id = args.audio.stem
    segments = transcribe_audio_with_gemini(args.audio)
    records = featurize_and_slice(args.audio, segments, video_id)
    save_dataset_manifests(records, video_id)


if __name__ == "__main__":
    main()
