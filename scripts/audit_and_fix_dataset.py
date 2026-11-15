#!/usr/bin/env python3
"""
Comprehensive Kurdish Sorani Grammar & Orthography Auditor using Gemini 3.5 Flash Lite.
Corrects phonetic ASR slips, enforces authentic Divine names & honorifics,
audits 100% pure Central Kurdish (Sorani), and preserves all Arabic Harakat in ﴿...﴾ and «...».
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "dataset" / "dataset.jsonl"
METADATA_PATH = BASE_DIR / "data" / "dataset" / "metadata.csv"


def get_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        env_f = BASE_DIR / ".env"
        if env_f.exists():
            for line in env_f.read_text(encoding="utf-8").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key:
        print("[ERROR] GEMINI_API_KEY not found in environment or .env")
        sys.exit(1)
    return genai.Client(api_key=key)


def apply_deterministic_fixes(text: str) -> str:
    """Applies known acoustic & orthographic corrections directly."""
    rules = [
        # User specified fixes:
        (r"\bئەبەیا\b", "ئەبوایە"),
        (r"\bکامرو\b", "کە ئەمڕۆ"),
        (r"\bتلارین\b", "تەلاریان"),
        (r"\b(بەدریجەی مەجری|بەدریژای مەجری|بەدریجەی مەجرێ|بەدریژەی مەجری)\b", "بەدرێژایی مێژوو"),
        (r"\bزوور\b", "زۆر"),
        (r"\bزوورتر\b", "زۆرتر"),
        (r"\bزوورترین\b", "زۆرترین"),
        # Divine Name: Always authentic Classical Arabic with Tashkeel
        (r"\b(ئەڵڵا|ئەڵڵەه|ئەلڵا|ئەللا|ئەڵا|اللە)\b", "اللَّه"),
        # Honorifics & formulas
        (r"\b(سبحان اللە|سوبحان اللە|سبحان اللَّه|سوبحان اللَّه)\b", "سُبْحَانَ اللَّهِ"),
        (r"\b(سبحانە و تەعالا|سوبحانە و تەعالا|سوبحانو تەعلا|سبحانه و تعالی)\b", "سُبْحَانَهُ وَتَعَالَى"),
        (r"\b(صلی الله علیە وسلم|صلی اللە علیە وسلم|صلى الله عليه وسلم)\b", "صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ"),
        # Sacred shrines folklore terms
        (r"\b(دار و بەرد و پەڕی|دار و بەرد پەڕی|دار و بەردوو پەڕەوە|دار و بەرد و پەڕەوە|دار و بەرد و پەڕەی)\b", "دار و بەرد و پەڕۆ"),
        # Theological terms
        (r"\b(تەواوی دەکەن بکەن|تەواتیتەکان)\b", "تەواغیتەکان"),
        (r"\b(زۆر بەیت|زۆربەیت)\b", "زۆربەی"),
    ]
    for pat, repl in rules:
        text = re.sub(pat, repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


SYSTEM_PROMPT = """You are a senior linguist specializing in Central Kurdish (Sorani / سۆرانی) and Classical Arabic Islamic speech.
Your task is to audit and correct a list of transcribed spoken segments by Islamic preacher Mala Halo.

CRITICAL EDITING RULES:
1. GRAMMAR & ORTHOGRAPHY:
   - Correct phonetic/acoustic errors: (ئەبەیا -> ئەبوایە، کامرو -> کە ئەمڕۆ، تلارین -> تەلاریان، بەدریجەی مەجری -> بەدرێژایی مێژوو، زوور -> زۆر).
   - Ensure standard Central Kurdish (Sorani) spelling and syntax.
   - Keep spoken verbatim colloquialisms intact (e.g. ناکەن، دەڵێن، دەکات، بۆچی) - DO NOT translate Sorani into Kurmanji or literary Persian.
   - NEVER introduce Kurmanji markers (دێ چم، هاتیە، هندە، ڕۆنکرنا).

2. ARABIC SCRIPTURE & DIVINE NAMES:
   - Always write Allah as: اللَّه
   - Always keep full Harakat (Tashkeel) on Quranic verses inside ﴿...﴾ and Hadiths inside «...».
   - Honorifics: سُبْحَانَهُ وَتَعَالَى, صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ, رَضِيَ اللَّهُ عَنْهُ, عَزَّ وَجَلَّ.

Return ONLY a JSON array of objects: [{"id": "...", "text": "..."}]
"""


def batch_audit(client: genai.Client, records: list[dict], batch_size: int = 25) -> list[dict]:
    total = len(records)
    print(f"\nAuditing and correcting {total} segments in batches of {batch_size}...")

    corrected_records = []
    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        batch_input = [{"id": r["id"], "text": r["text"]} for r in batch]
        prompt = SYSTEM_PROMPT + "\n\nINPUT JSON:\n" + json.dumps(batch_input, ensure_ascii=False)

        success = False
        for attempt in range(1, 4):
            try:
                resp = client.models.generate_content(
                    model="models/gemini-3.5-flash-lite",
                    contents=prompt,
                    config={"response_mime_type": "application/json", "temperature": 0.0},
                )
                res_list = json.loads(resp.text.strip())
                lookup = {item["id"]: item.get("text", "") for item in res_list if "id" in item}

                for r in batch:
                    new_text = lookup.get(r["id"])
                    if new_text and len(new_text) > 2:
                        r["text"] = apply_deterministic_fixes(new_text)
                    else:
                        r["text"] = apply_deterministic_fixes(r["text"])
                    corrected_records.append(r)

                success = True
                print(f"  Batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({min(i+batch_size, total)}/{total}) complete.")
                break
            except Exception as e:
                print(f"  Batch {i//batch_size + 1} attempt {attempt} failed ({e}). Retrying...")
                time.sleep(attempt * 2)

        if not success:
            print(f"  [WARN] Batch {i//batch_size + 1} failed LLM audit; applying deterministic rules.")
            for r in batch:
                r["text"] = apply_deterministic_fixes(r["text"])
                corrected_records.append(r)

    return corrected_records


def main():
    client = get_client()

    print(f"Loading dataset from {DATASET_PATH}...")
    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Pre-clean with deterministic rules first
    for r in records:
        r["text"] = apply_deterministic_fixes(r["text"])

    # Batch audit with Gemini
    audited_records = batch_audit(client, records, batch_size=25)

    # Save to dataset.jsonl
    print(f"\nWriting audited records to {DATASET_PATH}...")
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for r in audited_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save to metadata.csv
    print(f"Writing updated metadata to {METADATA_PATH}...")
    with open(METADATA_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "transcription", "language", "speaker", "duration", "arabic_scripture"])
        for r in audited_records:
            scrip_str = " | ".join(r.get("arabic_scripture", [])) if r.get("arabic_scripture") else ""
            writer.writerow([r["audio_file"], r["text"], r["language"], r["speaker"], r["duration"], scrip_str])

    print("\n[OK] Dataset audit complete!")


if __name__ == "__main__":
    main()
