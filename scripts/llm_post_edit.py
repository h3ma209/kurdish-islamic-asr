#!/usr/bin/env python3
"""
LLM Post-Editor for Sorani Kurdish + Classical Arabic Islamic ASR transcripts.
Restores proper Kurdish orthography (ڕ/ڵ/ێ/ۆ/ە), Arabic letters (ص/ض/ط/ظ/ذ/ة/أ/إ/آ),
salawat (ﷺ), Quran/Hadith phrasing, and punctuation without inventing content.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _s = _line.strip()
        if _s and not _s.startswith("#") and "=" in _s:
            _k, _v = _s.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

SYSTEM_PROMPT = """You are an expert Sorani Kurdish and Classical Arabic Islamic transcript editor.
You receive raw ASR output from a Sorani Kurdish Islamic lecture containing Classical Arabic Quranic verses, Hadiths, and scholarly quotes.
The ASR model lacks Arabic-only letters (ص ض ط ظ ذ ة أ إ آ) and occasionally merges/drops short Kurdish vowels or phoneticizes Arabic phrases.

Your strict rules:
1. Fix phonetic Kurdish ASR errors using context (e.g., "ئەسلەکەی" -> "ئەسڵەکەی", "چڕوپرتڕ وێرتر" -> "چڕوپڕتر، وردتر", "لەوە ئەچێ" / "لەواچێ" keep speaker dialect, "فری" -> "فەرمووی", "تیایە" -> "تیایا").
2. Restore Classical Arabic quotes, Quranic ayahs, Hadiths, and Islamic honorifics accurately with proper Arabic letters (ص ض ط ظ ذ ة أ إ آ) and diacritics where appropriate (e.g., "سلسل" / "سلی الله وسلم" -> "ﷺ" or "صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ", "سبحان و تەعالا" -> "سُبْحَانَهُ وَتَعَالَى", "هزه الشیعە" -> "هذه الشيعة", "لیس لها سدق" -> "ليس لها صدور", "للتحدم" -> "لا تحزن").
3. Add clean punctuation (، ؟ .) at natural clause boundaries.
4. NEVER summarize, translate, skip sentences, or invent words not grounded in the ASR phonetics. Preserve the speaker's exact dialectal wording.
5. Return ONLY the corrected transcript text, with zero commentary or markdown."""


class KurdishArabicPostEditor:
    def __init__(self, model: str = "gemini-3.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment/.env")
        self.api_key = api_key
        self.model = model

    def edit(self, raw_asr: str) -> str:
        if not raw_asr.strip():
            return raw_asr
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": f"Correct this raw ASR transcript verbatim:\n\n{raw_asr}"}]}],
            "generationConfig": {
                "temperature": 0.1,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        for attempt in range(4):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                out = "".join(p.get("text", "") for p in parts).strip()
                return out if out else raw_asr
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return raw_asr
