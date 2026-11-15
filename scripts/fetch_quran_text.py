#!/usr/bin/env python3
"""
Fetches Quran Arabic (quran-uthmani) and Kurdish translation (ku.asan) text
from AlQuran Cloud API and caches each Surah locally.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
QURAN_DIR = BASE_DIR / "data" / "quran"
SURAH_LIST_PATH = BASE_DIR / "data" / "surahs_meta.json"


def parse_surah_range(spec: str) -> list[int]:
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            for s in range(int(start), int(end) + 1):
                if 1 <= s <= 114:
                    result.add(s)
        else:
            s = int(part)
            if 1 <= s <= 114:
                result.add(s)
    return sorted(result)


def fetch_json(url: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuranStudySPA/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise


def fetch_all_surahs_meta() -> list[dict]:
    SURAH_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SURAH_LIST_PATH.exists():
        with open(SURAH_LIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    data = fetch_json("https://api.alquran.cloud/v1/surah")
    surahs = data.get("data", [])
    with open(SURAH_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(surahs, f, ensure_ascii=False, indent=2)
    return surahs


def fetch_surah_text(surah_num: int, force: bool = False) -> Path:
    QURAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QURAN_DIR / f"surah_{surah_num:03d}.json"
    if out_path.exists() and not force:
        return out_path

    url = f"https://api.alquran.cloud/v1/surah/{surah_num}/editions/quran-uthmani,ku.asan,en.sahih"
    payload = fetch_json(url)
    editions = payload.get("data", [])
    if len(editions) < 2:
        raise RuntimeError(f"Unexpected response for Surah {surah_num}")

    ar_ed = editions[0]
    ku_ed = editions[1]
    en_ed = editions[2] if len(editions) > 2 else {"ayahs": []}

    ar_ayahs = ar_ed.get("ayahs", [])
    ku_ayahs = ku_ed.get("ayahs", [])
    en_ayahs = en_ed.get("ayahs", [])

    verses = []
    for i, ar in enumerate(ar_ayahs):
        ku = ku_ayahs[i] if i < len(ku_ayahs) else {}
        en = en_ayahs[i] if i < len(en_ayahs) else {}
        ar_text = ar.get("text", "")
        # Strip leading Bismillah from verse 1 of Surahs other than 1 and 9 if attached
        verses.append({
            "number_in_surah": ar.get("numberInSurah", i + 1),
            "number_global": ar.get("number"),
            "juz": ar.get("juz"),
            "page": ar.get("page"),
            "arabic_text": ar_text,
            "kurdish_text": ku.get("text", ""),
            "english_text": en.get("text", ""),
        })

    result = {
        "surah_number": ar_ed.get("number", surah_num),
        "name_arabic": ar_ed.get("name", ""),
        "name_english": ar_ed.get("englishName", ""),
        "name_translation": ar_ed.get("englishNameTranslation", ""),
        "revelation_type": ar_ed.get("revelationType", ""),
        "number_of_ayahs": ar_ed.get("numberOfAyahs", len(verses)),
        "verses": verses,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Quran Text] Saved Surah {surah_num:03d} ({result['name_english']}, {len(verses)} verses)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Fetch Quran Arabic & Kurdish text.")
    parser.add_argument("--surahs", type=str, default="1,78-114")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fetch_all_surahs_meta()
    targets = parse_surah_range(args.surahs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(fetch_surah_text, s, args.force): s for s in targets}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"Error fetching Surah {s}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
