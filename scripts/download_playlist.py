#!/usr/bin/env python3
"""
YouTube Playlist to MP4 Downloader for Quran Kurdish Translation.
Playlist: https://youtube.com/playlist?list=PLyFKeTlAZeZvYxj8Bu_NMeGS1h5bQaYCc
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PLAYLIST_URL = "https://youtube.com/playlist?list=PLyFKeTlAZeZvYxj8Bu_NMeGS1h5bQaYCc"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
PLAYLIST_META_PATH = DATA_DIR / "playlist_index.json"
YTDLP_BIN = str(BASE_DIR / ".venv" / "bin" / "yt-dlp") if (BASE_DIR / ".venv" / "bin" / "yt-dlp").exists() else "yt-dlp"


def parse_surah_range(spec: str) -> list[int]:
    """Parse range spec like '1,78-114' into sorted list of surah numbers."""
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


def fetch_playlist_index(force: bool = False) -> dict[str, dict]:
    """Fetch playlist metadata and map surah_number -> video info."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PLAYLIST_META_PATH.exists() and not force:
        with open(PLAYLIST_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Fetching playlist metadata from {PLAYLIST_URL} ...")
    cmd = [
        YTDLP_BIN,
        "--no-update",
        "--flat-playlist",
        "--dump-single-json",
        PLAYLIST_URL,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = json.loads(proc.stdout)
    entries = raw.get("entries", [])

    index = {}
    for idx, entry in enumerate(entries, start=1):
        title = entry.get("title", "")
        # Try extracting [N] from title, fallback to 1-based playlist index
        m = re.search(r"\[(\d+)\]", title)
        surah_num = int(m.group(1)) if m else idx
        video_id = entry.get("id")
        index[str(surah_num)] = {
            "surah_number": surah_num,
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": title,
            "duration": entry.get("duration"),
        }

    with open(PLAYLIST_META_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"Saved playlist index with {len(index)} Surahs to {PLAYLIST_META_PATH}")
    return index


def download_surah_mp4(surah_num: int, video_info: dict, force: bool = False) -> Path:
    """Download a single Surah video as MP4."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEOS_DIR / f"surah_{surah_num:03d}.mp4"
    if out_path.exists() and out_path.stat().st_size > 10_000 and not force:
        print(f"[Surah {surah_num:03d}] Already downloaded: {out_path.name}")
        return out_path

    url = video_info["url"]
    print(f"[Surah {surah_num:03d}] Downloading {video_info['title']} ({url}) -> {out_path.name} ...")
    temp_template = str(VIDEOS_DIR / f"surah_{surah_num:03d}.%(ext)s")
    cmd = [
        YTDLP_BIN,
        "--no-update",
        "--no-playlist",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "-f",
        "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        temp_template,
        url,
    ]
    subprocess.run(cmd, check=True)
    if not out_path.exists():
        raise RuntimeError(f"Expected output {out_path} not found after yt-dlp")
    print(f"[Surah {surah_num:03d}] Completed: {out_path.name} ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Quran Kurdish Tafsir YouTube videos as MP4.")
    parser.add_argument(
        "--surahs",
        type=str,
        default="1,78-114",
        help="Comma-separated surah numbers or ranges (default: '1,78-114' for Al-Fatihah + Juz Amma)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--force", action="store_true", help="Re-download existing MP4 files")
    args = parser.parse_args()

    index = fetch_playlist_index()
    target_surahs = parse_surah_range(args.surahs)
    print(f"Target Surahs ({len(target_surahs)}): {target_surahs}")

    tasks = []
    for s in target_surahs:
        info = index.get(str(s))
        if not info:
            print(f"Warning: Surah {s} not found in playlist index", file=sys.stderr)
            continue
        tasks.append((s, info))

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(download_surah_mp4, s, info, args.force): s for s, info in tasks
        }
        for fut in as_completed(future_map):
            s = future_map[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"[Surah {s:03d}] FAILED: {exc}", file=sys.stderr)
                failed.append(s)

    if failed:
        print(f"Failed Surahs: {failed}", file=sys.stderr)
        sys.exit(1)
    print("All requested Surah MP4s downloaded successfully!")


if __name__ == "__main__":
    main()
