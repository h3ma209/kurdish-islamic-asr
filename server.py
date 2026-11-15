#!/usr/bin/env python3
"""
FastAPI Backend Server for Kurdish Quran Tafsir Study SPA.
Serves:
- Static SPA UI (/ and /static/*)
- HTTP Range-enabled MP4 video & VTT/SRT subtitle streaming (/media/*)
- Surah list & verse-aligned Kurdish Tafsir API (/api/surahs, /api/surah/{id})
- On-demand download & Whisper transcription endpoint (/api/process/{id})
"""
import json
import mimetypes
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
QURAN_DIR = DATA_DIR / "quran"
SUBTITLES_DIR = DATA_DIR / "subtitles"
ALIGNED_DIR = DATA_DIR / "aligned"
TTS_CACHE_DIR = DATA_DIR / "tts_cache"
VOICE_REFS_DIR = DATA_DIR / "voice_refs"
STATIC_DIR = BASE_DIR / "static"
PLAYLIST_META_PATH = DATA_DIR / "playlist_index.json"
SURAH_LIST_PATH = DATA_DIR / "surahs_meta.json"
KURDISH_TAFSIRS_DB = QURAN_DIR / "kurdish_tafsirs.db"
CLASSICAL_TAFSIRS_DIR = QURAN_DIR / "classical_tafsirs"
CLASSICAL_TAFSIRS_DIR.mkdir(parents=True, exist_ok=True)

AVAILABLE_TAFSIRS = [
    # Kurdish Sorani Mainstream
    {"id": "asan", "name": "تەفسیری ئاسان", "author": "بورهان موحەممەد ئەمین", "category": "kurdish_sorani"},
    {"id": "raman", "name": "تەفسیری ڕامان", "author": "مامۆستا مەلا ئەحمەد کاکە مەحموود", "category": "kurdish_sorani"},
    {"id": "puxta", "name": "تەفسیری پوختە", "author": "مەلا عوسمان عەبدولعەزیز", "category": "kurdish_sorani"},
    {"id": "rebar", "name": "تەفسیری ڕێبەر", "author": "ڕێبەر (لەگەڵ فەرموودە و فەزڵەکان)", "category": "kurdish_sorani"},
    {"id": "hazhar", "name": "وەرگێڕانی هەژار", "author": "هەژار موکریانی (شاکاری ئەدەبی)", "category": "kurdish_sorani"},
    {"id": "tawhid", "name": "تەفسیری تەوحید", "author": "دەستەیەک لە زانایان", "category": "kurdish_sorani"},
    {"id": "roshn", "name": "تەفسیری ڕۆشن", "author": "تەفسیری ڕۆشن لە تەفسیری قورئان", "category": "kurdish_sorani"},

    # Classical Tafsir Works Translated into Kurdish
    {"id": "maisar", "name": "تەفسیری مویەسەر (کوردی)", "author": "التفسير الميسر — کۆمەڵگەی مەلیک فەهد", "category": "kurdish_translated"},
    {"id": "mokhtasar", "name": "تەفسیری موختەسەر (کوردی)", "author": "المختصر في التفسير — سەنتەری تەفسیر", "category": "kurdish_translated"},

    # Kurdish Kurmanji / Badini Dialect
    {"id": "sanahi", "name": "تەفسیری ساناهی (بادینی)", "author": "ب زارێ بهدینی / بادینی", "category": "kurdish_badini"},
    {"id": "zhin", "name": "تەفسیری ژین (بادینی)", "author": "ب زارێ بهدینی / بادینی", "category": "kurdish_badini"},
    {"id": "runahi", "name": "تەفسیری ڕووناهی (بادینی)", "author": "ب زارێ بهدینی / بادینی", "category": "kurdish_badini"},

    # Classical Arabic References
    {"id": "ibn_kathir", "name": "تفسير ابن كثير (ئیبن کەسیر)", "author": "الإمام الحافظ ابن كثير الدمشقي", "category": "classical_arabic"},
    {"id": "tabari", "name": "تفسير الطبري (تەبەری)", "author": "الإمام أبو جعفر الطبري", "category": "classical_arabic"},
    {"id": "qurtubi", "name": "تفسير القرطبي (قورتوبی)", "author": "الإمام أبو عبد الله القرطبي", "category": "classical_arabic"},
    {"id": "saadi", "name": "تفسير السعدي (سەعدی)", "author": "الشيخ عبد الرحمن السعدي", "category": "classical_arabic"},
]

CLASSICAL_CDN_SLUGS = {
    "ibn_kathir": "ar-tafsir-ibn-kathir",
    "tabari": "ar-tafsir-al-tabari",
    "qurtubi": "ar-tafseer-al-qurtubi",
    "saadi": "ar-tafsir-as-saadi",
}


def _get_classical_tafsir_map(edition_id: str, surah_num: int) -> dict:
    slug = CLASSICAL_CDN_SLUGS.get(edition_id)
    if not slug:
        return {}
    cache_file = CLASSICAL_TAFSIRS_DIR / f"{edition_id}_{surah_num}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    url = f"https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@main/tafsir/{slug}/{surah_num}.json"
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            m = {str(item["ayah"]): item["text"].strip() for item in data if "ayah" in item and "text" in item}
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False)
            return m
    except Exception as e:
        print(f"Classical tafsir fetch notice ({edition_id} surah {surah_num}): {e}")
        return {}


def _attach_kurdish_tafsirs(surah_num: int, data: dict) -> dict:
    data["available_tafsirs"] = AVAILABLE_TAFSIRS
    if not KURDISH_TAFSIRS_DB.exists():
        return data
    try:
        conn = sqlite3.connect(str(KURDISH_TAFSIRS_DB))
        cur = conn.cursor()
        cur.execute(
            """SELECT ayah, asan, raman, puxta, hazhar, rebar,
                      maisar, mokhtasar, tawhid, roshn, sanahi, zhin, runahi
               FROM ayah_text WHERE suraId = ? ORDER BY ayah""",
            (str(surah_num),),
        )
        rows = {
            r[0]: {
                "asan": (r[1] or "").strip(),
                "raman": (r[2] or "").strip(),
                "puxta": (r[3] or "").strip(),
                "hazhar": (r[4] or "").strip(),
                "rebar": (r[5] or "").strip(),
                "maisar": (r[6] or "").strip(),
                "mokhtasar": (r[7] or "").strip(),
                "tawhid": (r[8] or "").strip(),
                "roshn": (r[9] or "").strip(),
                "sanahi": (r[10] or "").strip(),
                "zhin": (r[11] or "").strip(),
                "runahi": (r[12] or "").strip(),
            }
            for r in cur.fetchall()
        }
        conn.close()

        for v in data.get("verses", []):
            ayah_num = v.get("number_in_surah")
            t_map = rows.get(ayah_num, {})
            if not t_map.get("asan") and v.get("kurdish_text"):
                t_map["asan"] = v.get("kurdish_text")
            v["kurdish_tafsirs"] = t_map
    except Exception as e:
        print(f"Warning: Failed to attach Kurdish tafsirs for Surah {surah_num}: {e}")
    return data

app = FastAPI(title="Quran Kurdish Tafsir Study SPA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track in-progress background download/transcription jobs
PROCESSING_STATUS: dict[int, str] = {}


def load_json_safe(path: Path) -> Optional[dict | list]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.get("/api/surahs")
def list_surahs():
    """Return all 114 Surahs with their download & Whisper alignment status."""
    meta_list = load_json_safe(SURAH_LIST_PATH) or []
    playlist_idx = load_json_safe(PLAYLIST_META_PATH) or {}

    surahs = []
    for item in meta_list:
        num = item.get("number")
        video_exists = (VIDEOS_DIR / f"surah_{num:03d}.mp4").exists()
        aligned_exists = (ALIGNED_DIR / f"surah_{num:03d}.json").exists()
        pl_entry = playlist_idx.get(str(num), {})

        surahs.append({
            "number": num,
            "name_arabic": item.get("name"),
            "name_english": item.get("englishName"),
            "name_translation": item.get("englishNameTranslation"),
            "number_of_ayahs": item.get("numberOfAyahs"),
            "revelation_type": item.get("revelationType"),
            "is_juz_amma": num == 1 or num >= 78,
            "video_downloaded": video_exists,
            "whisper_aligned": aligned_exists,
            "processing_state": PROCESSING_STATUS.get(num, "ready" if aligned_exists else ("downloaded" if video_exists else "idle")),
            "youtube_url": pl_entry.get("url", ""),
            "youtube_title": pl_entry.get("title", ""),
            "duration": pl_entry.get("duration"),
        })

    return {"surahs": surahs}


@app.get("/api/tafsirs")
def list_available_tafsirs():
    """List all available Kurdish and classical tafsir editions."""
    return {"tafsirs": AVAILABLE_TAFSIRS}


@app.get("/api/tafsir/{edition_id}/{surah_num}")
def get_tafsir_edition(edition_id: str, surah_num: int):
    """Return verse-by-verse commentary for a specific tafsir edition."""
    if not (1 <= surah_num <= 114):
        raise HTTPException(status_code=400, detail="Surah number must be between 1 and 114")

    # 1. Classical Arabic Tafsirs (Ibn Kathir, Tabari, Qurtubi, Saadi)
    if edition_id in CLASSICAL_CDN_SLUGS:
        m = _get_classical_tafsir_map(edition_id, surah_num)
        return {"edition": edition_id, "surah": surah_num, "ayahs": m}

    # 2. Local Kurdish DB Tafsirs
    if KURDISH_TAFSIRS_DB.exists():
        valid_cols = {
            "asan", "raman", "puxta", "hazhar", "rebar",
            "maisar", "mokhtasar", "tawhid", "roshn",
            "sanahi", "zhin", "runahi"
        }
        if edition_id in valid_cols:
            try:
                conn = sqlite3.connect(str(KURDISH_TAFSIRS_DB))
                cur = conn.cursor()
                cur.execute(f"SELECT ayah, {edition_id} FROM ayah_text WHERE suraId = ? ORDER BY ayah", (str(surah_num),))
                rows = {str(r[0]): (r[1] or "").strip() for r in cur.fetchall()}
                conn.close()
                return {"edition": edition_id, "surah": surah_num, "ayahs": rows}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Database query error: {e}")

    raise HTTPException(status_code=404, detail=f"Tafsir edition '{edition_id}' not found")


@app.get("/api/surah/{surah_num}")
def get_surah(surah_num: int):
    """Return aligned Surah data (or fallback to Quran text if not yet transcribed)."""
    if not (1 <= surah_num <= 114):
        raise HTTPException(status_code=400, detail="Surah number must be between 1 and 114")

    aligned_path = ALIGNED_DIR / f"surah_{surah_num:03d}.json"
    video_path = VIDEOS_DIR / f"surah_{surah_num:03d}.mp4"
    playlist_idx = load_json_safe(PLAYLIST_META_PATH) or {}
    pl_entry = playlist_idx.get(str(surah_num), {})

    if aligned_path.exists():
        data = load_json_safe(aligned_path)
        if data:
            data["video_downloaded"] = video_path.exists()
            data["whisper_aligned"] = True
            data["processing_state"] = PROCESSING_STATUS.get(surah_num, "ready")
            data["youtube_url"] = pl_entry.get("url", "")
            data["youtube_title"] = pl_entry.get("title", "")
            return _attach_kurdish_tafsirs(surah_num, data)

    # Fallback: load or fetch Quran text
    quran_path = QURAN_DIR / f"surah_{surah_num:03d}.json"
    if not quran_path.exists():
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from fetch_quran_text import fetch_surah_text
        fetch_surah_text(surah_num)

    qdata = load_json_safe(quran_path)
    if not qdata:
        raise HTTPException(status_code=404, detail=f"Surah {surah_num} data not found")

    res = {
        **qdata,
        "video_downloaded": video_path.exists(),
        "whisper_aligned": False,
        "processing_state": PROCESSING_STATUS.get(surah_num, "downloaded" if video_path.exists() else "idle"),
        "video_file": f"/media/videos/surah_{surah_num:03d}.mp4" if video_path.exists() else None,
        "vtt_file": None,
        "srt_file": None,
        "youtube_url": pl_entry.get("url", ""),
        "youtube_title": pl_entry.get("title", ""),
        "segments": [],
    }
    return _attach_kurdish_tafsirs(surah_num, res)


def _run_pipeline_for_surah(surah_num: int):
    try:
        PROCESSING_STATUS[surah_num] = "downloading"
        py_bin = str(BASE_DIR / ".venv" / "bin" / "python")
        subprocess.run(
            [py_bin, str(BASE_DIR / "scripts" / "download_playlist.py"), "--surahs", str(surah_num)],
            check=True,
        )
        PROCESSING_STATUS[surah_num] = "transcribing"
        subprocess.run(
            [py_bin, str(BASE_DIR / "scripts" / "transcribe_and_align.py"), "--surahs", str(surah_num)],
            check=True,
        )
        PROCESSING_STATUS[surah_num] = "ready"
    except Exception as exc:
        print(f"Error processing Surah {surah_num}: {exc}", file=sys.stderr)
        PROCESSING_STATUS[surah_num] = f"error: {exc}"


@app.post("/api/process/{surah_num}")
def trigger_process_surah(surah_num: int, background_tasks: BackgroundTasks):
    """Trigger on-demand YouTube MP4 download + Whisper alignment for any Surah."""
    if not (1 <= surah_num <= 114):
        raise HTTPException(status_code=400, detail="Invalid Surah number")
    state = PROCESSING_STATUS.get(surah_num)
    if state in ("downloading", "transcribing"):
        return {"status": state, "surah_number": surah_num}
    PROCESSING_STATUS[surah_num] = "downloading"
    background_tasks.add_task(_run_pipeline_for_surah, surah_num)
    return {"status": "started", "surah_number": surah_num}


@app.post("/api/surah/{surah_num}/verse/{ayah_num}/timestamps")
async def update_verse_timestamps(surah_num: int, ayah_num: int, request: Request):
    """Persist user fine-tuned verse tafsir timestamps."""
    aligned_path = ALIGNED_DIR / f"surah_{surah_num:03d}.json"
    if not aligned_path.exists():
        raise HTTPException(status_code=404, detail="Aligned Surah file not found")
    body = await request.json()
    data = json.loads(aligned_path.read_text(encoding="utf-8"))
    for v in data.get("verses", []):
        if v.get("number_in_surah") == ayah_num:
            if "tafsir_start" in body:
                v["tafsir_start"] = round(float(body["tafsir_start"]), 2)
                v["tafsir_sub_start"] = v["tafsir_start"]
            if "tafsir_end" in body:
                v["tafsir_end"] = round(float(body["tafsir_end"]), 2)
                v["tafsir_sub_end"] = v["tafsir_end"]
            break
    aligned_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


@app.api_route("/media/videos/{filename}", methods=["GET", "HEAD"])
async def stream_video(filename: str, request: Request):
    """Stream MP4 video with full HTTP Range support for instant seeking to verse timestamps."""
    file_path = VIDEOS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        byte1, byte2 = 0, None
        m = range_header.strip().lower().replace("bytes=", "").split("-")
        if m[0]:
            byte1 = int(m[0])
        if len(m) > 1 and m[1]:
            byte2 = int(m[1])

        end = byte2 if byte2 is not None else file_size - 1
        end = min(end, file_size - 1)
        length = end - byte1 + 1

        def iterfile():
            with open(file_path, "rb") as f:
                f.seek(byte1)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {byte1}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers)

    return FileResponse(file_path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})


@app.get("/media/subtitles/{filename}")
def serve_subtitle(filename: str):
    file_path = SUBTITLES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Subtitle file not found")
    media_type = "text/vtt; charset=utf-8" if filename.endswith(".vtt") else "text/plain; charset=utf-8"
    return FileResponse(file_path, media_type=media_type)


@app.get("/media/tts/{filename}")
def serve_tts_audio(filename: str):
    file_path = TTS_CACHE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="TTS audio file not found")
    return FileResponse(file_path, media_type="audio/wav")


@app.get("/media/voice_refs/{filename}")
def serve_voice_ref(filename: str):
    file_path = VOICE_REFS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Reference voice file not found")
    return FileResponse(file_path, media_type="audio/wav")


@app.get("/api/tts/training_status")
def get_tts_training_status():
    status_file = DATA_DIR / "models" / "audio8_quran_kurdish" / "training_status.json"
    if not status_file.exists():
        return {"state": "starting"}
    return load_json_safe(status_file) or {"state": "running"}


@app.post("/api/tts/clone")
async def generate_zero_shot_clone(request: Request):
    """Synthesize native Sorani Kurdish speech + OpenVoice V2 Narrator Timbre Cloning."""
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required for voice cloning")

    engine = (body.get("engine") or "audio8_finetuned").strip()
    if engine not in ("mms_openvoice", "mms_native", "audio8_finetuned"):
        engine = "audio8_finetuned"
    tau = float(body.get("tau") or 0.3)
    force = bool(body.get("force", False))

    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from zero_shot_clone import DEFAULT_REF_WAV, normalize_ckb
    import hashlib

    norm_text = normalize_ckb(text)
    cache_key = hashlib.sha256(
        f"v2|{engine}|{DEFAULT_REF_WAV.name}|{tau}|{norm_text}".encode("utf-8")
    ).hexdigest()[:16]
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_wav = TTS_CACHE_DIR / f"kurdish_{engine}_{cache_key}.wav"

    if out_wav.exists() and not force:
        return {
            "ok": True,
            "cached": True,
            "engine": engine,
            "audio_url": f"/media/tts/{out_wav.name}",
            "kurdish_text": text,
            "phonetic_text": norm_text,
        }

    tts_python = BASE_DIR / ".venv-tts" / "bin" / "python"
    script_path = BASE_DIR / "scripts" / "zero_shot_clone.py"
    if not tts_python.exists():
        raise HTTPException(status_code=503, detail="Zero-shot TTS environment (.venv-tts) is not ready")

    proc = subprocess.run(
        [
            str(tts_python),
            str(script_path),
            "--text",
            text,
            "--engine",
            engine,
            "--tau",
            str(tau),
            "--output",
            str(out_wav),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out_wav.exists():
        err_msg = (proc.stderr or proc.stdout or "Unknown TTS error")[-600:]
        raise HTTPException(status_code=500, detail=f"Kurdish voice cloning failed: {err_msg}")

    meta = {}
    for line in proc.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            try:
                meta = json.loads(line.split("JSON_RESULT:", 1)[1])
            except Exception:
                pass

    return {
        "ok": True,
        "cached": False,
        "engine": engine,
        "audio_url": f"/media/tts/{out_wav.name}",
        "kurdish_text": text,
        "phonetic_text": meta.get("phonetic_text", norm_text),
        "duration_sec": meta.get("duration_sec"),
        "elapsed_sec": meta.get("elapsed_sec"),
    }



NODE_MODULES_DIR = BASE_DIR / "node_modules"
if NODE_MODULES_DIR.exists():
    app.mount("/node_modules", StaticFiles(directory=str(NODE_MODULES_DIR)), name="node_modules")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_spa():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
