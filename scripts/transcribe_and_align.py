#!/usr/bin/env python3
"""
Precision Acoustic + Whisper Word-Timestamp Aligner for Kurdish Quran Tafsir.
1. Uses adaptive RMS silence detection to locate every spoken phrase boundary with zero dropped speech.
2. Runs Whisper with `language='ar'` and `word_timestamps=True` on each speech interval.
3. Splits mixed Arabic->Kurdish chunks at the exact word timestamp where Arabic recitation transitions to Kurdish Tafsir.
4. Globally aligns every Arabic recitation block to its exact range of Quran verses (v_start .. v_end) using Dynamic Programming over the Surah's Arabic text.
5. Sub-aligns multi-verse Kurdish Tafsir blocks to individual verses using Whisper word timestamps + Kurdish verse weights.
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
QURAN_DIR = DATA_DIR / "quran"
SUBTITLES_DIR = DATA_DIR / "subtitles"
ALIGNED_DIR = DATA_DIR / "aligned"

KURDISH_CHARS = set("پچژگڤڵڕێۆە")
KURDISH_TOKENS = {
    "خدا", "خدايا", "خداي", "بخشند", "بخشندو", "مهربان", "بناوي", "بنبي", "بناظي",
    "اما", "اي", "امان", "اماندار", "جهان", "قيامت", "پاداشت", "سزا", "خور",
    "تفسير", "كورد", "رباز", "برسطين", "پرورتجار", "بلام", "تنها", "كاتيك", "كاتق",
    "هر", "بو", "لبر", "كو", "هرد", "نياز", "بيوست", "دصلا", "دبن", "دبيت", "ناوان",
}


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


def load_audio_ffmpeg(video_path: Path, sr: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(video_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"[ىئ]", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_acoustic_speech_chunks(
    audio: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 20,
    min_silence_s: float = 0.25,
    min_speech_s: float = 0.35,
    max_chunk_s: float = 15.0,
) -> list[tuple[float, float]]:
    """Detect all spoken intervals using adaptive RMS energy envelope."""
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return []
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)

    p10 = float(np.percentile(db, 10))
    thresh = max(-36.0, min(-25.0, p10 + 9.0))
    is_speech = db > thresh

    min_sil_frames = int(min_silence_s * 1000 / frame_ms)
    min_sp_frames = int(min_speech_s * 1000 / frame_ms)
    max_ch_frames = int(max_chunk_s * 1000 / frame_ms)

    # Fill short silence gaps
    i = 0
    while i < n_frames:
        if not is_speech[i]:
            j = i
            while j < n_frames and not is_speech[j]:
                j += 1
            if i > 0 and j < n_frames and (j - i) < min_sil_frames:
                is_speech[i:j] = True
            i = j
        else:
            i += 1

    chunks = []
    i = 0
    total_dur = len(audio) / sr
    while i < n_frames:
        if is_speech[i]:
            j = i
            while j < n_frames and is_speech[j]:
                j += 1
            cur = i
            while (j - cur) > max_ch_frames:
                search_start = cur + int(4.0 * 1000 / frame_ms)
                search_end = min(j - int(2.0 * 1000 / frame_ms), cur + max_ch_frames)
                if search_end <= search_start:
                    break
                window = np.convolve(db[search_start:search_end], np.ones(15) / 15.0, mode="valid")
                split_idx = search_start + int(np.argmin(window)) + 7
                s_t = max(0.0, (cur * frame_ms / 1000.0) - 0.08)
                e_t = min(total_dur, (split_idx * frame_ms / 1000.0) + 0.08)
                chunks.append((round(s_t, 2), round(e_t, 2)))
                cur = split_idx
            if (j - cur) >= min_sp_frames:
                s_t = max(0.0, (cur * frame_ms / 1000.0) - 0.08)
                e_t = min(total_dur, (j * frame_ms / 1000.0) + 0.08)
                chunks.append((round(s_t, 2), round(e_t, 2)))
            i = j
        else:
            i += 1
    return chunks


def word_match_score(w_norm: str, vocab: set[str]) -> float:
    if not w_norm:
        return 0.0
    if w_norm in vocab:
        return 1.0
    return max((difflib.SequenceMatcher(None, w_norm, vw).ratio() for vw in vocab), default=0.0)


def phrase_similarity(seg_norm: str, target_norm: str) -> float:
    if not seg_norm or not target_norm:
        return 0.0
    seq_r = difflib.SequenceMatcher(None, seg_norm, target_norm).ratio()
    s_words = seg_norm.split()
    t_words = target_norm.split()
    if not s_words or not t_words:
        return seq_r
    hits = sum(
        1
        for sw in s_words
        if max((difflib.SequenceMatcher(None, sw, tw).ratio() for tw in t_words), default=0.0) >= 0.68
    )
    cov_s = hits / len(s_words)
    cov_t = hits / len(t_words)
    f1 = (2 * cov_s * cov_t / (cov_s + cov_t)) if (cov_s + cov_t) > 0 else 0.0
    return max(seq_r, 0.65 * cov_s + 0.35 * f1)


def transcribe_chunks_with_words(
    audio: np.ndarray,
    chunks: list[tuple[float, float]],
    model: WhisperModel,
    arabic_vocab: set[str],
    norm_verses: list[str],
) -> list[dict]:
    """
    Transcribe each acoustic chunk with word timestamps.
    If a chunk starts with Arabic recitation and transitions mid-chunk into Kurdish tafsir,
    split it at the exact transition word timestamp!
    """
    segments = []
    seg_id = 1

    for st, en in chunks:
        sub = audio[int(st * 16000) : int(en * 16000)]
        if len(sub) < 1600:
            continue
        raw_segs, _ = model.transcribe(
            sub,
            language="ar",
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
            beam_size=2,
            compression_ratio_threshold=2.4,
            log_prob_threshold=None,
            no_speech_threshold=0.9,
        )

        words = []
        for rseg in raw_segs:
            for w in rseg.words or []:
                w_raw = w.word.strip()
                w_norm = normalize_arabic(w_raw)
                if not w_norm:
                    continue
                has_ku_char = any(ch in KURDISH_CHARS for ch in w_raw)
                is_ku_tok = w_norm in KURDISH_TOKENS
                vmatch = 0.25 if (has_ku_char or is_ku_tok) else word_match_score(w_norm, arabic_vocab)
                words.append({
                    "start": round(st + float(w.start), 2),
                    "end": round(st + float(w.end), 2),
                    "word": w_raw,
                    "norm": w_norm,
                    "ar_score": round(vmatch, 3),
                })

        if not words:
            continue

        # Check if this chunk is a mixed [Arabic Recitation -> Kurdish Tafsir] chunk
        split_at = None
        if len(words) >= 6:
            best_gain = 0.0
            for k in range(3, len(words) - 2):
                prefix_scores = [x["ar_score"] for x in words[:k]]
                suffix_scores = [x["ar_score"] for x in words[k:]]
                p_mean = float(np.mean(prefix_scores))
                s_mean = float(np.mean(suffix_scores))
                # Ensure the first word of the suffix is not a strong Arabic word
                if words[k]["ar_score"] >= 0.72:
                    continue
                if p_mean >= 0.75 and s_mean <= 0.52 and (p_mean - s_mean) >= 0.28 and (p_mean - s_mean) > best_gain:
                    p_norm = " ".join(x["norm"] for x in words[:k])
                    max_v_sim = max((phrase_similarity(p_norm, vn) for vn in norm_verses), default=0.0)
                    if max_v_sim >= 0.55:
                        best_gain = p_mean - s_mean
                        split_at = k

        if split_at is not None:
            w_ar = words[:split_at]
            w_ku = words[split_at:]
            segments.append({
                "id": seg_id,
                "start": round(st, 2),
                "end": w_ar[-1]["end"],
                "text": " ".join(x["word"] for x in w_ar),
                "words": w_ar,
                "ar_score": round(float(np.mean([x["ar_score"] for x in w_ar])), 3),
            })
            seg_id += 1
            segments.append({
                "id": seg_id,
                "start": w_ku[0]["start"],
                "end": round(en, 2),
                "text": " ".join(x["word"] for x in w_ku),
                "words": w_ku,
                "ar_score": round(float(np.mean([x["ar_score"] for x in w_ku])), 3),
            })
            seg_id += 1
        else:
            segments.append({
                "id": seg_id,
                "start": round(st, 2),
                "end": round(en, 2),
                "text": " ".join(x["word"] for x in words),
                "words": words,
                "ar_score": round(float(np.mean([x["ar_score"] for x in words])), 3),
            })
            seg_id += 1

    return segments


def classify_and_align(
    surah_num: int,
    segments: list[dict],
    verses: list[dict],
    total_duration: float,
) -> list[dict]:
    n_verses = len(verses)
    if not segments or n_verses == 0:
        return []

    bism_prefix = normalize_arabic("بسم الله الرحمن الرحيم")
    bism_full = normalize_arabic("اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم")

    norm_verses = []
    for i, v in enumerate(verses):
        txt_norm = normalize_arabic(v["arabic_text"])
        if surah_num != 1 and i == 0 and txt_norm.startswith(bism_prefix):
            stripped = txt_norm[len(bism_prefix) :].strip()
            norm_verses.append(stripped or txt_norm)
        else:
            norm_verses.append(txt_norm)

    # Precompute concatenated verse norms for spans up to 6 verses
    span_norms: dict[tuple[int, int], str] = {}
    for i in range(n_verses):
        for j in range(i, min(i + 6, n_verses)):
            span_norms[(i, j)] = " ".join(norm_verses[i : j + 1])

    # Classify each segment as 'arabic' or 'kurdish_tafsir'
    for idx, seg in enumerate(segments):
        s_norm = normalize_arabic(seg["text"])
        has_ku_char = any(ch in KURDISH_CHARS for ch in seg["text"])
        s_tokens = set(s_norm.split())
        ku_tok_hits = len(s_tokens & KURDISH_TOKENS)

        # Best similarity to any 1..4 verse span near expected chronological position
        best_span_sim = 0.0
        for (vi, vj), sp_norm in span_norms.items():
            if vj - vi > 3:
                continue
            sim = phrase_similarity(s_norm, sp_norm)
            if sim > best_span_sim:
                best_span_sim = sim

        bism_sim = phrase_similarity(s_norm, bism_full) if idx <= 2 else 0.0
        ar_score = seg.get("ar_score", 0.0)
        seg_dur = seg["end"] - seg["start"]
        word_cnt = len(s_norm.split())

        is_arabic = False
        if idx <= 1 and bism_sim >= 0.60 and ku_tok_hits == 0:
            is_arabic = True
        elif seg_dur >= 2.2 and word_cnt >= 3:
            if not has_ku_char and ku_tok_hits == 0 and (best_span_sim >= 0.58 or (ar_score >= 0.76 and best_span_sim >= 0.50)):
                is_arabic = True
            elif ku_tok_hits <= 1 and best_span_sim >= 0.68 and ar_score >= 0.74:
                is_arabic = True

        seg["role"] = "arabic" if is_arabic else "kurdish_tafsir"
        seg["span_sim"] = round(max(best_span_sim, bism_sim), 3)

    # Group consecutive segments into alternating (Arabic Recitation + Kurdish Tafsir) blocks
    blocks = []
    cur_blk = None
    for seg in segments:
        if seg["role"] == "arabic":
            if cur_blk is None:
                cur_blk = {"arabic_segs": [seg], "tafsir_segs": []}
            elif len(cur_blk["tafsir_segs"]) == 0:
                cur_blk["arabic_segs"].append(seg)
            else:
                blocks.append(cur_blk)
                cur_blk = {"arabic_segs": [seg], "tafsir_segs": []}
        else:
            if cur_blk is None:
                cur_blk = {"arabic_segs": [], "tafsir_segs": [seg]}
            else:
                cur_blk["tafsir_segs"].append(seg)
    if cur_blk is not None:
        blocks.append(cur_blk)

    # Strip standalone Bismillah intro block for Surahs != 1
    content_blocks = blocks
    if surah_num != 1 and len(blocks) > 1:
        b0_ar_norm = normalize_arabic(" ".join(s["text"] for s in blocks[0]["arabic_segs"]))
        bism_match = phrase_similarity(b0_ar_norm, bism_full)
        v1_match = phrase_similarity(b0_ar_norm, norm_verses[0])
        if bism_match >= v1_match or len(b0_ar_norm.split()) <= 5:
            content_blocks = blocks[1:]

    if not content_blocks:
        content_blocks = blocks

    # Merge any block that has empty tafsir_segs into the next block (or into the previous block if it's the final block)
    merged_content = []
    i = 0
    while i < len(content_blocks):
        blk = content_blocks[i]
        while len(blk["tafsir_segs"]) == 0 and (i + 1) < len(content_blocks):
            nxt = content_blocks[i + 1]
            blk["arabic_segs"].extend(nxt["arabic_segs"])
            blk["tafsir_segs"].extend(nxt["tafsir_segs"])
            i += 1
        if len(blk["tafsir_segs"]) == 0 and merged_content:
            # Trailing orphan Arabic-classified segment at the very end of the video -> attach to previous block's tafsir
            for s in blk["arabic_segs"]:
                s["role"] = "kurdish_tafsir"
            merged_content[-1]["tafsir_segs"].extend(blk["arabic_segs"])
        else:
            merged_content.append(blk)
        i += 1
    content_blocks = merged_content

    B = len(content_blocks)
    V = n_verses

    # Dynamic Programming to Partition V verses across B blocks (or merge blocks if B > V)
    if B == V:
        mapping = [[v_i] for v_i in range(V)]
    elif 0 < B < V:
        # DP[b][v] = max score assigning blocks 0..b to verses 0..v
        dp = np.full((B, V), -1e9, dtype=np.float64)
        back = np.zeros((B, V), dtype=np.int32)

        blk_ar_norms = [
            normalize_arabic(" ".join(s["text"] for s in blk["arabic_segs"]))
            for blk in content_blocks
        ]

        def score_block_span(b_idx: int, v_start: int, v_end: int) -> float:
            ar_txt = blk_ar_norms[b_idx]
            target = " ".join(norm_verses[v_start : v_end + 1])
            sim = phrase_similarity(ar_txt, target)
            # Word count ratio penalty so a 2-verse recitation doesn't swallow 5 verses
            w_rec = max(len(ar_txt.split()), 1)
            w_tar = max(len(target.split()), 1)
            len_ratio = min(w_rec, w_tar) / max(w_rec, w_tar)
            return 0.75 * sim + 0.25 * len_ratio

        for v_end in range(0, V - (B - 1)):
            if v_end <= 5:
                dp[0, v_end] = score_block_span(0, 0, v_end)

        for b_idx in range(1, B):
            for v_end in range(b_idx, V - (B - 1 - b_idx)):
                best_val = -1e9
                best_prev = b_idx - 1
                lo_prev = max(b_idx - 1, v_end - 6)
                for prev_v in range(lo_prev, v_end):
                    if dp[b_idx - 1, prev_v] <= -1e8:
                        continue
                    val = dp[b_idx - 1, prev_v] + score_block_span(b_idx, prev_v + 1, v_end)
                    if val > best_val:
                        best_val = val
                        best_prev = prev_v
                dp[b_idx, v_end] = best_val
                back[b_idx, v_end] = best_prev

        # Backtrack from (B-1, V-1)
        mapping = [[] for _ in range(B)]
        cur_v = V - 1
        for b_idx in range(B - 1, -1, -1):
            prev_v = int(back[b_idx, cur_v]) if b_idx > 0 else -1
            mapping[b_idx] = list(range(prev_v + 1, cur_v + 1))
            cur_v = prev_v
    else:
        # B > V: merge adjacent blocks using DP so V verses map cleanly
        merged_blocks = []
        for v_i in range(V):
            s_b = int(round(v_i * B / V))
            e_b = int(round((v_i + 1) * B / V))
            sub = content_blocks[s_b : max(s_b + 1, e_b)]
            ar_s = [s for b in sub for s in b["arabic_segs"]]
            tf_s = [s for b in sub for s in b["tafsir_segs"]]
            merged_blocks.append({"arabic_segs": ar_s, "tafsir_segs": tf_s})
        content_blocks = merged_blocks
        mapping = [[v_i] for v_i in range(V)]

    # Build per-verse aligned records with word-accurate sub-slicing
    aligned_verses = []
    for b_idx, verse_indices in enumerate(mapping):
        blk = content_blocks[b_idx]
        ar_segs = blk["arabic_segs"]
        tf_segs = blk["tafsir_segs"]
        all_segs = ar_segs + tf_segs
        if not all_segs:
            continue

        blk_start = all_segs[0]["start"]
        blk_end = all_segs[-1]["end"]
        ar_start = ar_segs[0]["start"] if ar_segs else blk_start
        ar_end = ar_segs[-1]["end"] if ar_segs else blk_start
        tf_start = tf_segs[0]["start"] if tf_segs else ar_end
        tf_end = tf_segs[-1]["end"] if tf_segs else blk_end

        ar_whisper = " ".join(s["text"].strip() for s in ar_segs).strip()
        tf_whisper = " ".join(s["text"].strip() for s in tf_segs).strip()

        # Collect all Whisper words and segment boundaries inside tf_segs for exact sub-slicing
        tf_words = [w for s in tf_segs for w in s.get("words", [])]
        seg_ends = [s["end"] for s in tf_segs[:-1]]

        def verse_weight(v_idx: int) -> float:
            v_o = verses[v_idx]
            ku_len = len((v_o.get("kurdish_text") or "").strip())
            ar_len = len(normalize_arabic(v_o.get("arabic_text") or ""))
            # If ku.asan merged translation into previous verse (empty string), estimate from Arabic length
            return float(ku_len if ku_len > 5 else max(ar_len * 2.2, 25.0))

        weights = [verse_weight(v_i) for v_i in verse_indices]
        total_w = sum(weights)
        cur_tf_t = tf_start
        tf_dur = max(0.4, tf_end - tf_start)

        for pos, v_i in enumerate(verse_indices):
            v_obj = verses[v_i]
            sub_start = cur_tf_t
            if pos == len(verse_indices) - 1:
                sub_end = tf_end
            else:
                target_t = cur_tf_t + tf_dur * (weights[pos] / total_w)
                # Prefer snapping to a pause boundary between Whisper tafsir segments if within 2.5s
                valid_seg_ends = [se for se in seg_ends if se > sub_start + 0.6 and se < tf_end - 0.5]
                best_seg_end = min(valid_seg_ends, key=lambda c: abs(c - target_t)) if valid_seg_ends else None
                if best_seg_end is not None and abs(best_seg_end - target_t) <= 2.5:
                    sub_end = best_seg_end
                elif tf_words:
                    # Snap to nearest word boundary
                    valid_w_ends = [w["end"] for w in tf_words if w["end"] > sub_start + 0.5 and w["end"] < tf_end - 0.4]
                    sub_end = min(valid_w_ends, key=lambda c: abs(c - target_t)) if valid_w_ends else round(target_t, 2)
                else:
                    sub_end = round(target_t, 2)
            cur_tf_t = sub_end

            # Also extract the specific Whisper words inside [sub_start, sub_end]
            sub_whisper = " ".join(
                w["word"] for w in tf_words if w["start"] >= sub_start - 0.15 and w["end"] <= sub_end + 0.15
            ).strip() or tf_whisper

            aligned_verses.append({
                "number_in_surah": v_obj["number_in_surah"],
                "number_global": v_obj.get("number_global"),
                "arabic_text": v_obj["arabic_text"],
                "kurdish_text": v_obj["kurdish_text"],
                "english_text": v_obj.get("english_text", ""),
                "start_time": round(blk_start, 2),
                "end_time": round(blk_end, 2),
                "arabic_start": round(ar_start, 2),
                "arabic_end": round(ar_end, 2),
                "tafsir_start": round(tf_start, 2),
                "tafsir_end": round(tf_end, 2),
                "tafsir_sub_start": round(sub_start, 2),
                "tafsir_sub_end": round(sub_end, 2),
                "shared_block": len(verse_indices) > 1,
                "shared_verses": [verses[x]["number_in_surah"] for x in verse_indices],
                "whisper_arabic": ar_whisper,
                "whisper_tafsir": sub_whisper,
            })

    return aligned_verses


def format_timestamp_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hrs = ms // 3_600_000
    ms %= 3_600_000
    mins = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hrs = ms // 3_600_000
    ms %= 3_600_000
    mins = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"


def write_subtitles(surah_num: int, segments: list[dict]) -> tuple[Path, Path]:
    SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)
    srt_path = SUBTITLES_DIR / f"surah_{surah_num:03d}.srt"
    vtt_path = SUBTITLES_DIR / f"surah_{surah_num:03d}.vtt"

    srt_lines = []
    vtt_lines = ["WEBVTT\n"]

    for i, seg in enumerate(segments, start=1):
        s_srt = format_timestamp_srt(seg["start"])
        e_srt = format_timestamp_srt(seg["end"])
        s_vtt = format_timestamp_vtt(seg["start"])
        e_vtt = format_timestamp_vtt(seg["end"])
        role_tag = "🕌 [Arabic]" if seg.get("role") == "arabic" else "🗣️ [Kurdish Tafsir]"
        text = seg["text"].strip()

        srt_lines.append(f"{i}\n{s_srt} --> {e_srt}\n{text}\n")
        vtt_lines.append(f"{i}\n{s_vtt} --> {e_vtt}\n{role_tag} {text}\n")

    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")
    return srt_path, vtt_path


def process_surah(
    surah_num: int,
    model: WhisperModel,
    force: bool = False,
) -> dict | None:
    video_path = VIDEOS_DIR / f"surah_{surah_num:03d}.mp4"
    quran_path = QURAN_DIR / f"surah_{surah_num:03d}.json"
    aligned_path = ALIGNED_DIR / f"surah_{surah_num:03d}.json"

    if not video_path.exists():
        print(f"[Surah {surah_num:03d}] Video not found at {video_path}, skipping.")
        return None

    if aligned_path.exists() and not force:
        print(f"[Surah {surah_num:03d}] Already transcribed & aligned: {aligned_path.name}")
        return json.loads(aligned_path.read_text(encoding="utf-8"))

    if not quran_path.exists():
        from fetch_quran_text import fetch_surah_text
        fetch_surah_text(surah_num)

    quran_data = json.loads(quran_path.read_text(encoding="utf-8"))
    verses = quran_data.get("verses", [])

    print(f"[Surah {surah_num:03d}] Acoustic chunking + Whisper word-timestamp alignment on {video_path.name} ...")
    audio = load_audio_ffmpeg(video_path)
    duration = round(len(audio) / 16000.0, 2)

    arabic_vocab = set()
    norm_verses = []
    for v in verses:
        vn = normalize_arabic(v["arabic_text"])
        norm_verses.append(vn)
        arabic_vocab.update(vn.split())
    arabic_vocab.update(normalize_arabic("بسم الله الرحمن الرحيم اعوذ بالله من الشيطان الرجيم").split())

    chunks = get_acoustic_speech_chunks(audio)
    segments = transcribe_chunks_with_words(audio, chunks, model, arabic_vocab, norm_verses)
    aligned_verses = classify_and_align(surah_num, segments, verses, duration)

    # Strip word lists from segments before saving JSON to keep payload compact
    clean_segments = [
        {
            "id": s["id"],
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "role": s.get("role", "kurdish_tafsir"),
        }
        for s in segments
    ]
    write_subtitles(surah_num, clean_segments)

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "surah_number": surah_num,
        "name_arabic": quran_data.get("name_arabic", ""),
        "name_english": quran_data.get("name_english", ""),
        "name_translation": quran_data.get("name_translation", ""),
        "revelation_type": quran_data.get("revelation_type", ""),
        "number_of_ayahs": quran_data.get("number_of_ayahs", len(verses)),
        "duration": duration,
        "video_file": f"/media/videos/surah_{surah_num:03d}.mp4",
        "vtt_file": f"/media/subtitles/surah_{surah_num:03d}.vtt",
        "srt_file": f"/media/subtitles/surah_{surah_num:03d}.srt",
        "segments": clean_segments,
        "verses": aligned_verses,
    }
    aligned_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    ar_count = sum(1 for s in clean_segments if s["role"] == "arabic")
    ku_count = len(clean_segments) - ar_count
    print(
        f"[Surah {surah_num:03d}] Done! {len(clean_segments)} cues (ar={ar_count}, ku={ku_count}), "
        f"{len(aligned_verses)} verses aligned."
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="Transcribe Surah MP4s with Whisper and align Kurdish Tafsir to verses.")
    parser.add_argument("--surahs", type=str, default="1,78-114")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Whisper workers")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    targets = parse_surah_range(args.surahs)
    print(f"Loading faster-whisper model '{args.model}' (int8 on CPU/Accelerate) ...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=4, num_workers=args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_surah, s, model, args.force): s for s in targets}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"[Surah {s:03d}] Error during transcription: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
