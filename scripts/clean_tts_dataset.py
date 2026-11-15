#!/usr/bin/env python3
"""
ASR-Verified Kurdish Tafsir TTS Dataset Builder (v3).

Problem with v1/v2 clips (timestamp-only slicing):
  - Some clips contain Arabic Quran recitation bleed.
  - Some clips' audio doesn't match the `ku.asan` text being paired with it.
  - Some clips run past the verse into the next Ayah's recitation/tafsir.

Fix — CTC forced alignment with Meta MMS Central Kurdish ASR (`facebook/mms-1b-all`, adapter `ckb`):
  1. For every Kurdish tafsir block, extract audio (+/- margin) at 44.1kHz.
  2. Run MMS `ckb` CTC acoustic model → per-frame (20ms) character log-probs.
  3. Forced-align the exact Kurdish text → per-word start/end time + confidence.
     Audio NOT covered by Kurdish words (Arabic recitation, next Ayah, silence) is discarded.
  4. Re-cut into 2.5s–14s clips at real word-gap pauses; any gap > 0.9s is a hard cut
     (inserted non-Kurdish content never ends up inside a clip).
  5. Quality gate each clip:
       - mean aligned-char confidence >= MIN_ALIGN_CONF
       - CER(greedy MMS transcript, target text) <= MAX_CER
       - speech rate 8–17 chars/s
  6. Writes `data/tts_dataset/metadata.jsonl` + `data/tts_dataset/rejected.jsonl` (with reasons).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoProcessor, Wav2Vec2ForCTC

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from zero_shot_clone import normalize_ckb

ALIGNED_DIR = BASE_DIR / "data" / "aligned"
VIDEOS_DIR = BASE_DIR / "data" / "videos"
DATASET_DIR = BASE_DIR / "data" / "tts_dataset"
WAVS_DIR = DATASET_DIR / "wavs"
CODES_DIR = DATASET_DIR / "codes"

MMS_ID = "facebook/mms-1b-all"
SR_OUT = 44100
SR_ASR = 16000
FRAME_SEC = 0.02  # MMS / wav2vec2 stride (320 samples @ 16kHz)

MARGIN_SEC = 0.8
MIN_DUR, MAX_DUR, TARGET_DUR = 2.5, 14.0, 7.5
SOFT_GAP, HARD_GAP = 0.22, 0.9
MIN_CPS, MAX_CPS = 8.0, 17.0
MIN_ALIGN_CONF = 0.45
MAX_CER = 0.38

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def extract_audio(video: Path, start: float, end: float) -> np.ndarray | None:
    dur = end - start
    if dur < 1.5:
        return None
    cmd = [
        "/opt/homebrew/bin/ffmpeg", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{dur:.3f}",
        "-vn", "-af", "highpass=f=65", "-ac", "1", "-ar", str(SR_OUT), "-f", "f32le",
        "-loglevel", "error", "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or len(res.stdout) < SR_OUT * 4:
        return None
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Text <-> MMS ckb vocab
# ---------------------------------------------------------------------------
def asr_char_form(word: str, vocab: dict[str, int]) -> list[int]:
    w = unicodedata.normalize("NFC", word)
    w = w.replace("\u200c", "").replace("\u200d", "")
    w = _ARABIC_DIACRITICS.sub("", w)
    w = w.lower()
    return [vocab[c] for c in w if c in vocab]


def tokenize_words(text: str) -> list[str]:
    text = text.replace("(", " ").replace(")", " ").replace("«", " ").replace("»", " ")
    return [w for w in text.split() if w.strip()]


def strip_punct(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.replace("\u200c", ""), flags=re.UNICODE)


def cer(ref: str, hyp: str) -> float:
    r = strip_punct(ref).replace(" ", "")
    h = strip_punct(hyp).replace(" ", "")
    if not r:
        return 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
class KurdishAligner:
    def __init__(self, device: str):
        self.processor = AutoProcessor.from_pretrained(MMS_ID, target_lang="ckb")
        self.model = Wav2Vec2ForCTC.from_pretrained(
            MMS_ID, target_lang="ckb", ignore_mismatched_sizes=True
        ).eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.vocab: dict[str, int] = self.processor.tokenizer.get_vocab()
        self.blank = int(self.processor.tokenizer.pad_token_id)
        self.word_delim = self.vocab.get("|")
        self.id2char = {i: c for c, i in self.vocab.items()}

    @torch.inference_mode()
    def emissions(self, wav16: np.ndarray) -> torch.Tensor:
        """Return log-probs [T, V] on CPU. Long audio processed in 30s windows."""
        chunk = SR_ASR * 30
        outs = []
        for s in range(0, len(wav16), chunk):
            seg = wav16[s : s + chunk]
            if len(seg) < SR_ASR * 0.3:
                # tiny tail: pad so the conv stack yields frames
                seg = np.pad(seg, (0, int(SR_ASR * 0.3) - len(seg)))
            inp = self.processor(seg, sampling_rate=SR_ASR, return_tensors="pt").input_values.to(self.device)
            logits = self.model(inp).logits[0].float()
            outs.append(torch.log_softmax(logits, dim=-1).cpu())
        return torch.cat(outs, dim=0)

    def greedy(self, lp: torch.Tensor) -> str:
        ids = lp.argmax(-1).tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i != self.blank:
                c = self.id2char.get(i, "")
                out.append(" " if c == "|" else c)
            prev = i
        return " ".join("".join(out).split())

    def align_words(self, lp: torch.Tensor, words: list[str]) -> list[dict] | None:
        """Forced-align words → [{word, start, end, conf}] (frame indices)."""
        tok_words = [(w, asr_char_form(w, self.vocab)) for w in words]
        targets, owners = [], []
        for wi, (_, toks) in enumerate(tok_words):
            if not toks:
                continue
            if targets and self.word_delim is not None:
                targets.append(self.word_delim)
                owners.append(-1)
            targets.extend(toks)
            owners.extend([wi] * len(toks))
        if not targets or len(targets) * 1 >= lp.shape[0]:
            return None

        tgt = torch.tensor([targets], dtype=torch.int32)
        try:
            ali, scores = AF.forced_align(lp.unsqueeze(0), tgt, blank=self.blank)
        except Exception:
            return None
        ali = ali[0].tolist()
        probs = scores[0].exp().tolist()

        # Map each frame to its target-token index (CTC path expansion)
        tok_frames: list[list[int]] = [[] for _ in targets]
        t_idx, prev = -1, self.blank
        for f, lab in enumerate(ali):
            if lab == self.blank:
                prev = lab
                continue
            if lab != prev or t_idx < 0:
                t_idx += 1
            if t_idx < len(targets):
                tok_frames[t_idx].append(f)
            prev = lab

        out: dict[int, dict] = {}
        for ti, frames in enumerate(tok_frames):
            wi = owners[ti]
            if wi < 0 or not frames:
                continue
            conf = float(np.mean([probs[f] for f in frames]))
            d = out.setdefault(wi, {"start": frames[0], "end": frames[-1] + 1, "confs": []})
            d["start"] = min(d["start"], frames[0])
            d["end"] = max(d["end"], frames[-1] + 1)
            d["confs"].append(conf)

        result = []
        for wi, (w, toks) in enumerate(tok_words):
            if wi in out:
                d = out[wi]
                result.append({"word": w, "start": d["start"], "end": d["end"], "conf": float(np.mean(d["confs"]))})
            elif result:
                # non-alignable token (pure punctuation / symbols): glue to previous word
                result[-1]["word"] += " " + w
        return result


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def segment_words(words: list[dict]) -> list[list[dict]]:
    """Group aligned words into clips cut at real pauses; hard-cut big gaps."""
    segs, cur = [], []
    for i, w in enumerate(words):
        if cur:
            gap = (w["start"] - cur[-1]["end"]) * FRAME_SEC
            dur = (cur[-1]["end"] - cur[0]["start"]) * FRAME_SEC
            punct_end = bool(re.search(r"[.؟!؛،:]$", cur[-1]["word"]))
            if gap >= HARD_GAP:
                segs.append(cur); cur = []
            elif dur >= TARGET_DUR and (gap >= SOFT_GAP or punct_end):
                segs.append(cur); cur = []
            elif dur >= MAX_DUR - 1.0 and gap >= 0.08:
                segs.append(cur); cur = []
        cur.append(w)
    if cur:
        segs.append(cur)

    # Merge too-short tail segments into the previous one when the joining gap is small
    merged: list[list[dict]] = []
    for s in segs:
        dur = (s[-1]["end"] - s[0]["start"]) * FRAME_SEC
        if merged and dur < MIN_DUR:
            prev = merged[-1]
            gap = (s[0]["start"] - prev[-1]["end"]) * FRAME_SEC
            total = (s[-1]["end"] - prev[0]["start"]) * FRAME_SEC
            if gap < HARD_GAP and total <= MAX_DUR:
                prev.extend(s)
                continue
        merged.append(s)
    return merged


def finish_clip(wav44: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    s = max(0, int(start_s * SR_OUT))
    e = min(len(wav44), int(end_s * SR_OUT))
    clip = wav44[s:e].copy()
    fade = min(int(SR_OUT * 0.012), len(clip) // 8)
    if fade > 0:
        clip[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        clip[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    peak = float(np.abs(clip).max()) if len(clip) else 0.0
    if peak > 1e-4:
        clip *= 0.89 / peak
    return clip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def collect_blocks(only_surahs: set[int] | None) -> list[dict]:
    jobs = []
    for af in sorted(ALIGNED_DIR.glob("surah_*.json")):
        data = json.loads(af.read_text(encoding="utf-8"))
        sn = int(data["surah_number"])
        if only_surahs and sn not in only_surahs:
            continue
        video = VIDEOS_DIR / f"surah_{sn:03d}.mp4"
        if not video.exists():
            continue
        total_dur = float(data.get("duration") or 1e9)
        blocks: dict[tuple[float, float], list[dict]] = {}
        for v in data["verses"]:
            ts, te = v.get("tafsir_start"), v.get("tafsir_end")
            if ts in (None, "") or te in (None, ""):
                continue
            ts, te = float(ts), float(te)
            if te > ts:
                blocks.setdefault((round(ts, 2), round(te, 2)), []).append(v)
        for (ts, te), verses in sorted(blocks.items()):
            texts = [str(v.get("kurdish_text", "")).strip() for v in verses]
            text = normalize_ckb(" . ".join(t for t in texts if t))
            if len(text) < 10:
                continue
            a0, a1 = int(verses[0]["number_in_surah"]), int(verses[-1]["number_in_surah"])
            jobs.append({
                "id": f"s{sn:03d}_a{a0:03d}_{a1:03d}", "surah": sn, "ayah_start": a0, "ayah_end": a1,
                "video": video, "start": max(0.0, ts - MARGIN_SEC), "end": min(total_dur, te + MARGIN_SEC),
                "text": text,
            })
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surahs", type=str, default="", help="comma list, e.g. 1,80 (default: all)")
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true", help="don't overwrite metadata / wavs")
    args = ap.parse_args()
    only = {int(x) for x in args.surahs.split(",") if x.strip()} or None

    jobs = collect_blocks(only)
    print(f"Blocks to align: {len(jobs)}", flush=True)
    aligner = KurdishAligner(args.device)
    print(f"MMS ckb loaded on {args.device} | vocab={len(aligner.vocab)}", flush=True)

    out_wavs = WAVS_DIR if not args.dry_run else DATASET_DIR / "wavs_dryrun"
    if not args.dry_run:
        for d in (WAVS_DIR, CODES_DIR):
            d.mkdir(parents=True, exist_ok=True)
            for f in list(d.glob("*.wav")) + list(d.glob("*.npy")):
                f.unlink()
    out_wavs.mkdir(parents=True, exist_ok=True)

    kept, rejected = [], []
    t0 = time.time()
    for bi, job in enumerate(jobs, 1):
        wav44 = extract_audio(job["video"], job["start"], job["end"])
        if wav44 is None:
            rejected.append({"id": job["id"], "reason": "extract_failed"})
            continue
        wav16 = AF.resample(torch.from_numpy(wav44), SR_OUT, SR_ASR).numpy()
        lp = aligner.emissions(wav16)
        words = aligner.align_words(lp, tokenize_words(job["text"]))
        if not words:
            rejected.append({"id": job["id"], "reason": "align_failed"})
            continue

        n_frames = lp.shape[0]
        for si, seg in enumerate(segment_words(words)):
            cid = f"{job['id']}_c{si+1:02d}"
            text = " ".join(w["word"] for w in seg).strip()
            f0, f1 = seg[0]["start"], seg[-1]["end"]
            # pad: never reach into neighbouring words
            prev_end = words[words.index(seg[0]) - 1]["end"] if words.index(seg[0]) > 0 else 0
            next_start = words[words.index(seg[-1]) + 1]["start"] if words.index(seg[-1]) + 1 < len(words) else n_frames
            pad_l = min(4, max(0, (f0 - prev_end) // 2))
            pad_r = min(6, max(0, (next_start - f1) // 2))
            s_sec, e_sec = (f0 - pad_l) * FRAME_SEC, (f1 + pad_r) * FRAME_SEC
            dur = e_sec - s_sec
            conf = float(np.mean([w["conf"] for w in seg]))
            low_words = sum(1 for w in seg if w["conf"] < 0.2) / len(seg)
            hyp = aligner.greedy(lp[max(0, f0 - 2) : f1 + 2])
            c = cer(text, hyp)
            cps = len(text) / max(dur, 1e-3)

            reason = None
            if dur < MIN_DUR or dur > MAX_DUR + 0.5:
                reason = f"duration {dur:.1f}s"
            elif conf < MIN_ALIGN_CONF:
                reason = f"low_align_conf {conf:.2f}"
            elif low_words > 0.25:
                reason = f"unmatched_words {low_words:.0%}"
            elif c > MAX_CER:
                reason = f"cer {c:.2f}"
            elif not (MIN_CPS <= cps <= MAX_CPS):
                reason = f"speech_rate {cps:.1f}"

            rec = {
                "id": cid, "surah": job["surah"], "ayah_start": job["ayah_start"], "ayah_end": job["ayah_end"],
                "duration": round(dur, 2), "chars_per_sec": round(cps, 2), "align_conf": round(conf, 3),
                "cer": round(c, 3), "text": text, "asr": hyp,
                "src_start": round(job["start"] + s_sec, 3), "src_end": round(job["start"] + e_sec, 3),
            }
            if reason:
                rejected.append({**rec, "reason": reason})
                continue
            wav_path = out_wavs / f"{cid}.wav"
            sf.write(str(wav_path), finish_clip(wav44, s_sec, e_sec), SR_OUT, subtype="PCM_16")
            kept.append({**rec, "audio_filepath": str(wav_path)})

        if bi % 10 == 0 or bi == len(jobs):
            mins = sum(r["duration"] for r in kept) / 60
            print(f"  [{bi}/{len(jobs)}] kept={len(kept)} ({mins:.1f} min) rejected={len(rejected)} | {time.time()-t0:.0f}s", flush=True)

    suffix = "_dryrun" if args.dry_run else ""
    with (DATASET_DIR / f"metadata{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for r in sorted(kept, key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DATASET_DIR / f"rejected{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    reasons: dict[str, int] = {}
    for r in rejected:
        key = r["reason"].split()[0]
        reasons[key] = reasons.get(key, 0) + 1
    mins = sum(r["duration"] for r in kept) / 60
    print(f"\n✅ Verified clips: {len(kept)} ({mins:.1f} min) | rejected: {len(rejected)} {reasons}", flush=True)


if __name__ == "__main__":
    main()
