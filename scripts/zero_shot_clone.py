#!/usr/bin/env python3
"""
Native Central Kurdish (Sorani, `ckb`) TTS + OpenVoice V2 Tone Color Voice Cloner.

Pipeline:
1. Native Sorani Kurdish Base TTS (`razhan/mms-tts-ckb` VITS):
   - Directly tokenizes Sorani Kurdish Arabic script (`ئ، ا، ب، پ، ت، ج، چ، ح، خ، د، ر، ڕ، ز، ژ، س، ش، ع، غ، ف، ڤ، ق، ک، گ، ل، ڵ، م، ن، و، ۆ، ه، ە، ی، ێ`).
   - Speaks 100% authentic Central Kurdish phonology and cadence with zero English accent.
2. OpenVoice V2 Audio-to-Audio Tone Color Converter (`Hinotsuba/OpenVoice-ONNX-v2`):
   - Extracts 256-dim speaker tone color embeddings from the base Kurdish TTS and from
     our Quran Kurdish Tafsir narrator (`data/voice_refs/kurdish_narrator_ref_12s.wav`).
   - Converts the base Kurdish speech spectrogram to match the Quran narrator's vocal
     timbre while preserving every Kurdish phoneme and consonant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None  # Imported lazily when synthesizing audio in .venv-tts

BASE_DIR = Path(__file__).resolve().parent.parent
VOICE_REFS_DIR = BASE_DIR / "data" / "voice_refs"
TTS_CACHE_DIR = BASE_DIR / "data" / "tts_cache"

DEFAULT_REF_WAV = VOICE_REFS_DIR / "kurdish_narrator_ref_12s.wav"
if not DEFAULT_REF_WAV.exists():
    DEFAULT_REF_WAV = VOICE_REFS_DIR / "kurdish_narrator_ref.wav"

MMS_MODEL_ID = "razhan/mms-tts-ckb"
OPENVOICE_REPO_ID = "Hinotsuba/OpenVoice-ONNX-v2"
OV_SAMPLE_RATE = 22050
FINETUNED_AUDIO8_DIR = BASE_DIR / "data" / "models" / "audio8_quran_kurdish"

# ---------------------------------------------------------------------------
# Sorani Kurdish (`ckb`) Text Normalization
# ---------------------------------------------------------------------------

_CKB_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ە",
        "ھ": "ه",
        "«": '"',
        "»": '"',
        "٬": ",",
        "٫": ".",
        "–": "-",
        "—": "-",
    }
)
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_DIACRITICS_RE = re.compile("[\u064b-\u065f\u0670\u06d6-\u06ed]")
_INVISIBLE_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

_CKB_ONES = [
    "سفر", "یەک", "دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت", "هەشت", "نۆ",
    "دە", "یازدە", "دوازدە", "سێزدە", "چواردە", "پازدە", "شازدە", "حەڤدە", "هەژدە", "نۆزدە",
]
_CKB_TENS = ["", "", "بیست", "سی", "چل", "پەنجا", "شەست", "حەفتا", "هەشتا", "نەوەد"]
_CKB_SCALES = [(10**9, "ملیار"), (10**6, "ملیۆن"), (1000, "هەزار")]


def _ckb_below_1000(n: int) -> str:
    parts = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        parts.append("سەد" if hundreds == 1 else f"{_CKB_ONES[hundreds]} سەد")
    if rest >= 20:
        tens, ones = divmod(rest, 10)
        parts.append(_CKB_TENS[tens] if not ones else f"{_CKB_TENS[tens]} و {_CKB_ONES[ones]}")
    elif rest:
        parts.append(_CKB_ONES[rest])
    return " و ".join(parts)


def ckb_number_words(value: int) -> str:
    if value == 0:
        return _CKB_ONES[0]
    parts = []
    for scale, name in _CKB_SCALES:
        count, value = divmod(value, scale)
        if count:
            parts.append(name if count == 1 and scale == 1000 else f"{ckb_number_words(count)} {name}")
    if value:
        parts.append(_ckb_below_1000(value))
    return " و ".join(parts)


def _ckb_int(value: str) -> str:
    return ckb_number_words(int(value)) if len(value) <= 12 else " ".join(_CKB_ONES[int(d)] for d in value)


def _ckb_number(match: re.Match[str]) -> str:
    token = match.group()
    if "," in token:
        whole, _, rest = token.partition(",")
        if len(rest) == 3:
            return _ckb_int(whole + rest)
        return f"{_ckb_int(whole)} ، {_ckb_int(rest)}"
    if "." in token:
        whole, _, frac = token.partition(".")
        return f"{_ckb_int(whole)} پۆینت {_ckb_int(frac)}"
    return _ckb_int(token)


def normalize_ckb(text: Any) -> str:
    """Normalize Sorani Kurdish text (including Tanzil ku.asan orthography and Islamic honorifics) to standard Sorani."""
    text = str(text or "")
    # Ignore / skip words in curly brackets (e.g. {بسم الله الرحمن الرحيم *ويل للمطففين}) and Quran brackets ﴾...﴿
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"﴾[^﴿]*﴿", " ", text)

    # Strip Quran verse citations and bracketed indices/footnotes
    text = re.sub(r"\[\s*(?:سورة|سوورەتی|سورەتی)[^\]]*\]", " ", text)
    text = re.sub(r"\(\s*(?:سورة|سوورەتی|سورەتی)[^)]*\)", " ", text)
    text = re.sub(r"\[\s*[\d٠-٩]+\s*\]", " ", text)
    text = re.sub(r"\(\s*[\d٠-٩]+\s*\)", " ", text)
    text = re.sub(r"\[\s*[^\]]{1,25}\s*\]", " ", text)

    # Islamic and Quranic honorific expansions to native Sorani phonetics (with or without brackets)
    # 1. Prophet Muhammad (صلی الله علیه وسلم / ﷺ / (ص) / (دخ))
    text = re.sub(
        r"\(?\s*(?:ص[لڵ]ّ?[ىيی]\s+[اأ]?[لڵ]+[هەھ]\s+(?:ع[لڵ][يی][هەھ]|عَلَيْهِ)\s+(?:وە?[سش][ەه]?ل+ّ?[ەه]?م|وسلم|وسلّم)|ﷺ|\(\s*ص\s*\)|\(\s*دخ\s*\))\s*\)?",
        " دروودی خوای لێ بێت ",
        text,
    )

    # 2. Prophets & Angels (علیه السلام / عەلیهی سەلام / (ع))
    text = re.sub(r"\(?\s*ع[لڵ][يی][هەھ]م\s*ال[سص]ّ?[لڵ][ەه]?ام\s*\)?", " سڵاوی خوایان لێ بێت ", text)
    text = re.sub(
        r"\(?\s*(?:ع[لڵ][يی][هەھ]\s*ال[سص]ّ?[لڵ][ەه]?ام|عەلیهی\s*سەلام)\s*\)?|\(\s*ع\s*\)",
        " سڵاوی خوای لێ بێت ",
        text,
    )

    # 3. Companions (رضي الله عنه / ڕەزای خوای لێبێت / (رض))
    text = re.sub(r"\(?\s*ر[ضظ][يی]\s*[اأ]?[لڵ]+[هەھ]\s*عن[هەھ]م\s*\)?", " ڕەزای خوایان لێ بێت ", text)
    text = re.sub(
        r"\(?\s*(?:ر[ضظ][يی]\s*[اأ]?[لڵ]+[هەھ]\s*عن[هەھ]ا?|ڕەزای\s*خوای\s*لێ\s*بێت|ڕەزای\s*خوای\s*لێبێت)\s*\)?|\(\s*رض\s*\)",
        " ڕەزای خوای لێ بێت ",
        text,
    )

    # 4. Scholars & Righteous (رحمه الله / ڕەحمەتی خوای لێبێت / (رح))
    text = re.sub(
        r"\(?\s*(?:رحم[هەھ]|ڕەحمەتی)\s*(?:[اأ]?[لڵ]+[هەھ]|خوای?)(?:\s*لێ\s*بێت|\s*لێبێت)?\s*\)?|\(\s*رح\s*\)",
        " ڕەحمەتی خوای لێ بێت ",
        text,
    )

    # 5. Divine expressions & glorifications
    text = re.sub(r"\(?\s*س?سبحان[هەھ]\s*و\s*تعال[ىيی]\s*\)?|\(\s*سبحان[هەھ]\s*\)", " سوبحانەهو وە تەعالا ", text)
    text = re.sub(r"\(?\s*سبحان\s*[اأ]?[لڵ]+[هەھ]\s*\)?", " سوبحانەڵڵا ", text)
    text = re.sub(r"\(?\s*ع[زژ]+ّ?\s*و\s*ج[لڵ]+ّ?\s*\)?|عَزَّ\s*وَجَلَّ", " عەززە وە جەلل ", text)
    text = re.sub(r"\(?\s*ج[لڵ]+ّ?\s*ج[لڵ][ەه]?ال[هەھ]\s*\)?", " جەللە جەلالەهو ", text)
    text = re.sub(r"\(?\s*تبار[كک]\s*و\s*تعال[ىيی]\s*\)?", " تەبارەکە وە تەعالا ", text)
    text = re.sub(r"\(?\s*لا\s*إ?ل[هەھ]\s*إ?لا\s*[اأ]?[لڵ]+[هەھ]\s*\)?", " لا ئیلاهە ئیللەلا ", text)
    text = re.sub(r"\(?\s*[اأ]?لحمد\s*ل[لڵ]+[هەھ]\s*\)?", " ئەلحەمدولیلا ", text)
    text = re.sub(r"\(?\s*[إا]?ن\s*شا[ءە]\s*[اأ]?[لڵ]+[هەھ]\s*\)?", " ئینشائەڵڵا ", text)
    text = re.sub(r"\(?\s*ما\s*شا[ءە]\s*[اأ]?[لڵ]+[هەھ]\s*\)?", " ماشائەڵڵا ", text)
    text = re.sub(r"\(?\s*[اأ]ستغفر\s*[اأ]?[لڵ]+[هەھ]\s*\)?", " ئەستەغفیرولڵا ", text)

    # Strip remaining redundant parentheses around clean honorific expressions
    text = re.sub(
        r"\(\s*(دروودی\s*خوای\s*لێ\s*بێت|سڵاوی\s*خوای\s*لێ\s*بێت|ڕەزای\s*خوای\s*لێ\s*بێت)\s*\)",
        r" \1 ",
        text,
    )

    text = unicodedata.normalize("NFC", text)
    # Convert Arabic-style final ه + space/punctuation or ه+ZWNJ into Sorani vowel ە
    text = text.replace("ه\u200c", "ە").replace("ه‌", "ە")
    text = re.sub(r"ه(?=$|[\s،,.!?؟:;؛«»()\[\]])", "ە", text)
    text = text.translate(_CKB_CHAR_MAP).translate(_DIGITS)
    text = _DIACRITICS_RE.sub("", text)
    text = _INVISIBLE_RE.sub("", text)
    text = text.replace("ـ", "")
    text = re.sub(r"(?<=\d)[,٬](?=\d{3}\b)", "", text)
    text = re.sub(r"(?:لەسەدا|لە\s*سەدا)\s*(\d+(?:[.,]\d+)?)\s*[%٪]", r" لەسەدا \1 ", text)
    text = re.sub(r"(?<!لەسەدا\s)(\d+(?:[.,]\d+)?)\s*[%٪]", r" لەسەدا \1 ", text)
    text = _NUMBER_RE.sub(lambda m: f" {_ckb_number(m)} ", text)
    text = re.sub(r"\s+([،,.!?؟:;؛])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_tail_bleed(
    wav: np.ndarray,
    sr: int,
    max_bleed_dur: float = 1.0,
    min_gap_dur: float = 0.22,
    thresh: float = 0.015,
) -> np.ndarray:
    """Detect and remove trailing autoregressive bleed (e.g. recitation murmur)

    occurring after speech has already concluded with a silence gap >= min_gap_dur.
    Leaves clean natural sentences untouched.
    """
    if len(wav) < int(sr * 1.0):
        return wav

    frame_sz = int(sr * 0.02)  # 20ms frames
    n_frames = len(wav) // frame_sz
    if n_frames < 30:
        return wav

    energies = np.array([np.mean(np.abs(wav[i * frame_sz : (i + 1) * frame_sz])) for i in range(n_frames)])
    active_indices = np.where(energies > thresh)[0]
    if len(active_indices) == 0:
        return wav
    last_active = active_indices[-1]

    max_bleed_frames = int(max_bleed_dur / 0.02)
    min_gap_frames = int(min_gap_dur / 0.02)
    search_limit = max(0, last_active - max_bleed_frames - min_gap_frames)

    gap_end = None
    for i in range(last_active, search_limit, -1):
        if energies[i] <= thresh:
            if gap_end is None:
                gap_end = i
        else:
            if gap_end is not None:
                gap_len = gap_end - i
                burst_len = last_active - gap_end
                if gap_len >= min_gap_frames and burst_len <= max_bleed_frames:
                    gap_start = i
                    cut_idx = min(len(wav), (gap_start + 2) * frame_sz)
                    res = wav[:cut_idx].copy()
                    fade = min(int(sr * 0.025), len(res))
                    if fade > 0:
                        res[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
                    return res
                gap_end = None

    return wav


def post_process_studio_audio(wav: np.ndarray, sr: int) -> np.ndarray:
    """Mastering chain:
    1. 80Hz highpass filter to eliminate subsonic boom / rumble.
    2. Silence trim leading & trailing digital silence.
    3. Strip post-pause tail bleed (if AR hallucination occurred).
    4. Smooth soft-knee peak limiter targeting -0.8 dBFS true-peak.
    """
    if len(wav) == 0:
        return wav

    try:
        import torch
        import torchaudio.functional as AF

        # Highpass biquad filter @ 80Hz
        t = torch.from_numpy(wav).unsqueeze(0)
        t = AF.highpass_biquad(t, sr, 80.0).squeeze(0)
        filtered = t.numpy()
    except Exception:
        filtered = wav.copy()

    # Trim leading & trailing silence (< -42 dB threshold)
    thresh = 0.008
    non_silent = np.where(np.abs(filtered) > thresh)[0]
    if len(non_silent) > 0:
        start = max(0, non_silent[0] - int(sr * 0.05))
        end = min(len(filtered), non_silent[-1] + int(sr * 0.08))
        filtered = filtered[start:end]

    # Clean any trailing bleed following an inter-speech pause
    filtered = strip_tail_bleed(filtered, sr)

    # Peak normalization with soft-knee headroom limiter
    peak = float(np.max(np.abs(filtered)))
    if peak > 1e-4:
        target_peak = 0.92  # -0.7 dBFS headroom
        scale = target_peak / peak
        filtered = filtered * scale
        filtered = np.clip(filtered, -0.98, 0.98)

    return filtered.astype(np.float32)


# Alias for backward compatibility with server.py
kurdish_to_phonetic = normalize_ckb


def split_ckb_clauses(text: str, max_chars: int = 160) -> list[str]:
    """Split long Sorani paragraphs into sentence/clause chunks for natural breathing pauses."""
    text = normalize_ckb(text)
    if len(text) <= max_chars:
        return [text] if text else []
    pieces = re.split(r"(?<=[.!?؟؛،,])\s+", text)
    chunks: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if chunks and len(chunks[-1]) + len(piece) + 1 <= max_chars:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Singleton Caches for MMS-TTS-CKB & OpenVoice V2 ONNX
# ---------------------------------------------------------------------------

_MMS_CACHE: dict[str, Any] = {}
_OV_CACHE: dict[str, Any] = {}
_AUDIO8_FT_CACHE: dict[str, Any] = {}


def synthesize_audio8_finetuned(text: str, ref_audio_path: Path = DEFAULT_REF_WAV) -> tuple[np.ndarray, int]:
    """Synthesize speech using our fine-tuned Audio8-TTS-Kurdish-Sorani model (or base Audio8 zero-shot clone while training)."""
    import torch
    from transformers import AutoModel, AutoProcessor

    ft_exists = (FINETUNED_AUDIO8_DIR / "model.safetensors").exists()
    model_source = str(FINETUNED_AUDIO8_DIR) if ft_exists else "ArminRahimi/Audio8-TTS-Kurdish-Sorani"
    if _AUDIO8_FT_CACHE.get("source") != model_source:
        device = "cpu"
        processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_source, trust_remote_code=True, dtype=torch.float32).eval().to(device)
        _AUDIO8_FT_CACHE["source"] = model_source
        _AUDIO8_FT_CACHE["processor"] = processor
        _AUDIO8_FT_CACHE["model"] = model
        _AUDIO8_FT_CACHE["device"] = device

    model = _AUDIO8_FT_CACHE["model"]
    processor = _AUDIO8_FT_CACHE["processor"]
    device = _AUDIO8_FT_CACHE["device"]

    ref_codes_path = FINETUNED_AUDIO8_DIR / "anchor_ref_codes.npy"
    ref_text_path = FINETUNED_AUDIO8_DIR / "anchor_ref_text.txt"
    ref_codes = np.load(str(ref_codes_path)) if ref_codes_path.exists() else None
    ref_text = ref_text_path.read_text(encoding="utf-8").strip() if ref_text_path.exists() else None

    # If anchor_ref_codes.npy isn't saved yet, encode kurdish_narrator_ref.wav on the fly and cache it
    if ref_codes is None:
        default_ref_wav = VOICE_REFS_DIR / "kurdish_narrator_ref.wav"
        default_ref_txt = normalize_ckb(
            "سوپاس و ستایش هەر شایستەی خودایە و بۆ خودایە، کە خاوەن و پەروەردگاری هەموو جیهانیان و هەموو بوونەوەرە."
        )
        if "_cached_ref_codes" not in _AUDIO8_FT_CACHE and default_ref_wav.exists():
            inp = processor(
                text=["سڵاو"],
                reference_audio=[str(default_ref_wav)],
                reference_text=[default_ref_txt],
                return_tensors="pt",
            )
            with torch.inference_mode():
                c, cl = model.encode_audio(inp["reference_audio_values"].to(device), inp["reference_audio_lengths"].to(device))
                _AUDIO8_FT_CACHE["_cached_ref_codes"] = c[0, :, : int(cl[0])].cpu().numpy()
                _AUDIO8_FT_CACHE["_cached_ref_text"] = default_ref_txt
        ref_codes = _AUDIO8_FT_CACHE.get("_cached_ref_codes")
        ref_text = _AUDIO8_FT_CACHE.get("_cached_ref_text")

    clauses = split_ckb_clauses(text, max_chars=135)
    if not clauses:
        raise ValueError("Empty Kurdish text after normalization")

    sr = int(model.config.codec_sample_rate)
    pieces: list[np.ndarray] = []

    def clear_caches():
        for layer in list(model.layers) + list(model.fast_layers):
            attn = getattr(layer, "attention", None)
            if attn is not None:
                attn.kv_cache = None

    try:
        for idx, clause in enumerate(clauses):
            clear_caches()
            kwargs: dict[str, Any] = {"text": [clause], "return_tensors": "pt"}
            if ref_codes is not None and ref_text:
                kwargs["reference_codes"] = [ref_codes]
                kwargs["reference_text"] = [normalize_ckb(ref_text)]
            inputs = processor(**kwargs)
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            est_frames = max(90, min(420, int(len(clause) * 2.4) + 40))
            gen = torch.Generator(device=device).manual_seed(1234 + idx)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=est_frames,
                    temperature=0.50,
                    top_p=0.84,
                    top_k=28,
                    do_sample=True,
                    return_dict_in_generate=True,
                    generator=gen,
                )
                wavs, lengths = model.decode_audio(out.codes)
                chunk = wavs[0, : int(lengths[0])].float().cpu().numpy()
                chunk = strip_tail_bleed(chunk, sr)
                fade_n = min(int(sr * 0.015), len(chunk) // 8)
                if fade_n > 0:
                    chunk[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
                    chunk[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
                pieces.append(chunk)
                if idx < len(clauses) - 1:
                    pieces.append(np.zeros(int(sr * 0.18), dtype=np.float32))
    finally:
        clear_caches()

    full_wav = np.concatenate(pieces)
    full_wav = post_process_studio_audio(full_wav, sr)
    return full_wav, sr


def _get_mms_model():
    import torch
    from transformers import AutoTokenizer, VitsModel

    if "model" not in _MMS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(MMS_MODEL_ID)
        model = VitsModel.from_pretrained(MMS_MODEL_ID).eval()
        # Slightly slower speaking rate for clear Quranic tafsir cadence
        model.speaking_rate = 0.92
        model.noise_scale = 0.5
        model.noise_scale_duration = 0.6
        _MMS_CACHE["tokenizer"] = tokenizer
        _MMS_CACHE["model"] = model
    return _MMS_CACHE["model"], _MMS_CACHE["tokenizer"]


def synthesize_mms_sorani(text: str) -> tuple[np.ndarray, int]:
    """Synthesize 100% native Central Kurdish speech via Meta MMS (`razhan/mms-tts-ckb`)."""
    import torch

    model, tokenizer = _get_mms_model()
    clauses = split_ckb_clauses(text, max_chars=160)
    if not clauses:
        raise ValueError("Empty Kurdish text after normalization")

    sr = int(model.config.sampling_rate)
    pieces: list[np.ndarray] = []

    for idx, clause in enumerate(clauses):
        inputs = tokenizer(clause, return_tensors="pt")
        if inputs["input_ids"].shape[1] == 0:
            continue
        with torch.inference_mode():
            torch.manual_seed(42)
            wav = model(**inputs).waveform[0].cpu().numpy().astype(np.float32)
            pieces.append(wav)
            if idx < len(clauses) - 1:
                pieces.append(np.zeros(int(sr * 0.22), dtype=np.float32))

    if not pieces:
        raise RuntimeError("No audio generated by MMS-TTS-CKB")
    return np.concatenate(pieces), sr


def _compute_openvoice_spectrogram(wav_22k: np.ndarray) -> np.ndarray:
    """
    Compute linear magnitude spectrogram matching OpenVoice V2 `spectrogram_torch`:
    n_fft=1024, hop_length=256, win_length=1024, reflect pad 384, center=False.
    Returns shape [1, 513, frames] float32.
    """
    import torch

    y = torch.from_numpy(np.asarray(wav_22k, dtype=np.float32)).unsqueeze(0)  # [1, T]
    # Normalize peak to ~0.95 to avoid clipping
    peak = y.abs().max()
    if peak > 1e-4:
        y = y / peak * 0.9

    pad = int((1024 - 256) / 2)  # 384
    y_padded = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    window = torch.hann_window(1024, periodic=True, dtype=torch.float32)
    spec_complex = torch.stft(
        y_padded,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(spec_complex.real.pow(2) + spec_complex.imag.pow(2) + 1e-6)
    return spec.numpy().astype(np.float32)  # [1, 513, frames]


def _resample_to_22050(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == OV_SAMPLE_RATE:
        return wav.astype(np.float32)
    import torch
    from torchaudio.functional import resample

    t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    if t.ndim > 1:
        t = t.mean(dim=-1)
    out = resample(t, int(orig_sr), OV_SAMPLE_RATE)
    return out.numpy().astype(np.float32)


def _get_openvoice_sessions():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    if "extract_sess" not in _OV_CACHE:
        extract_path = hf_hub_download(OPENVOICE_REPO_ID, "tone_extract.onnx")
        color_path = hf_hub_download(OPENVOICE_REPO_ID, "tone_color.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        _OV_CACHE["extract_sess"] = ort.InferenceSession(extract_path, sess_options=opts)
        _OV_CACHE["color_sess"] = ort.InferenceSession(color_path, sess_options=opts)
    return _OV_CACHE["extract_sess"], _OV_CACHE["color_sess"]


def _extract_tone_embedding(wav_22k: np.ndarray) -> np.ndarray:
    """Return speaker tone embedding of shape [1, 256, 1] float32."""
    extract_sess, _ = _get_openvoice_sessions()
    spec_f_t = _compute_openvoice_spectrogram(wav_22k)  # [1, 513, frames]
    spec_t_f = np.transpose(spec_f_t, (0, 2, 1))        # [1, frames, 513]
    emb = extract_sess.run(["tone_embedding"], {"input": spec_t_f})[0]  # [1, 256]
    return emb.reshape(1, 256, 1).astype(np.float32)


def _get_target_narrator_embedding(ref_audio_path: Path) -> np.ndarray:
    key = f"dest_tone_{ref_audio_path.name}"
    if key not in _OV_CACHE:
        # Combine both reference clips if available for an even richer speaker embedding
        wavs = []
        for p in [VOICE_REFS_DIR / "kurdish_narrator_ref.wav", VOICE_REFS_DIR / "kurdish_narrator_ref_12s.wav"]:
            if p.exists():
                w, sr = sf.read(str(p), dtype="float32")
                if w.ndim > 1:
                    w = w.mean(axis=1)
                wavs.append(_resample_to_22050(w, sr))
        if not wavs:
            w, sr = sf.read(str(ref_audio_path), dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            wavs.append(_resample_to_22050(w, sr))

        embs = [_extract_tone_embedding(w) for w in wavs]
        _OV_CACHE[key] = np.mean(embs, axis=0).astype(np.float32)
    return _OV_CACHE[key]


def _get_mms_base_embedding() -> np.ndarray:
    """Extract a stable speaker embedding for the base MMS-TTS-CKB voice."""
    if "src_tone_mms" not in _OV_CACHE:
        calib_text = (
            "سوپاس و ستایش هەر شایستەی خودایە و بۆ خودایە، کە خاوەن و پەروەردگاری هەموو جیهانیان و هەموو بوونەوەرە. "
            "بەڕاستی ئێمە خێر و چاکەیەکی زۆر و بێ سنوورمان بە تۆ بەخشیوە ئەی پێغەمبەر."
        )
        wav, sr = synthesize_mms_sorani(calib_text)
        wav_22k = _resample_to_22050(wav, sr)
        _OV_CACHE["src_tone_mms"] = _extract_tone_embedding(wav_22k)
    return _OV_CACHE["src_tone_mms"]


def convert_voice_timbre_openvoice(
    src_wav: np.ndarray,
    src_sr: int,
    ref_audio_path: Path = DEFAULT_REF_WAV,
    tau: float = 0.3,
) -> tuple[np.ndarray, int]:
    """
    Convert the native Kurdish TTS audio (`src_wav`) into the Quran narrator's
    vocal timbre using OpenVoice V2 Tone Color Converter.
    """
    _, color_sess = _get_openvoice_sessions()
    wav_22k = _resample_to_22050(src_wav, src_sr)
    spec = _compute_openvoice_spectrogram(wav_22k)  # [1, 513, frames]
    frames = spec.shape[2]

    src_tone = _get_mms_base_embedding()
    dest_tone = _get_target_narrator_embedding(ref_audio_path)

    converted = color_sess.run(
        ["converted_audio"],
        {
            "audio": spec,
            "audio_length": np.array([frames], dtype=np.int64),
            "src_tone": src_tone,
            "dest_tone": dest_tone,
            "tau": np.array([tau], dtype=np.float32),
        },
    )[0]
    out_wav = converted[0, 0, :].astype(np.float32)
    # Normalize output peak cleanly
    peak = np.max(np.abs(out_wav))
    if peak > 1e-4:
        out_wav = (out_wav / peak * 0.92).astype(np.float32)
    return out_wav, OV_SAMPLE_RATE


def clone_kurdish_voice(
    text: str,
    output_path: Path | None = None,
    ref_audio_path: Path = DEFAULT_REF_WAV,
    engine: str = "mms_openvoice",  # "mms_openvoice" (cloned timbre) | "mms_native" (pure base Sorani)
    tau: float = 0.3,
    **kwargs,
) -> dict:
    """
    Synthesize 100% native Sorani Kurdish speech and optionally clone the Quran
    narrator's vocal timbre via OpenVoice V2.
    """
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    norm_text = normalize_ckb(text)

    if output_path is None:
        cache_key = hashlib.sha256(
            f"v2|{engine}|{ref_audio_path.name}|{tau}|{norm_text}".encode("utf-8")
        ).hexdigest()[:16]
        output_path = TTS_CACHE_DIR / f"kurdish_{engine}_{cache_key}.wav"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if engine == "audio8_finetuned":
        final_wav, final_sr = synthesize_audio8_finetuned(norm_text)
    else:
        base_wav, base_sr = synthesize_mms_sorani(norm_text)
        if engine == "mms_native":
            final_wav, final_sr = base_wav, base_sr
        else:
            final_wav, final_sr = convert_voice_timbre_openvoice(
                base_wav, base_sr, ref_audio_path=ref_audio_path, tau=tau
            )
        final_wav = post_process_studio_audio(final_wav, final_sr)

    sf.write(str(output_path), final_wav, final_sr)
    elapsed = round(time.time() - t0, 2)
    duration = round(len(final_wav) / final_sr, 2)

    return {
        "output_path": str(output_path),
        "filename": output_path.name,
        "engine": engine,
        "kurdish_text": text,
        "phonetic_text": norm_text,
        "duration_sec": duration,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Native Sorani Kurdish TTS + OpenVoice V2 Voice Cloner")
    parser.add_argument(
        "--text",
        type=str,
        default="بەڕاستی ئێمە خێر و چاکەیەکی زۆر و بێ سنوورمان بە تۆ بەخشیوە ئەی پێغەمبەر.",
        help="Sorani Kurdish text to synthesize",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="mms_openvoice",
        choices=["mms_openvoice", "mms_native", "audio8_finetuned"],
        help="mms_openvoice, mms_native, or audio8_finetuned (Fine-Tuned on 38 Surahs)",
    )
    parser.add_argument("--tau", type=float, default=0.3, help="OpenVoice tone conversion temperature")
    parser.add_argument("--output", type=str, default=None, help="Output WAV file path")
    parser.add_argument("--json", action="store_true", help="Print JSON metadata")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    result = clone_kurdish_voice(
        text=args.text,
        output_path=out_path,
        engine=args.engine,
        tau=args.tau,
    )
    if args.json:
        print("JSON_RESULT:" + json.dumps(result, ensure_ascii=False))
    else:
        print(f"\n✅ Saved native Kurdish audio ({result['engine']}) to: {result['output_path']}")
        print(f"   Normalized Sorani : {result['phonetic_text']}")
        print(f"   Duration          : {result['duration_sec']}s (generated in {result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
