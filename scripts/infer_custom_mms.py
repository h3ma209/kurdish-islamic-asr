#!/usr/bin/env python3
"""
Meta MMS-1B + KenLM + Scripture Rescorer: Verbatim Kurdish Sorani & Arabic Harakat Transcriber.
Runs 100% locally on Apple Silicon MPS with zero cloud dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as AF
from pyctcdecode import build_ctcdecoder
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    Wav2Vec2ForCTC,
)

from scripts.scripture_rescorer import ScriptureRescorer

MODEL_DIR = BASE_DIR / "data" / "models" / "mms_1b_islamic_kurdish"
ARPA_PATH = MODEL_DIR / "kurdish_islamic_3gram.arpa"
HEAD_PATH = MODEL_DIR / "finetuned_ckb_head.pt"
ADAPTER_PATH = MODEL_DIR / "finetuned_ckb_adapter.pt"
VOCAB_PATH = MODEL_DIR / "vocab.json"
CORRECTOR_DIR = BASE_DIR / "data" / "models" / "byt5_kurdish_corrector"
WHISPER_AR_DIR = BASE_DIR / "data" / "models" / "whisper_large_v3_turbo"

KURDISH_HARD_CHARS = frozenset("ڕڵۆێگژچپڤ")
ARABIC_SCRIPT_CHARS = frozenset("ذضظصطةأإآكيىثءؤئ")
ARABIC_FUNC_WORDS = frozenset({
    "في", "من", "عن", "على", "إلى", "الى", "قال", "كان", "ليس", "إن", "أن",
    "هذا", "هذه", "تلك", "الذي", "التي", "وما", "لقد", "قد", "صلى", "الله",
    "رسول", "بن", "أبي", "ابي", "علي", "عمر", "أبو", "ابو", "بكر", "شيعة",
    "الشيعة", "لا", "إلا", "الا", "هو", "هي", "هم", "لها", "له", "فيهم",
    "عنه", "عنهم", "ما", "كل", "بعد", "قبل", "مع", "عند", "بين",
})


class IslamicKurdishTranscriber:
    def __init__(self, device: str = "mps", use_corrector: bool = False, use_arabic_whisper: bool = False):
        self.device = device if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
        print(f"Loading Meta MMS-1B backbone on {self.device}...")
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
        self.processor.tokenizer.set_target_lang("ckb")
        self.model.load_adapter("ckb")
        vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8")) if VOCAB_PATH.exists() else self.processor.tokenizer.get_vocab()
        if ADAPTER_PATH.exists():
            ckpt = torch.load(ADAPTER_PATH, map_location="cpu")
            for idx, state in ckpt["adapters"].items():
                self.model.wav2vec2.encoder.layers[int(idx)].adapter_layer.load_state_dict(state)
            head_out = ckpt["lm_head"]["weight"].shape[0]
            if head_out != self.model.lm_head.out_features:
                self.model.lm_head = nn.Linear(
                    self.model.lm_head.in_features,
                    head_out,
                    bias=self.model.lm_head.bias is not None,
                )
            self.model.lm_head.load_state_dict(ckpt["lm_head"])
            print(f"Loaded fine-tuned bilingual adapters ({ckpt['start_layer']}..47) + {head_out}-tok lm_head from {ADAPTER_PATH.name}!")
        elif HEAD_PATH.exists():
            self.model.lm_head.load_state_dict(torch.load(HEAD_PATH, map_location="cpu"))
            print(f"Loaded fine-tuned ckb lm_head from {HEAD_PATH.name}!")
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded in {time.time() - t0:.2f}s!")

        # Initialize PyCTCDecode + KenLM
        inv_vocab = {v: k for k, v in vocab.items()}
        self.inv_vocab = inv_vocab
        labels = []
        for i in range(len(vocab)):
            tok = inv_vocab.get(i, "")
            if i == 0 or tok == "<pad>":
                labels.append("")
            elif tok in ("<s>", "</s>", "<unk>"):
                labels.append(chr(i))
            elif tok == "|":
                labels.append(" ")
            else:
                labels.append(tok)

        words = set()
        in_1g = False
        with open(ARPA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s == "\\1-grams:":
                    in_1g = True
                    continue
                if s.startswith("\\") and in_1g:
                    break
                if in_1g and s:
                    parts = s.split("\t")
                    if len(parts) >= 2 and parts[1] not in ("<s>", "</s>", "<unk>"):
                        words.add(parts[1])

        print(f"Building KenLM Beam Search Decoder with {len(words)} lexicon words...")
        self.decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=str(ARPA_PATH),
            unigrams=sorted(list(words)),
            alpha=0.30,
            beta=0.50,
            unk_score_offset=-10.0,
        )

        print("Initializing Islamic Scripture & Harakat Rescorer...")
        self.rescorer = ScriptureRescorer()

        self.corrector_tok = None
        self.corrector_model = None
        if use_corrector and CORRECTOR_DIR.exists():
            t_c = time.time()
            self.corrector_tok = AutoTokenizer.from_pretrained(str(CORRECTOR_DIR))
            self.corrector_model = AutoModelForSeq2SeqLM.from_pretrained(str(CORRECTOR_DIR)).to(self.device)
            self.corrector_model.eval()
            print(f"Loaded ByT5 error corrector in {time.time() - t_c:.2f}s!")

        self.ar_whisper_proc = None
        self.ar_whisper_model = None
        if use_arabic_whisper and WHISPER_AR_DIR.exists():
            t_w = time.time()
            dtype = torch.float16 if self.device == "mps" else torch.float32
            self.ar_whisper_proc = AutoProcessor.from_pretrained(str(WHISPER_AR_DIR))
            self.ar_whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(str(WHISPER_AR_DIR), dtype=dtype).to(self.device)
            self.ar_whisper_model.eval()
            print(f"Loaded Arabic Whisper-large-v3-turbo fallback in {time.time() - t_w:.2f}s!")

        print(f"Ready! {len(self.rescorer.scriptures)} Quran/Hadith entries indexed.\n")

    def _correct_text(self, text: str) -> str:
        if self.corrector_model is None or not text.strip():
            return text
        words = text.split()
        chunks, cur, cur_bytes = [], [], 0
        for w in words:
            wb = len(w.encode("utf-8")) + 1
            if cur and cur_bytes + wb > 220:
                chunks.append(" ".join(cur))
                cur, cur_bytes = [w], wb
            else:
                cur.append(w)
                cur_bytes += wb
        if cur:
            chunks.append(" ".join(cur))

        prompts = ["fix: " + c for c in chunks]
        enc = self.corrector_tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=320).to(self.device)
        with torch.no_grad():
            gen = self.corrector_model.generate(**enc, max_new_tokens=320)
        decoded = self.corrector_tok.batch_decode(gen, skip_special_tokens=True)

        out_chunks = []
        for orig, fixed in zip(chunks, decoded):
            fixed = fixed.strip()
            if fixed and 0.65 <= (len(fixed) / max(1, len(orig))) <= 1.50:
                out_chunks.append(fixed)
            else:
                out_chunks.append(orig)
        return " ".join(out_chunks)

    def _is_arabic_word(self, word: str) -> bool:
        if any(c in KURDISH_HARD_CHARS for c in word):
            return False
        return any(c in ARABIC_SCRIPT_CHARS for c in word) or (word in ARABIC_FUNC_WORDS)

    def _refine_arabic_spans(
        self,
        wav_16k: torch.Tensor,
        dur: float,
        num_frames: int,
        word_spans: list[tuple[str, int, int]],
    ) -> str:
        if self.ar_whisper_model is None or not word_spans or num_frames <= 0:
            return " ".join(w for w, _, _ in word_spans)

        sec_per_frame = dur / num_frames
        n = len(word_spans)
        flags = [self._is_arabic_word(w) for w, _, _ in word_spans]
        # Allow a single non-hard-Kurdish bridge word inside an Arabic span
        bridged = list(flags)
        for idx in range(1, n - 1):
            if not bridged[idx] and flags[idx - 1] and flags[idx + 1]:
                w = word_spans[idx][0]
                if not any(c in KURDISH_HARD_CHARS for c in w):
                    bridged[idx] = True

        # Find contiguous spans with >= 4 Arabic words and >= 1.2s duration
        spans_to_replace = []
        i = 0
        while i < n:
            if not bridged[i]:
                i += 1
                continue
            j = i
            while j < n and bridged[j]:
                j += 1
            ar_cnt = sum(1 for k in range(i, j) if flags[k])
            st_f = word_spans[i][1]
            en_f = word_spans[j - 1][2]
            span_dur = max(0.0, (en_f - st_f) * sec_per_frame)
            if ar_cnt >= 4 and (j - i) >= 4 and 1.2 <= span_dur <= 25.0:
                spans_to_replace.append((i, j, st_f, en_f))
            i = j

        if not spans_to_replace:
            return " ".join(w for w, _, _ in word_spans)

        out_words = [w for w, _, _ in word_spans]
        wav_np = wav_16k.cpu().numpy()
        dtype = next(self.ar_whisper_model.parameters()).dtype

        # Apply replacements from right to left so indices stay valid
        for i, j, st_f, en_f in reversed(spans_to_replace):
            t0 = max(0.0, st_f * sec_per_frame - 0.25)
            t1 = min(dur, en_f * sec_per_frame + 0.25)
            seg = wav_np[int(t0 * 16000) : int(t1 * 16000)]
            if len(seg) < int(0.8 * 16000):
                continue
            inp = self.ar_whisper_proc(seg, sampling_rate=16000, return_tensors="pt").input_features.to(self.device, dtype=dtype)
            with torch.no_grad():
                pred_ids = self.ar_whisper_model.generate(
                    inp,
                    language="ar",
                    task="transcribe",
                    max_new_tokens=160,
                    repetition_penalty=1.1,
                )
            ar_txt = self.ar_whisper_proc.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            ar_txt = re.sub(r"[،؛.!؟\"«»]+", " ", ar_txt)
            ar_txt = re.sub(r"\s+", " ", ar_txt).strip()
            mms_span = " ".join(out_words[i:j])
            if ar_txt and 0.50 <= (len(ar_txt) / max(1, len(mms_span))) <= 1.65:
                out_words[i:j] = [ar_txt]

        return " ".join(out_words)

    def transcribe(self, audio_path: Path, beam_width: int = 15) -> dict:
        wav, sr = sf.read(str(audio_path))
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)

        t = torch.from_numpy(wav.astype("float32")).unsqueeze(0)
        if sr != 16000:
            wav_16k = AF.resample(t, sr, 16000).squeeze(0)
        else:
            wav_16k = t.squeeze(0)

        dur = len(wav_16k) / 16000.0
        t0 = time.time()

        inputs = self.processor(wav_16k, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            logits_np = logits.cpu().numpy()

        t_inf = time.time() - t0

        t0_dec = time.time()
        beams = self.decoder.decode_beams(logits_np, beam_width=beam_width)
        beam_raw = beams[0][0]
        word_spans: list[tuple[str, int, int]] = []
        for word, (st, en) in beams[0][2]:
            # If KenLM glued multiple OOV tokens into a long string, restore acoustic spaces from greedy CTC
            if len(word) > 12:
                span_ids = np.argmax(logits_np[max(0, st - 1):min(len(logits_np), en + 1)], axis=-1)
                # Decode greedy CTC tokens using expanded 120-token inv_vocab
                prev = -1
                chars = []
                for tid in span_ids:
                    if tid != prev:
                        if tid != 0 and tid in self.inv_vocab:
                            tok = self.inv_vocab[tid]
                            if tok not in ("<s>", "</s>", "<unk>", "<pad>"):
                                chars.append(" " if tok == "|" else tok)
                        prev = tid
                greedy_span = "".join(chars).strip()
                if " " in greedy_span:
                    for sub_w in greedy_span.split():
                        word_spans.append((sub_w, st, en))
                else:
                    word_spans.append((word, st, en))
            else:
                word_spans.append((word, st, en))
        selected_raw = self._refine_arabic_spans(wav_16k, dur, len(logits_np), word_spans)
        t_dec = time.time() - t0_dec

        t0_res = time.time()
        corrected_raw = self._correct_text(selected_raw)
        final_text = self.rescorer.rescore(corrected_raw)
        t_res = time.time() - t0_res

        total_time = t_inf + t_dec + t_res
        speed = dur / total_time if total_time > 0 else 0

        return {
            "audio": str(audio_path),
            "duration_sec": round(dur, 2),
            "inference_time_sec": round(t_inf, 3),
            "beam_decode_sec": round(t_dec, 3),
            "rescore_sec": round(t_res, 3),
            "total_time_sec": round(total_time, 3),
            "speedup_realtime": round(speed, 1),
            "raw_beam_text": beam_raw,
            "transcription": final_text,
        }


def main():
    parser = argparse.ArgumentParser(description="MMS-1B + KenLM + Harakat Kurdish Transcriber")
    parser.add_argument("audio", type=Path, help="Audio file to transcribe (.wav, .mp3, etc.)")
    parser.add_argument("--beam", type=int, default=30, help="Beam width (default: 30)")
    args = parser.parse_args()

    transcriber = IslamicKurdishTranscriber()
    result = transcriber.transcribe(args.audio, beam_width=args.beam)

    print("=" * 70)
    print(f"AUDIO:      {Path(result['audio']).name} ({result['duration_sec']}s)")
    print(f"SPEED:      {result['total_time_sec']}s total ({result['speedup_realtime']}x real-time)")
    print("-" * 70)
    print("FINAL TRANSCRIPTION:")
    print(result["transcription"])
    print("=" * 70)


if __name__ == "__main__":
    main()
