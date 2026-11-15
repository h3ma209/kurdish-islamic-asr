#!/usr/bin/env python3
"""
Dual-Model Segmented Transcriber:
1. Splits audio into natural 1.5-8s segments on acoustic pauses.
2. Runs Fine-Tuned MMS-1B (ckb) for Kurdish Sorani (greedy space-preserving CTC).
3. Detects Arabic segments (via MMS-1B ckb vs ara CTC confidence + Sorani-char ratio + ScriptureRescorer)
   and routes Arabic segments to stock Whisper-Large-V3-Turbo (language="ar") + ScriptureRescorer.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoProcessor, Wav2Vec2ForCTC, WhisperForConditionalGeneration, WhisperProcessor

from scripts.scripture_rescorer import ScriptureRescorer

MODEL_DIR = BASE_DIR / "data" / "models" / "mms_1b_islamic_kurdish"
ADAPTER_PATH = MODEL_DIR / "finetuned_ckb_adapter.pt"
WHISPER_ID = "openai/whisper-large-v3-turbo"
SORANI_ONLY_CHARS = set("پچژگڤڕڵێۆە")


def segment_on_pauses(
    wav: np.ndarray,
    sr: int = 16000,
    min_sec: float = 1.5,
    max_sec: float = 7.5,
    silence_db_below_rms: float = 14.0,
) -> list[tuple[float, float, np.ndarray]]:
    """Split audio at low-energy pauses into 1.5-7.5s chunks."""
    win = int(0.05 * sr)  # 50ms
    hop = int(0.02 * sr)  # 20ms
    if len(wav) <= int(max_sec * sr):
        return [(0.0, len(wav) / sr, wav)]

    rms_all = np.sqrt(np.mean(wav ** 2) + 1e-12)
    thresh = rms_all * (10 ** (-silence_db_below_rms / 20.0))

    energies = []
    for i in range(0, len(wav) - win, hop):
        energies.append((i + win // 2, float(np.sqrt(np.mean(wav[i : i + win] ** 2) + 1e-12))))

    cuts = [0]
    cur = 0
    total = len(wav)
    while total - cur > int(max_sec * sr):
        lo = cur + int(min_sec * sr)
        hi = min(cur + int(max_sec * sr), total - int(0.8 * sr))
        cands = [(pos, e) for pos, e in energies if lo <= pos <= hi]
        if not cands:
            nxt = hi
        else:
            # Prefer quietest frame near middle-to-end of window
            quiet = [p for p, e in cands if e < thresh]
            if quiet:
                nxt = quiet[len(quiet) // 2]
            else:
                nxt = min(cands, key=lambda x: x[1])[0]
        cuts.append(nxt)
        cur = nxt
    cuts.append(total)

    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= int(0.35 * sr):
            segs.append((a / sr, b / sr, wav[a:b]))
    return segs


class DualLangTranscriber:
    def __init__(self, device: str = "mps"):
        self.device = device if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
        t0 = time.time()
        print(f"Loading Dual-Model Pipeline on {self.device}...")

        # 1. Fine-tuned MMS-1B (ckb)
        self.mms_proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        self.mms_model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
        self.mms_proc.tokenizer.set_target_lang("ckb")
        self.mms_model.load_adapter("ckb")
        if ADAPTER_PATH.exists():
            ckpt = torch.load(ADAPTER_PATH, map_location="cpu")
            for idx, state in ckpt["adapters"].items():
                self.mms_model.wav2vec2.encoder.layers[int(idx)].adapter_layer.load_state_dict(state)
            self.mms_model.lm_head.load_state_dict(ckpt["lm_head"])
        self.mms_model.to(self.device).eval()

        # 2. Stock Whisper-Large-V3-Turbo (Arabic specialist)
        self.wh_proc = WhisperProcessor.from_pretrained(WHISPER_ID)
        self.wh_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_ID, dtype=torch.float16
        ).to(self.device).eval()
        self.ar_prompt_ids = self.wh_proc.get_decoder_prompt_ids(
            language="ar", task="transcribe", no_timestamps=True
        )
        # Arabic token id for Whisper LID check
        self.ar_tok_id = self.wh_proc.tokenizer.convert_tokens_to_ids("<|ar|>")
        self.sot_id = self.wh_model.config.decoder_start_token_id

        self.rescorer = ScriptureRescorer()
        print(f"Dual-Model Pipeline ready in {time.time() - t0:.2f}s!")

    def _detect_arabic_prob(self, input_features_fp16: torch.Tensor) -> tuple[float, torch.Tensor]:
        """Use Whisper encoder + 1-step decoder language token softmax to score P(Arabic)."""
        with torch.no_grad():
            enc_out = self.wh_model.get_encoder()(input_features_fp16)
            dec_in = torch.tensor([[self.sot_id]], device=self.device)
            logits = self.wh_model(
                encoder_outputs=enc_out,
                decoder_input_ids=dec_in,
            ).logits[0, 0].float()
            probs = torch.softmax(logits, dim=-1)
            ar_prob = float(probs[self.ar_tok_id].item())
        return ar_prob, enc_out

    def _whisper_arabic(self, enc_out, input_features_fp16: torch.Tensor) -> str:
        with torch.no_grad():
            ids = self.wh_model.generate(
                encoder_outputs=enc_out,
                forced_decoder_ids=self.ar_prompt_ids,
                max_new_tokens=160,
                num_beams=2,
                no_repeat_ngram_size=4,
            )
            return self.wh_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def _mms_ckb_greedy(self, seg_wav: np.ndarray) -> tuple[str, float]:
        inp = self.mms_proc(seg_wav.astype("float32"), sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.mms_model(**inp).logits[0]
            probs = torch.softmax(logits, dim=-1)
            max_p, ids = probs.max(dim=-1)
            non_blank = ids != 0
            conf = float(max_p[non_blank].mean().item()) if non_blank.any() else 0.0
        txt = self.mms_proc.decode(ids.cpu().numpy()).strip()
        return txt, conf

    def transcribe(self, audio_path: str | Path, ar_threshold: float = 0.25) -> dict:
        wav, sr = sf.read(str(audio_path))
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = AF.resample(torch.from_numpy(wav.astype("float32")).unsqueeze(0), sr, 16000).squeeze(0).numpy()

        dur = len(wav) / 16000.0
        t0 = time.time()
        segs = segment_on_pauses(wav, sr=16000)

        parts = []
        routing_log = []
        for st, en, seg_wav in segs:
            ckb_txt, ckb_conf = self._mms_ckb_greedy(seg_wav)
            sorani_chars = sum(1 for c in ckb_txt if c in SORANI_ONLY_CHARS)
            sorani_ratio = sorani_chars / max(1, len(ckb_txt.replace(" ", "")))

            inp_wh = self.wh_proc(seg_wav, sampling_rate=16000, return_tensors="pt").input_features.to(
                self.device, dtype=torch.float16
            )
            ar_prob, enc_out = self._detect_arabic_prob(inp_wh)

            # Route to Arabic if Whisper detects Arabic AND Sorani-specific char density is near zero
            use_ar = (ar_prob >= ar_threshold and sorani_ratio <= 0.08) or (ar_prob >= 0.80 and ckb_conf < 0.85)
            if use_ar:
                ar_txt = self._whisper_arabic(enc_out, inp_wh)
                chosen = self.rescorer.rescore(ar_txt)
                lang = "AR"
            else:
                chosen = self.rescorer.rescore(ckb_txt)
                lang = "CKB"

            if chosen:
                parts.append(chosen)
            routing_log.append(f"[{st:.1f}-{en:.1f}s] {lang} (ar={ar_prob:.2f}, sor={sorani_ratio:.2f}, conf={ckb_conf:.2f})")

        final_text = " ".join(parts)
        return {
            "audio": str(audio_path),
            "duration_sec": round(dur, 2),
            "total_time_sec": round(time.time() - t0, 2),
            "segments": len(segs),
            "routing": routing_log,
            "transcription": final_text,
        }


if __name__ == "__main__":
    import re
    clips = [
        ("test_1min_unseen", "data/mela_halo/test_1min_unseen.wav", "data/mela_halo/test_1min_unseen_transcription(correct-fully).txt"),
        ("test_tawhid_part_4b", "data/mela_halo/test_unseen_tawhid_part4b.wav", "data/mela_halo/test_tawhid_part_4b_transcription(correct-fully).txt"),
        ("test_lecture_2", "data/mela_halo/test_1min_unseen_lecture2.wav", "data/mela_halo/test_lecture_2_transcription(fully-correct).txt"),
        ("test_shia_lecture", "data/mela_halo/test_unseen_shia_lecture.wav", "data/mela_halo/test_shia_lecture_transcription(correct-fully).txt"),
    ]

    def norm(t):
        t = re.sub(r"\[\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\]\s*", "", t)
        t = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", t)
        t = re.sub(r"[،؛؟.!:«»﴿﴾\"()\-]", " ", t)
        return " ".join(t.split())

    def lev(a, b):
        dp = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            ndp = [i]
            for j, y in enumerate(b, 1):
                ndp.append(min(dp[j] + 1, ndp[j - 1] + 1, dp[j - 1] + (x != y)))
            dp = ndp
        return dp[-1]

    tr = DualLangTranscriber(device="mps")
    for name, wav_p, ref_p in clips:
        res = tr.transcribe(wav_p)
        ref = norm(open(ref_p).read())
        hyp = norm(res["transcription"])
        wer = lev(ref.split(), hyp.split()) / max(1, len(ref.split())) * 100
        cer = lev(list(ref), list(hyp)) / max(1, len(ref)) * 100
        print(f"\n=== {name}: WER={wer:.1f}% CER={cer:.1f}% ({res['total_time_sec']}s, {res['segments']} segs) ===")
        for rlog in res["routing"]:
            print(" ", rlog)
        print("HYP:", res["transcription"])
