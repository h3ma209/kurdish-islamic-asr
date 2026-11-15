#!/usr/bin/env python3
import json
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import soundfile as sf
import torch
import torchaudio.functional as AF
from scripts.infer_custom_mms import load_custom_mms_model, ctc_decode

def main():
    model, adapter, inv_vocab = load_custom_mms_model()

    test_ids = [
        "BJgnqMWSnhw_0004",         # Arabic Ayah
        "BJgnqMWSnhw_0006",         # Ayat al-Kursi
        "qhFaSCn-k0Y_3min_0001",    # Book intro
        "qhFaSCn-k0Y_part2_0010",   # Fast Kurdish
        "qhFaSCn-k0Y_part3_0026",   # Arabic grammar
        "qhFaSCn-k0Y_part3_0042",   # Pure Arabic quote
        "qhFaSCn-k0Y_part4a_0002",  # La ilaha illa Allah
        "qhFaSCn-k0Y_part4a_0020",  # Wujooha qawmika
        "qhFaSCn-k0Y_part4b_0035",  # Tawhid definition
        "qhFaSCn-k0Y_part4b_0054",  # Shirk definition
    ]

    records = {}
    with open("data/dataset/dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] in test_ids:
                records[item["id"]] = item

    print("\n" + "=" * 70)
    print("FORENSIC MMS-1B INFERENCE AUDIT ON 10 TARGET SEGMENTS")
    print("=" * 70 + "\n")

    for tid in test_ids:
        if tid not in records:
            continue
        rec = records[tid]
        wav_path = Path("data/dataset") / rec["audio_file"]
        wav, sr = sf.read(str(wav_path))
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        t = torch.from_numpy(wav.astype("float32")).unsqueeze(0)
        if sr != 16000:
            wav_16k = AF.resample(t, sr, 16000).squeeze(0)
        else:
            wav_16k = t.squeeze(0)
            
        with torch.no_grad():
            inputs = wav_16k.unsqueeze(0).to("mps")
            outputs = model.wav2vec2(inputs)
            logits = adapter(outputs.last_hidden_state)
            logits[..., 0] -= 1.0  # blank penalty
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
        hyp = ctc_decode(pred_ids, inv_vocab)
        ref = rec["text"]
        
        print(f"[{tid}] ({rec.get('language')})")
        print(f"REF: {ref}")
        print(f"HYP: {hyp}")
        print("-" * 70)

if __name__ == "__main__":
    main()
