# Kurdish Islamic ASR (`kurdish-islamic-asr`)

Standalone local Central Kurdish (Sorani, `ckb`) + Classical Arabic Islamic speech-to-text model (**zero LLM**).

## Architecture
1. **Acoustic Model**: Meta `facebook/mms-1b-all` (`ckb` adapter) fine-tuned on Top-24 transformer layers (`24..47`) + CTC `lm_head` (`data/models/mms_1b_islamic_kurdish/finetuned_ckb_adapter.pt`, ~4.7 MB).
2. **Language Model**: 3-gram Sorani Islamic KenLM ARPA (`data/models/mms_1b_islamic_kurdish/kurdish_islamic_3gram.arpa`, ~7.5 MB) with beam search decoding + frame-aligned word de-gluing (`scripts/infer_custom_mms.py`).
3. **Scripture & Harakat Rescorer**: Space-invariant 3-gram phonetic fuzzy matcher (`scripts/scripture_rescorer.py`) backed by `data/quran/scriptures.db` (`6,236` Quran ayahs, Hadiths, Azkar) + common Islamic formulas (`صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ`, `سُبْحَانَهُ وَتَعَالَى`, etc.) to restore spoken phonetic Arabic spans into authentic Uthmani Matn with full Tashkeel (Harakat).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/infer_custom_mms.py path/to/audio.wav
```

- First run downloads the base `facebook/mms-1b-all` weights from Hugging Face (`~4 GB` cached locally), then loads the included `ckb` fine-tuned adapter, KenLM ARPA, and scripture index.
- Runs on Apple Silicon (`mps`), CUDA (`cuda`), or `cpu`.

## Key Files
- `scripts/infer_custom_mms.py` — standalone inference CLI
- `scripts/scripture_rescorer.py` — phonetic Quran/Hadith/Honorifics harakat rescorer
- `scripts/finetune_mms_ckb.py` — Top-24 MMS-1B adapter + SpecAugment training script
- `scripts/build_kenlm_arpa.py` — KenLM 3-gram ARPA builder aligned to MMS `ckb` vocab
- `data/models/mms_1b_islamic_kurdish/` — fine-tuned adapter weights, vocab, and 3-gram ARPA
- `data/quran/scriptures.db` — compact SQLite index of Quran ayahs, hadiths, and azkar
