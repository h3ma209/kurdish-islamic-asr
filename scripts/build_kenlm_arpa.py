#!/usr/bin/env python3
"""
Build high-precision ARPA 3-gram Language Model for Kurdish Sorani + Classical Arabic Quran & Hadith.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import kenlm

STRIP_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640\uf072\ufeff\u200C-\u200F﴿﴾«»ﷺ]")
EXTRA_ARABIC_TOKENS = ("ذ", "ض", "ظ", "ص", "ط", "ة", "أ", "إ", "آ")


def _load_allowed_chars() -> set[str]:
    vocab_path = BASE_DIR / "data" / "models" / "mms_1b_islamic_kurdish" / "vocab.json"
    chars = set(EXTRA_ARABIC_TOKENS)
    if vocab_path.exists():
        v = json.loads(vocab_path.read_text(encoding="utf-8"))
        for k in v:
            if len(k) == 1 and k != "|":
                chars.add(k)
    return chars


ALLOWED_CHARS = _load_allowed_chars()


def clean_sentence(text: str) -> list[str]:
    """Preserves both Kurdish Sorani and Classical Arabic orthography; strips Harakat/punctuation."""
    cleaned = re.sub(r"\[\[.*?\]\]|<[^>]+>", " ", text)
    cleaned = STRIP_RE.sub("", cleaned).replace("ٱ", "ا")
    cleaned = re.sub(r"[«»\"()\[\]{},،؛;؟?!—–.:\-_/\\<>#*~`…\d٠-٩]+", " ", cleaned)
    words = []
    for w in cleaned.split():
        w = w.strip()
        if w and all(ch in ALLOWED_CHARS for ch in w):
            words.append(w)
    return words


def collect_corpus() -> list[list[str]]:
    corpus = []

    # 1. Lecture & Quran datasets (weighted x3 for preacher + recitation vocabulary)
    for jname in ("dataset.jsonl", "quran_arabic.jsonl"):
        jpath = BASE_DIR / "data" / "dataset" / jname
        if not jpath.exists():
            continue
        with open(jpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    words = clean_sentence(obj["text"])
                    if words:
                        for _ in range(3):
                            corpus.append(words)

    # 2. All 6,236 Quran Ayahs + Hadiths + Dhikr
    db_path = BASE_DIR / "data" / "quran" / "kurdish_tafsirs.db"
    if not db_path.exists():
        db_path = BASE_DIR / "data" / "quran" / "scriptures.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        for row in c.execute("SELECT text FROM ayah_text"):
            words = clean_sentence(row[0])
            if words:
                corpus.append(words)
        for row in c.execute("SELECT ar_desc FROM farmuda"):
            if row[0]:
                words = clean_sentence(row[0])
                if words:
                    corpus.append(words)
        for row in c.execute("SELECT ardhikr FROM dhikr"):
            if row[0]:
                words = clean_sentence(row[0])
                if words:
                    corpus.append(words)
        conn.close()

    # 3. Classical Arabic Tafsirs (Ibn Kathir, Qurtubi, Tabari) for Arabic n-gram priors
    for tf in sorted(glob.glob(str(BASE_DIR / "data" / "quran" / "classical_tafsirs" / "*.json"))):
        try:
            data = json.loads(Path(tf).read_text(encoding="utf-8"))
        except Exception:
            continue
        texts = data.values() if isinstance(data, dict) else data
        for raw in texts:
            if not isinstance(raw, str):
                continue
            for seg in re.split(r"[.!؟\n]+", raw):
                words = clean_sentence(seg)
                if len(words) >= 3:
                    corpus.append(words)

    return corpus

def build_arpa(corpus: list[list[str]], output_path: Path):
    unigram_counts = collections.Counter()
    bigram_counts = collections.Counter()
    trigram_counts = collections.Counter()
    
    for sentence in corpus:
        padded = ["<s>"] + sentence + ["</s>"]
        for w in padded:
            unigram_counts[w] += 1
        for b in zip(padded[:-1], padded[1:]):
            bigram_counts[b] += 1
        for t in zip(padded[:-2], padded[1:-1], padded[2:]):
            trigram_counts[t] += 1

    total_unigrams = sum(unigram_counts.values())

    # Filter rare n-grams to keep model tight and fast
    vocab = set(unigram_counts.keys())
    vocab.add("<unk>")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ARPA Header
    lines = [
        "\\data\\",
        f"ngram 1={len(vocab)}",
        f"ngram 2={len(bigram_counts)}",
        f"ngram 3={len(trigram_counts)}",
        "",
        "\\1-grams:",
    ]

    # 1-grams: log10(prob) word [backoff]
    for w in sorted(vocab):
        if w in ("<s>", "<unk>"):
            prob = -99.0
            backoff = -0.4
        elif w == "</s>":
            prob = math.log10(max(unigram_counts[w], 1) / total_unigrams)
            backoff = 0.0
        else:
            prob = math.log10(max(unigram_counts[w], 1) / total_unigrams)
            backoff = -0.4
        lines.append(f"{prob:.4f}\t{w}\t{backoff:.4f}")

    lines.append("")
    lines.append("\\2-grams:")
    for (w1, w2), count in bigram_counts.items():
        w1_count = unigram_counts[w1]
        prob = math.log10(count / w1_count)
        backoff = -0.3
        lines.append(f"{prob:.4f}\t{w1} {w2}\t{backoff:.4f}")

    lines.append("")
    lines.append("\\3-grams:")
    for (w1, w2, w3), count in trigram_counts.items():
        w12_count = bigram_counts[(w1, w2)]
        prob = math.log10(count / w12_count)
        lines.append(f"{prob:.4f}\t{w1} {w2} {w3}")

    lines.append("")
    lines.append("\\end\\")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"ARPA model written to {output_path} ({len(vocab)} unigrams, {len(bigram_counts)} bigrams, {len(trigram_counts)} trigrams)")

def main():
    print("Collecting Kurdish + Islamic Arabic corpus...")
    corpus = collect_corpus()
    print(f"Collected {len(corpus)} sentences across dataset, Quran, Hadith & Tafsir.")
    
    out_path = BASE_DIR / "data" / "models" / "mms_1b_islamic_kurdish" / "kurdish_islamic_3gram.arpa"
    build_arpa(corpus, out_path)
    
    print("Verifying loading into KenLM C++ engine...")
    lm = kenlm.Model(str(out_path))
    test_phrase = "لَا إِلَٰهَ إِلَّا اللَّهُ"
    score = lm.score(test_phrase)
    print(f"Test phrase: \"{test_phrase}\" -> KenLM Score: {score:.2f}")

if __name__ == "__main__":
    main()
