#!/usr/bin/env python3
"""
Islamic Scripture & Honorifics Rescorer for Kurdish Sorani Preacher Transcripts.
Restores exact Classical Arabic Quran Ayahs, Hadiths, and Azkar with full Tashkeel (Harakat)
and standardizes Islamic Kurdish honorifics.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
_FULL_DB = BASE_DIR / "data" / "quran" / "kurdish_tafsirs.db"
_COMPACT_DB = BASE_DIR / "data" / "quran" / "scriptures.db"
DB_PATH = _FULL_DB if _FULL_DB.exists() else _COMPACT_DB

# Common Kurdish-only function words that never begin/end an Arabic scripture span
KURDISH_BOUNDARY_STOPWORDS = frozenset({
    "ئەو", "ئەم", "ئەمانە", "ئەوان", "ئێمە", "ئێوە", "ئەوەی", "ئەمەی",
    "چونکە", "بەڵام", "بۆیە", "کەواتە", "هەموو", "هیچ", "زۆر", "کەم",
    "لەبەر", "لەسەر", "لەگەڵ", "بەسەر", "بەبێ", "پێش", "دوای",
    "ئەبێ", "ئەبێت", "دەبێ", "دەبێت", "نییە", "هەیە", "بوو", "بێت",
    "ئەکا", "ئەکات", "دەکا", "دەکات", "ئەڵێ", "ئەڵێت", "دەڵێ", "دەڵێت",
    "ئەفەرموێ", "ئەفەرموێت", "فەرمووی", "فەرموویەتی", "واتە", "یانی",
    "کە", "و", "بۆ", "لە", "بە", "تا", "یان", "یا", "خۆ", "خۆی", "خۆت",
    "ئاوا", "ئێستا", "ئێستە", "پاشان", "ئینجا", "هەروەها", "تەنانەت",
})

# Common spoken Arabic formulas (Salawat, Duas, Istighfar, Tasbih) not always in ayah_text
ISLAMIC_FORMULAS: list[tuple[str, str, str]] = [
    ("formula", "صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ", "صلوات"),
    ("formula", "عَلَيْهِ الصَّلَاةُ وَالسَّلَامُ", "صلوات"),
    ("formula", "اللَّهُمَّ صَلِّ وَسَلِّمْ عَلَى نَبِيِّنَا مُحَمَّدٍ وَعَلَى آلِهِ وَأَصْحَابِهِ وَمَنْ وَالَاهُ", "صلوات"),
    ("formula", "اللَّهُمَّ صَلِّ عَلَى مُحَمَّدٍ وَعَلَى آلِ مُحَمَّدٍ", "صلوات"),
    ("formula", "سُبْحَانَهُ وَتَعَالَى", "تقديس"),
    ("formula", "عَزَّ وَجَلَّ", "تقديس"),
    ("formula", "تَبَارَكَ وَتَعَالَى", "تقديس"),
    ("formula", "رَضِيَ اللَّهُ عَنْهُ", "ترضي"),
    ("formula", "رَضِيَ اللَّهُ عَنْهَا", "ترضي"),
    ("formula", "رَضِيَ اللَّهُ عَنْهُمْ", "ترضي"),
    ("formula", "رَحِمَهُ اللَّهُ", "ترحم"),
    ("formula", "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ", "بسملة"),
    ("formula", "أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ", "استعاذة"),
    ("formula", "لَا إِلَٰهَ إِلَّا اللَّهُ", "تهليل"),
    ("formula", "لَا إِلَٰهَ إِلَّا اللَّهُ وَحْدَهُ لَا شَرِيكَ لَهُ", "تهليل"),
    ("formula", "أَسْتَغْفِرُ اللَّهَ الْعَظِيمَ", "استغفار"),
    ("formula", "لَا حَوْلَ وَلَا قُوَّةَ إِلَّا بِاللَّهِ", "حوقلة"),
    ("formula", "حَسْبُنَا اللَّهُ وَنِعْمَ الْوَكِيلُ", "حسبلة"),
    ("formula", "إِنَّا لِلَّهِ وَإِنَّا إِلَيْهِ رَاجِعُونَ", "استرجاع"),
    ("formula", "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ", "حمدلة"),
]


def fold_phonetic(s: str) -> str:
    """Aggressive phonetic folding mapping Kurdish speech letters to Arabic Quran phonemes."""
    s = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u200C-\u200F]", "", s)
    s = re.sub(r"[.,،؛:!?؟﴿﴾«»\(\)\[\]\"'ـ]", "", s)
    s = re.sub(r"[أإآٱئؤء]", "ا", s)
    s = s.replace("ة", "ه").replace("ى", "ي")
    s = s.replace("ک", "ك").replace("گ", "ك")
    s = s.replace("ی", "ي").replace("ێ", "ي")
    s = s.replace("ۆ", "و")
    s = s.replace("ڕ", "ر").replace("ڵ", "ل")
    s = s.replace("ڤ", "ف").replace("چ", "ج").replace("ژ", "ز").replace("پ", "ب")
    s = s.replace("ذ", "ز").replace("ظ", "ز").replace("ض", "ز")
    s = s.replace("ث", "س").replace("ص", "س")
    s = s.replace("ط", "ت")
    s = s.replace("ە", "")
    s = re.sub(r"\s+", "", s)
    # Collapse gemination/doubled letters (e.g. ڕەببەنا -> ربنا, زەککاها -> زكها)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def clean_for_matching(text: str) -> str:
    """Strips Tashkeel and normalizes Arabic/Kurdish characters for matching."""
    # Remove diacritics & Quranic symbols
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u200C-\u200F]", "", text)
    # Normalize alifs
    text = re.sub(r"[أإآٱ]", "ا", text)
    # Normalize taa marbuta
    text = text.replace("ة", "ه").replace("ى", "ي")
    # Normalize Kurdish kaf/ye/waw to standard Arabic
    text = text.replace("ک", "ك").replace("ی", "ي").replace("ێ", "ي").replace("ۆ", "و").replace("ڕ", "ر").replace("ڵ", "ل")
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ScriptureRescorer:
    def __init__(self, db_path: Path = DB_PATH):
        self.scriptures: list[dict] = []
        self.trigram_index: dict[str, list[int]] = defaultdict(list)
        try:
            from rapidfuzz import fuzz
            self.fuzz = fuzz
            self.has_rapidfuzz = True
        except ImportError:
            self.fuzz = None
            self.has_rapidfuzz = False
        self._load_scriptures(db_path)
        self._build_phonetic_index()

    def _add_scripture(self, kind: str, ref: str, harakat_text: str, min_words: int = 2):
        t = harakat_text.strip().rstrip(":：")
        cleaned = clean_for_matching(t)
        words = cleaned.split()
        if len(words) < min_words:
            return
        harakat_words = t.split()
        # Keep alignment between cleaned words and harakat words when possible
        if len(harakat_words) != len(words):
            harakat_words = t.split()
        folded_words = [fold_phonetic(w) for w in words]
        folded_full = fold_phonetic("".join(words))
        if len(folded_full) < 8:
            return
        self.scriptures.append({
            "kind": kind,
            "ref": ref,
            "harakat": t,
            "harakat_words": harakat_words,
            "cleaned": cleaned,
            "words": words,
            "folded_words": folded_words,
            "folded_full": folded_full,
        })

    def _load_scriptures(self, db_path: Path):
        # 0. Common Islamic Formulas
        for kind, text, ref in ISLAMIC_FORMULAS:
            self._add_scripture(kind, ref, text, min_words=2)

        if not db_path.exists():
            return
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # 1. Quran Ayahs (6,236)
        cursor.execute("SELECT suraId, ayah, text FROM ayah_text")
        for sura_id, ayah_num, ayah_text in cursor.fetchall():
            if ayah_text:
                self._add_scripture("quran", f"سورة {sura_id}:{ayah_num}", ayah_text, min_words=2)

        # 2. Hadith (farmuda)
        cursor.execute("SELECT ar_desc FROM farmuda")
        for (ar_text,) in cursor.fetchall():
            if ar_text:
                for match in re.findall(r"\[(.*?)\]", ar_text):
                    self._add_scripture("hadith", "فەرموودە", match, min_words=3)

        # 3. Dhikr & Duas
        cursor.execute("SELECT ardhikr FROM dhikr")
        for (ar_text,) in cursor.fetchall():
            if ar_text:
                self._add_scripture("dhikr", "ویرد", ar_text, min_words=3)

        conn.close()
        # Sort scriptures longest-first for greedy matching
        self.scriptures.sort(key=lambda s: len(s["words"]), reverse=True)

    def _build_phonetic_index(self):
        self.trigram_index.clear()
        for idx, item in enumerate(self.scriptures):
            ff = item["folded_full"]
            if len(ff) < 3:
                continue
            seen = set()
            for k in range(len(ff) - 2):
                tg = ff[k : k + 3]
                if tg not in seen:
                    seen.add(tg)
                    self.trigram_index[tg].append(idx)

    def rescore_honorifics(self, text: str) -> str:
        """Standardizes spoken contractions into proper Islamic honorifics."""
        salawat_patterns = [
            (r"\b(رەسول|ڕەسول|رەسوڵ|ڕەسوڵ|رەسولەڵلا|ڕەسووڵ|رسول)\s+(له|الە|الهی|اللە|الله|ئەڵڵا|ئەڵا)?\s*(سڵسەلم|سسەلن|سەلاسەلەم|سەلا سەلەم|سلسل\s*م?|سەلەم|سەلام|سلی\s+الله\s+(?:مالی|علیه|علی)?\s*وسلم)\b", "رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ"),
            (r"\b(پێغەمبەر|پێغەمبەری خودا)\s*(سڵسەلم|سسەلن|سەلاسەلەم|سلسل\s*م?|سەلەم|سەلام)\b", "پێغەمبەر صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ"),
            (r"\bرَسُولُ اللَّهِ\s*(سڵسەلم|سسەلن|سەلاسەلەم|سلسل\s*م?|سەلەم|سەلام)\b", "رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ"),
            (r"\b(سلی\s+الله\s+(?:مالی|علیه|علی|عەلەیهی)?\s*(?:و)?سلم|صلی الله علیە وسلم|دروودی خوای لێ بێت|سەلامی خوای لێ بێ)\b", "صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ"),
            (r"\b(رسول|رەسول|ڕەسول)\s+(علی|عەلی|عەلەیهی)\s+(سەلاتی?\s+و\s+سەلام|سەڵاتی?\s+و\s+سەلام)\b", "رَسُولُ اللَّهِ عَلَيْهِ الصَّلَاةُ وَالسَّلَامُ"),
            (r"\b(رسول|رەسول)\s+(له|الهی|الە)\b", "رَسُولُ اللَّهِ"),
            (r"\b(سوبحانەئ?\s*و\s*تەعالا|سوبحانەئ?\s*وەتەعالە?|سوبحانەوتەعالا|سوبحانو\s*تەعلا|سبحانە\s*و\s*تەعالا|سبحان\s*و\s*تەعالا|سبحانه\s+ئەتی\s+علا|سبحانه\s*و\s*تعالی)\b", "سُبْحَانَهُ وَتَعَالَى"),
            (r"\b(لا\s+ئیلاه\s+لە\s+لای|لا\s+ئیلاهی\s+لەڵە|لا\s+اله\s+لا|لَا\s*إِلَٰهَ\s*إِلَّا\s*اللَّهُ|لا\s*إله\s*إلا\s*الله|لا\s+اله\s+الا\s+الله)\b", "لَا إِلَٰهَ إِلَّا اللَّهُ"),
            (r"\bسەلەفی ساڵحین\b", "سەلەفی ساڵح"),
            (r"\b(عەززە و جەلل|عەززەوەجەلل|عەزەوەجەل)\b", "عَزَّ وَجَلَّ"),
            (r"\b(ڕەزای خوای لێ بێت|ڕەزای خوای لێبێت)\b", "رَضِيَ اللَّهُ عَنْهُ"),
        ]
        res = text
        for pat, repl in salawat_patterns:
            res = re.sub(pat, repl, res)
        res = re.sub(r"(وَسَلَّمَ|وَالسَّلَامُ|وَتَعَالَى|عَنْهُ|اللَّهُ)(?=[^\s،.؟!﴿﴾«»])", r"\1 ", res)
        return res

    def rescore_spoken_quran_patterns(self, text: str) -> str:
        """Restores common spoken preacher recitations directly to authentic Uthmani Quran text."""
        patterns = [
            # Surah Sad 38:5: أجعَل الآلهة إلها واحدا
            (
                r"(?:\b)(جە\s*حەل|جا\s*حەل|جەعەلە|جعەل|جعل|ئەجەعەل|ئەجەعەلە)\s*(لە|ل|لئالی|لالی|لە\s*ئالی)?\s*(ئالی\s*هەتا|ئالی\s*هەتە|ئالیهەتا|ئالی\s*هە|ئالیهە|هەتا)?\s*(ئیلاهن|ئیلاهەن|ئیلاه|ئال\s*هن|ئال\s*هەن)?(\s*واحیدە|\s*واحید|\s*واحدا|\s*وا٧دە)?(?:\b)",
                "﴿أَجَعَلَ ٱلۡأٓلِهَةَ إِلَٰهٗا وَٰحِدًاۖ﴾",
            ),
            # Surah Adh-Dhariyat 51:56: وما خلقت الجن والإنس إلا ليعبدون
            (
                r"(?:\b)(وەما\s*خەلەقتول\s*جینا|وما\s*خلقت\s*الجن)\s*(وەل\s*ئینس|والإنس)?\s*(ئیللا\s*لیەعبودون|إلا\s*ليعبدون)(?:\b)",
                "﴿وَمَا خَلَقۡتُ ٱلۡجِنَّ وَٱلۡإِنسَ إِلَّا لِيَعۡبُدُونِ﴾",
            ),
            # Surah Al-Fatihah 1:5: إياك نعبد وإياك نستعين
            (
                r"(?:\b)(ئیاکە\s*نەعبودو|إياك\s*نعبد)\s*(وەئیاکە\s*نەستەعین|وإياك\s*نستعين)(?:\b)",
                "﴿إِيَّاكَ نَعۡبُدُ وَإِيَّاكَ نَسۡتَعِينُ﴾",
            ),
            # Surah Al-Ikhlas 112:1: قل هو الله أحد
            (
                r"(?:\b)(قول\s*هوەڵڵاهو\s*ئەحەد|قُل\s*هُوَ\s*اللَّهُ\s*أَحَد)(?:\b)",
                "﴿قُلۡ هُوَ ٱللَّهُ أَحَدٌ﴾",
            ),
            # Surah Muhammad 47:19: فاعلم أنه لا إله إلا الله
            (
                r"(?:\b)(فەعلەم\s*ئەنەهو|فاعلم\s*أنه)\s*(لا\s*ئیلاه|لَا\s*إِلَٰهَ)\s*(ئیللەڵڵا|إِلَّا\s*اللَّهُ)(?:\b)",
                "﴿فَٱعۡلَمۡ أَنَّهُۥ لَآ إِلَٰهَ إِلَّا ٱللَّهُ﴾",
            ),
            # Surah Al-Baqarah 2:256: لا إكراه في الدين
            (
                r"(?:\b)(لا\s*ئیکراهە\s*فیددین|لَا\s*إِكۡرَاهَ\s*فِي\s*ٱلدِّينِ)(?:\b)",
                "﴿لَآ إِكۡرَاهَ فِي ٱلدِّينِۖ﴾",
            ),
        ]
        res = text
        for pat, repl in patterns:
            res = re.sub(pat, repl, res)
        return res

    def _phonetic_candidates(self, folded_chunk: str, max_candidates: int = 25) -> list[int]:
        """Returns top scripture indices sharing 3-gram phonetic skeletons with folded_chunk."""
        n = len(folded_chunk)
        if n < 6:
            return []
        counts: dict[int, int] = defaultdict(int)
        for k in range(n - 2):
            tg = folded_chunk[k : k + 3]
            for idx in self.trigram_index.get(tg, ()):
                counts[idx] += 1
        if not counts:
            return []
        min_hits = max(3, n // 4)
        filtered = [(cnt, idx) for idx, cnt in counts.items() if cnt >= min_hits]
        if not filtered:
            return []
        filtered.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in filtered[:max_candidates]]

    def _best_subspan(self, folded_chunk: str, item: dict, threshold: float) -> tuple[float, str] | None:
        """Finds best contiguous word sub-span inside a longer scripture matching folded_chunk."""
        if self.fuzz.partial_ratio(folded_chunk, item["folded_full"]) < threshold - 3.0:
            return None
        fw = item["folded_words"]
        hw = item["harakat_words"]
        m = len(fw)
        target_len = len(folded_chunk)
        min_len = int(target_len * 0.72)
        max_len = int(target_len * 1.35)

        best_score = 0.0
        best_span = None
        for j in range(m):
            acc = []
            acc_len = 0
            for k in range(j, m):
                acc.append(fw[k])
                acc_len += len(fw[k])
                if acc_len < min_len:
                    continue
                if acc_len > max_len:
                    break
                if k - j + 1 < 3:
                    continue
                cand = "".join(acc)
                score = self.fuzz.ratio(folded_chunk, cand)
                if score > best_score:
                    best_score = score
                    best_span = (j, k + 1)
        if best_span and best_score >= threshold and best_span[1] <= len(hw):
            return best_score, " ".join(hw[best_span[0] : best_span[1]])
        return None

    def _wrap_match(self, kind: str, harakat_text: str) -> str:
        if kind == "quran":
            return f"﴿{harakat_text}﴾"
        if kind in ("hadith", "dhikr"):
            return f"«{harakat_text}»"
        return harakat_text

    def match_scriptures(self, text: str, min_words: int = 3) -> str:
        """Finds spoken Quran / Hadith / Dhikr phrases (exact + space-invariant phonetic) and restores full Harakat."""
        words = text.split()
        if len(words) < min_words:
            return text

        cleaned_words = [clean_for_matching(w) for w in words]
        matched_flags = [False] * len(words)

        # Mark already quoted spans (e.g. within ﴿...﴾ or «...») and already-restored honorifics
        in_quote = False
        honorific_tokens = frozenset({
            "صَلَّى", "اللَّهُ", "عَلَيْهِ", "وَسَلَّمَ", "عَلَيْهِ", "الصَّلَاةُ", "وَالسَّلَامُ",
            "سُبْحَانَهُ", "وَتَعَالَى", "رَضِيَ", "عَنْهُ", "رَسُولُ", "اللَّهِ", "عَزَّ", "وَجَلَّ",
        })
        for idx, w in enumerate(words):
            if "﴿" in w or "«" in w:
                in_quote = True
            if in_quote or w in honorific_tokens:
                matched_flags[idx] = True
            if "﴾" in w or "»" in w:
                in_quote = False

        folded_word_tokens = [fold_phonetic(w) for w in words] if self.has_rapidfuzz else []
        kind_priority = {"quran": 3, "formula": 2, "hadith": 1, "dhikr": 0}
        max_win = min(22, len(words))

        # Unified greedy longest-window-first loop (exact match first, then space-invariant phonetic)
        for window_size in range(max_win, min_words - 1, -1):
            for i in range(len(words) - window_size + 1):
                if any(matched_flags[i : i + window_size]):
                    continue

                # 1. Exact word-level match (when window_size <= 15)
                if window_size <= 15:
                    chunk = " ".join(cleaned_words[i : i + window_size])
                    for item in self.scriptures:
                        if chunk in item["cleaned"]:
                            scrip_cleaned_words = item["words"]
                            scrip_harakat_words = item["harakat_words"]
                            if len(scrip_cleaned_words) != len(scrip_harakat_words):
                                continue

                            for j in range(len(scrip_cleaned_words) - window_size + 1):
                                if " ".join(scrip_cleaned_words[j : j + window_size]) == chunk:
                                    matched_harakat = " ".join(scrip_harakat_words[j : j + window_size])
                                    words[i] = self._wrap_match(item["kind"], matched_harakat)
                                    for k in range(i + 1, i + window_size):
                                        words[k] = ""
                                        matched_flags[k] = True
                                    matched_flags[i] = True
                                    break
                            if matched_flags[i]:
                                break
                if matched_flags[i] or not self.has_rapidfuzz:
                    continue

                # 2. Space-invariant phonetic fuzzy match
                if words[i] in KURDISH_BOUNDARY_STOPWORDS or words[i + window_size - 1] in KURDISH_BOUNDARY_STOPWORDS:
                    continue

                folded_chunk = fold_phonetic("".join(folded_word_tokens[i : i + window_size]))
                flen = len(folded_chunk)
                if flen < 12:
                    continue

                threshold = 78.0 if flen >= 30 else (82.0 if flen >= 18 else 88.0)
                candidates = self._phonetic_candidates(folded_chunk)
                if not candidates:
                    continue

                best_key = (-1.0, -1)
                best_replacement = None

                for idx in candidates:
                    item = self.scriptures[idx]
                    ff = item["folded_full"]
                    ratio_len = flen / len(ff)
                    prio = kind_priority.get(item["kind"], 0)
                    if 0.72 <= ratio_len <= 1.35:
                        score = self.fuzz.ratio(folded_chunk, ff)
                        key = (round(score, 1), prio)
                        if score >= threshold and key > best_key:
                            best_key = key
                            best_replacement = self._wrap_match(item["kind"], item["harakat"])
                    elif len(ff) > flen * 1.25:
                        sub = self._best_subspan(folded_chunk, item, threshold)
                        if sub:
                            key = (round(sub[0], 1), prio)
                            if key > best_key:
                                best_key = key
                                best_replacement = self._wrap_match(item["kind"], sub[1])

                if best_replacement is not None:
                    words[i] = best_replacement
                    for k in range(i + 1, i + window_size):
                        words[k] = ""
                        matched_flags[k] = True
                    matched_flags[i] = True

        result = " ".join([w for w in words if w]).strip()
        return result

    def rescore_lexicon(self, text: str) -> str:
        """Standardizes Islamic terminology, authentic Divine names, and preacher ASR confusions."""
        lexicon_patterns = [
            # User specified grammar & phonetic corrections:
            (r"\bئەبەیا\b", "ئەبوایە"),
            (r"\bکامرو\b", "کە ئەمڕۆ"),
            (r"\bتلارین\b", "تەلاریان"),
            (r"\b(بەدریجەی مەجری|بەدریژای مەجری|بەدریجەی مەجرێ|بەدریژەی مەجری)\b", "بەدرێژایی مێژوو"),
            (r"\bزوور\b", "زۆر"),
            (r"\bزوورتر\b", "زۆرتر"),
            (r"\bزوورترین\b", "زۆرترین"),
            # Authentic spelling of Allah: always اللَّه
            (r"\b(ئەڵڵا|ئەڵڵەه|ئەلڵا|ئەللا|ئەڵا|اللە)\b", "اللَّه"),
            # زۆر بەیت -> زۆربەی
            (r"\b(زۆر بەیت|زۆربەیت)\b", "زۆربەی"),
            # دار و بەرد و پەڕۆیە (sacred trees, stones, and rags/cloth)
            (r"\b(دار و بەرد و پەڕی|دار و بەرد پەڕی|دار و بەردوو پەڕەوە|دار و بەرد و پەڕەوە|دار و بەرد و پەڕەی|دار و بەرد و پەڕە|دار و بەرد و پەڕۆ)\b", "دار و بەرد و پەڕۆیە"),
            # طاغوت / طەواغیت
            (r"\b(تەواوی دەکا|تەواوی دەکەن|تەواتیتەکان|تاغوتەکان)\b", "طاغوتەکان"),
            (r"\bتەواغیتەکان\b", "طەواغیتەکان"),
            # Theological & Arabic root terms
            (r"\b(ئیسباتەکە|ئیسپاتەکە)\b", "ئیثباتەکە"),
            (r"\bنەفیەکە\b", "نەفییەکە"),
            (r"\b(زەربی|زەبی)\s+سفر\b", "ضەربی سفر"),
            (r"\bحەرام و حەڵاڵیان بۆ دەڵێ\b", "حەرام و حەڵاڵیان بۆ دادەڕێژن"),
        ]
        res = text
        for pat, repl in lexicon_patterns:
            res = re.sub(pat, repl, res)
        return res

    def rescore(self, text: str) -> str:
        text = self.rescore_spoken_quran_patterns(text)
        text = self.rescore_honorifics(text)
        text = self.rescore_lexicon(text)
        text = self.match_scriptures(text)
        text = self.rescore_lexicon(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
