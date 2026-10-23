"""Word error rate and simple statistics.

WER = (substitutions + deletions + insertions) / reference words, after normalizing both sides. English uses the
Whisper English normalizer (lower case, no punctuation, no fillers such as "um", numbers as digits), the same as
scripts/stt_bench.py and decision 0005. Other languages (Hindi) get lower case with punctuation and symbols removed.
"""
from __future__ import annotations

import unicodedata

_english = None


def normalize(text: str, lang: str = "en") -> str:
    global _english
    if lang.startswith("en"):
        if _english is None:
            from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
            _english = EnglishTextNormalizer({})
        return _english(text or "")
    kept = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in (text or "").lower())
    return " ".join(kept.split())  # Devanagari vowel signs (category M) stay, so words stay whole


def edits(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def score(ref: str, hyp: str, lang: str = "en") -> tuple[int, int]:
    """(edits, reference words) after normalizing."""
    r, h = normalize(ref, lang).split(), normalize(hyp, lang).split()
    return edits(r, h), len(r)


def wer(pairs: list[tuple[int, int]]) -> float | None:
    words = sum(n for _, n in pairs)
    return sum(e for e, _ in pairs) / words if words else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)
