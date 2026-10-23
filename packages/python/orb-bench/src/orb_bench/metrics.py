"""Word error rate and simple statistics.

WER = (substitutions + deletions + insertions) / reference words, after normalizing both sides. English uses the
Whisper English normalizer (lower case, no punctuation, no fillers such as "um", numbers as digits), the same as
scripts/stt_bench.py and decision 0005. Other languages (Hindi) get lower case with punctuation and symbols removed.

The written-form metrics (fwer, entity, punct_f1, added_words) apply no normalizer: they see case, punctuation and
how numbers are written.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

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


# ---- written-form metrics: no normalizer, so case, punctuation and number formatting count ----

_TOKEN = re.compile(r"""
    (?<![\w-])--?(?=[A-Za-z])[\w-]*[A-Za-z0-9]   # a command flag: -b, --watch
  | [^\W_]+(?:[.,:'’/@_-][^\W_]+)*                # a word; joiners inside it keep emails, 3:30, 12,450.30 and paths whole
  | [^\w\s]                                       # any other mark, one token each
""", re.VERBOSE)
PUNCT = frozenset(".,?!;:")
_STOPWORDS = frozenset("""a an the and or but of to in on at for from by with as is are was were be been it its this that
    these those i you he she we they me my your our their so if then than not no yes do does did have has had will would
    can could should just""".split())


def written_tokens(text: str) -> list[str]:
    """Tokens that keep case. Punctuation marks are their own tokens, except inside a word (priya.sharma@x.ai, 3:30)."""
    return _TOKEN.findall(text or "")


def fscore(ref: str, hyp: str) -> tuple[int, int]:
    """(edits, reference tokens) on written tokens: case and punctuation count, no normalizer."""
    r = written_tokens(ref)
    return edits(r, written_tokens(hyp)), len(r)


def entity_hits(keys: list[str], hyp: str) -> tuple[int, int]:
    """(keys found, keys). A key counts when it appears exactly (same case) and is not part of a longer word."""
    found = 0
    for k in keys:
        left = r"(?<![^\W_])" if re.match(r"[^\W_]", k) else ""
        right = r"(?![^\W_])" if re.search(r"[^\W_]$", k) else ""
        found += bool(re.search(left + re.escape(k) + right, hyp or ""))
    return found, len(keys)


def _words_and_marks(text: str) -> tuple[list[str], list[tuple[int, str]]]:
    """Lower-case words, and each punctuation mark as (index of the word it follows, mark)."""
    words, marks = [], []
    for t in written_tokens(text):
        if t in PUNCT:
            marks.append((len(words) - 1, t))
        elif any(c.isalnum() for c in t):
            words.append(t.lower())
    return words, marks


def _align(ref: list[str], hyp: list[str]) -> dict[int, int]:
    """Map the index of each hypothesis word that is matched or substituted to its reference word."""
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, mapping = len(ref), len(hyp), {}
    while i and j:
        if d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            mapping[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return mapping


def punct_counts(ref: str, hyp: str) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) of the marks . , ? ! ; : after the right word.

    The words of both texts are aligned first, so a wrong word elsewhere does not move a mark.
    """
    rw, rm = _words_and_marks(ref)
    hw, hm = _words_and_marks(hyp)
    mapping = _align(rw, hw)
    want = Counter(rm)
    got = Counter((mapping.get(i, ("x", i)), m) if i >= 0 else (-1, m) for i, m in hm)
    tp = sum((want & got).values())
    return tp, sum(got.values()) - tp, sum(want.values()) - tp


def f1(tp: int, fp: int, fn: int) -> float | None:
    """None when there are no marks on either side."""
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None


def punct_f1(ref: str, hyp: str) -> float | None:
    return f1(*punct_counts(ref, hyp))


def content_words(text: str) -> set[str]:
    """Lower-case letter runs of the words (camelCase and dotted names split), without function words and without
    tokens that start with a digit (3rd, 3:30), so digits and symbols are not content."""
    out = set()
    for tok in (text or "").split():
        if re.match(r"\W*\d", tok):
            continue
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tok)
        out |= {w.lower() for w in re.findall(r"[A-Za-z]+", spaced)} - _STOPWORDS
    return out


def _token_words(text: str):
    """(token, camel-split content words, lower-case letter runs) for each token that is not a number."""
    for tok in (text or "").split():
        if re.match(r"\W*\d", tok):
            continue
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tok)
        yield (tok, {w.lower() for w in re.findall(r"[A-Za-z]+", spaced)} - _STOPWORDS,
               {w.lower() for w in re.findall(r"[A-Za-z]+", tok)} - _STOPWORDS)


def added_words(stage_in: str, stage_out: str, ref: str) -> list[str]:
    """Content words of a stage's output that are in neither its input nor the reference. A good stage keeps this empty.

    Case and joining do not make a word new: "DevSlabs.aI" -> "devslabs.ai" and "use effect" -> "useEffect" add none.
    A word is known when it is a camelCase part or a whole letter run of any token of the input or the reference.
    """
    known = set()
    for text in (stage_in, ref):
        for _, parts, runs in _token_words(text):
            known |= parts | runs
    out = set()
    for _, parts, runs in _token_words(stage_out):
        if runs <= known or parts <= known:
            continue
        out |= (parts if parts - known else runs) - known
    return sorted(out)


def row_scores(ref: str, hyp: str, lang: str = "en", written: bool = False, keys: list[str] | None = None) -> dict:
    """The scores of one clip: `e`, `n` (normalized WER parts); for written-form refs also `fw` (fwer edits and ref
    tokens), `punct` (tp, fp, fn) and, when the clip has keys, `entity` (found, total)."""
    e, n = score(ref, hyp, lang) if ref else (0, 0)
    out = {"e": e, "n": n}
    if written and ref:
        out["fw"] = list(fscore(ref, hyp))
        out["punct"] = list(punct_counts(ref, hyp))
        if keys:
            out["entity"] = list(entity_hits(keys, hyp))
    return out
