"""Vocabulary: put the user's own words (identifiers, names, product words) back into a transcript.

`correct(text, terms)` is a pure, deterministic and idempotent function of its two arguments. It never touches audio,
a model or the network. It reads word n-grams (1 to 3 words) of the text and replaces a span by a term when all of
these hold (design 4.4, decision 0011):
- the phonetic key of the span matches the key of the term, or differs by one edit (keys of 4 or more characters);
- the edit distance of the letters (case, spaces and punctuation ignored) is at most 34 % of the longer side, and the
  terms of 4 letters or fewer must match exactly;
- the span is not an ordinary English word (about 12,500 words of SCOWL size 20), unless the term or the span has several words;
- a span of ordinary words only must spell the term exactly ("use effect" gives `useEffect`, "like it" never `LiveKit`);
- the span has no extra word: dropping its first or its last word does not make it match the term as well.

The term is written as given, so "use effect" gives `useEffect`, "post gress" and "poster's" give `Postgres`. Text in
double quotes or backticks, emails, URLs, paths, `#N`, money and times are never touched. A term that is already in
the text stays as it is, and a term is never changed by another term, which makes the function idempotent.

Phonetic key: a compact English key in the spirit of Metaphone (no jellyfish or metaphone package is installed).
"""
from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MAX_TERMS = 200            # per source (the team file, the request header); the merged list has at most 2 sources
MAX_TERM_CHARS = 64
MAX_TOTAL_BYTES = 4096     # the sum of the terms of one source, in UTF-8
MAX_FILE_BYTES = 65536     # the team file as read: a bigger file is a mistake, not a vocabulary
MAX_SPAN = 3               # words
MAX_DISTANCE = 0.34        # edits / letters of the longer side; "day or base" -> database is 3 of 9 = 0.33
MIN_LETTERS = 2            # a term with fewer letters or digits never matches
FUZZY_MIN_LETTERS = 5      # below this, only the same letters match ("oath" and "OAuth" are 4 and 5: one edit)

COMMON_WORDS_FILE = Path(__file__).parent / "data" / "common_words.txt"
PROTECTED = re.compile(r'("[^"\n]*"|“[^”\n]*”|`[^`\n]*`)')
TOKEN = re.compile(r"(\s*)(\S+)")
SPLIT = re.compile(r"^([(\[{\"“‘]*)(.*?)([.,;:!?)\]}\"”’…]*)$", re.S)
PLAIN = re.compile(r"^[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*$")
DOTTED = re.compile(r"^[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+$")
NAME_WORD = re.compile(r"^[A-Za-z]{2,}$")
COMMENT = re.compile(r"(?:^|\s)#.*$")


class VocabularyError(ValueError):
    """A vocabulary that breaks a limit. The message says which rule, and never repeats a term."""


@functools.lru_cache(maxsize=1)
def common_words() -> frozenset[str]:
    lines = COMMON_WORDS_FILE.read_text(encoding="utf-8").splitlines()
    return frozenset(w for w in (line.strip() for line in lines) if w and not w.startswith("#"))


# --- phonetic key and edit distance -------------------------------------------------------------------------------

INITIAL = (("kn", "n"), ("gn", "n"), ("pn", "n"), ("wr", "r"), ("ps", "s"), ("wh", "w"), ("x", "s"))
SIMPLE = {"b": "B", "d": "D", "f": "F", "j": "J", "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "q": "K",
          "r": "R", "v": "F", "z": "S"}
VOWELS = "aeiouy"


@functools.lru_cache(maxsize=8192)
def phonetic_key(letters: str) -> str:
    """A key that equal-sounding English spellings share: vowels (except a first one), `h`, `w` and double letters
    drop out, and `ph`, `ck`, `sh`, `ch`, `th`, soft `c` and `g`, `x`, `q`, `v` and `z` are mapped to one sound each.
    The input is lower case letters and digits."""
    s = letters
    for old, new in INITIAL:
        if s.startswith(old):
            s = new + s[len(old):]
            break
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c, nxt = s[i], s[i + 1:i + 2]
        code, step = "", 1
        if c in VOWELS:
            code = "A" if i == 0 else ""
        elif c.isdigit():
            code = c
        elif c == "c":
            code, step = ("X", 2) if nxt == "h" else ("K", 2) if nxt == "k" else ("S" if nxt in ("e", "i", "y") else "K"), 1
            if nxt == "k":
                code, step = "K", 2
            elif nxt == "h":
                code, step = "X", 2
        elif c == "s":
            code, step = ("X", 2) if nxt == "h" else ("S", 1)
        elif c == "t":
            if s[i + 1:i + 3] == "ch":
                code, step = "X", 3
            elif nxt == "h":
                code, step = "0", 2
            else:
                code = "T"
        elif c == "p":
            code, step = ("F", 2) if nxt == "h" else ("P", 1)
        elif c == "g":
            if nxt == "h":
                code, step = ("K" if i == 0 else ""), 2
            else:
                code = "J" if nxt in ("e", "i", "y") else "K"
        elif c == "x":
            code = "KS"
        elif c in ("h", "w"):
            code = ""
        else:
            code = SIMPLE.get(c, "")
        if code and (not out or out[-1][-1] != code[-1] or len(code) > 1):
            out.append(code)
        i += step
    return "".join(out)


def edit_distance(a: str, b: str, limit: int | None = None) -> int:
    """Levenshtein distance. With `limit`, it stops early and returns limit + 1 for any larger distance."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if limit is not None and len(a) - len(b) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if limit is not None and min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def budget(longer: int) -> int:
    """Most edits allowed between two letter strings, by the length of the longer one."""
    return 0 if longer < FUZZY_MIN_LETTERS else int(MAX_DISTANCE * longer)


def keys_match(a: str, b: str) -> bool:
    return a == b or (max(len(a), len(b)) >= 4 and edit_distance(a, b, 1) <= 1)


# --- terms: limits, parsing, merging -----------------------------------------------------------------------------

def check_terms(terms: Iterable[str], source: str = "vocabulary") -> tuple[str, ...]:
    """The terms as a tuple, or VocabularyError. At most 200 terms, 1 to 64 characters each, 4,096 bytes in total,
    and no control characters. Empty entries and repeats (ignoring case) are dropped; the first spelling stays."""
    out: dict[str, str] = {}
    total = 0
    for term in terms:
        term = " ".join(term.split())
        if not term:
            continue
        if len(term) > MAX_TERM_CHARS:
            raise VocabularyError(f"{source}: a term is longer than {MAX_TERM_CHARS} characters.")
        if any(ord(c) < 32 or ord(c) == 127 for c in term):
            raise VocabularyError(f"{source}: a term has a control character.")
        if term.casefold() not in out:
            out[term.casefold()] = term
            total += len(term.encode("utf-8"))
        if len(out) > MAX_TERMS:
            raise VocabularyError(f"{source}: at most {MAX_TERMS} terms.")
        if total > MAX_TOTAL_BYTES:
            raise VocabularyError(f"{source}: the terms are longer than {MAX_TOTAL_BYTES} bytes in total.")
    return tuple(out.values())


def parse_header(value: str | None) -> tuple[str, ...]:
    """The terms of an `X-Orb-Vocabulary` header: comma separated. None or empty gives no terms. The header is
    user-controlled input: its size is checked before it is split."""
    if not value or not value.strip():
        return ()
    try:
        value = value.encode("latin-1").decode("utf-8")  # HTTP headers arrive as latin-1; clients send UTF-8 bytes
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    if len(value.encode("utf-8")) > 2 * MAX_TOTAL_BYTES:
        raise VocabularyError(f"vocabulary: the header is longer than {2 * MAX_TOTAL_BYTES} bytes.")
    return check_terms(value.split(","), "vocabulary")


def load_file(path: str | os.PathLike) -> tuple[str, ...]:
    """The terms of a team file: one per line. A line that starts with `#`, or has ` #` and a comment after its term,
    is a comment. `C#` is a term. Raises VocabularyError when the file is missing, too big or breaks a limit."""
    p = Path(path)
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            raise VocabularyError(f"vocabulary file {p.name}: bigger than {MAX_FILE_BYTES} bytes.")
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise VocabularyError(f"vocabulary file {p.name}: cannot be read ({e.strerror or type(e).__name__}).") from None
    except UnicodeDecodeError:
        raise VocabularyError(f"vocabulary file {p.name}: not UTF-8 text.") from None
    return check_terms((COMMENT.sub("", line) for line in text.splitlines()), f"vocabulary file {p.name}")


def merge(*lists: Iterable[str]) -> tuple[str, ...]:
    """The lists in order, without repeats (ignoring case). The first spelling stays, so the team file wins."""
    seen: dict[str, str] = {}
    for terms in lists:
        for term in terms:
            seen.setdefault(term.casefold(), term)
    return tuple(seen.values())


# --- the compiled vocabulary -------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Variant:
    """One way to hear a term. `shape` is "words" (plain words) or "token" (one word with dots inside)."""
    letters: str       # what a span must sound like: lower case letters and digits
    key: str
    out: str           # the text that replaces the span
    shape: str
    multi: bool        # the term has several words: the common-word rule does not apply
    min_words: int
    after: str         # "at": the next word must be `at` (the local part of an email), or ""
    order: int


def variants_of(term: str, order: int) -> list[Variant]:
    flat = re.sub(r"[^a-z0-9]", "", term.lower())
    if len(flat) < MIN_LETTERS:
        return []
    multi = " " in term
    out: list[Variant] = []

    def add(letters, text, shape, min_words=1, after=""):
        out.append(Variant(letters, phonetic_key(letters), text, shape, multi, min_words, after, order))

    parts = term.split()
    if DOTTED.match(term):  # a domain: one dotted word, or "x dot y"
        add(flat, term, "token")
        add(re.sub(r"[^a-z0-9]", "", term.lower().replace(".", "dot")), term, "words", min_words=2)
    elif len(parts) == 2 and all(NAME_WORD.match(w) for w in parts):  # a name: also the start of its email
        local = ".".join(w.lower() for w in parts)
        add(flat, term, "words")
        add(flat, local, "token", after="at")
        add(parts[0].lower() + "dot" + parts[1].lower(), local, "words", min_words=3, after="at")
    else:
        add(flat, term, "words")
    return out


def humps(term: str) -> list[str]:
    """The words that a term's spaces and capitals mark, lower case: useEffect -> [use, effect], README -> [readme]."""
    return [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", term)]


def inner_capital(term: str) -> bool:
    """A capital letter or a digit after the first character: useEffect, OAuth, README; not Sanjay or Parakeet."""
    return any(c.isupper() or c.isdigit() for c in term[1:])


def deletions(key: str) -> set[str]:
    return {key[:i] + key[i + 1:] for i in range(len(key))}


class Tok:
    __slots__ = ("gap", "lead", "core", "end", "kind", "low", "surf", "raw")

    def __init__(self, gap: str, raw: str):
        self.gap, self.raw = gap, gap + raw
        self.lead, self.core, self.end = SPLIT.match(raw).groups()
        self.kind = 1 if PLAIN.match(self.core) else 2 if DOTTED.match(self.core) else 0  # 0: not a word to match
        self.low = re.sub(r"[^a-z0-9]", "", self.core.lower()) if self.kind else ""
        self.surf = self.core.lower().replace("’", "'")


class Vocabulary:
    """A term list, compiled once: the variants of each term, and an index of their phonetic keys."""

    def __init__(self, terms: tuple[str, ...]):
        self.terms = terms
        self.single = frozenset(t for t in terms if " " not in t)
        self.exact = frozenset(terms)
        self.variants = [v for i, t in enumerate(terms) for v in variants_of(t, i)]
        self.index: dict[str, list[int]] = {}
        self.lengths: set[int] = set()
        for n, v in enumerate(self.variants):
            for k in {v.key} | deletions(v.key):
                self.index.setdefault(k, []).append(n)
            m = len(v.letters)
            self.lengths.update(x for x in range(1, 2 * m + 3) if abs(x - m) <= budget(max(x, m)))
        self.common = common_words()

    def candidates(self, key: str) -> set[int]:
        found = set(self.index.get(key, ()))
        for d in deletions(key):
            found.update(self.index.get(d, ()))
        return found

    def best(self, span: list[Tok], nxt: Tok | None) -> Variant | None:
        letters = "".join(t.low for t in span)
        if letters.isdigit() or len(letters) < MIN_LETTERS or len(letters) not in self.lengths:
            return None
        shape, k = ("token" if span[0].kind == 2 else "words"), len(span)
        key = phonetic_key(letters)
        ordinary = all(t.surf in self.common for t in span)  # "like it": only the exact letters of a term may match
        found: list[tuple[int, int, Variant]] = []
        for n in self.candidates(key):
            v = self.variants[n]
            if v.shape != shape or k < v.min_words or not keys_match(key, v.key):
                continue
            if v.after and (nxt is None or nxt.low != "at" or nxt.gap.count("\n") or span[-1].end):
                continue
            d = edit_distance(letters, v.letters, budget(max(len(letters), len(v.letters))))
            if d > budget(max(len(letters), len(v.letters))):
                continue
            if k == 1 and not v.multi and span[0].surf in self.common:
                continue
            if ordinary and (d > 0 or (k > 1 and [t.low for t in span] != humps(v.out))):
                continue  # "use effect" is useEffect; "like it" is not LiveKit, "red is" not Redis, "read me" not README
            if k == 1 and d == 0 and v.shape == "words" and not inner_capital(v.out):
                continue  # only the case differs: "parakeet" stays a bird unless the term is useEffect-like
            found.append((d, v.order, v))
        for d, _, v in sorted(found, key=lambda f: f[:2]):
            if k == 1 or all(edit_distance("".join(t.low for j, t in enumerate(span) if j != drop), v.letters, d) > d
                             for drop in range(k)):  # no extra word: "parse the config" is not parseConfig
                return v
        return None

    def match(self, toks: list[Tok], i: int):
        """(words used, replacement or None to keep them) for the longest span at `i`, or None."""
        for k in range(min(MAX_SPAN, len(toks) - i), 0, -1):
            span = toks[i:i + k]
            if any(not t.kind for t in span) or (k > 1 and not self.joined(span)):
                continue
            if " ".join(t.core for t in span) in self.exact:
                return k, None
            if k == 1 and re.sub(r"['’]s$", "", span[0].core) in self.exact:
                return k, None  # "Sanjay's" keeps its possessive
            if k > 1 and any(t.core in self.single for t in span):  # a term is an atom: never glued to a neighbour
                continue
            v = self.best(span, toks[i + k] if i + k < len(toks) else None)
            if v is not None:
                return k, v.out
        return self.subdomain(toks[i])

    def subdomain(self, tok: Tok):
        """A host with a subdomain: "docs.devslabs.ai" gives "docs.devxlabs.ai" when devxlabs.ai is a term. Only the
        last two labels are compared, and the labels before them stay as they are."""
        labels = tok.core.split(".")
        if tok.kind != 2 or len(labels) < 3:
            return None
        tail = ".".join(labels[-2:])
        if tail in self.exact:
            return None
        v = self.best([Tok("", tail)], None)
        return (1, ".".join(labels[:-2]) + "." + v.out) if v is not None and v.shape == "token" else None

    @staticmethod
    def joined(span: list[Tok]) -> bool:
        """Words of one phrase: plain words, no punctuation or line break between them."""
        return all(t.kind == 1 for t in span) and all(not t.end for t in span[:-1]) \
            and all(not t.lead and "\n" not in t.gap for t in span[1:])

    def run(self, text: str) -> str:
        toks = [Tok(g, w) for g, w in TOKEN.findall(text)]
        tail = text[len("".join(t.raw for t in toks)):]
        out, i = [], 0
        while i < len(toks):
            m = self.match(toks, i)
            if m is None or m[1] is None:
                used = 1 if m is None else m[0]
                out.extend(t.raw for t in toks[i:i + used])
                i += used
                continue
            k, rep = m
            out.append(toks[i].gap + toks[i].lead + rep + toks[i + k - 1].end)
            i += k
        return "".join(out) + tail


@functools.lru_cache(maxsize=16)
def compile_terms(terms: tuple[str, ...]) -> Vocabulary:
    """The compiled vocabulary of a term list. The cache is keyed by the list itself (its hash), so a service that
    sends the same terms on every call compiles them once."""
    return Vocabulary(check_terms(terms) if len(terms) <= MAX_TERMS * 2 else check_terms(terms[:0] or terms))


def correct(text: str, terms: Iterable[str] | Vocabulary) -> str:
    """`text` with the spans that sound like a term replaced by the term. See the module docstring."""
    if not text:
        return text or ""
    voc = terms if isinstance(terms, Vocabulary) else compile_terms(tuple(terms))
    if not voc.variants:
        return text
    parts = PROTECTED.split(text)
    for k in range(0, len(parts), 2):
        if parts[k].strip():
            parts[k] = voc.run(parts[k])
    return "".join(parts)


@functools.lru_cache(maxsize=1)
def _team_terms() -> tuple[str, ...]:
    return load_file(os.environ["ORB_STT_VOCABULARY_FILE"])


def file_stage(text: str) -> str:
    """A text stage for the bench's format suite: `correct` with the terms of the team file `ORB_STT_VOCABULARY_FILE`.
    Register it with `--define vocab=orb_speech.vocab:file_stage`."""
    return correct(text, _team_terms())
