"""Spoken form to written form for dictation text (inverse text normalization), by orb's own rules.

    written("Please review pull request number four thousand eight hundred and twenty-one at 3 30 pm")
        -> "Please review pull request #4821 at 3:30 PM"

A pure function: the same text always gives the same text, and `written(written(x)) == written(x)`. It never adds a
content word. It adds only digits, the symbols `# @ : . / $ %`, the suffix of an ordinal (14th), and it changes the
case of words that it already has (AM, PM, OK, API). It drops only the spoken words that the symbol replaces
("dot", "at", "percent", "number"). Text in "double quotes" or `backticks`, and written forms that are already
there (digits, `$12,450.30`, "Gemma 4"), stay as they are. Decision 0010 has the choice of engine and the numbers.

Symbols inside spoken commands ("dash b") and identifiers ("use effect") are not here: they are for the vocabulary
and polish stages of docs/design/stt-accuracy.md.
"""
from __future__ import annotations

import re

STYLES = ("default",)

UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
TEENS = {w: 10 + i for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 20 + 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9, "trillion": 10 ** 12}
BIG = ("million", "billion", "trillion")
ORD_UNITS = {w: i + 1 for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth".split())}
ORD_TEENS = {w: 10 + i for i, w in enumerate(
    "tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth".split())}
ORD_TENS = {w: 20 + 10 * i for i, w in enumerate(
    "twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth".split())}
ORD_SCALES = {"hundredth": 100, "thousandth": 10 ** 3, "millionth": 10 ** 6, "billionth": 10 ** 9}
ORD_ALL = {**ORD_UNITS, **ORD_TEENS, **ORD_TENS, **ORD_SCALES}
CARD = {*UNITS, *TEENS, *TENS, *SCALES, "hundred"}

MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
AMBIGUOUS_MONTHS = {"march", "may", "august"}  # also ordinary words: they need a capital letter

TIME_UNITS = {"second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month",
              "months", "year", "years", "times"}
METRIC_UNITS = {"byte", "bytes", "kilobyte", "kilobytes", "megabyte", "megabytes", "gigabyte", "gigabytes",
                "terabyte", "terabytes", "millisecond", "milliseconds", "microsecond", "microseconds",
                "nanosecond", "nanoseconds", "kilometer", "kilometers", "kilometre", "kilometres", "meter",
                "meters", "metre", "metres", "centimeter", "centimeters", "millimeter", "millimeters", "mile",
                "miles", "feet", "inches", "inch", "kilo", "kilos", "kilogram", "kilograms", "gram", "grams",
                "liter", "liters", "litre", "litres", "degrees", "degree", "hertz", "kilohertz", "megahertz",
                "gigahertz", "pixels", "pixel", "cents", "cent", "pounds", "pound", "euros", "euro", "rupees",
                "rupee"}
NUM_LABELS = {"version", "step", "page", "line", "chapter", "section", "item", "port", "build", "release",
              "level", "row", "column", "slide", "question", "problem", "exercise", "figure", "table"}
ISSUE_WORDS = {"issue", "ticket", "bug"}
NOT_A_LABEL_NUMBER = (TIME_UNITS | METRIC_UNITS | {"percent", "dollars", "dollar", "million", "billion", "trillion",
                                                   "thousand", "hundred", "o'clock", "oclock", "am", "pm", "a.m", "p.m"})
NUMBER_NOT_AFTER = {"the", "a", "an", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these",
                    "those", "phone", "telephone", "mobile", "cell", "account", "order", "serial", "tracking",
                    "reference", "confirmation", "card", "invoice", "license", "licence", "flight", "house", "zip",
                    "fax", "pin", "id", "passport", "policy", "case", "room", "seat", "plate", "model", "part",
                    "street", "total", "any", "some", "no", "what", "which", "whose", "random", "prime",
                    "whole", "even", "odd", "same", "exact", "average", "lucky", "magic", "large", "small",
                    "big", "high", "low", "of"}
YEAR_PREPS = {"in", "since", "until", "till", "by", "from", "during", "before", "after", "through", "of", "around"}

TLDS = {"com", "net", "org", "io", "ai", "co", "dev", "app", "edu", "gov", "me", "in", "uk", "us", "de", "fr",
        "jp", "au", "ca", "ie", "info", "biz", "xyz", "tech", "cloud", "sh", "tv", "so", "ly", "gg", "eu", "nl",
        "ch", "se", "no", "it", "es", "br", "ru", "cn", "sg", "nz", "za", "ae"}
SPOKEN_TLD_NOT = {"so", "no", "it", "in", "es"}  # also words: a spoken "dot in" needs a longer domain around it
LABEL_STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "from", "is", "it",
              "this", "that", "my", "your", "our", "their", "his", "her", "its", "as", "by", "be", "i", "we", "you",
              "he", "she", "they", "dot", "not", "no", "so", "if", "than", "then", "was", "are", "were", "will",
              "can", "do", "did", "has", "have", "had", "me", "us", "them", "him", "also", "just", "very", "all",
              "any", "some", "polka", "full", "decimal", "period", "slash", "colon", "com"}
NUMBERISH = re.compile(r"^\d+(?:st|nd|rd|th)?$")
EMAIL_CUES = {"email", "e-mail", "emailed", "mail", "mailed", "to", "cc", "bcc", "contact", "address", "from",
              "ping", "reach", "message", "send", "forward", "invite", "tag"}

ACRONYMS = {w.lower(): w for w in (
    "API CI PR README URL URI SQL AWS GCP HTTP HTTPS JSON HTML CSS YAML XML CSV PDF UI UX CLI GUI GPU CPU SSH SSL "
    "TLS DNS CDN SDK IDE LLM JWT CORS SSO QA FAQ ETA KPI OKR ROI CEO CTO CFO AI OCR SaaS USB VPN TCP UDP IP "
    "OK").split()}
ACRONYMS["saas"] = "SaaS"
PLURAL_ACRONYMS = {"api", "pr", "url", "uri", "sdk", "llm", "pdf", "gpu", "cpu", "cli", "kpi", "okr", "faq"}

MAX_PARTS = 16  # most words of an email name or a domain; a longer run is not an address, and a cap keeps the scan linear
MARKERS = {"am": "AM", "pm": "PM", "a.m": "AM", "p.m": "PM"}

NUMBER_WORDS = CARD | set(ORD_ALL) | {"oh", "point"}
PROTECTED = re.compile(r'("[^"\n]*"|“[^”\n]*”|`[^`\n]*`)')
TOKEN = re.compile(r"(\s*)(\S+)")
SPLIT = re.compile(r"^([(\[{\"“‘]*)(.*?)([.,;:!?)\]}\"”’…]*)$", re.S)
DIGITS = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^\d+(?:\.\d+)?$")
INTEGER = re.compile(r"^\d+$")
CLOCK = re.compile(r"^(?:(\d{1,2})[:.](\d{2})|(\d{1,2})(\d{2}))$")
ATTACHED = re.compile(r"^(\d{1,2})(?:[:.](\d{2}))?(am|pm|a\.m|p\.m)$", re.I)
LABEL = re.compile(r"^[a-z0-9][a-z0-9-]*$")
LOCAL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+-]*$")
DOTTED = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9][A-Za-z0-9-]*)+$")
DAY = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?$")
SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _clock(text: str):
    """(hour, minutes text) for '3:30', '3.30', '330' or '1030' when it can be a time on a 12-hour clock, else None."""
    m = CLOCK.match(text)
    if not m:
        return None
    h, minute = int(m.group(1) or m.group(3)), m.group(2) or m.group(4)
    return (h, minute) if 1 <= h <= 12 and int(minute) <= 59 else None


def _label(low: str) -> bool:
    """A word that can be a part of a spoken domain: letters and digits, but not a number."""
    return bool(LABEL.match(low)) and not NUMBERISH.match(low) and not all(w in NUMBER_WORDS for w in low.split("-"))


def _suffix(n: int) -> str:
    return "th" if 10 < n % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _ordinal(n: int) -> str:
    return f"{n}{_suffix(n)}"


class Tok:
    __slots__ = ("raw", "gap", "lead", "core", "end", "low", "words")

    def __init__(self, gap: str, raw: str):
        self.gap, self.raw = gap, raw
        self.lead, self.core, self.end = SPLIT.match(raw).groups()
        self.low = self.core.lower()
        self.words = None
        if self.core.isalpha():
            self.words = (self.low,)
        elif "-" in self.core and self.core.replace("-", "").isalpha():
            parts = self.low.split("-")
            if len(parts) == 2 and parts[0] in TENS and (parts[1] in UNITS and parts[1] != "zero"
                                                         or parts[1] in ORD_UNITS):
                self.words = tuple(parts)


class Num:
    """A number read from the tokens i..j: its written text, and what the matchers need to decide."""
    __slots__ = ("text", "j", "value", "ordinal", "spelled", "decimal", "scaled", "skip")

    def __init__(self, text="", j=0, value=None, ordinal=False, spelled=False, decimal=False, scaled=False,
                 skip=False):
        self.text, self.j, self.value, self.ordinal, self.spelled = text, j, value, ordinal, spelled
        self.decimal, self.scaled, self.skip = decimal, scaled, skip


def _below100(a, i):
    n = len(a)
    if i >= n:
        return None
    w = a[i]
    if w in TENS:
        v = TENS[w]
        if i + 1 < n and a[i + 1] in UNITS and a[i + 1] != "zero":
            return v + UNITS[a[i + 1]], i + 2, False
        if i + 1 < n and a[i + 1] in ORD_UNITS:
            return v + ORD_UNITS[a[i + 1]], i + 2, True
        return v, i + 1, False
    if w in ORD_TENS:
        return ORD_TENS[w], i + 1, True
    if w in TEENS:
        return TEENS[w], i + 1, False
    if w in ORD_TEENS:
        return ORD_TEENS[w], i + 1, True
    if w in UNITS and w != "zero":
        return UNITS[w], i + 1, False
    if w in ORD_UNITS:
        return ORD_UNITS[w], i + 1, True
    return None


def _group(a, i):
    """The longest 1 to 999 at a[i]: (value, next index, is ordinal), or None."""
    n = len(a)
    if i + 1 < n and (a[i] in UNITS and a[i] != "zero" or a[i] in TEENS) and a[i + 1] in ("hundred", "hundredth"):
        count = UNITS.get(a[i]) or TEENS[a[i]]
        base = count * 100
        if a[i + 1] == "hundredth":
            return base, i + 2, True
        k = i + 3 if i + 2 < n and a[i + 2] == "and" else i + 2
        rest = _below100(a, k)
        if rest:
            return base + rest[0], rest[1], rest[2]
        return base, i + 2, False
    return _below100(a, i)


def _cardinal(a, i):
    """A whole number at a[i]: (value, next index, is ordinal), or None. Valid English only: 'twenty twenty-six' is not."""
    n = len(a)
    if i < n and a[i] == "zero":
        return 0, i + 1, False
    total, j, seen = 0, i, False
    for name, scale in (("billion", 10 ** 9), ("million", 10 ** 6), ("thousand", 10 ** 3)):
        g = _group(a, j)
        if not g or g[2] or g[1] >= n:
            continue
        if a[g[1]] == name:
            total, j, seen = total + g[0] * scale, g[1] + 1, True
            if j + 1 < n and a[j] == "and" and _group(a, j + 1):
                j += 1
        elif a[g[1]] == name + "th":
            return total + g[0] * scale, g[1] + 1, True
    g = _group(a, j)
    if g:
        return total + g[0], g[1], g[2]
    if seen:
        return total, j, False
    return None


def _parse_chain(a):
    """All the atoms as one number, or None. -> (value, ordinal, fraction digits or None, scale word or None)."""
    if "point" in a:
        p = a.index("point")
        whole, frac = a[:p], a[p + 1:]
        scale = None
        if frac and frac[-1] in BIG:
            scale, frac = frac[-1], frac[:-1]
        if not frac or any(w not in UNITS and w != "oh" for w in frac):
            return None
        r = _cardinal(whole, 0)
        if not r or r[1] != len(whole) or r[2]:
            return None
        return r[0], False, "".join("0" if w in ("oh", "zero") else str(UNITS[w]) for w in frac), scale
    r = _cardinal(a, 0)
    if not r or r[1] != len(a):
        return None
    return r[0], r[2], None, None


class Fmt:
    """The tokens of one text segment, and the matchers that read them."""

    def __init__(self, text: str):
        pairs = TOKEN.findall(text)
        self.t = [Tok(g, w) for g, w in pairs]
        self.tail = text[len("".join(g + w for g, w in pairs)):]
        self.n = len(self.t)

    def low(self, k: int) -> str:
        return self.t[k].low if 0 <= k < self.n else ""

    def joined(self, k: int) -> bool:
        """Token k and token k + 1 sit in one phrase: no punctuation between them."""
        return k + 1 < self.n and not self.t[k].end and not self.t[k + 1].lead

    def prev(self, i: int) -> str:
        return self.t[i - 1].low if i > 0 and not self.t[i - 1].end else ""

    # ---- numbers

    def chain(self, i: int):
        """The run of number words that starts at token i: (atoms, owner token of each atom, end token)."""
        a, own, k, after_point = [], [], i, False
        stop = False
        while k < self.n and not stop:
            if k > i and not self.joined(k - 1):
                break
            words = self.t[k].words
            if words is None:
                break
            for w in words:
                p = a[-1] if a else None
                if after_point:
                    ok = w in UNITS or w == "oh" or (w in BIG and p not in ("point",))
                    if not ok:
                        stop = True
                        break
                    if w in BIG:
                        a.append(w), own.append(k)
                        stop = True
                        break
                elif w in CARD:
                    pass
                elif w in ORD_ALL:
                    joinable = (p is None or (p in TENS and w in ORD_UNITS)
                                or p in ("hundred", "thousand", "million", "billion", "trillion", "and"))
                    if not joinable:
                        stop = True
                        break
                    a.append(w), own.append(k)
                    stop = True
                    break
                elif w == "and":
                    if p not in ("hundred", "thousand", "million", "billion", "trillion"):
                        stop = True
                        break
                elif w == "oh":  # "twenty oh five": never one number, but it must stay whole
                    if p is None or p == "and":
                        stop = True
                        break
                elif w == "point":
                    if p is None or p == "and":
                        stop = True
                        break
                    after_point = True
                else:
                    stop = True
                    break
                a.append(w), own.append(k)
            else:
                k += 1
                continue
            break
        while a and a[-1] in ("and", "point"):  # a dangling joiner is a word of its own
            a.pop(), own.pop()
        if not a or a[0] in ("and", "point", "oh"):
            return [], [], i
        end = own[-1] + 1
        if len(self.t[own[-1]].words or ()) > 1 and sum(1 for o in own if o == own[-1]) < len(self.t[own[-1]].words):
            end = own[-1]
            while a and own[-1] == end:
                a.pop(), own.pop()
            if not a:
                return [], [], i
        return a, own, end

    def read_num(self, i: int):
        t = self.t[i]
        if DIGITS.match(t.core):
            return Num(t.core, i + 1, int(t.core) if INTEGER.match(t.core) else None, decimal="." in t.core)
        if t.words is None:
            return None
        a, own, end = self.chain(i)
        if not a:
            return None
        r = _parse_chain(a)
        if r is None:
            return Num(j=end, skip=True)
        value, ordinal, frac, scale = r
        if ordinal:
            return Num(str(value), end, value, True, True)
        if frac is not None:
            text = f"{value:,}.{frac}" if value >= 1000 else f"{value}.{frac}"
            if scale:
                return Num(f"{text} {scale}", end, None, False, True, True, True)
            return Num(text, end, None, False, True, True)
        g = _cardinal(a[:-1], 0) if a[-1] in BIG and len(a) > 1 else None
        if g and g[1] == len(a) - 1 and not g[2] and g[0] < 1000:
            return Num(f"{g[0]} {a[-1]}", end, value, False, True, False, True)
        year = (len(a) >= 3 and a[0] == "two" and a[1] == "thousand" and 2001 <= value <= 2099
                and "hundred" not in a)
        return Num(str(value) if year else f"{value:,}", end, value, False, True)

    def pair_year(self, k: int):
        """'twenty twenty-six' or 'nineteen oh five' at token k as (text, end token), or None."""
        words, own = [], []
        j = k
        while j < self.n and len(words) < 4 and (j == k or self.joined(j - 1)) and self.t[j].words:
            for w in self.t[j].words:
                words.append(w), own.append(j)
            j += 1
        if not words or words[0] not in ("nineteen", "twenty"):
            return None
        head = "19" if words[0] == "nineteen" else "20"
        for m in (3, 2):
            rest = words[1:m]
            if len(rest) != m - 1:
                continue
            last_tok = own[m - 1]
            if m < len(words) and own[m] == last_tok:
                continue
            tail = None
            if rest[0] in ("oh", "zero") and len(rest) == 2 and rest[1] in UNITS and rest[1] != "zero":
                tail = f"0{UNITS[rest[1]]}"
            elif len(rest) == 1 and rest[0] in TEENS:
                tail = str(TEENS[rest[0]])
            elif len(rest) == 1 and rest[0] in TENS:
                tail = str(TENS[rest[0]])
            elif len(rest) == 2 and rest[0] in TENS and rest[1] in UNITS and rest[1] != "zero":
                tail = str(TENS[rest[0]] + UNITS[rest[1]])
            if tail is None:
                continue
            nxt = last_tok + 1
            if nxt < self.n and self.joined(last_tok) and (
                    self.t[nxt].words and (self.t[nxt].words[0] in CARD or self.t[nxt].words[0] in ORD_ALL)
                    or self.t[nxt].low in NOT_A_LABEL_NUMBER or self.t[nxt].core[:1].isdigit()):
                return None
            return head + tail, last_tok + 1
        return None

    def read_year(self, k: int):
        """A spelled year after a month or a date: (text, end token) or None."""
        if k >= self.n or not self.joined(k - 1) and not (k > 0 and self.t[k - 1].end == ","):
            return None
        if self.t[k].lead:
            return None
        num = self.read_num(k)
        if num and not num.skip and num.spelled and not num.ordinal and not num.decimal and num.value is not None \
                and 1900 <= num.value <= 2099:
            if num.j < self.n and self.joined(num.j - 1) and self.t[num.j].low in NOT_A_LABEL_NUMBER:
                return None
            return str(num.value), num.j
        if self.t[k].words:
            return self.pair_year(k)
        return None

    # ---- the matchers: each returns (replacement text, end token) or (None, end token) to keep the tokens as they are

    def m_email(self, i: int):
        t = self.t[i]
        if t.lead or not (LOCAL.match(t.core) or DOTTED.match(t.core)):
            return None
        local, p = [t.core], i + 1
        literal = "." in t.core or "_" in t.core or any(c.isdigit() for c in t.core) or "-" in t.core or "+" in t.core
        while len(local) < MAX_PARTS and p + 1 < self.n and self.joined(p - 1) \
                and self.t[p].low in ("dot", "underscore") and LOCAL.match(self.t[p + 1].core) and not self.t[p + 1].lead:
            local.append("." if self.t[p].low == "dot" else "_")
            local.append(self.t[p + 1].core)
            literal = True
            p += 2
        if not (p < self.n and self.joined(p - 1) and self.t[p].low == "at" and not self.t[p].end):
            return None
        if len(local) == 1 and (local[0].lower() in LABEL_STOP or NUMBERISH.match(local[0])):
            return None
        dom = self.read_domain(p + 1, email=True)
        if dom is None:
            return None
        cue = i > 0 and self.t[i - 1].low in EMAIL_CUES and not self.t[i - 1].end
        cue = cue or (i > 1 and self.t[i - 1].low == "is" and not self.t[i - 1].end
                      and self.t[i - 2].low in ("email", "e-mail", "mail", "address"))  # "my email is arjun at …"
        digit_or_sep = literal or any(c.isdigit() for c in "".join(local))
        if not (digit_or_sep or cue):
            return None
        return ("".join(local).lower() + "@" + dom[0], dom[1])

    def read_domain(self, k: int, email: bool = False):
        """A domain at token k, spoken ('example dot com') or written ('example.com'): (text, end token) or None."""
        if k >= self.n or self.t[k].lead:
            return None
        labels, p, spoken = [], k, False
        while len(labels) <= MAX_PARTS:
            tk = self.t[p]
            if DOTTED.match(tk.core) and not labels:
                parts = tk.core.lower().split(".")
                if parts[-1].isalpha() and (2 <= len(parts[-1]) <= 10 if email else parts[-1] in TLDS):
                    return tk.core.lower(), p + 1
                return None
            if not _label(tk.low):
                break
            if not labels and tk.low in LABEL_STOP:
                return None
            labels.append(tk.low)
            if (p + 2 < self.n and self.joined(p) and self.t[p + 1].low == "dot" and not self.t[p + 1].end
                    and not self.t[p + 2].lead and (_label(self.t[p + 2].low) or DOTTED.match(self.t[p + 2].core))):
                spoken = True
                p += 2
                if DOTTED.match(self.t[p].core) and not _label(self.t[p].low):
                    parts = self.t[p].core.lower().split(".")
                    labels.extend(parts)
                    p += 1
                    break
                continue
            p += 1
            break
        if not spoken or len(labels) < 2:
            return None
        tld = labels[-1]
        if tld not in TLDS or tld in SPOKEN_TLD_NOT and len(labels) < 3:
            return None
        return ".".join(labels), p

    def m_url(self, i: int):
        t = self.t[i]
        k, scheme = i, ""
        if t.low in ("http", "https") and i + 4 < self.n and all(self.joined(i + d) for d in range(3)) \
                and self.t[i + 1].low == "colon" and self.t[i + 2].low == "slash" and self.t[i + 3].low == "slash":
            scheme, k = t.low + "://", i + 4
        elif t.lead or not _label(t.low) and not DOTTED.match(t.core):
            return None
        elif not (_label(t.low) and "dot" in (self.low(i + 1), self.low(i + 3))) \
                and not (DOTTED.match(t.core) and self.low(i + 1) == "slash" and self.joined(i)):
            return None  # a written domain is joined with a path only when "slash" follows it
        dom = self.read_domain(k)
        if dom is None:
            return None
        text, j = dom
        while text.count("/") < MAX_PARTS and j + 1 < self.n and self.joined(j - 1) and self.t[j].low == "slash" and not self.t[j].end \
                and SEGMENT.match(self.t[j + 1].core) and not self.t[j + 1].lead and self.t[j + 1].low not in LABEL_STOP:
            text += "/" + self.t[j + 1].core
            j += 2
        if j == i + 1 and not scheme:
            return None  # a written domain with no path: nothing to do
        if j > i + 1 and self.low(j + 1) == "slash" and self.low(j) not in LABEL_STOP and self.joined(j - 1) \
                and "/" in text and self.low(j - 1) != "slash":
            return None  # "slash dev slabs slash orb": a segment of two words is not a path we can write
        return scheme + text, j

    def m_label(self, i: int):
        """'pull request number N', 'PR N', 'issue N', 'ticket N', 'number N' -> '... #N'."""
        t = self.t[i]
        low, j = t.low, i + 1
        if low == "pr":
            head = "PR"
        elif low == "pull" and self.low(i + 1) == "request" and self.joined(i):
            head, j = t.core + self.t[i + 1].gap + self.t[i + 1].core, i + 2
        elif low in ISSUE_WORDS:
            head = t.core
        elif low == "number" and self.prev(i) not in NUMBER_NOT_AFTER:
            head = ""
        else:
            return None
        if not self.joined(j - 1):
            return None
        k = j
        if head and self.low(k) == "number" and self.joined(k):
            k += 1
        num = self.read_num(k)
        if num is not None and num.skip:
            ident = self.read_id(k)
            return None if ident is None else ((head + self.t[j].gap + f"#{ident[0]}" if head else f"#{ident[0]}"), ident[1])
        if num is None or num.ordinal or num.decimal or num.scaled or num.value is None:
            return None
        if num.spelled and num.value < 10 and (not head or self.low(num.j) == "of"):
            return None  # "number one", "issue one of them"
        digits, end = str(num.value) if num.spelled else num.text, num.j
        if num.value >= 1000 and num.value % 100 == 0 and self.t[end - 1].end in ("", ",") \
                and self.low(end) == "and" and self.t[end].end in ("", ",") and end + 1 < self.n \
                and not self.t[end + 1].lead:
            rest = self.read_num(end + 1)  # Parakeet splits "4821" as "4800 and 21" or "4800, and 21,"
            if rest and not rest.skip and not rest.ordinal and not rest.decimal and not rest.scaled \
                    and rest.value is not None and 10 <= rest.value < (1000 if num.value % 1000 == 0 else 100):
                digits, end = str(num.value + rest.value), rest.j
        if self.joined(end - 1) and re.split(r"[^a-z']", self.low(end))[0] in NOT_A_LABEL_NUMBER:
            return None  # "issue 10 AM", "PR 5 minutes": the number belongs to the word after it
        return (head + self.t[j].gap + f"#{digits}" if head else f"#{digits}"), end

    def read_id(self, k: int):
        """An id read in pairs after a marker: 'three eighteen' -> 318, 'forty eight twenty one' -> 4821, 'three oh two'
        -> 302, 'one zero four two' -> 1042. (digits, end token), or None when the words are not such an id."""
        a, own, end = self.chain(k)
        if not a or len(a) < 2 or (end < self.n and self.joined(end - 1) and self.low(end) in NOT_A_LABEL_NUMBER):
            return None
        if all(w in UNITS or w == "oh" for w in a):  # digit by digit
            return "".join("0" if w == "oh" else str(UNITS[w]) for w in a), end
        chunks, i = [], 0
        while i < len(a):
            w = a[i]
            if w in ("oh", "zero") and i + 1 < len(a) and a[i + 1] in UNITS and a[i + 1] != "zero":
                chunks.append(f"0{UNITS[a[i + 1]]}")
                i += 2
            elif w in TENS:
                if i + 1 < len(a) and a[i + 1] in UNITS and a[i + 1] != "zero":
                    chunks.append(str(TENS[w] + UNITS[a[i + 1]]))
                    i += 2
                else:
                    chunks.append(str(TENS[w]))
                    i += 1
            elif w in TEENS:
                chunks.append(str(TEENS[w]))
                i += 1
            elif w in UNITS or w == "oh":
                chunks.append("0" if w == "oh" else str(UNITS[w]))
                i += 1
            else:
                return None
        if len(chunks[0]) > 2 or any(len(c) != 2 for c in chunks[1:]) or len(chunks) > 4:
            return None
        return "".join(chunks), end

    def m_time(self, i: int):
        """'3 30 pm', 'three thirty p.m.', '3.30pm', '10 am', "three o'clock" -> '3:30 PM', '10 AM', "3 o'clock"."""
        t = self.t[i]
        if t.lead:
            return None
        attached = ATTACHED.match(t.core)
        if attached:
            h, minute, mark = int(attached.group(1)), attached.group(2), MARKERS[attached.group(3).lower()]
            if not 1 <= h <= 12 or (minute is not None and int(minute) > 59):
                return None
            return (f"{h}:{minute} {mark}" if minute else f"{h} {mark}"), i + 1, self.mark_end(i)
        if not self.joined(i):
            return None
        clock = _clock(t.core)
        hour = None
        if clock:
            (h, minute), j = clock, i + 1
        elif CLOCK.match(t.core):
            return None
        else:
            if INTEGER.match(t.core) and len(t.core) <= 2 and 1 <= int(t.core) <= 12:
                h = int(t.core)
            elif t.words and len(t.words) == 1 and (t.words[0] in UNITS and t.words[0] != "zero"
                                                     or t.words[0] in ("ten", "eleven", "twelve")):
                h = UNITS.get(t.words[0]) or TEENS[t.words[0]]
            else:
                return None
            hour, j, minute = h, i + 1, None
            m = self.minute_at(j)
            if m:
                minute, j, mark = m
                if mark:
                    return f"{h}:{minute} {mark}", j, self.mark_end(j - 1)
        if j >= self.n or not self.joined(j - 1) or self.t[j].lead:
            return None
        nxt = self.t[j]
        if nxt.low in ("o'clock", "oclock") and minute is None and hour is not None:
            return f"{hour} o'clock", j + 1
        mark = MARKERS.get(nxt.low)
        if mark is None:
            return None
        return (f"{h}:{minute} {mark}" if minute is not None else f"{h} {mark}"), j + 1, self.mark_end(j)

    def mark_end(self, k: int) -> str:
        """The punctuation after 'p.m.': its dot is the sentence's too, unless a lower-case word goes on."""
        tok, end = self.t[k], self.t[k].end
        if end.startswith(".") and tok.low.endswith(("a.m", "p.m")):
            after = self.t[k + 1] if k + 1 < self.n else None
            if after is not None and not after.core[:1].isupper():
                return end[1:]
        return end

    def minute_at(self, k: int):
        """Minutes at token k: ('05', next token) for '05', 'oh five', 'fifteen', 'forty-five', 'thirty'."""
        if k >= self.n:
            return None
        t = self.t[k]
        if t.lead:
            return None
        tail = ATTACHED.match(t.core)
        if tail and tail.group(2) is None and len(tail.group(1)) == 2 and int(tail.group(1)) <= 59:
            return tail.group(1), k + 1, MARKERS[tail.group(3).lower()]
        if not self.joined(k):
            return None
        if INTEGER.match(t.core) and len(t.core) == 2 and int(t.core) <= 59:
            return t.core, k + 1, None
        w = t.words
        if not w:
            return None
        if len(w) == 1 and w[0] in ("oh", "zero") and k + 1 < self.n and self.joined(k):
            u = self.t[k + 1].words
            if u and len(u) == 1 and u[0] in UNITS and u[0] != "zero":
                return f"0{UNITS[u[0]]}", k + 2, None
            return None
        if len(w) == 1 and w[0] in TEENS:
            return str(TEENS[w[0]]), k + 1, None
        if len(w) == 1 and w[0] in TENS and TENS[w[0]] <= 50:
            u = self.t[k + 1].words if k + 1 < self.n and self.joined(k) else None
            if u and len(u) == 1 and u[0] in UNITS and u[0] != "zero":
                return str(TENS[w[0]] + UNITS[u[0]]), k + 2, None
            return str(TENS[w[0]]), k + 1, None
        if len(w) == 2 and w[0] in TENS and TENS[w[0]] <= 50 and w[1] in UNITS:
            return str(TENS[w[0]] + UNITS[w[1]]), k + 1, None
        return None

    def month_at(self, k: int):
        if k >= self.n or self.t[k].lead:
            return None
        t = self.t[k]
        if t.low not in MONTHS or not t.core.isalpha():
            return None
        if t.low in AMBIGUOUS_MONTHS and not t.core[:1].isupper():
            return None
        return t.low.capitalize()

    def m_date(self, i: int):
        """'November fourteenth twenty twenty-six' -> 'November 14th, 2026' (the month needs a capital if it is also a word)."""
        month = self.month_at(i)
        if month is None or not self.joined(i):
            return None
        k, t = i + 1, self.t[i + 1]
        day = None
        if t.words:
            num = self.read_num(k)
            if num and not num.skip and num.ordinal and 1 <= num.value <= 31:
                day, k = _ordinal(num.value), num.j
        elif DAY.match(t.core) and 1 <= int(DAY.match(t.core).group(1)) <= 31:
            day, k = t.core, k + 1
        if day is None:
            yr = self.read_year(k)
            return None if yr is None else (f"{month} {yr[0]}", yr[1], self.t[yr[1] - 1].end)
        end = self.t[k - 1].end
        if end in ("", ","):
            yr = self.read_year(k)
            if yr is not None:
                return f"{month} {day}{end} {yr[0]}", yr[1], self.t[yr[1] - 1].end
        return f"{month} {day}", k, end

    def m_number(self, i: int):
        t = self.t[i]
        num = self.read_num(i)
        if num is None:
            return None
        prev = self.prev(i)
        if num.skip:
            if prev in YEAR_PREPS:
                y = self.pair_year(i)
                if y:
                    return y[0], y[1], self.t[y[1] - 1].end
            return None, num.j
        j = num.j
        nxt = self.t[j] if j < self.n and self.joined(j - 1) else None
        nlow = nxt.low if nxt else ""
        if num.ordinal:
            value = num.value
            if not t.words or value is None:
                return None
            month = None
            if nlow == "of" and j + 1 < self.n and self.joined(j):
                month = self.month_at(j + 1)
            if value >= 10 or (month and value <= 31):
                text = _ordinal(value)
                if month:
                    tok = self.t[j + 1]
                    tok.core, tok.low = month, month.lower()
                    tok.raw = tok.lead + month + tok.end
                    return f"{text}{self.t[j].gap}of", j + 1
                return text, j
            return None, j
        if nlow == "percent":
            return num.text + "%", j + 1, nxt.end
        if nlow in ("dollars", "dollar") and not num.decimal:
            text, k = "$" + num.text, j + 1
            cents = self.cents_at(k) if not nxt.end and k < self.n else None
            if cents and not num.scaled:
                return f"{text}.{cents[0]}", cents[1], self.t[cents[1] - 1].end
            return text, k, nxt.end
        if not num.spelled:
            return None
        if nxt and nxt.low in MARKERS and not nxt.lead and _clock(num.text):
            h, minute = _clock(num.text)
            return f"{h}:{minute} {MARKERS[nxt.low]}", j + 1, self.mark_end(j)
        small = num.value is not None and num.value < 10
        convert = num.decimal or num.scaled or not small
        if small and nlow in METRIC_UNITS:
            convert = True
        elif small and num.value >= 2 and nlow in TIME_UNITS:
            convert = True
        elif small and prev in NUM_LABELS:
            convert = True
        elif small and nlow in ("o'clock",):
            convert = True
        if convert:
            return num.text, j
        return None, j

    def cents_at(self, k: int):
        """'and thirty cents' after 'dollars': ('30', end token), or None."""
        if k < self.n and self.t[k].low == "and" and self.joined(k):
            k += 1
        if k >= self.n:
            return None
        num = self.read_num(k)
        if num is None or num.skip or num.ordinal or num.decimal or num.scaled or num.value is None \
                or not 1 <= num.value <= 99 or num.j >= self.n or not self.joined(num.j - 1):
            return None
        if self.t[num.j].low not in ("cents", "cent"):
            return None
        return f"{num.value:02d}", num.j + 1

    def m_case(self, i: int):
        t = self.t[i]
        if not t.core.isalpha() or self.prev(i) == "dot":  # "settings dot json" is a file name, not JSON the format
            return None
        low = t.low
        if low in ACRONYMS and (t.core.islower() or t.core.istitle() or t.core.isupper()):
            return ACRONYMS[low], i + 1
        if low.endswith("s") and low[:-1] in PLURAL_ACRONYMS and (t.core.islower() or t.core.istitle()):
            return ACRONYMS[low[:-1]] + "s", i + 1
        return None

    MATCHERS = ("m_email", "m_url", "m_date", "m_label", "m_time", "m_number", "m_case")

    def run(self) -> str:
        out, i = [], 0
        t = self.t
        while i < self.n:
            hit = None
            for name in self.MATCHERS:
                hit = getattr(self, name)(i)
                if hit is not None:
                    break
            if hit is None:
                out.append(t[i].gap + t[i].raw)
                i += 1
                continue
            text, j = hit[0], hit[1]
            if text is None:
                out.extend(t[k].gap + t[k].raw for k in range(i, j))
            else:
                end = hit[2] if len(hit) > 2 else t[j - 1].end
                out.append(t[i].gap + t[i].lead + text + end)
            i = j
        return "".join(out) + self.tail


def written(text: str, style: str = "default") -> str:
    """The written form of dictated text. Pure, deterministic and idempotent. See the module docstring."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {', '.join(STYLES)}, not {style!r}.")
    if not text:
        return text or ""
    parts = PROTECTED.split(text)
    for k in range(0, len(parts), 2):
        if parts[k].strip():
            parts[k] = Fmt(parts[k]).run()
    return "".join(parts)
