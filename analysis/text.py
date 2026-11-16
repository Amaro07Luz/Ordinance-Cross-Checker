"""
Text helpers shared by the validator and the conflict detector:
normalization for verbatim matching, sentence splitting and number extraction.
"""

import re
import unicodedata
from typing import List, Set

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
})


def normalize(s: str) -> str:
    """Lowercase, unify quotes/dashes, collapse whitespace. Used for verbatim checks so
    a quote still matches when the model changes curly quotes or line breaks."""
    s = unicodedata.normalize("NFKC", s or "").translate(_TRANSLATE)
    return re.sub(r"\s+", " ", s).strip().lower()


# Sentence boundary: . ; or : followed by whitespace and a capital, "(" or "§"
_SENT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(§])")


def sentences(text: str, min_len: int = 15) -> List[str]:
    flat = re.sub(r"\s+", " ", text or "").strip()
    return [s.strip() for s in _SENT_RE.split(flat) if len(s.strip()) >= min_len]


WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty-five": 25, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}
WORD_NUM_PATTERN = "|".join(sorted(map(re.escape, WORD_NUMBERS), key=len, reverse=True))

DIGIT_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_DIGIT_RE = re.compile(rf"(?<![\w.]){DIGIT_NUM}")
_WORD_RE = re.compile(rf"\b(?:{WORD_NUM_PATTERN})\b", re.I)

# Identifiers that contain digits but are not quantities: section numbers,
# district codes (R-1, TH-2), ordinance numbers (§ 220-4), ZIP codes.
_IDENT_RES = [
    re.compile(r"\b(?:section|sec\.|article|art\.|chapter|ch\.|ordinance|no\.)\s*[\d][\w.\-]*", re.I),
    re.compile(r"§+\s*[\d][\w.\-]*"),
    re.compile(r"\b[A-Za-z]{1,4}-\d+[A-Za-z]?\b"),
    re.compile(r"\b\d{5}\b(?!\s*(?:square|sq|feet|ft|acres?))", re.I),
]


def canon_number(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in WORD_NUMBERS:
        return f"{WORD_NUMBERS[raw]:g}"
    try:
        return f"{float(raw.replace(',', '')):g}"
    except ValueError:
        return raw


def strip_identifiers(text: str) -> str:
    for rx in _IDENT_RES:
        text = rx.sub(" ", text)
    return text


def quantities(text: str, include_words: bool = True) -> Set[str]:
    """Numbers that state a quantity (identifiers removed), canonicalized:
    '7,200' -> '7200', '25.0' -> '25', 'two' -> '2'."""
    text = strip_identifiers(text or "")
    nums = {canon_number(m.group()) for m in _DIGIT_RE.finditer(text)}
    if include_words:
        nums |= {canon_number(m.group()) for m in _WORD_RE.finditer(text)}
    return nums


def all_numbers(text: str) -> Set[str]:
    """Every number in the text, identifiers included (used as the 'allowed' side)."""
    t = text or ""
    return ({canon_number(m.group()) for m in _DIGIT_RE.finditer(t)}
            | {canon_number(m.group()) for m in _WORD_RE.finditer(t)})
