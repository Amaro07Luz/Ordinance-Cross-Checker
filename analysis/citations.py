"""
Citation validator. Deterministic, no LLM. A claim passes only if:

  1. it cites a passage that was actually retrieved (P1..Pn),
  2. its quote is an exact excerpt of that passage (after whitespace/quote/case
     normalization; "..." may join excerpts that appear in order),
  3. every quantity in the statement appears in the quote (or in the passage's
     section heading, for section numbers), so no number can be invented, and
  4. if the statement names a jurisdiction, it is the passage's jurisdiction.

This is the mechanism behind the 95% citation-accuracy target: anything the
model writes that can't be traced word-for-word to a source is revised or dropped.
"""

import re
from typing import Dict, List, Sequence

from analysis.models import Claim
from analysis.text import all_numbers, normalize, quantities
from retrieval.query_agent import GAZETTEER

_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*")

# Lowercased names/aliases -> canonical jurisdiction, for check 4.
_NAME_TO_JUR: Dict[str, str] = {}
for _canon, _info in GAZETTEER.items():
    _NAME_TO_JUR[_canon.lower()] = _canon
    for _a in _info["aliases"]:
        _NAME_TO_JUR[_a.lower()] = _canon
    for _p in _info["parents"]:
        _NAME_TO_JUR.setdefault(_p.lower(), _p)
# "montgomery" alone is ambiguous in prose (county vs. city); only match the full name.
_NAME_TO_JUR.pop("montgomery", None)
_NAME_RES = [(re.compile(rf"\b{re.escape(n)}\b"), j)
             for n, j in sorted(_NAME_TO_JUR.items(), key=lambda kv: -len(kv[0]))]


def is_verbatim(quote: str, content: str) -> bool:
    """True if quote appears in content (normalized). Ellipses split the quote into
    pieces that must appear in order."""
    hay = normalize(content)
    pieces = [normalize(p) for p in _ELLIPSIS_RE.split(quote or "") if normalize(p)]
    if not pieces:
        return False
    pos = 0
    for p in pieces:
        i = hay.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


def jurisdictions_named(text: str) -> set:
    t = normalize(text)
    return {j for rx, j in _NAME_RES if rx.search(t)}


def validate_claims(claims: Sequence[Claim], ref_map: Dict[str, object],
                    min_quote_words: int = 4, check_jurisdiction: bool = True) -> List[Claim]:
    """Fills chunk_id/jurisdiction/section_title on each claim and records problems.
    ref_map: {"P1": RetrievedPassage, ...}. Returns the same claims (mutated)."""
    for c in claims:
        c.problems = []
        p = ref_map.get((c.passage_ref or "").strip().upper())
        if p is None:
            c.problems.append(f"cites {c.passage_ref or 'no passage'}, which was not provided")
            continue
        c.chunk_id, c.jurisdiction, c.section_title = p.chunk_id, p.jurisdiction, p.section_title

        if not (c.statement or "").strip():
            c.problems.append("empty statement")
        if len(normalize(c.quote).split()) < min_quote_words:
            c.problems.append(f"quote must be at least {min_quote_words} words copied from "
                              f"{c.passage_ref}")
        elif not is_verbatim(c.quote, p.content):
            c.problems.append(f"quote is not an exact excerpt of {c.passage_ref}; copy the text "
                              "character for character")

        allowed = all_numbers(c.quote) | all_numbers(p.section_title)
        unsupported = sorted(quantities(c.statement) - allowed, key=lambda x: float(x))
        if unsupported:
            c.problems.append("number(s) " + ", ".join(unsupported) + " in the statement do not "
                              "appear in the quote")

        named = jurisdictions_named(c.statement) if check_jurisdiction else set()
        if named and p.jurisdiction not in named:
            c.problems.append(f"statement names {', '.join(sorted(named))} but {c.passage_ref} "
                              f"is from {p.jurisdiction}")
    return list(claims)


def problems_report(claims: Sequence[Claim]) -> str:
    """Human-readable list of failures, fed back to the LLM for the revise pass."""
    lines = []
    for i, c in enumerate(claims, 1):
        if c.problems:
            lines.append(f"- claim {i} ({c.passage_ref}: {c.statement[:120]!r}): "
                         + "; ".join(c.problems))
    return "\n".join(lines)
