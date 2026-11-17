"""
Cross-referencing: numeric requirement extraction, conflict detection and gap detection.
Deterministic (no LLM), so it is testable and can't hallucinate a conflict.

Conflict kinds
  overlapping_jurisdictions  a municipality and its county (or the state) give different
                             values for the same requirement -> both may apply, flag for review
  within_jurisdiction        two different documents of the same jurisdiction disagree
                             (e.g. code vs. a later amendment) -> flag for review
  comparison                 the user asked about several jurisdictions and they differ;
                             that's the expected answer, not a conflict, so no review flag
"""

import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence

from analysis.models import Conflict, Constraint, Gap
from analysis.text import DIGIT_NUM, WORD_NUM_PATTERN, canon_number, normalize, sentences
from retrieval.query_agent import REGULATION_TYPES, STATE_DOT

# ---------------------------------------------------------------------------
# Numeric requirement extraction
# ---------------------------------------------------------------------------
# Order matters: longer units first so "square feet" isn't read as "feet".
_UNITS = [
    (r"units?\s+per\s+(?:gross\s+|net\s+)?acre", "units/acre"),
    (r"square\s+feet|sq\.?\s*ft\.?|sf\b", "sq ft"),
    (r"acres?", "acre"),
    (r"feet|foot|ft\.?|'", "ft"),
    (r"percent|%", "%"),
    (r"stories|story", "stories"),
    (r"(?:off-street\s+)?(?:parking\s+)?spaces?", "spaces"),
]
_VALUE = rf"(?:{DIGIT_NUM}|\b(?:{WORD_NUM_PATTERN})\b)"
_QTY_RE = re.compile(
    rf"(?P<value>{_VALUE})\s*(?:-\s*)?(?P<unit>{'|'.join(f'(?:{u})' for u, _ in _UNITS)})",
    re.I)
_UNIT_CANON = [(re.compile(rf"^(?:{u})$", re.I), c) for u, c in _UNITS]

# measure name -> (pattern, regulation type, units it can be measured in). Specific first.
MEASURES = [
    ("front setback", r"front (?:yard|setback|building line)", "setback", {"ft"}),
    ("side setback", r"side (?:yard|setback)", "setback", {"ft"}),
    ("rear setback", r"rear (?:yard|setback)", "setback", {"ft"}),
    ("setback", r"setback|set back|\byards?\b|from (?:the|any) (?:property|lot) line", "setback", {"ft"}),
    ("building height", r"height|\btall\b", "height", {"ft", "stories"}),
    ("lot width", r"lot width|frontage", "lot_density", {"ft"}),
    ("lot coverage", r"lot coverage|impervious", "lot_density", {"%"}),
    ("lot area", r"lot area|lot size|minimum lot\b|lot served", "lot_density", {"sq ft", "acre"}),
    ("density", r"units per|density", "lot_density", {"units/acre"}),
    ("parking", r"parking", "parking", {"spaces"}),
    ("right-of-way width", r"right-of-way|right of way", "access_dot", {"ft"}),
    ("driveway spacing", r"driveway|spaced", "access_dot", {"ft"}),
    ("floor elevation", r"flood elevation|finished floor", "environmental", {"ft"}),
]
_MEASURE_RES = [(name, re.compile(pat, re.I), rtype, units) for name, pat, rtype, units in MEASURES]
MEASURE_TYPE = {name: rtype for name, _, rtype, _ in MEASURES}
# Units that identify the measure on their own.
UNIT_IMPLIES = {"spaces": "parking", "units/acre": "density"}

_MIN_RE = re.compile(r"\bminimum\b|\bat least\b|not (?:be )?less than|no less than|or more\b|"
                     r"\bmin\.", re.I)
_MAX_RE = re.compile(r"\bmaximum\b|\bexceed|no more than|\bat most\b|or less\b|"
                     r"\bup to\b|\bmax\.", re.I)
_DISTRICT_RE = re.compile(r"\b([A-Z]{1,3}-\d{1,2}[A-Z]?)\b|\b([A-Z]{2,3})\s+(?:zoning\s+)?district\b")


def _unit(raw: str) -> str:
    for rx, c in _UNIT_CANON:
        if rx.match(raw.strip()):
            return c
    return raw.lower()


def _district(text: str) -> Optional[str]:
    m = _DISTRICT_RE.search(text)
    return (m.group(1) or m.group(2)) if m else None


def _first_measure(window: str, unit: str) -> Optional[str]:
    """Most specific measure mentioned in the window that can be expressed in `unit`
    (MEASURES is ordered specific-first)."""
    for name, rx, _, units in _MEASURE_RES:
        if unit in units and rx.search(window):
            return name
    return None


def extract_constraints(passage) -> List[Constraint]:
    """For each quantity, the measure is looked for in the text between the previous
    quantity and this one (else just after it, up to the next quantity), so
    '... front yard setback shall be 25 feet and the rear yard setback shall be 20 feet'
    gives front=25 and rear=20."""
    out: List[Constraint] = []
    title = passage.section_title or ""
    body = passage.content or ""
    if body.startswith(title):
        body = body[len(title):]
    title_district = _district(title)
    for sent in sentences(body):
        sent_district = _district(sent) or title_district
        qtys = [m for m in _QTY_RE.finditer(sent)
                if not sent[m.end():m.end() + 6].lower().startswith(("-year", " year"))]
        for i, m in enumerate(qtys):
            start = qtys[i - 1].end() if i else 0
            nxt = qtys[i + 1].start() if i + 1 < len(qtys) else len(sent)
            before, after = sent[start:m.start()], sent[m.end():nxt]
            unit = _unit(m.group("unit"))
            measure = (UNIT_IMPLIES.get(unit) or _first_measure(before, unit)
                       or _first_measure(after, unit))
            if not measure:
                continue
            ctx = before + sent[m.start():m.end() + 30]
            comp = ("min" if _MIN_RE.search(ctx) else
                    "max" if _MAX_RE.search(ctx) else "unspecified")
            out.append(Constraint(
                jurisdiction=passage.jurisdiction, chunk_id=passage.chunk_id,
                doc_id=passage.doc_id, section_title=title, scope=passage.scope,
                measure=measure, comparator=comp,
                value=float(canon_number(m.group("value"))), unit=unit,
                district=sent_district, text=sent))
    return out


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------
def _fmt(v: float) -> str:
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def find_conflicts(constraints: Sequence[Constraint], pq) -> List[Conflict]:
    wanted_types = set(pq.regulation_types)
    asked = set(pq.jurisdictions)
    groups: Dict[tuple, List[Constraint]] = defaultdict(list)
    for c in constraints:
        if wanted_types and MEASURE_TYPE.get(c.measure) not in wanted_types:
            continue
        groups[(c.measure, c.comparator, c.unit)].append(c)

    conflicts: List[Conflict] = []
    seen = set()
    for (measure, comp, unit), cs in groups.items():
        for a, b in combinations(cs, 2):
            if a.value == b.value:
                continue
            if a.district and b.district and a.district != b.district:
                continue                      # different zoning districts may differ legitimately
            if a.jurisdiction == b.jurisdiction:
                if a.doc_id == b.doc_id:
                    continue                  # same document: usually different situations
                kind = "within_jurisdiction"
            elif a.jurisdiction in asked and b.jurisdiction in asked:
                kind = "comparison"
            else:
                kind = "overlapping_jurisdictions"
            pair = frozenset({(a.jurisdiction, a.value), (b.jurisdiction, b.value)})
            key = (kind, measure, comp, unit, pair)
            if key in seen:
                continue
            seen.add(key)
            entries = [{"jurisdiction": x.jurisdiction, "value": x.value, "unit": x.unit,
                        "section_title": x.section_title, "chunk_id": x.chunk_id,
                        "scope": x.scope, "district": x.district} for x in (a, b)]
            label = f"{'minimum ' if comp == 'min' else 'maximum ' if comp == 'max' else ''}{measure}"
            vals = " vs. ".join(f"{e['jurisdiction']} ({e['section_title']}): {_fmt(e['value'])} {unit}"
                                for e in entries)
            if kind == "overlapping_jurisdictions":
                note = (f"Different {label} requirements: {vals}. Both may apply to the same site; "
                        "confirm which governs with the local planning office.")
            elif kind == "within_jurisdiction":
                note = (f"{a.jurisdiction} documents disagree on the {label}: {vals}. One may "
                        "supersede the other; confirm the current rule.")
            else:
                note = f"The {label} differs: {vals}."
            conflicts.append(Conflict(kind, measure, comp, unit, entries,
                                      needs_review=kind != "comparison", note=note))
    return conflicts


# ---------------------------------------------------------------------------
# Gaps (missing information)
# ---------------------------------------------------------------------------
def _covers(rtype: str, passages: Iterable) -> bool:
    terms = REGULATION_TYPES[rtype]["triggers"] + REGULATION_TYPES[rtype]["expand"]
    for p in passages:
        hay = normalize(f"{p.section_title} {p.content}")
        if any(re.search(rf"\b{re.escape(t)}", hay) for t in terms):
            return True
    return False


def detect_gaps(pq, retrieved: Sequence, used: Sequence) -> List[Gap]:
    """retrieved: everything Part C returned; used: passages the answer cites."""
    gaps: List[Gap] = []
    got = {p.jurisdiction for p in retrieved}
    used_j = {p.jurisdiction for p in used}

    for j in pq.jurisdictions:
        if j not in got:
            gaps.append(Gap("jurisdiction_not_indexed",
                            f"No indexed regulations for {j} matched this question."))
        elif used and j not in used_j:
            gaps.append(Gap("jurisdiction_not_answered",
                            f"The retrieved {j} sections didn't directly answer this question."))

    for parent in pq.parent_jurisdictions:
        if parent not in got:
            gaps.append(Gap("parent_jurisdiction_missing",
                            f"{parent} rules may also apply, but no {parent} documents were found "
                            "in the index."))

    if pq.needs_state_dot:
        has_state = any(p.scope == "state" or (p.metadata or {}).get("jurisdiction_type") == "State"
                        for p in retrieved)
        if not has_state:
            dots = ", ".join(STATE_DOT[s] for s in pq.states if s in STATE_DOT) or "State DOT"
            gaps.append(Gap("state_dot_missing",
                            f"{dots} requirements may apply (state highway access), but none are "
                            "in the index."))

    if used:
        for rtype in pq.regulation_types:
            if not _covers(rtype, used):
                gaps.append(Gap("topic_not_covered",
                                f"The cited sections don't address {rtype.replace('_', ' ')}."))

    if used and not any((p.metadata or {}).get(k) for p in used
                        for k in ("effective_date", "adopted_date", "effective")):
        gaps.append(Gap("effective_date_unknown",
                        "Effective dates aren't recorded for these documents; confirm they are "
                        "current before relying on them."))
    return gaps
