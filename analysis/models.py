"""Data structures produced by Part D. Everything serializes to plain dicts."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class Claim:
    """One atomic statement in the answer, tied to one passage and a verbatim quote."""
    statement: str
    passage_ref: str                  # "P1".. as shown to the LLM
    quote: str
    chunk_id: str = ""
    jurisdiction: str = ""
    section_title: str = ""
    problems: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems


@dataclass
class Constraint:
    """A numeric requirement found in a passage, e.g. front setback >= 25 ft."""
    jurisdiction: str
    chunk_id: str
    doc_id: str
    section_title: str
    scope: str
    measure: str                      # "front setback", "building height", ...
    comparator: str                   # "min" | "max" | "unspecified"
    value: float
    unit: str                         # "ft", "%", "acre", "sq ft", "stories", ...
    district: Optional[str]
    text: str                         # the sentence it came from


@dataclass
class Conflict:
    kind: str                         # "overlapping_jurisdictions" | "within_jurisdiction" | "comparison"
    measure: str
    comparator: str
    unit: str
    entries: List[Dict]               # [{jurisdiction, value, section_title, chunk_id, scope}]
    needs_review: bool
    note: str


@dataclass
class Gap:
    kind: str                         # see conflicts.detect_gaps
    detail: str


@dataclass
class AnalysisResult:
    answer: str
    citations: List[Dict]
    abstained: bool
    confidence: float
    warnings: List[str] = field(default_factory=list)
    # Extra fields: Part E's adapter ignores these today; they are here so the API/UI
    # can show them once E passes them through.
    claims: List[Claim] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    gaps: List[Gap] = field(default_factory=list)
    needs_human_review: bool = False
    mode: str = "extractive"          # "llm" | "extractive"
    model: Optional[str] = None
    revisions: int = 0
    dropped_claims: int = 0
    timing_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)
