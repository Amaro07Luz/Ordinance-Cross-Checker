"""
Answer generation seam for Part D (analysis agent).

PART D IS NOT ON A BRANCH YET. Until it is, the API uses ExtractiveAnswerer,
a no-LLM stand-in that quotes the best-matching sentences from the top
passages and abstains when Part C reports low confidence.

When Part D is merged, nothing here needs to change IF Part D provides:

    analysis/agent.py
        class AnalysisAgent:
            def answer(self, question: str, parsed: ParsedQuery,
                       passages: List[RetrievedPassage]) -> dict | object

returning (as dict keys or attributes):
    answer / text   str            the answer shown to the user
    citations       list           chunk_id strings, or dicts/objects with chunk_id
    abstained       bool           True when D declines to answer
    confidence      float 0..1     optional
    warnings        list[str]      optional

ANSWERER=auto (default) then picks up Part D automatically; ANSWERER=stub
forces the stand-in; ANSWERER=partd fails loudly if Part D is missing.
If Part D's module/class/method names differ, edit only PART_D_MODULE /
PART_D_CLASS / PART_D_METHOD below or PartDAdapter._normalize().
"""

import importlib
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from retrieval.query_agent import ParsedQuery
from retrieval.rerank import low_confidence
from retrieval.search import RetrievedPassage

log = logging.getLogger(__name__)

PART_D_MODULE = "analysis.agent"
PART_D_CLASS = "AnalysisAgent"
PART_D_METHOD = "answer"


@dataclass
class Citation:
    n: int
    chunk_id: str
    jurisdiction: str
    section_title: str
    source_url: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    quote: Optional[str] = None


@dataclass
class Answer:
    text: str
    citations: List[Citation]
    abstained: bool
    confidence: float
    answerer: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def citation_from(p: RetrievedPassage, n: int, quote: Optional[str] = None) -> Citation:
    m = p.metadata or {}
    return Citation(n=n, chunk_id=p.chunk_id, jurisdiction=p.jurisdiction,
                    section_title=p.section_title, source_url=p.source_url,
                    page_start=m.get("page_start"), page_end=m.get("page_end"), quote=quote)


# ----------------------------------------------------------------------
# Stand-in until Part D lands
# ----------------------------------------------------------------------
SENT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
NUM_RE = re.compile(r"\d")


class ExtractiveAnswerer:
    name = "extractive-stub"

    def __init__(self, max_passages: int = 3, sentences_per_passage: int = 2,
                 relative_cutoff: float = 0.8):
        self.max_passages = max_passages
        self.sentences_per_passage = sentences_per_passage
        self.relative_cutoff = relative_cutoff   # quote only passages scoring near the best one

    def answer(self, question: str, parsed: ParsedQuery,
               passages: List[RetrievedPassage]) -> Answer:
        warnings = list(parsed.warnings)
        if low_confidence(passages):
            where = ", ".join(parsed.jurisdictions) or "the indexed jurisdictions"
            return Answer(
                text=(f"I couldn't find a reliable answer for {where} in the indexed regulations. "
                      "Try naming the city or county, or check the source documents directly."),
                citations=[citation_from(p, i) for i, p in enumerate(passages[:2], 1)],
                abstained=True, confidence=0.0, answerer=self.name, warnings=warnings)

        keywords = {k for k in parsed.keywords if len(k) > 2}
        lines, cites = [], []
        top = passages[0].score
        strong = [p for p in passages[: self.max_passages] if p.score >= self.relative_cutoff * top]
        for n, p in enumerate(strong, start=1):
            best = self._best_sentences(p.content, p.section_title, keywords)
            if not best:
                continue
            quote = " ".join(best)
            cites.append(citation_from(p, n, quote))
            lines.append(f"{p.jurisdiction}, {p.section_title.rstrip('.:')}: {quote} [{n}]")
        if not lines:
            return Answer("The retrieved sections did not contain a direct answer.", [],
                          True, 0.0, self.name, warnings)
        confidence = round(min(1.0, passages[0].score / 1.5), 2)
        if any((p.metadata or {}).get("ocr") for p in passages[:len(cites)]):
            warnings.append("Some text came from OCR of a scanned page; verify numbers in the source.")
        return Answer("\n".join(lines), cites, False, confidence, self.name, warnings)

    def _best_sentences(self, content: str, title: str, keywords) -> List[str]:
        body = content[len(title):].strip() if content.startswith(title) else content
        sents = [s.strip() for s in SENT_RE.split(body.replace("\n", " ")) if len(s.strip()) > 15]
        scored = sorted(
            ((sum(k in s.lower() for k in keywords) + 0.5 * bool(NUM_RE.search(s)), i, s)
             for i, s in enumerate(sents)), reverse=True)
        top = sorted(scored[: self.sentences_per_passage], key=lambda x: x[1])
        return [s for score, _, s in top if score > 0] or sents[:1]


# ----------------------------------------------------------------------
# Part D adapter
# ----------------------------------------------------------------------
class PartDAdapter:
    def __init__(self, agent: Any):
        self.agent = agent
        self.name = f"part-d:{type(agent).__name__}"

    def answer(self, question, parsed, passages) -> Answer:
        raw = getattr(self.agent, PART_D_METHOD)(question, parsed, passages)
        return self._normalize(raw, parsed, passages)

    def _normalize(self, raw: Any, parsed: ParsedQuery, passages: List[RetrievedPassage]) -> Answer:
        get = (lambda k, d=None: raw.get(k, d)) if isinstance(raw, dict) else \
              (lambda k, d=None: getattr(raw, k, d))
        by_id = {p.chunk_id: p for p in passages}
        cites: List[Citation] = []
        for i, c in enumerate(get("citations", []) or [], start=1):
            cid = c if isinstance(c, str) else (c.get("chunk_id") if isinstance(c, dict)
                                                else getattr(c, "chunk_id", None))
            quote = None if isinstance(c, str) else (c.get("quote") if isinstance(c, dict)
                                                     else getattr(c, "quote", None))
            if cid in by_id:
                cites.append(citation_from(by_id[cid], i, quote))
        return Answer(
            text=get("answer") or get("text") or "",
            citations=cites,
            abstained=bool(get("abstained", False)),
            confidence=float(get("confidence", 0.0) or 0.0),
            answerer=self.name,
            warnings=list(parsed.warnings) + list(get("warnings", []) or []),
        )


def load_answerer(mode: str = "auto"):
    mode = (mode or "auto").lower()
    if mode == "stub":
        return ExtractiveAnswerer()
    try:
        module = importlib.import_module(PART_D_MODULE)
        agent = getattr(module, PART_D_CLASS)()
        log.info("using Part D analysis agent", extra={"module": PART_D_MODULE})
        return PartDAdapter(agent)
    except (ImportError, AttributeError) as e:
        if mode == "partd":
            raise RuntimeError(f"ANSWERER=partd but Part D is unavailable: {e}") from e
        log.info("Part D not found; using extractive stub", extra={"reason": str(e)})
        return ExtractiveAnswerer()
