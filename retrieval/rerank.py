"""
Part C (3/3): Reranker
Re-scores the fused BM25+vector candidates with regulation-specific signals.
Optional: pass a cross_encoder(query, text) -> float to blend in a model score.
"""

import re
from typing import Callable, List, Optional, Tuple

from ingestion_pipeline_part_a import DocumentChunk
from retrieval.query_agent import ParsedQuery, REGULATION_TYPES

# Regulation types whose answers usually contain a number with a unit
QUANTITATIVE = {"setback", "height", "lot_density", "parking"}
NUMBER_WITH_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:feet|foot|ft|'|%|percent|acres?|sq\.?\s*ft|square feet|stories|spaces)\b",
    re.I)

SCOPE_BONUS = {"exact": 0.20, "parent": 0.08, "state": 0.05, "fallback": 0.0}


class HeuristicReranker:
    def __init__(self, cross_encoder: Optional[Callable[[str, str], float]] = None,
                 weights: Optional[dict] = None):
        self.cross_encoder = cross_encoder
        self.w = {"base": 1.0, "scope": 1.0, "title": 0.15, "number": 0.10,
                  "coverage": 0.20, "ce": 0.30}
        if weights:
            self.w.update(weights)

    def rerank(self, pq: ParsedQuery,
               items: List[Tuple[DocumentChunk, float, str]]):
        """items: (chunk, normalized_fused_score, scope). Returns
        [(chunk, final_score, scope, reasons)] sorted best-first."""
        scored = []
        kw = {k for k in pq.keywords if len(k) > 2}
        type_terms = {t for rt in pq.regulation_types
                      for t in REGULATION_TYPES[rt]["expand"] + REGULATION_TYPES[rt]["triggers"]}

        for chunk, base, scope in items:
            reasons = [f"fused={base:.2f}", f"scope={scope}"]
            score = self.w["base"] * base + self.w["scope"] * SCOPE_BONUS[scope]

            title = chunk.section_title.lower()
            if any(t in title for t in type_terms):
                score += self.w["title"]
                reasons.append("section title matches regulation type")

            if QUANTITATIVE & set(pq.regulation_types) and NUMBER_WITH_UNIT.search(chunk.content):
                score += self.w["number"]
                reasons.append("contains numeric requirement")

            if kw:
                body = chunk.content.lower()
                cov = sum(1 for k in kw if k in body) / len(kw)
                score += self.w["coverage"] * cov
                reasons.append(f"keyword coverage={cov:.0%}")

            if self.cross_encoder:
                ce = float(self.cross_encoder(pq.raw_query, chunk.content))
                score += self.w["ce"] * ce
                reasons.append(f"cross-encoder={ce:.2f}")

            scored.append((chunk, score, scope, reasons))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def low_confidence(passages, min_top_score: float = 0.9) -> bool:
    """Helper for workstream D's abstain path. Scores are not capped at 1.0
    (base + bonuses, roughly 0-1.7); tune min_top_score on the real eval set."""
    return (not passages) or passages[0].score < min_top_score or passages[0].scope == "fallback"
