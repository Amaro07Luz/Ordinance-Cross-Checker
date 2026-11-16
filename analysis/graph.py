"""
The analysis pipeline (Part D's "graph"). Plain Python: the flow is linear with one
bounded revise loop, so a graph framework would add a dependency without adding control.

    precheck ──abstain──────────────────────────────────────────────┐
       │                                                            │
    select evidence                                                 │
       │                                                            │
    draft (LLM, or extractive if no LLM / LLM failed)               │
       │                                                            │
    validate citations ──failures & time left──► revise (once) ─┐   │
       │◄───────────────────────────────────────────────────────┘   │
    drop unverifiable claims (fall back to extractive if none left) │
       │                                                            │
    cross-reference: conflicts + gaps                               │
       │                                                            │
    compose answer + citations + confidence ◄───────────────────────┘
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from analysis.citations import jurisdictions_named, problems_report, validate_claims
from analysis.conflicts import detect_gaps, extract_constraints, find_conflicts
from analysis.llm import LLMClient, LLMError
from analysis.models import AnalysisResult, Claim, Gap
from analysis.prompts import (
    SYSTEM_PROMPT,
    build_revision_message,
    build_user_message,
    previous_turn,
    refs_for,
)
from analysis.text import all_numbers, quantities, sentences
from retrieval.query_agent import GAZETTEER, REGULATION_TYPES
from retrieval.rerank import QUANTITATIVE, low_confidence

log = logging.getLogger(__name__)

_REF_MARK_RE = re.compile(r"\s*\[(?:P\d+(?:\s*,\s*P\d+)*|\d+)\]")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class PipelineConfig:
    budget_s: float = 20.0               # total LLM time; keep below Part E's ANSWER_TIMEOUT_S (30)
    min_revise_s: float = 4.0            # only revise if at least this much budget is left
    max_llm_passages: int = 5
    llm_relative_cutoff: float = 0.6     # passages scoring below 60% of the best aren't sent
    max_extractive_passages: int = 3
    extractive_relative_cutoff: float = 0.8
    sentences_per_passage: int = 2
    min_quote_words: int = 4
    min_top_score: float = 0.9           # Part C's low_confidence threshold

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(budget_s=_env_float("ANALYSIS_TIMEOUT_S", cls.budget_s),
                   min_top_score=_env_float("ANALYSIS_MIN_TOP_SCORE", cls.min_top_score))


class AnalysisPipeline:
    def __init__(self, llm: Optional[LLMClient] = None, config: Optional[PipelineConfig] = None,
                 llm_problem: Optional[str] = None):
        self.llm = llm
        self.cfg = config or PipelineConfig()
        self.llm_problem = llm_problem

    # ------------------------------------------------------------------
    def run(self, question: str, pq, passages: Sequence) -> AnalysisResult:
        t0 = time.perf_counter()
        deadline = t0 + self.cfg.budget_s
        passages = list(passages or [])
        warnings: List[str] = [self.llm_problem] if self.llm_problem else []

        reason = self._precheck(pq, passages)
        if reason:
            return self._abstain(reason, pq, passages, warnings, t0)

        in_scope = [p for p in passages if p.scope != "fallback"]
        mode, claims, summary, extra_gaps = "extractive", [], None, []
        revisions = dropped = 0
        t_llm = 0.0

        if self.llm is not None:
            tl = time.perf_counter()
            try:
                evidence = self._select(in_scope, pq, self.cfg.max_llm_passages,
                                        self.cfg.llm_relative_cutoff)
                claims, summary, missing, answerable, revisions, dropped = \
                    self._llm_draft(question, pq, evidence, deadline, warnings)
                extra_gaps = [Gap("not_in_passages", m) for m in missing]
                if claims:
                    mode = "llm"
                elif not answerable:
                    t_llm = time.perf_counter() - tl
                    why = "; ".join(missing) or "the retrieved sections don't answer this question"
                    res = self._abstain(f"The retrieved regulations don't answer this: {why}.",
                                        pq, passages, warnings, t0, extra_gaps)
                    res.mode, res.model = "llm", self.llm.name
                    res.timing_ms["llm"] = round(t_llm * 1000, 1)
                    return res
                else:
                    warnings.append("The AI summary couldn't be verified against the sources, so "
                                    "this answer quotes them directly.")
            except LLMError as e:
                log.warning("analysis LLM failed; using extractive mode", extra={"error": str(e)})
                warnings.append("The language model was unavailable, so this answer quotes the "
                                "sources directly.")
                claims, summary = [], None
            t_llm = time.perf_counter() - tl

        if mode == "extractive":
            claims = self._extractive(pq, in_scope)
            summary = None
        if not claims:
            return self._abstain("The retrieved sections did not contain a direct answer.",
                                 pq, passages, warnings, t0, extra_gaps)

        # ---- cross-reference ----------------------------------------
        by_id = {p.chunk_id: p for p in passages}
        cited_ids = list(dict.fromkeys(c.chunk_id for c in claims))
        used = [by_id[c] for c in cited_ids]
        constraints = [k for p in in_scope for k in extract_constraints(p)]
        conflicts = find_conflicts(constraints, pq)
        gaps = detect_gaps(pq, passages, used) + extra_gaps
        needs_review = any(c.needs_review for c in conflicts)

        if summary and not self._summary_ok(summary, claims, pq):
            log.info("dropping unverifiable summary", extra={"summary": summary[:200]})
            summary = None

        text, citations = self._compose(summary, claims, used, conflicts)
        for c in conflicts:
            if c.needs_review:
                warnings.append(f"Possible conflict, flagged for human review: {c.note}")
        for g in gaps:
            prefix = "Note" if g.kind == "effective_date_unknown" else "Not found"
            warnings.append(f"{prefix}: {g.detail}")
        if dropped:
            warnings.append(f"{dropped} statement(s) were removed because they couldn't be matched "
                            "to the source text.")
        if any((p.metadata or {}).get("ocr") for p in used):
            warnings.append("Some cited text came from OCR of a scanned page; verify the numbers "
                            "in the source.")

        return AnalysisResult(
            answer=text, citations=citations, abstained=False,
            confidence=self._confidence(passages, len(claims), dropped, needs_review, gaps),
            warnings=list(dict.fromkeys(warnings)), claims=claims, conflicts=conflicts, gaps=gaps,
            needs_human_review=needs_review, mode=mode,
            model=self.llm.name if (self.llm is not None and mode == "llm") else None,
            revisions=revisions, dropped_claims=dropped,
            timing_ms={"total": round((time.perf_counter() - t0) * 1000, 1),
                       "llm": round(t_llm * 1000, 1)})

    # ------------------------------------------------------------------
    def _precheck(self, pq, passages) -> Optional[str]:
        supported = ", ".join(GAZETTEER)
        if not passages:
            return "No regulations were retrieved for this question."
        if not pq.jurisdictions and not pq.states:
            return ("I couldn't match this question to a supported jurisdiction. Name one of: "
                    f"{supported} (or a ZIP code).")
        if pq.jurisdictions:
            wanted = set(pq.jurisdictions) | set(pq.parent_jurisdictions)
            if not any(p.jurisdiction in wanted for p in passages):
                return (f"No indexed regulations for {', '.join(pq.jurisdictions)} matched this "
                        "question, and rules from other places don't apply there.")
        if low_confidence(passages, self.cfg.min_top_score):
            where = ", ".join(pq.jurisdictions or pq.states)
            return (f"I couldn't find a reliable answer for {where} in the indexed regulations. "
                    "Try rephrasing with the specific requirement (e.g. front setback, plat "
                    "approval), or check the source documents directly.")
        return None

    def _abstain(self, reason: str, pq, passages, warnings, t0, extra_gaps=()) -> AnalysisResult:
        gaps = detect_gaps(pq, passages, []) + list(extra_gaps)
        text = reason
        details = [g.detail for g in gaps if g.kind != "effective_date_unknown"]
        if details:
            text += "\n\n" + "\n".join(f"• {d}" for d in dict.fromkeys(details))
        return AnalysisResult(answer=text, citations=[], abstained=True, confidence=0.0,
                              warnings=list(dict.fromkeys(warnings)), gaps=gaps,
                              timing_ms={"total": round((time.perf_counter() - t0) * 1000, 1)})

    # ------------------------------------------------------------------
    @staticmethod
    def _select(in_scope, pq, limit: int, cutoff: float) -> list:
        if not in_scope:
            return []
        top = in_scope[0].score
        chosen = [p for p in in_scope[:limit] if p.score >= cutoff * top]
        # In multi-jurisdiction questions make sure each place gets its best passage.
        for j in pq.jurisdictions:
            if not any(p.jurisdiction == j for p in chosen):
                best = next((p for p in in_scope if p.jurisdiction == j), None)
                if best is not None:
                    chosen.append(best)
        return chosen

    def _llm_draft(self, question, pq, evidence, deadline, warnings
                   ) -> Tuple[List[Claim], Optional[str], List[str], bool, int, int]:
        refs = refs_for(evidence)
        messages = [{"role": "user", "content": build_user_message(question, pq, refs)}]
        draft = self.llm.complete_json(SYSTEM_PROMPT, messages, deadline - time.perf_counter())
        claims = validate_claims(_parse_claims(draft), refs, self.cfg.min_quote_words)
        revisions = 0

        if any(not c.valid for c in claims) and deadline - time.perf_counter() >= self.cfg.min_revise_s:
            report = problems_report(claims)
            follow = messages + previous_turn(draft) + [
                {"role": "user", "content": build_revision_message(report)}]
            try:
                revised = self.llm.complete_json(SYSTEM_PROMPT, follow, deadline - time.perf_counter())
                draft = revised
                claims = validate_claims(_parse_claims(draft), refs, self.cfg.min_quote_words)
                revisions = 1
            except LLMError as e:
                log.warning("revision call failed; keeping verified claims from the first draft",
                            extra={"error": str(e)})

        valid = [c for c in claims if c.valid]
        dropped = len(claims) - len(valid)
        if dropped:
            log.info("dropped unverifiable claims",
                     extra={"problems": problems_report(claims)[:1000]})
        for c in valid:
            c.statement = _REF_MARK_RE.sub("", c.statement).strip()
        summary = draft.get("summary")
        summary = _REF_MARK_RE.sub("", summary).strip() if isinstance(summary, str) else None
        missing = [str(m) for m in (draft.get("missing") or []) if str(m).strip()]
        answerable = draft.get("answerable", True) is not False
        return valid, summary or None, missing, answerable, revisions, dropped

    def _extractive(self, pq, in_scope) -> List[Claim]:
        evidence = self._select(in_scope, pq, self.cfg.max_extractive_passages,
                                self.cfg.extractive_relative_cutoff)
        refs = refs_for(evidence)
        keywords = {k for k in pq.keywords if len(k) > 2}
        type_res = [re.compile(rf"\b{re.escape(t)}") for rt in pq.regulation_types
                    for t in REGULATION_TYPES[rt]["triggers"]]
        want_numbers = bool(QUANTITATIVE & set(pq.regulation_types))

        def score(s: str) -> float:
            low = s.lower()
            words = re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", low)
            kw = sum(any(w == k or (w.startswith(k) and len(w) - len(k) <= 2) for w in words)
                     for k in keywords)          # whole words, allowing plurals
            return kw + 0.5 * sum(bool(rx.search(low)) for rx in type_res) + \
                (0.5 if want_numbers and quantities(s) else 0.0)

        picked_by_ref = []
        for rank, (ref, p) in enumerate(refs.items()):
            body = p.content[len(p.section_title):] if p.content.startswith(p.section_title) else p.content
            sents = [s for s in sentences(body) if not _is_heading(s)]
            if not sents:
                continue
            scored = sorted(((score(s), i, s) for i, s in enumerate(sents)), key=lambda x: (-x[0], x[1]))
            best = [x for x in scored[: self.cfg.sentences_per_passage] if x[0] >= 1.0]
            if not best:
                if rank != 0:
                    continue                  # not about the question; leave it out
                best = scored[:1]             # top-ranked passage: keep its best sentence
            picked_by_ref.append((best[0][0], rank, ref, [s for _, _, s in sorted(best, key=lambda x: x[1])]))

        claims = []
        for _, _, ref, sents_ in sorted(picked_by_ref, key=lambda x: (-x[0], x[1])):
            claims += [Claim(statement=s, passage_ref=ref, quote=s) for s in sents_]
        # A verbatim sentence shown under its own heading can't misattribute a jurisdiction.
        validate_claims(claims, refs, min_quote_words=1, check_jurisdiction=False)
        return [c for c in claims if c.valid]

    @staticmethod
    def _summary_ok(summary: str, claims: List[Claim], pq) -> bool:
        allowed = set().union(*(all_numbers(c.quote) | all_numbers(c.section_title) for c in claims))
        if quantities(summary) - allowed:
            return False
        named = jurisdictions_named(summary)
        known = {c.jurisdiction for c in claims} | set(pq.jurisdictions) | set(pq.parent_jurisdictions)
        return named <= known

    @staticmethod
    def _compose(summary, claims: List[Claim], used, conflicts) -> Tuple[str, List[Dict]]:
        number = {p.chunk_id: n for n, p in enumerate(used, 1)}
        lines: List[str] = []
        if summary:
            lines += [summary, ""]
        for p in used:
            mine = [c for c in claims if c.chunk_id == p.chunk_id]
            scope = {"parent": " (county rules that may also apply)",
                     "state": " (state rules)"}.get(p.scope, "")
            lines.append(f"{p.jurisdiction} — {p.section_title.rstrip('.: ')}{scope}")
            lines += [f"• {c.statement.rstrip()} [{number[p.chunk_id]}]" for c in mine]
            lines.append("")
        comparisons = [c for c in conflicts if c.kind == "comparison"]
        if comparisons:
            lines.append("Differences between jurisdictions")
            lines += [f"• {c.note}" for c in comparisons]
        citations = []
        for p in used:
            quotes = list(dict.fromkeys(c.quote.strip() for c in claims if c.chunk_id == p.chunk_id))
            citations.append({"n": number[p.chunk_id], "chunk_id": p.chunk_id,
                              "quote": " … ".join(quotes), "jurisdiction": p.jurisdiction,
                              "section_title": p.section_title, "source_url": p.source_url})
        return "\n".join(lines).strip(), citations

    @staticmethod
    def _confidence(passages, n_valid, dropped, needs_review, gaps) -> float:
        retrieval = min(1.0, passages[0].score / 1.5)
        verified = n_valid / (n_valid + dropped) if (n_valid + dropped) else 0.0
        conf = retrieval * (0.6 + 0.4 * verified)
        if needs_review:
            conf *= 0.85
        serious = [g for g in gaps if g.kind not in ("effective_date_unknown", "parent_jurisdiction_missing")]
        conf -= min(0.2, 0.05 * len(serious))
        return round(max(0.05, min(1.0, conf)), 2)


def _is_heading(s: str) -> bool:
    """All-caps document titles like 'HARRIS COUNTY REGULATIONS ... (SYNTHETIC EXCERPT)'."""
    letters = [ch for ch in s if ch.isalpha()]
    return bool(letters) and sum(ch.isupper() for ch in letters) / len(letters) > 0.8


def _parse_claims(draft: dict) -> List[Claim]:
    out = []
    for item in draft.get("claims") or []:
        if not isinstance(item, dict):
            out.append(Claim(statement=str(item), passage_ref="", quote=""))
            continue
        out.append(Claim(
            statement=str(item.get("statement") or item.get("claim") or "").strip(),
            passage_ref=str(item.get("passage") or item.get("passage_ref")
                            or item.get("source") or "").strip(),
            quote=str(item.get("quote") or "").strip()))
    return out
