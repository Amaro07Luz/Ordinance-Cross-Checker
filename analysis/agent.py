"""
Part D entry point. Part E loads this automatically (ANSWERER=auto), see INTEGRATION.md:

    class AnalysisAgent:
        def answer(self, question, parsed, passages) -> dict

`parsed` is retrieval.query_agent.ParsedQuery, `passages` is
List[retrieval.search.RetrievedPassage]. The returned dict has the keys E's
PartDAdapter reads (answer, citations, abstained, confidence, warnings) plus
extra keys (claims, conflicts, gaps, needs_human_review, mode, ...) that E
ignores until it passes them through.

Notes for E:
  * The constructor takes no required arguments and never raises, so it can't
    break startup.
  * answer() doesn't raise on LLM problems; it falls back to quoting the sources
    and says so in `warnings`. Its LLM budget (ANALYSIS_TIMEOUT_S, default 20 s)
    stays below E's ANSWER_TIMEOUT_S (30 s).
  * Every citation returned refers to a passage it was given, in order 1..n,
    matching the [n] markers in the answer text, so E's adapter numbering lines up.
  * parsed.warnings are NOT repeated here: E's adapter already adds them.
"""

import logging
from typing import List, Optional

from analysis.graph import AnalysisPipeline, PipelineConfig
from analysis.llm import LLMClient, llm_from_env

log = logging.getLogger(__name__)


class AnalysisAgent:
    def __init__(self, llm: Optional[LLMClient] = None, config: Optional[PipelineConfig] = None,
                 use_env_llm: bool = True):
        problem = None
        if llm is None and use_env_llm:
            llm, problem = llm_from_env()
        if problem:
            log.error(problem)
        self.llm = llm
        self.pipeline = AnalysisPipeline(llm, config or PipelineConfig.from_env(), llm_problem=problem)
        log.info("Part D analysis agent ready",
                 extra={"llm": getattr(llm, "name", None) or "none (extractive)"})

    @property
    def mode(self) -> str:
        return "llm" if self.llm is not None else "extractive"

    def answer(self, question: str, parsed, passages: List) -> dict:
        return self.pipeline.run(question, parsed, passages).to_dict()
