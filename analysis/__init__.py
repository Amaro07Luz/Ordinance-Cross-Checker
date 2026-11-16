"""
Part D: Regulatory Analysis Agent.

Turns Part C's retrieved passages into a grounded answer:
summarize -> validate every citation -> revise once -> detect conflicts and gaps
-> answer, or abstain when the evidence doesn't support one.

Entry point (the contract Part E loads automatically, see INTEGRATION.md):

    from analysis.agent import AnalysisAgent
    AnalysisAgent().answer(question, parsed_query, passages) -> dict
"""

from analysis.agent import AnalysisAgent

__all__ = ["AnalysisAgent"]
