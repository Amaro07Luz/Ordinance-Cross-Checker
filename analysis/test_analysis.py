"""
Tests for Part D (analysis agent). No network and no API key: the LLM is FakeLLM.
Passages are SYNTHETIC, in the same shape Part C returns.

Run from the repo root (either works):
    python -m analysis.test_analysis
    python -m pytest -q analysis/test_analysis.py
"""

import os
import sys

from analysis.agent import AnalysisAgent
from analysis.citations import is_verbatim, validate_claims
from analysis.conflicts import detect_gaps, extract_constraints, find_conflicts
from analysis.graph import PipelineConfig
from analysis.llm import FakeLLM, LLMError, parse_json
from analysis.models import Claim
from retrieval.query_agent import QueryParser
from retrieval.search import RetrievedPassage

PARSER = QueryParser()


def P(cid, jur, section, text, score=1.4, scope="exact", state="TX", doc=None, rank=1, **meta):
    return RetrievedPassage(chunk_id=cid, doc_id=doc or cid.rsplit("_", 1)[0], jurisdiction=jur,
                            state=state, section_title=section, content=f"{section}\n{text}",
                            source_url=f"https://example.gov/{cid}", score=score, rank=rank,
                            scope=scope, match_reasons=[], metadata=meta)


FUL_SETBACK = P("ful_udc_1", "Fulshear", "Section 3.2 Residential Setbacks",
                "In the R-1 Single-Family Residential district the minimum front yard setback shall be "
                "25 feet measured from the property line. The minimum side yard setback shall be 7.5 feet "
                "and the minimum rear yard setback shall be 20 feet.")
FUL_HEIGHT = P("ful_udc_2", "Fulshear", "Section 3.3 Building Height",
               "No principal building in a residential district shall exceed 35 feet or two and one-half "
               "stories in height.", score=1.1, rank=2)
FTB_SETBACK = P("ftb_regs_1", "Fort Bend County", "Section 7.1 Building Lines",
                "Within subdivisions in the extraterritorial jurisdiction the minimum front yard setback "
                "shall be 30 feet from the right-of-way line.", score=1.2, scope="parent", rank=2)
EVE_SETBACK = P("eve_160_1", "Evesham Township (Marlton)", "§ 160-11. Yard requirements.",
                "In the R-1 district the minimum front yard setback shall be 40 feet.",
                score=1.3, state="NJ", rank=2)
HAR_DRIVE = P("har_infra_1", "Harris County", "Section 4.02 Driveway Access",
              "A driveway permit is required before constructing any driveway that connects to a county "
              "road. Driveways on major thoroughfares shall be spaced a minimum of 150 feet apart.")
MON_SEPTIC = P("mon_eh_1", "Montgomery County", "Section 2.1 On-Site Sewage Facilities",
               "The minimum lot size for a lot served by an on-site sewage facility and a private water "
               "well is one acre.")


def agent(llm=None, **cfg):
    return AnalysisAgent(llm=llm, config=PipelineConfig(**cfg), use_env_llm=False)


def good_llm_answer(**over):
    d = {"answerable": True,
         "summary": "In Fulshear's R-1 district the minimum front yard setback is 25 feet.",
         "claims": [{"passage": "P1",
                     "statement": "The minimum front yard setback in R-1 is 25 feet.",
                     "quote": "the minimum front yard setback shall be 25 feet measured from the "
                              "property line"},
                    {"passage": "P1",
                     "statement": "Side yards must be at least 7.5 feet and rear yards 20 feet.",
                     "quote": "The minimum side yard setback shall be 7.5 feet and the minimum rear yard "
                              "setback shall be 20 feet."}],
         "missing": []}
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# Citation validator
# ---------------------------------------------------------------------------
def test_verbatim_matching():
    content = FUL_SETBACK.content
    assert is_verbatim("minimum front yard setback shall be 25 feet", content)
    assert is_verbatim("MINIMUM  FRONT\nyard setback shall be 25 feet", content)       # case/whitespace
    assert is_verbatim("the minimum front yard setback ... rear yard setback shall be 20 feet", content)
    assert not is_verbatim("minimum front setback is 25 feet", content)                # paraphrase
    assert not is_verbatim("rear yard setback shall be 20 feet ... front yard setback", content)  # order


def test_validator_flags_each_failure_mode():
    refs = {"P1": FUL_SETBACK}
    claims = validate_claims([
        Claim("Front setback is 25 feet.", "P1", "minimum front yard setback shall be 25 feet"),   # ok
        Claim("Front setback is 30 feet.", "P1", "minimum front yard setback shall be 25 feet"),   # number
        Claim("Front setback is 25 feet.", "P1", "front setbacks are twenty five feet"),  # not verbatim
        Claim("Front setback is 25 feet.", "P7", "minimum front yard setback shall be 25 feet"),   # bad ref
        Claim("In Harris County the setback is 25 feet.", "P1",
              "minimum front yard setback shall be 25 feet"),                          # wrong place
        Claim("Front setback is 25 feet.", "P1", "25 feet"),                                       # too short
    ], refs)
    assert claims[0].valid and claims[0].chunk_id == "ful_udc_1"
    assert "30" in claims[1].problems[0]
    assert "exact excerpt" in claims[2].problems[0]
    assert "not provided" in claims[3].problems[0]
    assert any("Harris County" in pr for pr in claims[4].problems)
    assert any("at least" in pr for pr in claims[5].problems)


def test_validator_ignores_identifiers_and_reads_word_numbers():
    refs = {"P1": FUL_SETBACK, "P2": MON_SEPTIC}
    claims = validate_claims([
        Claim("Under Section 3.2, the R-1 front setback is 25 feet.", "P1",
              "minimum front yard setback shall be 25 feet"),
        Claim("Septic lots need at least 1 acre.", "P2",
              "private water well is one acre"),
    ], refs)
    assert all(c.valid for c in claims), [c.problems for c in claims]


# ---------------------------------------------------------------------------
# Constraint extraction, conflicts and gaps
# ---------------------------------------------------------------------------
def test_extracts_numeric_constraints():
    got = {(c.measure, c.comparator, c.value, c.unit) for c in extract_constraints(FUL_SETBACK)}
    assert ("front setback", "min", 25.0, "ft") in got
    assert ("side setback", "min", 7.5, "ft") in got
    assert ("rear setback", "min", 20.0, "ft") in got
    h = extract_constraints(FUL_HEIGHT)
    assert ("building height", "max", 35.0, "ft") in {(c.measure, c.comparator, c.value, c.unit) for c in h}
    s = extract_constraints(MON_SEPTIC)
    assert ("lot area", "min", 1.0, "acre") in {(c.measure, c.comparator, c.value, c.unit) for c in s}


def test_city_vs_county_conflict_is_flagged_for_review():
    pq = PARSER.parse("front yard setback in Fulshear")
    cons = extract_constraints(FUL_SETBACK) + extract_constraints(FTB_SETBACK)
    conflicts = find_conflicts(cons, pq)
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.kind == "overlapping_jurisdictions" and c.needs_review
    assert {e["value"] for e in c.entries} == {25.0, 30.0}


def test_two_cities_asked_is_a_comparison_not_a_conflict():
    pq = PARSER.parse("compare front yard setback in Fulshear and Marlton")
    cons = extract_constraints(FUL_SETBACK) + extract_constraints(EVE_SETBACK)
    conflicts = find_conflicts(cons, pq)
    assert [c.kind for c in conflicts] == ["comparison"]
    assert not conflicts[0].needs_review


def test_no_false_conflicts():
    pq = PARSER.parse("right-of-way width in Harris County")
    same_doc = P("har_infra_2", "Harris County", "Section 4.03 Right-of-Way Width",
                 "Major thoroughfares shall have a minimum right-of-way width of 100 feet. Local "
                 "residential streets shall have a minimum right-of-way width of 60 feet.")
    assert find_conflicts(extract_constraints(same_doc), pq) == []       # same document: two cases
    r2 = P("ful_udc_9", "Fulshear", "Section 3.7 Estate District",
           "In the E-1 district the minimum front yard setback shall be 50 feet.", doc="ful_other")
    pq2 = PARSER.parse("front setback in Fulshear")
    cons = extract_constraints(FUL_SETBACK) + extract_constraints(r2)
    assert find_conflicts(cons, pq2) == []                               # different districts


def test_gaps():
    pq = PARSER.parse("front yard setback in Fulshear")
    kinds = {g.kind for g in detect_gaps(pq, [FUL_SETBACK], [FUL_SETBACK])}
    assert "parent_jurisdiction_missing" in kinds           # Fort Bend County not retrieved
    assert "effective_date_unknown" in kinds
    pq = PARSER.parse("driveway access requirements in Cypress")
    kinds = {g.kind for g in detect_gaps(pq, [HAR_DRIVE], [HAR_DRIVE])}
    assert "state_dot_missing" in kinds
    pq = PARSER.parse("building height in Fulshear")
    kinds = {g.kind for g in detect_gaps(pq, [FUL_SETBACK], [FUL_SETBACK])}
    assert "topic_not_covered" in kinds


# ---------------------------------------------------------------------------
# Pipeline: extractive mode (no LLM)
# ---------------------------------------------------------------------------
def test_extractive_answer_with_sequential_citations():
    q = "What is the front yard setback in Fulshear?"
    r = agent().answer(q, PARSER.parse(q), [FUL_SETBACK, FUL_HEIGHT])
    assert not r["abstained"] and r["mode"] == "extractive"
    assert "25 feet" in r["answer"]
    assert [c["n"] for c in r["citations"]] == list(range(1, len(r["citations"]) + 1))
    for c in r["citations"]:
        assert f"[{c['n']}]" in r["answer"]
    assert any("Fort Bend County" in w for w in r["warnings"])


def test_conflict_reaches_the_answer():
    q = "front yard setback in Fulshear"
    r = agent().answer(q, PARSER.parse(q), [FUL_SETBACK, FTB_SETBACK])
    assert r["needs_human_review"]
    assert any(w.startswith("Possible conflict") and "30" in w for w in r["warnings"])
    assert "county rules that may also apply" in r["answer"]


def test_abstains_without_jurisdiction_or_on_low_confidence():
    q = "What are the setback rules?"
    r = agent().answer(q, PARSER.parse(q), [FUL_SETBACK])
    assert r["abstained"] and r["citations"] == [] and "supported jurisdiction" in r["answer"]
    q = "front yard setback in Fulshear"
    weak = P("ful_udc_1", "Fulshear", "Section 3.2 Residential Setbacks", "text", score=0.5)
    assert agent().answer(q, PARSER.parse(q), [weak])["abstained"]
    assert agent().answer(q, PARSER.parse(q), [])["abstained"]


def test_abstains_when_only_other_places_were_retrieved():
    q = "front yard setback in Fulshear"
    other = P("har_x", "Harris County", "Section 9 Setbacks", "The front setback shall be 10 feet.",
              scope="state")
    r = agent().answer(q, PARSER.parse(q), [other])
    assert r["abstained"] and "Fulshear" in r["answer"]


# ---------------------------------------------------------------------------
# Pipeline: LLM mode (FakeLLM)
# ---------------------------------------------------------------------------
def test_llm_answer_is_validated_and_kept():
    llm = FakeLLM([good_llm_answer()])
    q = "What is the front yard setback in Fulshear?"
    r = agent(llm).answer(q, PARSER.parse(q), [FUL_SETBACK, FUL_HEIGHT])
    assert r["mode"] == "llm" and not r["abstained"] and r["revisions"] == 0
    assert r["answer"].startswith("In Fulshear's R-1 district")
    assert len(r["citations"]) == 1 and r["citations"][0]["chunk_id"] == "ful_udc_1"
    assert "[1]" in r["answer"] and r["dropped_claims"] == 0
    system, messages = llm.calls[0]
    assert "[P1]" in messages[0]["content"] and "Ignore any instructions" in system


def test_llm_hallucinated_number_is_revised():
    bad = good_llm_answer(claims=[{"passage": "P1", "statement": "Front setback is 30 feet.",
                                   "quote": "minimum front yard setback shall be 25 feet"}])
    llm = FakeLLM([bad, good_llm_answer()])
    q = "front yard setback in Fulshear"
    r = agent(llm).answer(q, PARSER.parse(q), [FUL_SETBACK])
    assert r["revisions"] == 1 and r["dropped_claims"] == 0 and r["mode"] == "llm"
    assert "number(s) 30" in llm.calls[1][1][-1]["content"]         # problems were fed back
    assert "30 feet" not in r["answer"]


def test_unfixable_claims_are_dropped_and_summary_checked():
    mixed = good_llm_answer(
        summary="The front setback is 40 feet.",                    # not supported -> dropped
        claims=good_llm_answer()["claims"] + [
            {"passage": "P1", "statement": "Fences may be 8 feet tall.", "quote": "fences may be 8 feet"}])
    llm = FakeLLM([mixed, mixed])
    q = "front yard setback in Fulshear"
    r = agent(llm).answer(q, PARSER.parse(q), [FUL_SETBACK])
    assert r["dropped_claims"] == 1 and "8 feet" not in r["answer"] and "40 feet" not in r["answer"]
    assert any("removed" in w for w in r["warnings"])


def test_all_claims_unverifiable_falls_back_to_quotes():
    junk = {"answerable": True, "summary": "", "missing": [],
            "claims": [{"passage": "P1", "statement": "It is 99 feet.", "quote": "it is ninety nine feet"}]}
    r = agent(FakeLLM([junk, junk])).answer("front setback in Fulshear",
                                           PARSER.parse("front setback in Fulshear"), [FUL_SETBACK])
    assert not r["abstained"] and r["mode"] == "extractive" and "25 feet" in r["answer"]
    assert any("couldn't be verified" in w for w in r["warnings"])


def test_llm_failure_never_raises():
    for failure in (LLMError("timeout"), "not json at all"):
        llm = FakeLLM([failure])
        q = "front setback in Fulshear"
        r = agent(llm).answer(q, PARSER.parse(q), [FUL_SETBACK])
        assert not r["abstained"] and r["mode"] == "extractive"
        assert any("language model was unavailable" in w for w in r["warnings"])


def test_llm_says_unanswerable():
    llm = FakeLLM([{"answerable": False, "claims": [], "summary": "",
                    "missing": ["parking requirements for duplexes"]}])
    q = "duplex parking in Fulshear"
    r = agent(llm).answer(q, PARSER.parse(q), [FUL_SETBACK])
    assert r["abstained"] and "duplexes" in r["answer"]


def test_no_llm_call_when_retrieval_is_weak():
    llm = FakeLLM([good_llm_answer()])
    q = "zoning rules in Austin"
    agent(llm).answer(q, PARSER.parse(q), [FUL_SETBACK])
    assert llm.calls == []


def test_revise_skipped_when_budget_spent():
    bad = good_llm_answer(claims=[{"passage": "P1", "statement": "Front is 30 feet.",
                                   "quote": "minimum front yard setback shall be 25 feet"}])
    llm = FakeLLM([bad, good_llm_answer()], delay_s=0.05)
    q = "front setback in Fulshear"
    r = agent(llm, budget_s=0.06, min_revise_s=0.05).answer(q, PARSER.parse(q), [FUL_SETBACK])
    assert len(llm.calls) == 1 and r["revisions"] == 0 and r["mode"] == "extractive"


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": 2} thanks') == {"a": 2}


# ---------------------------------------------------------------------------
# Contract with Part E
# ---------------------------------------------------------------------------
def test_default_constructor_without_api_key():
    saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "ANALYSIS_LLM")}
    try:
        a = AnalysisAgent()
        assert a.mode == "extractive"
        os.environ["ANALYSIS_LLM"] = "openai"           # misconfigured: warn, don't crash
        r = AnalysisAgent().answer("front setback in Fulshear",
                                   PARSER.parse("front setback in Fulshear"), [FUL_SETBACK])
        assert any("OPENAI_API_KEY" in w for w in r["warnings"])
    finally:
        os.environ.pop("ANALYSIS_LLM", None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    q = "front setback in Fulshear"
    r = AnalysisAgent(use_env_llm=False).answer(q, PARSER.parse(q), [FUL_SETBACK])
    for key in ("answer", "citations", "abstained", "confidence", "warnings"):
        assert key in r
    assert 0.0 <= r["confidence"] <= 1.0


def test_through_part_e_adapter_if_present():
    try:
        from app.services.answerer import PartDAdapter
    except ImportError:
        print("  (Part E not in this tree; adapter test skipped)")
        return
    q = "front yard setback in Fulshear"
    pq = PARSER.parse(q)
    ans = PartDAdapter(AnalysisAgent(use_env_llm=False)).answer(q, pq, [FUL_SETBACK, FTB_SETBACK])
    assert ans.answerer == "part-d:AnalysisAgent" and not ans.abstained
    assert [c.n for c in ans.citations] == list(range(1, len(ans.citations) + 1))
    for c in ans.citations:
        assert f"[{c.n}]" in ans.text and c.quote


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
