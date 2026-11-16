# Ordinance-Cross-Checker — M3-D: Regulatory Analysis Agent

Part D of the pipeline: takes Part C's retrieved passages and returns a grounded answer
with verified citations, flags conflicting or missing information, and abstains when the
evidence doesn't support an answer.

```
 A  crawlers ─► B  processing/ ─► C  retrieval/ ─► D  analysis/ ─► E  app/ (API, UI, eval)
                                                   ▲ this branch
```

This branch contains only `analysis/`. Merge order: **M3-B → M3-C → M3-D → M3-E**.
It needs Part C (`retrieval/`) and Part A's `ingestion_pipeline_part_a.py` on the path,
which M3-E already has.

## How it plugs into Part E

E loads it automatically (`ANSWERER=auto`), exactly as `INTEGRATION.md` specifies:

```python
# analysis/agent.py
class AnalysisAgent:
    def answer(self, question: str, parsed: ParsedQuery,
               passages: List[RetrievedPassage]) -> dict
```

No names differ, so `PART_D_MODULE / PART_D_CLASS / PART_D_METHOD` and
`PartDAdapter._normalize()` in E stay as they are. Guarantees E relies on:

- `AnalysisAgent()` takes no required arguments and **never raises** at construction
  (E only tolerates ImportError/AttributeError at startup).
- `answer()` doesn't raise on LLM problems. It falls back to quoting the sources and says so
  in `warnings`. Its LLM budget (`ANALYSIS_TIMEOUT_S`, 20 s) stays below E's
  `ANSWER_TIMEOUT_S` (30 s).
- Citations are only chunk_ids from the passages it was given, numbered 1..n in order and
  matching the `[n]` markers in the text, so E's adapter numbering lines up.
- `parsed.warnings` are not repeated (E's adapter already adds them).

Returned dict: `answer`, `citations` (`[{n, chunk_id, quote, jurisdiction, section_title,
source_url}]`), `abstained`, `confidence`, `warnings`, **plus** extras E ignores until it
passes them through: `claims`, `conflicts`, `gaps`, `needs_human_review`, `mode`
(`llm`|`extractive`), `model`, `revisions`, `dropped_claims`, `timing_ms`.

## How it works

| File | Role |
|---|---|
| `agent.py` | `AnalysisAgent`, the E-facing entry point |
| `graph.py` | Pipeline: precheck → select evidence → draft → validate → revise once → cross-reference → compose |
| `prompts.py` | System prompt and message builders. Passages are labelled P1..Pn; the model returns JSON claims `{passage, statement, quote}` |
| `citations.py` | Deterministic citation validator |
| `conflicts.py` | Numeric requirement extraction, conflict detection, gap detection (no LLM) |
| `llm.py` | OpenAI client (any OpenAI-compatible endpoint), `FakeLLM` for tests, env config |
| `models.py`, `text.py` | Data classes; normalization and number helpers |
| `test_analysis.py` | 23 tests, no network needed |

**The LLM writes, code checks.** Every claim must cite a passage it was given, quote it
word for word, and contain no number that isn't in the quote. It also can't attribute a rule
to a different jurisdiction. Failing claims go back to the model once with the specific
problems listed. Anything still failing is dropped, and the user is told how many. If nothing
survives, Part D quotes the sources directly instead. The 1–2 sentence summary is held to the
same number check and is dropped if it fails.

**Abstains (no LLM call)** when there are no passages, no jurisdiction was recognized,
retrieval came back from other places only, or Part C's `low_confidence()` fires.
It also abstains when the model says the passages don't answer the question.

**Conflicts.** Extracts requirements like *front setback ≥ 25 ft* or *height ≤ 35 ft*
from the passages and compares them:

- `overlapping_jurisdictions`: e.g. Fulshear vs. Fort Bend County. Both may apply, so
  `needs_human_review` is set and a warning is added.
- `within_jurisdiction`: two documents of the same place disagree. Also flagged for review.
- `comparison`: the user asked about several places and they differ. That's the answer,
  so it's listed in the text but not flagged.

Values from different zoning districts (R-1 vs. E-1), or from the same document, aren't
treated as conflicts.

**Gaps:** requested jurisdiction not indexed or not answered, parent county not indexed,
TxDOT/NJDOT rules needed but not indexed, regulation type not covered by the cited text, and
no effective date on the source documents. These are returned as `gaps` and as `warnings`.

**Without an API key** Part D runs in *extractive* mode: it picks the most relevant sentences
verbatim and still runs conflict and gap detection. CI uses this mode.

## Configuration

| Variable | Default | |
|---|---|---|
| `ANALYSIS_LLM` | `auto` | `auto`: OpenAI if `OPENAI_API_KEY` is set and `openai` is installed. `openai`: same, but warn on every answer if unavailable. `none`: never call an LLM |
| `OPENAI_API_KEY` | – | |
| `ANALYSIS_MODEL` | `gpt-4o-mini` | Any chat model; pick a current one |
| `OPENAI_BASE_URL` | – | Any OpenAI-compatible endpoint (e.g. Gemini's) |
| `ANALYSIS_TIMEOUT_S` | `20` | Total LLM budget per answer; keep below E's `ANSWER_TIMEOUT_S` |
| `ANALYSIS_MIN_TOP_SCORE` | `0.9` | Abstain threshold passed to C's `low_confidence()` |

## Tests

```bash
python -m analysis.test_analysis        # or: python -m pytest -q analysis/test_analysis.py
```

Verified inside M3-E's tree, with no changes to E:

- `analysis.test_analysis`: 23/23.
- E's `pytest`: 18/19. The one failure is expected; see below.
- Ruff: clean.
- `make eval` with `answerer=part-d:AnalysisAgent`: success 100%, hit@5 100%, abstain
  accuracy 100%, p95 1.8 ms (extractive mode).

## For Part E when merging (small, all in E's files)

1. `tests/test_api.py` line 17 asserts `D_analysis_agent.startswith("stand-in")`. Once D is
   present it is `"ok"` (as INTEGRATION.md step 4 expects), so change it to `== "ok"`.
2. Add `openai>=1.40` to `requirements.txt`. Add `ANALYSIS_LLM`, `ANALYSIS_MODEL` and
   `ANALYSIS_TIMEOUT_S` to `.env.example` and to `docker-compose.yml` → `api.environment`.
   `OPENAI_API_KEY` is already there.
3. Add `python -m analysis.test_analysis` to the Makefile `test` target and to CI next to
   the B/C smoke tests.
4. Keep CI without `OPENAI_API_KEY`, so the 1500 ms p95 gate tests the extractive path. Run the
   LLM eval separately with a looser gate (design target: < 15 s), e.g.
   `ANALYSIS_LLM=openai python -m eval.harness --max-p95-ms 15000`.
5. Optional: pass `conflicts`, `gaps` and `needs_human_review` through `PartDAdapter` and show
   them in the UI. Today they reach the user only as warning boxes.

## Known limitations

- **The real-LLM path hasn't been run against OpenAI yet.** No key was available while
  building it. It has been tested with `FakeLLM` (good answers, invented numbers, bad quotes,
  revise, timeouts, invalid JSON) and with the real client pointed at an unreachable endpoint,
  which falls back cleanly. Run the eval with a key before the demo.
- **Conflict detection is only tested on hand-written fixtures.** The seed corpus has one
  document set per place, with no Fort Bend County and no TxDOT/NJDOT documents. Adding
  overlapping real documents (A/B) is what will exercise it.
- **Outdated rules can't be detected yet.** B stores `extracted_at` and `doc_version` but no
  effective/adoption date, so every answer carries an "effective dates aren't recorded" note
  until B adds one (`effective_date` in chunk metadata is picked up automatically).
- **Requirement extraction is regex-based.** It covers setbacks, height, lot
  area/width/coverage, density, parking, right-of-way, driveway spacing and flood elevation.
  Real ordinances will have phrasings it misses: a missed requirement means a missed conflict,
  never an invented one.
