# Integration status (M3-E)

M3-E is the integration branch: it contains Part A (via M3-B), Part B (M3-B),
Part C (M3-C) and Part D (M3-D), plus the Part E platform.

| Part | Branch | In M3-E? | How E uses it |
|---|---|---|---|
| A: Source registry, crawlers, raw storage | M3-A | yes (merged through M3-B) | `SourceRegistry` for `/api/jurisdictions`; raw files in `data/raw/<jurisdiction-slug>/` |
| B: Processing and indexing | M3-B | yes | `IndexService` runs B's `ProcessingPipeline`, reads B's pgvector store |
| C: Query understanding and retrieval | M3-C | yes | `QueryParser` + `HybridRetriever` over B's chunks (reusing B's stored vectors) |
| D: Analysis agent | M3-D | yes | loaded automatically as `part-d:AnalysisAgent`; its extra output is under `answer.details` |
| E: Platform, API and eval | M3-E | yes | this branch |

Merge order into `main`: **M3-B → M3-C → M3-D → M3-E**. Because M3-E already
contains B's and C's commits, those merges won't conflict with it.

---

## Part D (merged)

Steps 1–5 below are done. `ExtractiveAnswerer` remains as the fallback if Part D
errors or times out (timeouts are not retried; other errors are retried once).

**Open issue in Part D (`analysis/conflicts.py`):** constraints are extracted from every
retrieved passage, not just the cited ones, and accessory-structure limits are read as
building height. Repro: "Compare max building height in Fulshear and Marlton" reports
"35 ft vs. 15 ft". The 15 ft is § 160-14 (sheds), while Marlton's building height (§ 160-12)
is also 35 ft. Likely fix: only use constraints from cited passages, and skip sentences about
accessory structures, sheds, fences or signs for "building height".

The API looks for Part D automatically (`ANSWERER=auto`). Part D needs to provide:

```python
# analysis/agent.py
class AnalysisAgent:
    def answer(self, question: str, parsed: ParsedQuery,
               passages: List[RetrievedPassage]) -> dict:   # or an object with these attributes
        return {
            "answer": "...",            # text shown to the user ("text" also accepted)
            "citations": [...],         # chunk_id strings, or dicts/objects with chunk_id (+ optional quote)
            "abstained": False,         # True when D declines to answer
            "confidence": 0.82,         # optional, 0..1
            "warnings": [],             # optional
        }
```

`parsed` comes from `retrieval.query_agent.QueryParser.parse()` and `passages`
from `retrieval.search.HybridRetriever.retrieve()`, exactly C's contract
with D. `retrieval.rerank.low_confidence(passages)` is available for the
abstain path.

Then:
1. Merge M3-D into M3-E (or M3-E into main after D).
2. Add D's dependencies (e.g. `anthropic` / `openai`) to `requirements.txt` and any
   API key to `.env.example` and `docker-compose.yml` → `api.environment`.
3. **Only if D's names differ:** edit `PART_D_MODULE`, `PART_D_CLASS`, `PART_D_METHOD`
   at the top of `app/services/answerer.py`. If D's return shape differs, edit only
   `PartDAdapter._normalize()`.
4. Check `GET /api/health`: `components.D_analysis_agent` should be `"ok"` and
   `answerer` should start with `part-d:`. Set `ANSWERER=partd` in production so a
   broken import fails loudly instead of silently falling back.
5. Run `make eval`. If D paraphrases, `expect_answer_contains` still passes as long
   as the figure appears in the answer or a cited passage. Raise `ANSWER_TIMEOUT_S`
   if LLM calls are slow, and add a p95 target for the LLM path to CI.

Already handled for D: each call is bounded by `ANSWER_TIMEOUT_S`, retried once,
and if it still fails the API returns the extractive answer with a warning
instead of a 500 (see `tests/test_api.py::test_part_d_failure_falls_back`).
Responses are cached per (question, top_k, jurisdiction, index version, answerer).

## When Part A's crawler is wired up

- Save downloads to `data/raw/<slugify(jurisdiction_name)>/...`. `processing.pipeline.slugify`
  gives the folder name, e.g. `evesham-township-marlton`.
- Re-index with `make reindex` or `POST /api/admin/reindex` (header `X-Admin-Token`).
  Unchanged documents are skipped.
- A scheduler can just call that endpoint after each crawl.
- Known bug in `ingestion_pipeline_part_a.py`: `from BeautifulSoup4 import BeautifulSoup`
  should be `from bs4 import BeautifulSoup`. Nothing in B or E depends on it.

## When real documents replace the synthetic seed corpus

- `eval/seed_corpus/` is synthetic (made-up numbers) so CI can run without downloads.
  Point `SEED_DIR` at `data/raw` (or leave unset and use reindex).
- Update `eval/queries.jsonl` so `expect_section` / `expect_answer_contains` match the
  real section numbers and values. Keep the edge cases (`expect_abstain`).
- For real retrieval quality use `EMBEDDING_BACKEND=sentence-transformers`
  (`make install-ml`, or `docker compose build --build-arg WITH_ML=true`). Changing
  the embedding model changes the vector size: drop B's tables and re-index.

## When Part C changes

- E imports `QueryParser`, `ParsedQuery`, `GAZETTEER` from `retrieval.query_agent`,
  `HybridRetriever`, `RetrievedPassage` from `retrieval.search`, and `low_confidence`
  from `retrieval.rerank`. Renames there need matching edits in `app/services/index.py`
  and `app/services/answerer.py`.
- If C moves to querying pgvector directly (`store.similarity_search` / `keyword_search`),
  replace `IndexService.reload()` / `search()`. The API, cache and eval don't change.
