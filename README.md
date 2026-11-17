# Ordinance-Cross-Checker

Government Regulatory Intelligence Agent (M3 prototype): ask about zoning, subdivision,
setback and permitting rules in Texas and New Jersey jurisdictions and get answers that
cite the exact ordinance section.

Team: Vincent Zheng, Riley Beaudion, Amaro Da Luz, Harlon Sprague, Ishaan Sayal

```
 A  Source registry / crawlers  ──► data/raw/<jurisdiction>/
 B  processing/   extract · OCR · clean · section chunks · versions · embeddings ──► pgvector
 C  retrieval/    query parsing · hybrid BM25 + vector search · rerank
 D  analysis/     answer + citations (not merged yet; stand-in in app/services/answerer.py)
 E  app/ eval/    FastAPI · web UI · logging · retries · caching · metrics · eval · CI
```

See [INTEGRATION.md](INTEGRATION.md) for what each part provides and what to change when Part D lands.

## Quick start

**Docker (Postgres + pgvector + API):**
```bash
docker compose up --build          # http://localhost:8000  (UI)   /docs (API)
```

**Local, no database:**
```bash
pip install -r requirements-dev.txt
make run                            # in-memory index seeded from eval/seed_corpus
```

**Local with Postgres:**
```bash
make db
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/regintel
make run
```

The index is seeded from `SEED_DIR` on first start. `eval/seed_corpus` is a **synthetic**
corpus for development and CI. Real documents go in `data/raw/<jurisdiction-slug>/`, then
run `make reindex`.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/query` | `{"question": "...", "jurisdiction"?: "...", "top_k"?: 5}` → answer, citations, passages, parsed query, warnings, `timing_ms` |
| GET | `/api/jurisdictions` | Part A registry + indexed chunk counts |
| GET | `/api/health` | readiness, index size, and which parts are real vs stand-in |
| GET | `/api/metrics?source=api` | query success rate, abstain/error rate, cache hit rate, p50/p95/p99 latency |
| POST | `/api/feedback` | `{"request_id": "...", "helpful": true}` |
| POST | `/api/admin/reindex` | runs Part B over `raw_dir`; requires `X-Admin-Token` |

Every response carries `X-Request-ID` and `X-Response-Time-ms`. Errors are always
`{"error", "detail", "request_id"}`. Logs are JSON lines tagged with the request id.

## Reliability

- **Retries:** DB connections retry with exponential backoff and jitter. A dropped
  connection is reopened once. Retrieval and the answer step retry once.
- **Timeouts and fallback:** the answer step (Part D / LLM) is bounded by
  `ANSWER_TIMEOUT_S`. If it fails, the API returns an extractive answer with a warning
  instead of an error.
- **Caching:** TTL + LRU response cache keyed on the normalized question, filters, index
  version and answerer. Cleared on reindex.
- **Persistence:** every request goes to `query_log`. Feedback and eval runs are stored too
  (`db/schema.sql`).

## Evaluation

```bash
make eval                                   # in-process
python -m eval.harness --url http://localhost:8000
```

Runs `eval/queries.jsonl` (20 jurisdiction questions + 2 edge cases that should abstain)
and writes `eval/reports/latest.{json,md}`.

- **Response time:** p50 and p95 with the cache off, plus the cached p50.
- **Query success rate:** HTTP 200, plus correct answer-vs-abstain, expected section in
  the top k, and expected values present.
- Also reported: hit@k, MRR, jurisdiction accuracy and abstain accuracy.

CI fails below 85% success or above 1500 ms p95.

## CI

`.github/workflows/ci.yml` on every push and PR:
- lint
- Part B and Part C smoke tests
- Part E pytest suite against a pgvector service
- eval harness gate (report shown in the job summary and uploaded as an artifact)
- Docker build + `/api/health` boot check

## Configuration

See `.env.example`. Main settings:
- `DATABASE_URL`
- `EMBEDDING_BACKEND` (`hashing` for tests; `sentence-transformers` or `openai` for real use)
- `ANSWERER` (`auto` | `stub` | `partd`)
- `SEED_DIR`, `TOP_K`, `CACHE_TTL_S`, `ANSWER_TIMEOUT_S`, `ADMIN_TOKEN`
