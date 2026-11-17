-- Platform tables (Part E). Applied by the API at startup and by the
-- Postgres container on first boot (docker-entrypoint-initdb.d).
--
-- Part B's document tables (documents, document_versions, chunks) are created
-- by processing/store.py, because the embedding column is sized to the
-- embedding model's dimension at runtime.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per /query request: drives response-time and success-rate metrics
CREATE TABLE IF NOT EXISTS query_log (
    request_id      TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    question        TEXT NOT NULL,
    jurisdictions   TEXT[],
    regulation_types TEXT[],
    status          TEXT NOT NULL,            -- answered | abstained | error
    error           TEXT,
    latency_ms      DOUBLE PRECISION NOT NULL,
    retrieval_ms    DOUBLE PRECISION,
    answer_ms       DOUBLE PRECISION,
    cached          BOOLEAN NOT NULL DEFAULT FALSE,
    top_score       DOUBLE PRECISION,
    n_citations     INT,
    answerer        TEXT,
    source          TEXT NOT NULL DEFAULT 'api'  -- api | eval
);
CREATE INDEX IF NOT EXISTS query_log_created ON query_log (created_at DESC);

-- Thumbs up/down from the web UI
CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    request_id  TEXT NOT NULL,
    helpful     BOOLEAN NOT NULL,
    comment     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per eval-harness run, so metrics can be compared across commits
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id       TEXT PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    git_sha      TEXT,
    query_set    TEXT,
    n_queries    INT,
    success_rate DOUBLE PRECISION,
    hit_at_k     DOUBLE PRECISION,
    p50_ms       DOUBLE PRECISION,
    p95_ms       DOUBLE PRECISION,
    report       JSONB
);
