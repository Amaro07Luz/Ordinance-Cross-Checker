"""
Platform persistence: query log, feedback, eval runs (schema in db/schema.sql).
PgPlatformDB writes to Postgres; MemoryPlatformDB keeps the same data in
process for local runs and tests. Logging failures never fail a request.
"""

import logging
import os
import statistics
import threading
from collections import deque
from typing import Dict, List, Optional

from app.resilience import retry

log = logging.getLogger(__name__)
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "schema.sql")

QUERY_LOG_COLS = ["request_id", "question", "jurisdictions", "regulation_types", "status",
                  "error", "latency_ms", "retrieval_ms", "answer_ms", "cached", "top_score",
                  "n_citations", "answerer", "source"]


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1], 1)


def summarize(rows: List[Dict]) -> Dict:
    """Response-time and success-rate metrics from query_log rows."""
    n = len(rows)
    lat = [r["latency_ms"] for r in rows]
    ok = [r for r in rows if r["status"] != "error"]
    return {
        "queries": n,
        "success_rate": round(len(ok) / n, 4) if n else None,
        "answered_rate": round(sum(r["status"] == "answered" for r in rows) / n, 4) if n else None,
        "abstain_rate": round(sum(r["status"] == "abstained" for r in rows) / n, 4) if n else None,
        "error_rate": round((n - len(ok)) / n, 4) if n else None,
        "cache_hit_rate": round(sum(bool(r["cached"]) for r in rows) / n, 4) if n else None,
        "latency_ms": {"p50": percentile(lat, 50), "p95": percentile(lat, 95),
                       "p99": percentile(lat, 99), "max": round(max(lat), 1) if lat else None},
    }


class MemoryPlatformDB:
    kind = "memory"

    def __init__(self, maxlen: int = 10000):
        self.queries: deque = deque(maxlen=maxlen)
        self.feedback: List[Dict] = []
        self.eval_runs: List[Dict] = []
        self._lock = threading.Lock()

    def log_query(self, row: Dict):
        with self._lock:
            self.queries.append(row)

    def add_feedback(self, request_id: str, helpful: bool, comment: Optional[str]):
        self.feedback.append({"request_id": request_id, "helpful": helpful, "comment": comment})

    def save_eval_run(self, run: Dict):
        self.eval_runs.append(run)

    def metrics(self, last_n: int = 1000, source: str = "api") -> Dict:
        with self._lock:
            rows = [r for r in self.queries if r.get("source", "api") == source][-last_n:]
        return summarize(rows)

    def ping(self) -> bool:
        return True


class PgPlatformDB:
    kind = "postgres"

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._lock = threading.Lock()
        self.conn = self._connect()
        with open(SCHEMA_PATH) as f, self.conn.cursor() as cur:
            cur.execute(f.read())
        self.conn.commit()

    @retry(attempts=5, base_delay=1.0)   # the DB container may still be starting
    def _connect(self):
        import psycopg
        return psycopg.connect(self.dsn, autocommit=False, connect_timeout=5)

    def _exec(self, sql: str, params=None, fetch: bool = False):
        """Runs one statement; reconnects once if the connection was dropped."""
        import psycopg
        for attempt in (1, 2):
            with self._lock:
                try:
                    with self.conn.cursor() as cur:
                        cur.execute(sql, params)
                        rows = cur.fetchall() if fetch else None
                    self.conn.commit()
                    return rows
                except psycopg.OperationalError:
                    if attempt == 2:
                        raise
                    log.warning("platform DB connection lost; reconnecting")
                    self.conn = self._connect()
                except Exception:
                    self.conn.rollback()
                    raise

    def log_query(self, row: Dict):
        cols = ", ".join(QUERY_LOG_COLS)
        marks = ", ".join(["%s"] * len(QUERY_LOG_COLS))
        try:
            self._exec(f"INSERT INTO query_log ({cols}) VALUES ({marks}) "
                       "ON CONFLICT (request_id) DO NOTHING",
                       [row.get(c) for c in QUERY_LOG_COLS])
        except Exception as e:  # logging must never break a request
            log.error("failed to write query_log", extra={"error": str(e)})

    def add_feedback(self, request_id: str, helpful: bool, comment: Optional[str]):
        self._exec("INSERT INTO feedback (request_id, helpful, comment) VALUES (%s, %s, %s)",
                   (request_id, helpful, comment))

    def save_eval_run(self, run: Dict):
        from psycopg.types.json import Jsonb
        self._exec("""INSERT INTO eval_runs (run_id, git_sha, query_set, n_queries, success_rate,
                      hit_at_k, p50_ms, p95_ms, report) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                   (run["run_id"], run.get("git_sha"), run.get("query_set"), run["n_queries"],
                    run["success_rate"], run.get("hit_at_k"), run["p50_ms"], run["p95_ms"],
                    Jsonb(run)))

    def metrics(self, last_n: int = 1000, source: str = "api") -> Dict:
        rows = self._exec("""SELECT status, latency_ms, cached FROM query_log WHERE source = %s
                             ORDER BY created_at DESC LIMIT %s""", (source, last_n), fetch=True)
        return summarize([{"status": s, "latency_ms": ms, "cached": c} for s, ms, c in rows])

    def ping(self) -> bool:
        try:
            self._exec("SELECT 1", fetch=True)
            return True
        except Exception:
            return False
