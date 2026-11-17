"""
QueryEngine: question -> Part C parse/retrieve -> Part D (or stub) answer,
with response caching, timing, timeouts, retries, graceful degradation and
a query_log row for every request.
"""

import logging
import re
import time
import uuid
from typing import Dict, Optional

from app.config import Settings
from app.resilience import TTLCache, call_with_timeout, retry
from app.services.answerer import Answer, ExtractiveAnswerer

log = logging.getLogger(__name__)


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?.! ")


class QueryEngine:
    def __init__(self, settings: Settings, index, answerer, db):
        self.settings = settings
        self.index = index
        self.answerer = answerer
        self.fallback = ExtractiveAnswerer()
        self.db = db
        self.cache = TTLCache(settings.cache_size, settings.cache_ttl_s)

    def run(self, question: str, top_k: Optional[int] = None, jurisdiction: Optional[str] = None,
            request_id: Optional[str] = None, source: str = "api", use_cache: bool = True) -> Dict:
        request_id = request_id or uuid.uuid4().hex[:16]
        top_k = top_k or self.settings.top_k
        t0 = time.perf_counter()
        key = (normalize_question(question), top_k, jurisdiction, self.index.version,
               self.answerer.name)

        cached = self.cache.get(key) if use_cache else None
        if cached is not None:
            resp = {**cached, "request_id": request_id, "cached": True,
                    "timing_ms": {**cached["timing_ms"],
                                  "total": round((time.perf_counter() - t0) * 1000, 2)}}
            self._log(resp, question, source)
            return resp

        try:
            pq = self.index.parse(question, jurisdiction)
            t1 = time.perf_counter()
            passages = self._retrieve(pq, top_k)
            t2 = time.perf_counter()
            answer = self._answer(question, pq, passages)
            t3 = time.perf_counter()
        except ValueError:
            raise                                 # bad input -> 400 in the API layer
        except Exception as e:
            self._log({"request_id": request_id, "status": "error", "cached": False,
                       "error": f"{type(e).__name__}: {e}",
                       "timing_ms": {"total": round((time.perf_counter() - t0) * 1000, 2)}},
                      question, source)
            raise

        resp = {
            "request_id": request_id,
            "question": question,
            "status": "abstained" if answer.abstained else "answered",
            "answer": answer.to_dict(),
            "passages": [{
                "rank": p.rank, "chunk_id": p.chunk_id, "jurisdiction": p.jurisdiction,
                "state": p.state, "section_title": p.section_title, "score": p.score,
                "scope": p.scope, "source_url": p.source_url, "content": p.content,
                "page_start": (p.metadata or {}).get("page_start"),
                "page_end": (p.metadata or {}).get("page_end"),
                "match_reasons": p.match_reasons} for p in passages],
            "parsed": {"jurisdictions": pq.jurisdictions, "parent_jurisdictions": pq.parent_jurisdictions,
                       "states": pq.states, "zip_codes": pq.zip_codes,
                       "regulation_types": pq.regulation_types, "keywords": pq.keywords},
            "warnings": answer.warnings,
            "cached": False,
            "index_version": self.index.version,
            "timing_ms": {"parse": round((t1 - t0) * 1000, 2), "retrieval": round((t2 - t1) * 1000, 2),
                          "answer": round((t3 - t2) * 1000, 2),
                          "total": round((time.perf_counter() - t0) * 1000, 2)},
        }
        self.cache.set(key, resp)
        self._log(resp, question, source, pq=pq)
        return resp

    # ------------------------------------------------------------------
    @retry(attempts=2, base_delay=0.2)
    def _retrieve(self, pq, top_k):
        return self.index.search(pq, top_k)

    def _answer(self, question, pq, passages) -> Answer:
        """Part D may call an LLM: bound it with a timeout and fall back to the extractive
        answer rather than failing the whole request. Errors are retried once; timeouts
        are not (Part D already spent its LLM budget, a retry would double the wait)."""
        def call():
            return call_with_timeout(self.answerer.answer, self.settings.answer_timeout_s,
                                     question, pq, passages)
        try:
            try:
                return call()
            except TimeoutError:
                raise
            except Exception as first:
                log.warning("answerer error; retrying once",
                            extra={"error": f"{type(first).__name__}: {first}"})
                time.sleep(0.3)
                return call()
        except Exception as e:
            if isinstance(self.answerer, ExtractiveAnswerer):
                raise
            log.error("answerer failed; falling back to extractive answer",
                      extra={"answerer": self.answerer.name, "error": f"{type(e).__name__}: {e}"})
            ans = self.fallback.answer(question, pq, passages)
            ans.warnings.append("The analysis agent was unavailable, so this is an extractive answer.")
            return ans

    def _log(self, resp: Dict, question: str, source: str, pq=None):
        ans = resp.get("answer") or {}
        passages = resp.get("passages") or []
        parsed = resp.get("parsed") or {}
        t = resp.get("timing_ms") or {}
        row = {
            "request_id": resp["request_id"], "question": question[:2000],
            "jurisdictions": parsed.get("jurisdictions"),
            "regulation_types": parsed.get("regulation_types"),
            "status": resp["status"], "error": resp.get("error"),
            "latency_ms": t.get("total", 0.0), "retrieval_ms": t.get("retrieval"),
            "answer_ms": t.get("answer"), "cached": resp.get("cached", False),
            "top_score": passages[0]["score"] if passages else None,
            "n_citations": len(ans.get("citations") or []),
            "answerer": ans.get("answerer") or self.answerer.name, "source": source,
        }
        self.db.log_query(row)
        log.info("query", extra={k: row[k] for k in ("status", "latency_ms", "cached",
                                                      "jurisdictions", "n_citations")})

