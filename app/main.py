"""
FastAPI service for the Government Regulatory Intelligence Agent.

    uvicorn app.main:app --reload            # http://localhost:8000  (UI)
                                             # http://localhost:8000/docs (OpenAPI)
Endpoints
    GET  /                 minimal web UI
    POST /api/query        ask a question -> answer + citations + passages + timing
    GET  /api/jurisdictions  Part A's source registry + indexed chunk counts
    GET  /api/health       liveness/readiness + which parts are real vs stand-in
    GET  /api/metrics      response time (p50/p95/p99) and query success rate
    POST /api/feedback     thumbs up/down for a request_id
    POST /api/admin/reindex  run Part B over a raw-docs folder (needs X-Admin-Token)
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.db import MemoryPlatformDB, PgPlatformDB
from app.logging_setup import request_id_var, setup_logging
from app.services.answerer import load_answerer
from app.services.engine import QueryEngine
from app.services.index import IndexService

log = logging.getLogger("app")
STATIC = os.path.join(os.path.dirname(__file__), "static")


# ----------------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          examples=["What is the front yard setback in Fulshear?"])
    top_k: Optional[int] = Field(None, ge=1, le=20)
    jurisdiction: Optional[str] = Field(None, description="Force a jurisdiction, e.g. 'Fulshear'")
    use_cache: bool = True


class FeedbackRequest(BaseModel):
    request_id: str
    helpful: bool
    comment: Optional[str] = Field(None, max_length=2000)


class ReindexRequest(BaseModel):
    raw_dir: str = "data/raw"
    force: bool = False


# ----------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------
def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        db = PgPlatformDB(settings.database_url) if settings.database_url else MemoryPlatformDB()
        index = IndexService(settings)
        await run_in_threadpool(index.startup)
        answerer = load_answerer(settings.answerer)
        app.state.settings, app.state.db, app.state.index = settings, db, index
        app.state.engine = QueryEngine(settings, index, answerer, db)
        log.info("startup complete", extra={
            "store": index.store_kind, "platform_db": db.kind, "chunks": index.chunk_count,
            "embedder": index.embedder.name, "answerer": answerer.name,
            "ms": round((time.perf_counter() - t0) * 1000)})
        yield

    app = FastAPI(title="Government Regulatory Intelligence Agent",
                  version="0.3.0-m3", lifespan=lifespan)

    # ---- middleware: request id + response time on every response -----
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            ms = (time.perf_counter() - t0) * 1000
            response.headers["X-Request-ID"] = rid
            response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
            if request.url.path.startswith("/api"):
                log.info("http", extra={"method": request.method, "path": request.url.path,
                                        "status": response.status_code, "ms": round(ms, 1)})
            return response
        finally:
            request_id_var.reset(token)

    # ---- error handling: always JSON {error, detail, request_id} -------
    def _err(status: int, error: str, detail=None):
        return JSONResponse(status_code=status, content={
            "error": error, "detail": detail, "request_id": request_id_var.get()})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return _err(422, "invalid_request",
                    [{"field": ".".join(map(str, e["loc"][1:])), "msg": e["msg"]} for e in exc.errors()])

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return _err(exc.status_code, "http_error", exc.detail)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error", extra={"path": request.url.path})
        return _err(500, "internal_error", "Something went wrong; the error was logged.")

    # ---- routes --------------------------------------------------------
    def engine(request: Request) -> QueryEngine:
        return request.app.state.engine

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.post("/api/query")
    async def query(body: QueryRequest, eng: QueryEngine = Depends(engine),
                    x_query_source: Optional[str] = Header(None)):
        rid = request_id_var.get()
        # the eval harness sends X-Query-Source: eval so it doesn't skew live metrics
        source = "eval" if x_query_source == "eval" else "api"
        try:
            return await run_in_threadpool(eng.run, body.question, body.top_k, body.jurisdiction,
                                           rid, source, body.use_cache)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/jurisdictions")
    def jurisdictions(request: Request) -> List[dict]:
        return request.app.state.index.jurisdictions()

    @app.get("/api/health")
    def health(request: Request):
        s = request.app.state
        db_ok = s.db.ping()
        ready = db_ok and s.index.retriever is not None
        body = {
            "status": "ok" if ready else "degraded",
            "index": {"store": s.index.store_kind, "chunks": s.index.chunk_count,
                      "version": s.index.version, "embedder": s.index.embedder.name},
            "platform_db": {"kind": s.db.kind, "ok": db_ok},
            "components": {
                "A_source_registry": "ok",
                "B_processing_indexing": "ok",
                "C_retrieval": "ok",
                "D_analysis_agent": ("ok" if s.engine.answerer.name.startswith("part-d")
                                     else "stand-in (Part D not merged)"),
            },
            "answerer": s.engine.answerer.name,
            "cache": {"entries": len(s.engine.cache), "hits": s.engine.cache.hits,
                      "misses": s.engine.cache.misses},
        }
        return JSONResponse(body, status_code=200 if ready else 503)

    @app.get("/api/metrics")
    def metrics(request: Request, last_n: int = 1000, source: str = "api"):
        return {"window": last_n, "source": source, **request.app.state.db.metrics(last_n, source)}

    @app.post("/api/feedback")
    def feedback(body: FeedbackRequest, request: Request):
        request.app.state.db.add_feedback(body.request_id, body.helpful, body.comment)
        return {"ok": True}

    @app.post("/api/admin/reindex")
    async def reindex(body: ReindexRequest, request: Request,
                      x_admin_token: Optional[str] = Header(None)):
        s = request.app.state
        if not s.settings.admin_token or x_admin_token != s.settings.admin_token:
            raise HTTPException(403, "Set ADMIN_TOKEN and send it as X-Admin-Token.")
        if not os.path.isdir(body.raw_dir):
            raise HTTPException(400, f"raw_dir not found: {body.raw_dir}")
        reports = await run_in_threadpool(s.index.reindex, body.raw_dir, body.force)
        s.engine.cache.clear()
        return {"index_version": s.index.version, "chunks": s.index.chunk_count,
                "documents": [str(r) for r in reports]}

    return app


app = create_app()
