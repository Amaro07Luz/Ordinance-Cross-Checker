"""
Index service: glue between Part B (store + embeddings + processing pipeline)
and Part C (query parser + HybridRetriever).

- Opens Part B's store: PgVectorStore when DATABASE_URL is set, else InMemoryStore.
- Seeds the index from SEED_DIR on first start if it is empty (runs Part B's pipeline).
- Builds Part C's HybridRetriever over the current chunks, reusing the vectors
  Part B already stored instead of re-embedding every chunk at startup.
- reindex() runs Part B again and hot-swaps the retriever.
"""

import logging
import threading
import time
from typing import Dict, List, Optional

from app.config import Settings
from app.resilience import retry
from ingestion_pipeline_part_a import SourceRegistry
from processing.embeddings import get_embedder
from processing.pipeline import ProcessingPipeline, discover_raw_files
from processing.store import InMemoryStore
from retrieval.query_agent import GAZETTEER, ParsedQuery, QueryParser
from retrieval.search import HybridRetriever, RetrievedPassage

log = logging.getLogger(__name__)


class StoredVectorEmbedder:
    """Part C's embedder interface (.embed(text) -> sparse dict). Returns the vector
    Part B stored for a chunk when C asks for that chunk's text; embeds anything
    else (i.e. queries) live with the same model."""

    def __init__(self, embedder, stored: Dict[str, List[float]]):
        self.embedder = embedder
        self.stored = stored

    def embed(self, text: str) -> Dict[int, float]:
        vec = self.stored.get(text)
        if vec is None:
            vec = self.embedder.embed_query(text)
        return {i: x for i, x in enumerate(vec) if x}


class IndexService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = get_embedder(settings.embedding_backend, settings.embedding_model)
        self.store = self._open_store()
        self.parser = QueryParser()
        self.retriever: Optional[HybridRetriever] = None
        self.version = 0                    # bumps on every reload; part of the cache key
        self.chunk_count = 0
        self.loaded_at: Optional[float] = None
        self._reindex_lock = threading.Lock()

    @retry(attempts=5, base_delay=1.0)
    def _open_store(self):
        if self.settings.database_url:
            from processing.store import PgVectorStore
            return PgVectorStore(self.settings.database_url, dim=self.embedder.dim)
        return InMemoryStore(dim=self.embedder.dim)

    @property
    def store_kind(self) -> str:
        return "pgvector" if not isinstance(self.store, InMemoryStore) else "memory"

    # ------------------------------------------------------------------
    def startup(self):
        if self.settings.seed_dir and not self.store.load_chunks():
            log.info("index empty; seeding", extra={"seed_dir": self.settings.seed_dir})
            self.reindex(self.settings.seed_dir, reload=False)
        self.reload()

    def reindex(self, raw_dir: str, force: bool = False, reload: bool = True):
        """Runs Part B over raw_dir. Unchanged documents are skipped by Part B."""
        with self._reindex_lock:
            files = discover_raw_files(raw_dir, SourceRegistry())
            pipe = ProcessingPipeline(self.store, self.embedder, force=force, verbose=False)
            reports = pipe.run(files)
            for r in reports:
                log.info("indexed document", extra={"report": str(r)})
            if reload:
                self.reload()
            return reports

    def reload(self):
        t0 = time.perf_counter()
        chunks = self.store.load_chunks()
        stored = self._stored_vectors()
        by_text = {f"{c.section_title} {c.content}": stored[c.chunk_id]
                   for c in chunks if c.chunk_id in stored}
        self.retriever = HybridRetriever(chunks, embedder=StoredVectorEmbedder(self.embedder, by_text))
        self.chunk_count = len(chunks)
        self.version += 1
        self.loaded_at = time.time()
        log.info("retriever loaded", extra={"chunks": len(chunks), "index_version": self.version,
                                            "reused_vectors": len(by_text),
                                            "ms": round((time.perf_counter() - t0) * 1000, 1)})

    def _stored_vectors(self) -> Dict[str, List[float]]:
        if isinstance(self.store, InMemoryStore):
            return {r["chunk"].chunk_id: r["embedding"] for r in self.store.rows
                    if r["is_current"] and r["model"] == self.embedder.name}
        from processing.store import _to_list
        with self.store.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, embedding FROM chunks WHERE is_current "
                        "AND embedding_model = %s AND embedding IS NOT NULL", (self.embedder.name,))
            rows = cur.fetchall()
        self.store.conn.commit()
        return {cid: _to_list(e) for cid, e in rows}

    # ------------------------------------------------------------------
    def parse(self, question: str, jurisdiction: Optional[str] = None) -> ParsedQuery:
        pq = self.parser.parse(question)
        if jurisdiction:   # explicit filter from the UI/API overrides what the parser guessed
            if jurisdiction not in GAZETTEER:
                raise ValueError(f"Unknown jurisdiction {jurisdiction!r}")
            pq.jurisdictions, pq.parent_jurisdictions, pq.states = [], [], []
            QueryParser._add_jurisdiction(pq, jurisdiction)
            pq.warnings = [w for w in pq.warnings if not w.startswith("No jurisdiction")]
        return pq

    def search(self, pq: ParsedQuery, top_k: int) -> List[RetrievedPassage]:
        retriever = self.retriever          # local ref: safe if reload swaps it mid-request
        return retriever.retrieve(pq, top_k=top_k) if retriever else []

    def jurisdictions(self) -> List[Dict]:
        counts: Dict[str, int] = {}
        for c in (self.retriever.chunks if self.retriever else []):
            counts[c.jurisdiction] = counts.get(c.jurisdiction, 0) + 1
        return [{"name": s.jurisdiction_name, "type": s.jurisdiction_type, "state": s.state,
                 "document_type": s.document_type, "source_url": s.source_url,
                 "indexed_chunks": counts.get(s.jurisdiction_name, 0)}
                for s in SourceRegistry().get_active_sources()]
