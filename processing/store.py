"""
Part B (6/6): Writing to pgvector
PgVectorStore persists documents, their versions, and current/historical
chunks with embeddings in Postgres + pgvector. InMemoryStore has the same
interface for tests and for running without a database.

Read side for Part C / D:
  store.load_chunks(jurisdictions=..., states=...) -> List[DocumentChunk]
      (current version only; drop straight into HybridRetriever)
  store.similarity_search(query_vec, top_k, jurisdictions=..., states=...)
      -> List[(DocumentChunk, cosine_similarity)]
  store.keyword_search(query, top_k, ...) -> List[(DocumentChunk, ts_rank)]
"""

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ingestion_pipeline_part_a import DocumentChunk

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id            TEXT PRIMARY KEY,
    jurisdiction      TEXT NOT NULL,
    jurisdiction_type TEXT,
    state             TEXT,
    doc_title         TEXT,
    source_url        TEXT,
    document_type     TEXT,
    current_version   INT  NOT NULL DEFAULT 0,
    current_hash      TEXT,
    first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_checked_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_changed_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS document_versions (
    doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    version        INT  NOT NULL,
    content_hash   TEXT NOT NULL,
    section_hashes JSONB NOT NULL DEFAULT '{{}}',
    file_type      TEXT,
    page_count     INT,
    ocr_pages      INT,
    extracted_at   TIMESTAMPTZ,
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    change_summary JSONB,
    PRIMARY KEY (doc_id, version)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    version           INT  NOT NULL,
    chunk_index       INT  NOT NULL,
    jurisdiction      TEXT NOT NULL,
    jurisdiction_type TEXT,
    state             TEXT,
    section_title     TEXT,
    content           TEXT NOT NULL,
    source_url        TEXT,
    metadata          JSONB NOT NULL DEFAULT '{{}}',
    content_hash      TEXT NOT NULL,
    embedding         vector({dim}),
    embedding_model   TEXT,
    is_current        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('english', coalesce(section_title, '') || ' ' || content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin        ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_current_jur    ON chunks (jurisdiction) WHERE is_current;
CREATE INDEX IF NOT EXISTS chunks_current_state  ON chunks (state) WHERE is_current;
CREATE INDEX IF NOT EXISTS chunks_hash_model     ON chunks (content_hash, embedding_model);
"""

CHUNK_COLS = ("chunk_id, doc_id, jurisdiction, state, section_title, content, "
              "chunk_index, source_url, metadata")


def _to_list(vec) -> List[float]:
    """pgvector returns numpy arrays or pgvector.Vector depending on its version."""
    if hasattr(vec, "to_list"):
        return [float(x) for x in vec.to_list()]
    return [float(x) for x in vec]


def _row_to_chunk(row) -> DocumentChunk:
    meta = row[8] if isinstance(row[8], dict) else json.loads(row[8] or "{}")
    return DocumentChunk(chunk_id=row[0], doc_id=row[1], jurisdiction=row[2], state=row[3],
                         section_title=row[4], content=row[5], chunk_index=row[6],
                         source_url=row[7], metadata=meta)


class PgVectorStore:
    def __init__(self, dsn: Optional[str] = None, dim: int = 384, init_schema: bool = True):
        import psycopg
        from pgvector.psycopg import register_vector

        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise ValueError("Pass dsn= or set DATABASE_URL, e.g. "
                             "postgresql://postgres:postgres@localhost:5432/regintel")
        self.dim = dim
        self.conn = psycopg.connect(self.dsn, autocommit=False)
        if init_schema:
            self.init_schema()
        register_vector(self.conn)

    def init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA_SQL.format(dim=self.dim))
            cur.execute("""SELECT atttypmod FROM pg_attribute
                           WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'""")
            existing = cur.fetchone()[0]
        self.conn.commit()
        if existing != self.dim:
            raise ValueError(
                f"chunks.embedding is vector({existing}) but the embedder produces {self.dim}-d "
                "vectors. Use the same embedding model, or drop the tables and re-index.")

    def close(self):
        self.conn.close()

    # ---- write side ---------------------------------------------------
    def get_document(self, doc_id: str) -> Optional[Dict]:
        with self.conn.cursor() as cur:
            cur.execute("""SELECT d.current_version, d.current_hash, v.section_hashes
                           FROM documents d LEFT JOIN document_versions v
                             ON v.doc_id = d.doc_id AND v.version = d.current_version
                           WHERE d.doc_id = %s""", (doc_id,))
            row = cur.fetchone()
        self.conn.commit()
        if not row:
            return None
        return {"current_version": row[0], "current_hash": row[1], "section_hashes": row[2] or {}}

    def touch_document(self, doc_id: str):
        with self.conn.cursor() as cur:
            cur.execute("UPDATE documents SET last_checked_at = now() WHERE doc_id = %s", (doc_id,))
        self.conn.commit()

    def find_embeddings(self, content_hashes: Iterable[str], model: str) -> Dict[str, List[float]]:
        hashes = list(set(content_hashes))
        if not hashes:
            return {}
        with self.conn.cursor() as cur:
            cur.execute("""SELECT DISTINCT ON (content_hash) content_hash, embedding FROM chunks
                           WHERE content_hash = ANY(%s) AND embedding_model = %s
                             AND embedding IS NOT NULL""", (hashes, model))
            rows = cur.fetchall()
        self.conn.commit()
        return {h: _to_list(e) for h, e in rows}

    def save_version(self, document: Dict, version: Dict, chunks: List[DocumentChunk],
                     embeddings: Sequence[Sequence[float]], embedding_model: str):
        """Atomically: upsert document, add version row, retire old chunks, insert new ones."""
        import numpy as np
        from psycopg.types.json import Jsonb

        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (doc_id, jurisdiction, jurisdiction_type, state, doc_title,
                    source_url, document_type, current_version, current_hash, last_changed_at)
                VALUES (%(doc_id)s, %(jurisdiction)s, %(jurisdiction_type)s, %(state)s, %(doc_title)s,
                    %(source_url)s, %(document_type)s, %(version)s, %(content_hash)s, now())
                ON CONFLICT (doc_id) DO UPDATE SET
                    current_version = EXCLUDED.current_version,
                    current_hash    = EXCLUDED.current_hash,
                    doc_title       = EXCLUDED.doc_title,
                    source_url      = EXCLUDED.source_url,
                    last_checked_at = now(), last_changed_at = now()""",
                        {**document, "version": version["version"],
                         "content_hash": version["content_hash"]})
            cur.execute("""
                INSERT INTO document_versions (doc_id, version, content_hash, section_hashes,
                    file_type, page_count, ocr_pages, extracted_at, change_summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (document["doc_id"], version["version"], version["content_hash"],
                         Jsonb(version["section_hashes"]), version["file_type"],
                         version["page_count"], version["ocr_pages"], version["extracted_at"],
                         Jsonb(version.get("change_summary"))))
            cur.execute("UPDATE chunks SET is_current = FALSE WHERE doc_id = %s AND is_current",
                        (document["doc_id"],))
            cur.executemany(f"""
                INSERT INTO chunks ({CHUNK_COLS}, version, jurisdiction_type, content_hash,
                                    embedding, embedding_model, is_current)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)""",
                [(c.chunk_id, c.doc_id, c.jurisdiction, c.state, c.section_title, c.content,
                  c.chunk_index, c.source_url, Jsonb(c.metadata), version["version"],
                  c.metadata.get("jurisdiction_type"), c.metadata["content_hash"],
                  np.asarray(e, dtype=np.float32), embedding_model)
                 for c, e in zip(chunks, embeddings)])

    # ---- read side (Part C / D) ----------------------------------------
    @staticmethod
    def _filters(jurisdictions, states) -> Tuple[str, list]:
        where, params = ["is_current"], []
        if jurisdictions:
            where.append("jurisdiction = ANY(%s)")
            params.append(list(jurisdictions))
        if states:
            where.append("state = ANY(%s)")
            params.append(list(states))
        return " AND ".join(where), params

    def load_chunks(self, jurisdictions: Optional[List[str]] = None,
                    states: Optional[List[str]] = None) -> List[DocumentChunk]:
        where, params = self._filters(jurisdictions, states)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT {CHUNK_COLS} FROM chunks WHERE {where} "
                        "ORDER BY jurisdiction, doc_id, chunk_index", params)
            rows = cur.fetchall()
        self.conn.commit()
        return [_row_to_chunk(r) for r in rows]

    def similarity_search(self, query_vec: Sequence[float], top_k: int = 10,
                          jurisdictions: Optional[List[str]] = None,
                          states: Optional[List[str]] = None) -> List[Tuple[DocumentChunk, float]]:
        import numpy as np
        where, params = self._filters(jurisdictions, states)
        q = np.asarray(query_vec, dtype=np.float32)
        with self.conn.cursor() as cur:
            cur.execute(f"""SELECT {CHUNK_COLS}, 1 - (embedding <=> %s) AS sim FROM chunks
                            WHERE {where} ORDER BY embedding <=> %s LIMIT %s""",
                        [q] + params + [q, top_k])
            rows = cur.fetchall()
        self.conn.commit()
        return [(_row_to_chunk(r), float(r[9])) for r in rows]

    def keyword_search(self, query: str, top_k: int = 10,
                       jurisdictions: Optional[List[str]] = None,
                       states: Optional[List[str]] = None) -> List[Tuple[DocumentChunk, float]]:
        where, params = self._filters(jurisdictions, states)
        with self.conn.cursor() as cur:
            cur.execute(f"""SELECT {CHUNK_COLS}, ts_rank(tsv, q) AS r
                            FROM chunks, websearch_to_tsquery('english', %s) q
                            WHERE {where} AND tsv @@ q ORDER BY r DESC LIMIT %s""",
                        [query] + params + [top_k])
            rows = cur.fetchall()
        self.conn.commit()
        return [(_row_to_chunk(r), float(r[9])) for r in rows]


class InMemoryStore:
    """Same interface as PgVectorStore, kept in Python dicts. For tests / no-DB runs."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.documents: Dict[str, Dict] = {}
        self.versions: Dict[Tuple[str, int], Dict] = {}
        self.rows: List[Dict] = []   # each: {chunk, embedding, model, version, is_current}

    def get_document(self, doc_id):
        d = self.documents.get(doc_id)
        if not d:
            return None
        v = self.versions.get((doc_id, d["current_version"]), {})
        return {"current_version": d["current_version"], "current_hash": d["current_hash"],
                "section_hashes": v.get("section_hashes", {})}

    def touch_document(self, doc_id):
        pass

    def find_embeddings(self, content_hashes, model):
        wanted = set(content_hashes)
        return {r["chunk"].metadata["content_hash"]: r["embedding"] for r in self.rows
                if r["model"] == model and r["chunk"].metadata["content_hash"] in wanted}

    def save_version(self, document, version, chunks, embeddings, embedding_model):
        self.documents[document["doc_id"]] = {**document, "current_version": version["version"],
                                              "current_hash": version["content_hash"]}
        self.versions[(document["doc_id"], version["version"])] = version
        for r in self.rows:
            if r["chunk"].doc_id == document["doc_id"]:
                r["is_current"] = False
        self.rows += [{"chunk": c, "embedding": list(e), "model": embedding_model,
                       "version": version["version"], "is_current": True}
                      for c, e in zip(chunks, embeddings)]

    def _current(self, jurisdictions=None, states=None):
        return [r for r in self.rows if r["is_current"]
                and (not jurisdictions or r["chunk"].jurisdiction in jurisdictions)
                and (not states or r["chunk"].state in states)]

    def load_chunks(self, jurisdictions=None, states=None):
        return [r["chunk"] for r in self._current(jurisdictions, states)]

    def similarity_search(self, query_vec, top_k=10, jurisdictions=None, states=None):
        def cos(a, b):
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return sum(x * y for x, y in zip(a, b)) / (na * nb)
        scored = [(r["chunk"], cos(query_vec, r["embedding"]))
                  for r in self._current(jurisdictions, states)]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
