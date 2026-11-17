"""
Part B (5/6): Embeddings
One interface, three backends:

  sentence-transformers  local, free; default BAAI/bge-small-en-v1.5 (384-d)
  openai                 text-embedding-3-small (1536-d), needs OPENAI_API_KEY
  hashing                deterministic, dependency-free; for tests/CI only

Pick with EMBEDDING_BACKEND / EMBEDDING_MODEL env vars or get_embedder().
The pgvector column is sized to embedder.dim, so changing models means
re-indexing (the store refuses to mix dimensions).

RetrievalEmbedderAdapter lets Part C's HybridRetriever (which expects
.embed(text) -> Dict[int, float]) use these real embeddings unchanged.
"""

import hashlib
import math
import os
import re
from typing import Dict, List, Optional, Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


class Embedder:
    name: str = "base"
    dim: int = 0

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder(Embedder):
    # bge models retrieve better when queries carry this instruction prefix
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32,
                 device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import
        self.model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.query_prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts):
        vecs = self.model.encode(list(texts), batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text):
        return self.embed_documents([self.query_prefix + text])[0]


class OpenAIEmbedder(Embedder):
    DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 100):
        from openai import OpenAI  # lazy
        self.client = OpenAI()
        self.name = model_name
        self.dim = self.DIMS.get(model_name, 1536)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        out: List[List[float]] = []
        texts = list(texts)
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.name, input=texts[i:i + self.batch_size])
            out.extend(d.embedding for d in resp.data)
        return out


class HashingEmbedder(Embedder):
    """Signed feature hashing of unigrams+bigrams into a dense, L2-normalized vector.
    Uses blake2b (not Python's hash(), which changes every run) so stored vectors
    stay comparable across processes. Good enough for tests, not for the demo."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> List[float]:
        toks = TOKEN_RE.findall(text.lower())
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


def get_embedder(backend: Optional[str] = None, model_name: Optional[str] = None) -> Embedder:
    backend = (backend or os.getenv("EMBEDDING_BACKEND") or "sentence-transformers").lower()
    model_name = model_name or os.getenv("EMBEDDING_MODEL")
    if backend in ("sentence-transformers", "st", "local"):
        return SentenceTransformerEmbedder(model_name or "BAAI/bge-small-en-v1.5")
    if backend == "openai":
        return OpenAIEmbedder(model_name or "text-embedding-3-small")
    if backend == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")


def build_embedding_text(chunk) -> str:
    """What gets embedded: a short context header + the chunk text. The header lets a
    chunk that only says 'shall be 25 feet' still match 'Fulshear front setback'."""
    m = chunk.metadata or {}
    path = " > ".join(m.get("section_path") or [chunk.section_title])
    header = f"{chunk.jurisdiction}, {chunk.state}"
    if m.get("document_type"):
        header += f" | {m['document_type']}"
    return f"{header} | {path}\n{chunk.content}"


class RetrievalEmbedderAdapter:
    """Adapter for Part C: HybridRetriever(chunks, embedder=RetrievalEmbedderAdapter(e)).
    C's cosine() works on sparse dicts, so a dense vector becomes {index: value}."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def embed(self, text: str) -> Dict[int, float]:
        return {i: x for i, x in enumerate(self.embedder.embed_query(text)) if x}
