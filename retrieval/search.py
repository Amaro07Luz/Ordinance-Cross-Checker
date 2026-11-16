"""
Part C (2/3): Hybrid Retrieval
BM25 keyword search + vector search, fused with Reciprocal Rank Fusion,
restricted by a jurisdiction filter, then reranked (see rerank.py).

Contract with workstream D:
    retriever.retrieve(parsed_query, top_k=5) -> List[RetrievedPassage]
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from ingestion_pipeline_part_a import DocumentChunk  # swap to shared/schemas.py once it exists
from retrieval.query_agent import ParsedQuery
from retrieval.rerank import HeuristicReranker

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedPassage:
    chunk_id: str
    doc_id: str
    jurisdiction: str
    state: str
    section_title: str
    content: str
    source_url: str
    score: float                 # reranked score (not capped at 1.0, roughly 0-1.7)
    rank: int
    scope: str                   # "exact" | "parent" | "state" | "fallback"
    match_reasons: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# ------------------------------------------------------------------
# Embedders
# ------------------------------------------------------------------
class Embedder(Protocol):
    def embed(self, text: str) -> Dict[int, float]: ...


class HashingEmbedder:
    """Dependency-free stand-in: hashed unigram+bigram sparse vectors.
    Lets C run and be tested before B's real embeddings / pgvector exist.
    Replace with a real embedding model (same .embed interface) for the demo."""

    def __init__(self, dims: int = 4096):
        self.dims = dims

    def embed(self, text: str) -> Dict[int, float]:
        toks = tokenize(text)
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        vec: Dict[int, float] = Counter()
        for g in grams:
            vec[hash(g) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}


def cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


# ------------------------------------------------------------------
# BM25 (pure Python)
# ------------------------------------------------------------------
class BM25:
    def __init__(self, docs: Sequence[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avgdl = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query_tokens: List[str], i: int) -> float:
        s, tf = 0.0, self.tf[i]
        for t in set(query_tokens):
            f = tf.get(t, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avgdl or 1))
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
        return s


# ------------------------------------------------------------------
# Hybrid retriever
# ------------------------------------------------------------------
class HybridRetriever:
    def __init__(
        self,
        chunks: List[DocumentChunk],
        embedder: Optional[Embedder] = None,
        reranker: Optional[HeuristicReranker] = None,
        rrf_k: int = 60,
        candidate_pool: int = 25,
    ):
        self.chunks = chunks
        self.embedder = embedder or HashingEmbedder()
        self.reranker = reranker or HeuristicReranker()
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        # Index text includes the section title so headings are searchable
        texts = [f"{c.section_title} {c.content}" for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.vectors = [self.embedder.embed(t) for t in texts]

    # ---- jurisdiction filter -------------------------------------
    def _scope_of(self, chunk: DocumentChunk, pq: ParsedQuery) -> Optional[str]:
        if chunk.jurisdiction in pq.jurisdictions:
            return "exact"
        if chunk.jurisdiction in pq.parent_jurisdictions:
            return "parent"
        if (pq.needs_state_dot and chunk.state in pq.states
                and chunk.metadata.get("jurisdiction_type") == "State"):
            return "state"
        return None

    def _candidates(self, pq: ParsedQuery):
        """Returns {index: scope}. Falls back to wider scopes rather than returning nothing."""
        if pq.jurisdictions:
            scoped = {i: s for i, c in enumerate(self.chunks)
                      if (s := self._scope_of(c, pq))}
            if scoped:
                return scoped
            pq.warnings.append(
                f"No indexed documents for {', '.join(pq.jurisdictions)}; "
                "widened to same-state sources.")
        if pq.states:
            in_state = {i: "state" for i, c in enumerate(self.chunks) if c.state in pq.states}
            if in_state:
                return in_state
        pq.warnings.append("Searched all indexed jurisdictions (no usable filter).")
        return {i: "fallback" for i in range(len(self.chunks))}

    # ---- main entry point ----------------------------------------
    def retrieve(self, pq: ParsedQuery, top_k: int = 5) -> List[RetrievedPassage]:
        if not self.chunks:
            return []
        cands = self._candidates(pq)
        q_tokens = tokenize(pq.search_text)
        q_vec = self.embedder.embed(pq.search_text)

        bm25_scores = {i: self.bm25.score(q_tokens, i) for i in cands}
        vec_scores = {i: cosine(q_vec, self.vectors[i]) for i in cands}

        def ranks(scores: Dict[int, float]) -> Dict[int, int]:
            ordered = sorted(scores, key=scores.get, reverse=True)
            return {i: r for r, i in enumerate(ordered, 1) if scores[i] > 0}

        bm_rank, vec_rank = ranks(bm25_scores), ranks(vec_scores)
        fused: Dict[int, float] = {}
        for i in cands:
            s = 0.0
            if i in bm_rank:
                s += 1 / (self.rrf_k + bm_rank[i])
            if i in vec_rank:
                s += 1 / (self.rrf_k + vec_rank[i])
            if s:
                fused[i] = s

        pool = sorted(fused, key=fused.get, reverse=True)[: self.candidate_pool]
        max_f = max((fused[i] for i in pool), default=1.0) or 1.0
        items = [(self.chunks[i], fused[i] / max_f, cands[i]) for i in pool]
        reranked = self.reranker.rerank(pq, items)[:top_k]

        out = []
        for rank, (chunk, score, scope, reasons) in enumerate(reranked, 1):
            out.append(RetrievedPassage(
                chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
                jurisdiction=chunk.jurisdiction, state=chunk.state,
                section_title=chunk.section_title, content=chunk.content,
                source_url=chunk.source_url, score=round(score, 4), rank=rank,
                scope=scope, match_reasons=reasons, metadata=chunk.metadata))
        return out
