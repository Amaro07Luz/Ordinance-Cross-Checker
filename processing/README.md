# Part B: Document Processing & Indexing

Turns the raw files Part A collects into section-aware, embedded chunks in
Postgres + pgvector, and only re-processes what actually changed.

```
Part A raw files (data/raw/<jurisdiction>/...) + SourceRegistry metadata
   -> extract.py     PyMuPDF text layer; Tesseract OCR for scanned pages; HTML/TXT
   -> clean.py       unicode fixes, running headers/footers, page numbers, hyphenation
   -> versioning.py  stable doc_id + content hash: NEW / UNCHANGED / CHANGED
   -> chunker.py     Chapter > Article > Section/§ chunks (one section per chunk)
   -> embeddings.py  bge-small (default) / OpenAI / hashing; reuse unchanged vectors
   -> store.py       pgvector tables: documents, document_versions, chunks
```

## Setup

```bash
pip install -r processing/requirements.txt
brew install tesseract            # or: sudo apt install tesseract-ocr  (OCR fallback)
docker compose up -d              # Postgres 16 + pgvector
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/regintel
```

## Run

Put Part A's downloaded files in one folder per jurisdiction, named with the
slug of `GovernmentSource.jurisdiction_name`:

```
data/raw/fulshear/                        data/raw/harris-county/
data/raw/montgomery-county/               data/raw/washington-township-sewell/
data/raw/evesham-township-marlton/
```

```bash
python -m processing.pipeline --raw-dir data/raw                 # writes to pgvector
python -m processing.pipeline --raw-dir data/raw --force         # re-index everything
python -m processing.pipeline --raw-dir data/raw --in-memory --embedder hashing   # no DB
python -m processing.test_processing                             # smoke test
```

Re-running is cheap: unchanged documents are skipped, and for changed ones only
chunks whose text changed are re-embedded. Each run prints a change report,
e.g. `[CHANGED] Fulshear / udc.pdf -> v2 (+0 -0 ~1 sections)`, and the same
summary is stored in `document_versions.change_summary`.

## How it connects

**Part A.** Uses `GovernmentSource`, `SourceRegistry`, `ProcessedDocument` and
`DocumentChunk` from `ingestion_pipeline_part_a.py`. `ProcessingPipeline.run()`
takes the same `{file_path: GovernmentSource}` dict as
`run_offline_ingestion_pipeline`, and `discover_raw_files()` builds it from
the raw-storage folder. Part A's own `DocumentProcessor`, `SectionAwareChunker`
and `VectorStoreIndexer` are prototypes that this package replaces. B computes
its own `doc_id` from jurisdiction + source URL + title, because A's hashes the
local file path, which breaks version tracking when files are re-downloaded.

**Part C.** Chunks come back as Part A `DocumentChunk`s, so they drop straight
into `HybridRetriever`:

```python
from processing.store import PgVectorStore
from processing.embeddings import get_embedder, RetrievalEmbedderAdapter
from retrieval.search import HybridRetriever

emb = get_embedder()
store = PgVectorStore(dim=emb.dim)
retriever = HybridRetriever(store.load_chunks(),
                            embedder=RetrievalEmbedderAdapter(emb))  # real embeddings
```

For larger corpora, C can call `store.similarity_search(emb.embed_query(q), top_k,
jurisdictions=[...])` (HNSW index) and `store.keyword_search(q, ...)` (Postgres
full-text) instead of rebuilding indexes in Python.

**Chunk metadata** (in `chunk.metadata`): `jurisdiction_type`, `document_type`,
`file_type`, `doc_title`, `section_number`, `section_path`, `page_start`,
`page_end`, `ocr` (text came from OCR, so numbers may need a second look),
`part` (`k/n` when a long section was split), `content_hash`, `doc_version`.

## Config

| Env var | Default | |
|---|---|---|
| `DATABASE_URL` | none | Postgres DSN |
| `EMBEDDING_BACKEND` | `sentence-transformers` | or `openai`, `hashing` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | `text-embedding-3-small` for OpenAI |

The `chunks.embedding` column is sized to the model when the tables are first
created. Switching to a model with a different dimension means dropping the
tables and re-indexing (the store raises an error rather than mixing them).
