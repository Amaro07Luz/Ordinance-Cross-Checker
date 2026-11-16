"""
Workstream B: document processing and indexing.

extract.py     PyMuPDF text extraction, Tesseract OCR fallback, HTML/TXT
clean.py       normalization, header/footer + page-number removal, de-hyphenation
chunker.py     LegalSectionChunker: Chapter > Article > Section/§ aware chunks
versioning.py  stable doc ids, content hashes, section-level change detection
embeddings.py  sentence-transformers / OpenAI / hashing embedders + Part C adapter
store.py       PgVectorStore (Postgres + pgvector) and InMemoryStore
pipeline.py    ProcessingPipeline + CLI (python -m processing.pipeline)

Depends on Part A (ingestion_pipeline_part_a.py) for GovernmentSource,
SourceRegistry, ProcessedDocument and DocumentChunk; emits DocumentChunk
for Part C. Smoke test: python -m processing.test_processing
"""
