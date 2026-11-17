"""
Part B: Document Processing & Indexing pipeline

    Part A raw files + GovernmentSource metadata
        -> extract (PyMuPDF, OCR fallback, HTML)
        -> clean
        -> version/change detection   (skip unchanged documents)
        -> section-aware chunking     (Part A's DocumentChunk)
        -> embeddings                 (reuse vectors for unchanged chunk text)
        -> pgvector                   (documents / document_versions / chunks)

Library use (same input shape as Part A's run_offline_ingestion_pipeline):

    from ingestion_pipeline_part_a import SourceRegistry
    from processing.pipeline import ProcessingPipeline, discover_raw_files
    from processing.store import PgVectorStore
    from processing.embeddings import get_embedder

    emb = get_embedder()
    pipe = ProcessingPipeline(PgVectorStore(dim=emb.dim), emb)
    pipe.run(discover_raw_files("data/raw", SourceRegistry()))

CLI:
    python -m processing.pipeline --raw-dir data/raw            # uses DATABASE_URL
    python -m processing.pipeline --raw-dir data/raw --in-memory --embedder hashing
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional

from ingestion_pipeline_part_a import GovernmentSource, ProcessedDocument, SourceRegistry
from processing.chunker import LegalSectionChunker
from processing.clean import TextCleaner
from processing.embeddings import Embedder, build_embedding_text, get_embedder
from processing.extract import TextExtractor
from processing.versioning import ChangeReport, content_hash, diff_sections, make_doc_id

SUPPORTED_EXTS = {".pdf", ".html", ".htm", ".txt"}


class ProcessingPipeline:
    def __init__(self, store, embedder: Embedder,
                 extractor: Optional[TextExtractor] = None,
                 cleaner: Optional[TextCleaner] = None,
                 chunker: Optional[LegalSectionChunker] = None,
                 force: bool = False, embed_batch_size: int = 64, verbose: bool = True):
        if getattr(store, "dim", embedder.dim) != embedder.dim:
            raise ValueError(f"Store dim {store.dim} != embedder dim {embedder.dim}")
        self.store = store
        self.embedder = embedder
        self.extractor = extractor or TextExtractor()
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or LegalSectionChunker()
        self.force = force
        self.embed_batch_size = embed_batch_size
        self.verbose = verbose

    # ------------------------------------------------------------------
    def process_file(self, path: str, source: GovernmentSource,
                     doc_title: Optional[str] = None) -> ChangeReport:
        doc_title = doc_title or os.path.basename(path)
        doc_id = make_doc_id(source, doc_title)

        # 1-2. extract + clean
        extracted = self.extractor.extract(path)
        pages = self.cleaner.clean_pages(extracted.pages)
        full_text = "\n\n".join(f"--- Page {p.page_number} ---\n{p.text}" for p in pages)
        text_hash = content_hash("\n".join(p.text for p in pages))

        # 3. version / change detection
        existing = self.store.get_document(doc_id)
        if existing and existing["current_hash"] == text_hash and not self.force:
            self.store.touch_document(doc_id)
            return ChangeReport(doc_id, source.jurisdiction_name, doc_title, "UNCHANGED",
                                version=existing["current_version"],
                                previous_version=existing["current_version"],
                                ocr_pages=extracted.ocr_pages, warnings=extracted.warnings)

        version = (existing["current_version"] + 1) if existing else 1
        section_hashes = self.chunker.section_hashes(pages)
        report = ChangeReport(doc_id, source.jurisdiction_name, doc_title,
                              "CHANGED" if existing else "NEW", version=version,
                              previous_version=existing["current_version"] if existing else None,
                              ocr_pages=extracted.ocr_pages, warnings=list(extracted.warnings))
        if existing:
            d = diff_sections(existing["section_hashes"], section_hashes)
            report.sections_added, report.sections_removed, report.sections_modified = (
                d["added"], d["removed"], d["modified"])

        # 4. section-aware chunking -> Part A's ProcessedDocument / DocumentChunk
        doc = ProcessedDocument(
            doc_id=doc_id, jurisdiction=source.jurisdiction_name,
            jurisdiction_type=source.jurisdiction_type, state=source.state,
            doc_title=doc_title, source_url=source.source_url, raw_text=full_text,
            file_type=extracted.file_type, extracted_at=_utcnow())
        chunks = self.chunker.chunk_document(
            doc, pages, version=version,
            extra_metadata={"document_type": source.document_type})
        if not chunks:
            report.warnings.append("No text could be extracted; nothing indexed.")

        # 5. embeddings (reuse vectors for chunk text that already exists)
        hashes = [c.metadata["content_hash"] for c in chunks]
        known = self.store.find_embeddings(hashes, self.embedder.name)
        todo = [c for c in chunks if c.metadata["content_hash"] not in known]
        for i in range(0, len(todo), self.embed_batch_size):
            batch = todo[i:i + self.embed_batch_size]
            vecs = self.embedder.embed_documents([build_embedding_text(c) for c in batch])
            for c, v in zip(batch, vecs):
                known[c.metadata["content_hash"]] = v
        report.embeddings_computed = len({c.metadata["content_hash"] for c in todo})
        report.embeddings_reused = len(chunks) - len(todo)

        # 6. write to the vector store
        self.store.save_version(
            document={"doc_id": doc_id, "jurisdiction": source.jurisdiction_name,
                      "jurisdiction_type": source.jurisdiction_type, "state": source.state,
                      "doc_title": doc_title, "source_url": source.source_url,
                      "document_type": source.document_type},
            version={"version": version, "content_hash": text_hash,
                     "section_hashes": section_hashes, "file_type": extracted.file_type,
                     "page_count": len(pages), "ocr_pages": extracted.ocr_pages,
                     "extracted_at": doc.extracted_at, "change_summary": report.summary()},
            chunks=chunks, embeddings=[known[h] for h in hashes],
            embedding_model=self.embedder.name)
        report.chunks_written = len(chunks)
        return report

    def run(self, source_files: Dict[str, GovernmentSource]) -> List[ChangeReport]:
        self._log("=" * 70)
        self._log("PART B: DOCUMENT PROCESSING & INDEXING")
        self._log("=" * 70)
        reports = []
        for path, source in source_files.items():
            if not os.path.exists(path):
                self._log(f"[Warning] File not found: {path}. Skipping.")
                continue
            try:
                r = self.process_file(path, source)
            except Exception as e:  # one bad file must not stop the batch
                self._log(f"[Error] {path}: {type(e).__name__}: {e}")
                continue
            reports.append(r)
            self._log(" " + str(r))
            for w in r.warnings:
                self._log(f"   warning: {w}")
        counts = {s: sum(r.status == s for r in reports) for s in ("NEW", "CHANGED", "UNCHANGED")}
        self._log(f"Done: {counts}, {sum(r.chunks_written for r in reports)} chunks written.")
        return reports

    def _log(self, msg):
        if self.verbose:
            print(msg)


# ----------------------------------------------------------------------
# Hook into Part A's raw document storage
# ----------------------------------------------------------------------
def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def discover_raw_files(raw_dir: str, registry: Optional[SourceRegistry] = None
                       ) -> Dict[str, GovernmentSource]:
    """Maps files saved by Part A to their GovernmentSource.
    Expected layout: <raw_dir>/<slug of jurisdiction_name>/**/<file>
      e.g. data/raw/fulshear/udc.pdf, data/raw/evesham-township-marlton/chapter-160.html
    If a jurisdiction has several registry entries, the first active one is used."""
    registry = registry or SourceRegistry()
    by_slug: Dict[str, GovernmentSource] = {}
    for s in registry.get_active_sources():
        by_slug.setdefault(slugify(s.jurisdiction_name), s)

    found: Dict[str, GovernmentSource] = {}
    if not os.path.isdir(raw_dir):
        return found
    for entry in sorted(os.listdir(raw_dir)):
        src = by_slug.get(entry)
        folder = os.path.join(raw_dir, entry)
        if not src or not os.path.isdir(folder):
            continue
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                    found[os.path.join(root, f)] = src
    return found


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Part B: process raw documents into pgvector")
    ap.add_argument("--raw-dir", default="data/raw",
                    help="Part A raw storage (one sub-folder per jurisdiction slug)")
    ap.add_argument("--dsn", default=os.getenv("DATABASE_URL"), help="Postgres DSN")
    ap.add_argument("--embedder", default=os.getenv("EMBEDDING_BACKEND", "sentence-transformers"),
                    choices=["sentence-transformers", "openai", "hashing"])
    ap.add_argument("--model", default=os.getenv("EMBEDDING_MODEL"))
    ap.add_argument("--in-memory", action="store_true", help="don't write to Postgres")
    ap.add_argument("--force", action="store_true", help="re-index even if unchanged")
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args(argv)

    files = discover_raw_files(args.raw_dir)
    if not files:
        print(f"No files found under {args.raw_dir}. Expected sub-folders: "
              + ", ".join(slugify(s.jurisdiction_name) for s in SourceRegistry().get_active_sources()))
        return 1

    embedder = get_embedder(args.embedder, args.model)
    if args.in_memory:
        from processing.store import InMemoryStore
        store = InMemoryStore(dim=embedder.dim)
    else:
        from processing.store import PgVectorStore
        store = PgVectorStore(args.dsn, dim=embedder.dim)

    ProcessingPipeline(store, embedder, extractor=TextExtractor(ocr=not args.no_ocr),
                       force=args.force).run(files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
