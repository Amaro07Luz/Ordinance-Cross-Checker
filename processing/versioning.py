"""
Part B (4/6): Version & Change Detection
Each source document gets a stable doc_id (jurisdiction + source URL + title),
so a re-crawl of the same ordinance is recognized as a new *version* of it.

  NEW        never seen before                 -> chunk, embed, store as v1
  UNCHANGED  cleaned-text hash matches latest  -> skip (just mark as checked)
  CHANGED    hash differs                      -> store v(n+1), report which
                                                  sections were added/removed/
                                                  modified, re-embed only chunks
                                                  whose text actually changed

Old chunks are kept with is_current = false, so answers can be audited
against the version that was live at the time.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ingestion_pipeline_part_a import GovernmentSource


def content_hash(text: str) -> str:
    """Hash of cleaned text with whitespace normalized, so re-flowed PDFs don't count as edits."""
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def make_doc_id(source: GovernmentSource, doc_title: str) -> str:
    """Stable across re-crawls (unlike Part A's id, which hashes the local file path)."""
    key = f"{source.jurisdiction_name}|{source.state}|{source.source_url}|{doc_title}".lower()
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


@dataclass
class ChangeReport:
    doc_id: str
    jurisdiction: str
    doc_title: str
    status: str                       # NEW | UNCHANGED | CHANGED
    version: int
    previous_version: Optional[int] = None
    sections_added: List[str] = field(default_factory=list)
    sections_removed: List[str] = field(default_factory=list)
    sections_modified: List[str] = field(default_factory=list)
    chunks_written: int = 0
    embeddings_computed: int = 0
    embeddings_reused: int = 0
    ocr_pages: int = 0
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "status": self.status, "version": self.version,
            "previous_version": self.previous_version,
            "sections_added": self.sections_added,
            "sections_removed": self.sections_removed,
            "sections_modified": self.sections_modified,
        }

    def __str__(self) -> str:
        line = f"[{self.status}] {self.jurisdiction} / {self.doc_title} -> v{self.version}"
        if self.status == "CHANGED":
            line += (f" (+{len(self.sections_added)} -{len(self.sections_removed)} "
                     f"~{len(self.sections_modified)} sections)")
        if self.status != "UNCHANGED":
            line += (f" | {self.chunks_written} chunks, {self.embeddings_computed} embedded, "
                     f"{self.embeddings_reused} reused")
        if self.ocr_pages:
            line += f" | {self.ocr_pages} OCR page(s)"
        return line


def diff_sections(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, List[str]]:
    return {
        "added": [k for k in new if k not in old],
        "removed": [k for k in old if k not in new],
        "modified": [k for k in new if k in old and old[k] != new[k]],
    }
