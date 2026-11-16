"""
Part B (3/6): Section-Aware Chunking
Splits cleaned ordinance text on its legal structure (Chapter > Article >
Section/§) so every chunk belongs to exactly one section and carries that
section's heading. Long sections are split on paragraph, then sentence,
boundaries with a small overlap; a chunk never straddles two sections.

Output is Part A's DocumentChunk, so Part C's HybridRetriever can consume it
unchanged. Extra fields Part C/D can use are in chunk.metadata:
  jurisdiction_type, document_type, file_type, extracted_at,
  section_number, section_path, page_start, page_end, ocr,
  part (k of n when a section was split), content_hash, doc_version
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ingestion_pipeline_part_a import DocumentChunk, ProcessedDocument
from processing.extract import PageText

# ----------------------------------------------------------------------
# Heading detection
# level 1 = containers (Chapter/Article/Subchapter/Division/Part)
# level 2 = sections (Section 3.2, Sec. 4-12, § 220-4, 3.2.1 Title)
# ----------------------------------------------------------------------
# What may follow the number: punctuation (not starting another number), a Capitalized
# title, or end of line. "Section 160-12 of this chapter" is a reference, not a heading.
_TAIL = r"(?:\s*[.:\-](?![0-9A-Za-z])\s*|\s+(?=[A-Z(\"'])|\s*$)"
HEADING_PATTERNS: List[Tuple[int, re.Pattern]] = [
    (1, re.compile(r"^(?P<kind>(?i:chapter|subchapter|article|division|part))\s+"
                   r"(?P<num>[0-9]+[A-Z]?(?:[.\-][0-9A-Z]+)*|[IVXLC]+)\b" + _TAIL)),
    (2, re.compile(r"^(?P<kind>§§?|(?i:section|sec\.))\s*"
                   r"(?P<num>[0-9]+[A-Z]?(?:[.\-][0-9A-Z]+)*)" + _TAIL)),
    # Numbered outline headings used by many Texas UDCs: "3.2.1 Front Yard Setbacks"
    (2, re.compile(r"^(?P<kind>)(?P<num>[0-9]{1,2}(?:\.[0-9]{1,3}){1,3})\.?\s+(?=[A-Z])")),
]
MAX_HEADING_LEN = 140
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\"'0-9])")


CONTAINER_RANK = {"chapter": 0, "subchapter": 1, "article": 2, "part": 3, "division": 4}


def match_heading(line: str) -> Optional[Tuple[int, str, str]]:
    """Returns (level, section_number, kind) if the line is a heading, else None."""
    s = line.strip()
    if not s or len(s) > MAX_HEADING_LEN or s.endswith((",", ";")):
        return None
    for level, pat in HEADING_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        # Numbered-outline headings must look like a title, not a sentence
        if not m.group("kind") and (s.endswith(".") and len(s.split()) > 12):
            continue
        return level, m.group("num"), m.group("kind").lower()
    return None


@dataclass
class Section:
    title: str
    number: Optional[str]
    path: List[str]
    lines: List[Tuple[str, int]] = field(default_factory=list)  # (text, page_number)

    @property
    def body(self) -> str:
        return "\n".join(t for t, _ in self.lines)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LegalSectionChunker:
    def __init__(self, target_chars: int = 1200, max_chars: int = 1800,
                 overlap_chars: int = 200, min_chars: int = 40,
                 preamble_title: str = "General Provision"):
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars
        self.preamble_title = preamble_title  # same label Part A uses

    # ------------------------------------------------------------------
    def split_sections(self, pages: List[PageText]) -> List[Section]:
        containers: List[str] = []     # current Chapter/Article path
        current = Section(self.preamble_title, None, [])
        sections = [current]
        for page in pages:
            for raw in page.text.split("\n"):
                line = raw.strip()
                if not line:
                    continue
                h = match_heading(line)
                if h:
                    level, num, kind = h
                    if level == 1:
                        # keep only the containers above this one (Chapter > Article > Division)
                        rank = CONTAINER_RANK[kind]
                        containers = [c for c in containers
                                      if CONTAINER_RANK[c.split()[0].lower()] < rank]
                        containers.append(line)
                    current = Section(line, num, list(containers))
                    sections.append(current)
                current.lines.append((line, page.page_number))
        return [s for s in sections if len(self._body_without_heading(s)) >= self.min_chars
                or (s is sections[0] and len(s.body) >= self.min_chars)]

    @staticmethod
    def _body_without_heading(s: Section) -> str:
        return "\n".join(t for t, _ in s.lines[1:]) if s.number is not None else s.body

    # ------------------------------------------------------------------
    def _split_long(self, lines: List[Tuple[str, int]]) -> List[List[Tuple[str, int]]]:
        """Packs lines (paragraphs) into pieces of ~target_chars with overlap."""
        units: List[Tuple[str, int]] = []
        for text, pg in lines:
            if len(text) <= self.max_chars:
                units.append((text, pg))
            else:  # very long paragraph: fall back to sentences
                units.extend((s, pg) for s in SENTENCE_SPLIT_RE.split(text) if s)

        pieces, buf, size, fresh = [], [], 0, False
        for text, pg in units:
            if fresh and size + len(text) > self.target_chars:
                pieces.append(buf)
                # carry the tail of the previous piece forward as overlap
                tail, tail_len = [], 0
                for u in reversed(buf):
                    if tail_len + len(u[0]) > self.overlap_chars:
                        break
                    tail.insert(0, u)
                    tail_len += len(u[0])
                buf, size = tail, tail_len
            buf.append((text, pg))
            size += len(text) + 1
            fresh = True   # buf holds text not yet emitted
        if fresh:
            pieces.append(buf)
        # Hard cap: any piece still over max_chars is cut on character boundaries
        out = []
        for p in pieces:
            joined = "\n".join(t for t, _ in p)
            if len(joined) <= self.max_chars:
                out.append(p)
            else:
                for i in range(0, len(joined), self.target_chars):
                    out.append([(joined[i:i + self.max_chars], p[0][1])])
        return out

    # ------------------------------------------------------------------
    def chunk_document(self, doc: ProcessedDocument, pages: List[PageText],
                       version: int = 1,
                       extra_metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """chunk_id = <doc_id>_v<version>_c<index>, unique across versions."""
        ocr_pages = {p.page_number for p in pages if p.method == "ocr"}
        chunks: List[DocumentChunk] = []
        idx = 0
        for sec in self.split_sections(pages):
            pieces = self._split_long(sec.lines) if len(sec.body) > self.max_chars else [sec.lines]
            for k, piece in enumerate(pieces, start=1):
                content = "\n".join(t for t, _ in piece)
                # Later pieces of a split section start mid-section; restate the heading
                if k > 1 and not content.startswith(sec.title):
                    content = f"{sec.title} (continued)\n{content}"
                pages_in = [pg for _, pg in piece]
                meta = {
                    "jurisdiction_type": doc.jurisdiction_type,
                    "file_type": doc.file_type,
                    "extracted_at": doc.extracted_at,
                    "doc_title": doc.doc_title,
                    "section_number": sec.number,
                    "section_path": sec.path + [sec.title],
                    "page_start": min(pages_in),
                    "page_end": max(pages_in),
                    "ocr": any(pg in ocr_pages for pg in pages_in),
                    "part": f"{k}/{len(pieces)}",
                    "content_hash": sha(content),
                    "doc_version": version,
                }
                meta.update(extra_metadata or {})
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc.doc_id}_v{version}_c{idx}",
                    doc_id=doc.doc_id,
                    jurisdiction=doc.jurisdiction,
                    state=doc.state,
                    section_title=sec.title,
                    content=content,
                    chunk_index=idx,
                    source_url=doc.source_url,
                    metadata=meta,
                ))
                idx += 1
        return chunks

    def section_hashes(self, pages: List[PageText]) -> Dict[str, str]:
        """{section key: hash of its text}, used for version/change detection."""
        out: Dict[str, str] = {}
        for sec in self.split_sections(pages):
            key = sec.number or sec.title
            if key in out:  # duplicate numbers (e.g. two "Section 1" in different articles)
                key = " > ".join(sec.path + [sec.title])
            out[key] = sha(re.sub(r"\s+", " ", sec.body).strip())
        return out
