"""
Part A: Offline Data Ingestion Pipeline
Government Regulatory Intelligence Agent (M3 Prototype)

Team: Vincent Zheng, Riley Beaudion, Amaro Da Luz, Harlon Sprague, Ishaan Sayal
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Mandatory dependency checks for PDF and HTML parsing
try:
    import fitz  # PyMuPDF for reading PDFs
except ImportError:
    fitz = None

try:
    from BeautifulSoup4 import BeautifulSoup  # BeautifulSoup for cleaning HTML pages
except ImportError:
    BeautifulSoup = None


# ==========================================
# 1. SOURCE REGISTRY (Manages .gov portals)
# ==========================================

@dataclass
class GovernmentSource:
    jurisdiction_name: str
    jurisdiction_type: str  # Municipal, County, State
    state: str
    source_url: str
    document_type: str  # Zoning, Subdivision, DOT Access
    active: bool = True


class SourceRegistry:
    """Maintains approved government website URLs and jurisdiction metadata[cite: 3]."""
    
    # Pre-configured list of approved municipal and county portals for our prototype scope
    DEFAULT_SOURCES = [
        GovernmentSource(
            jurisdiction_name="Fulshear",
            jurisdiction_type="Municipal",
            state="TX",
            source_url="https://www.fulsheartexas.gov/business/community-development/planning-zoning",
            document_type="Subdivision & Zoning Code"
        ),
        GovernmentSource(
            jurisdiction_name="Harris County",
            jurisdiction_type="County",
            state="TX",
            source_url="https://oce.harriscountytx.gov/Services/Permits/Compliance",
            document_type="Development Regulations & Infrastructure"
        ),
        GovernmentSource(
            jurisdiction_name="Montgomery County",
            jurisdiction_type="County",
            state="TX",
            source_url="https://www.mctx.org/departments/departments_d_-_f/environmental_health/permitting/index.php",
            document_type="Subdivision & Permitting"
        ),
        GovernmentSource(
            jurisdiction_name="Washington Township (Sewell)",
            jurisdiction_type="Municipal",
            state="NJ",
            source_url="https://e-code.com",
            document_type="Chapter 220 Land Subdivision & Chapter 199 Site Plan"
        ),
        GovernmentSource(
            jurisdiction_name="Evesham Township (Marlton)",
            jurisdiction_type="Municipal",
            state="NJ",
            source_url="https://evesham-nj.org/community-development-page-list/1135-obtaining-a-zoning-permit",
            document_type="Chapter 160 Zoning Code"
        )
    ]

    def __init__(self, custom_sources: Optional[List[GovernmentSource]] = None):
        self.sources = custom_sources if custom_sources else self.DEFAULT_SOURCES

    def get_active_sources(self) -> List[GovernmentSource]:
        """Returns only the sources marked as active."""
        return [s for s in self.sources if s.active]


# ==========================================
# 2. DATA MODELS FOR DOCUMENTS & CHUNKS
# ==========================================

@dataclass
class ProcessedDocument:
    doc_id: str
    jurisdiction: str
    jurisdiction_type: str
    state: str
    doc_title: str
    source_url: str
    raw_text: str
    file_type: str
    extracted_at: str


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    jurisdiction: str
    state: str
    section_title: str
    content: str
    chunk_index: int
    source_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==========================================
# 3. DOCUMENT PROCESSING ENGINE (Extractors)
# ==========================================

class DocumentProcessor:
    """Extracts text, metadata, and structural content from PDFs and HTML[cite: 3]."""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extracts text page-by-page using PyMuPDF (fitz)[cite: 3]."""
        if not fitz:
            raise ImportError("PyMuPDF is required. Run `pip install PyMuPDF`.")
        
        doc = fitz.open(pdf_path)
        full_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                full_text.append(f"--- Page {page_num + 1} ---\n" + text)
            else:
                full_text.append(f"--- Page {page_num + 1} [Scanned Document / Needs OCR] ---\n")
        return "\n".join(full_text)

    @staticmethod
    def extract_text_from_html(html_content: str) -> str:
        """Strips website boilerplate and extracts clean text using BeautifulSoup."""
        if not BeautifulSoup:
            clean = re.sub('<[^<]+?>', '', html_content)
            return clean.strip()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove navigation bars, scripts, and footers to keep only core content
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        return soup.get_text(separator='\n', strip=True)

    def process_file(self, file_path: str, source: GovernmentSource) -> ProcessedDocument:
        """Determines file type (PDF vs HTML) and routes to the correct extractor."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            text = self.extract_text_from_pdf(file_path)
            file_type = "PDF"
        elif ext in [".html", ".htm"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = self.extract_text_from_html(f.read())
            file_type = "HTML"
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            file_type = "TXT"

        # Generate a unique hash ID for tracking the document
        doc_id = hashlib.md5(f"{source.jurisdiction_name}_{file_path}".encode()).hexdigest()[:12]
        
        return ProcessedDocument(
            doc_id=doc_id,
            jurisdiction=source.jurisdiction_name,
            jurisdiction_type=source.jurisdiction_type,
            state=source.state,
            doc_title=os.path.basename(file_path),
            source_url=source.source_url,
            raw_text=text,
            file_type=file_type,
            extracted_at=datetime.utcnow().isoformat()
        )


# ==========================================
# 4. SECTION-AWARE CHUNKING ENGINE
# ==========================================

class SectionAwareChunker:
    """Splits legal text along structural section boundaries (e.g., Section, Article)[cite: 2]."""

    def __init__(self, target_chunk_size: int = 1000, overlap: int = 150):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: ProcessedDocument) -> List[DocumentChunk]:
        chunks = []
        # Regex pattern to catch standard legal headings like 'Section 220-4' or 'Article IV'
        section_pattern = r'(?i)(^(?:Section|Article|Chapter|§)\s+\d+[A-Z0-9.-]*)'
        
        lines = doc.raw_text.splitlines()
        current_section = "General Provision"
        current_buffer = []
        current_len = 0
        chunk_idx = 0

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
                
            # Check if line marks a new regulatory section header
            match = re.match(section_pattern, line_str)
            if match:
                current_section = match.group(0)

            current_buffer.append(line_str)
            current_len += len(line_str)

            # When chunk reaches the target size limit, package it as a chunk object
            if current_len >= self.target_chunk_size:
                chunk_text = "\n".join(current_buffer)
                chunk_id = f"{doc.doc_id}_c{chunk_idx}"
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    jurisdiction=doc.jurisdiction,
                    state=doc.state,
                    section_title=current_section,
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    source_url=doc.source_url,
                    metadata={
                        "jurisdiction_type": doc.jurisdiction_type,
                        "file_type": doc.file_type,
                        "extracted_at": doc.extracted_at
                    }
                ))
                
                chunk_idx += 1
                # Retain a small line overlap to preserve context between text blocks
                overlap_lines = current_buffer[-2:] if len(current_buffer) > 2 else []
                current_buffer = overlap_lines
                current_len = sum(len(l) for l in current_buffer)

        # Catch and package any remaining text at the end of the file
        if current_buffer:
            chunk_text = "\n".join(current_buffer)
            chunk_id = f"{doc.doc_id}_c{chunk_idx}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                jurisdiction=doc.jurisdiction,
                state=doc.state,
                section_title=current_section,
                content=chunk_text,
                chunk_index=chunk_idx,
                source_url=doc.source_url,
                metadata={
                    "jurisdiction_type": doc.jurisdiction_type,
                    "file_type": doc.file_type,
                    "extracted_at": doc.extracted_at
                }
            ))

        return chunks


# ==========================================
# 5. VECTOR STORE INDEXER (Knowledge Base)
# ==========================================

class VectorStoreIndexer:
    """Simulates storing document chunks and embeddings into a shared knowledge base[cite: 3]."""

    def __init__(self, collection_name: str = "regulatory_knowledge_base"):
        self.collection_name = collection_name
        self.storage: Dict[str, DocumentChunk] = {}

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Saves chunks into the searchable storage dictionary."""
        for chunk in chunks:
            self.storage[chunk.chunk_id] = chunk
        print(f" Successfully indexed {len(chunks)} chunks into vector store '{self.collection_name}'.")


# ==========================================
# PIPELINE EXECUTION ENTRY POINT
# ==========================================

def run_offline_ingestion_pipeline(source_files: Dict[str, GovernmentSource]):
    print("=" * 70)
    print("STARTING PART A: OFFLINE DATA INGESTION PIPELINE")
    print("=" * 70)

    processor = DocumentProcessor()
    chunker = SectionAwareChunker()
    indexer = VectorStoreIndexer()
    total_chunks = []

    for file_path, source in source_files.items():
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}. Skipping.")
            continue
        
        print(f" Processing source: {source.jurisdiction_name} ({source.document_type}) -> {file_path}")
        doc = processor.process_file(file_path, source)
        chunks = chunker.chunk_document(doc)
        total_chunks.extend(chunks)
        print(f"   Generated {len(chunks)} section-aware chunks.")

    if total_chunks:
        indexer.index_chunks(total_chunks)
    
    print("=" * 70)
    print("PART A PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    registry = SourceRegistry()
    print(f"Loaded {len(registry.get_active_sources())} active sources in Source Registry.")