"""
Part B (1/6): Text Extraction
PyMuPDF text extraction for PDFs, with a per-page OCR fallback (Tesseract)
for scanned pages, plus HTML and plain-text extraction for crawled pages.

Input:  a raw file saved by Part A (crawler / raw document storage)
Output: ExtractionResult (a list of PageText), turned into Part A's
        ProcessedDocument by the pipeline.
"""

import io
import os
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # older PyMuPDF only exposes the "fitz" name
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None


@dataclass
class PageText:
    page_number: int          # 1-based
    text: str
    method: str               # "text" | "ocr" | "html" | "plain" | "empty"


@dataclass
class ExtractionResult:
    pages: List[PageText]
    file_type: str            # "PDF" | "HTML" | "TXT"
    title: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def ocr_pages(self) -> int:
        return sum(1 for p in self.pages if p.method == "ocr")

    @property
    def empty_pages(self) -> int:
        return sum(1 for p in self.pages if p.method == "empty")


class TextExtractor:
    """Routes a file to the right extractor based on its extension."""

    # HTML elements that never contain ordinance text
    HTML_NOISE = ["script", "style", "noscript", "nav", "footer", "header",
                  "aside", "form", "iframe", "svg", "button"]
    # Common containers for the main content on .gov / eCode360 / Municode pages
    HTML_MAIN_SELECTORS = ["main", "article", "#codeContent", "#content",
                           ".content", "#main-content", "#mainContent", "body"]

    def __init__(self, ocr: bool = True, ocr_dpi: int = 300, ocr_lang: str = "eng",
                 min_chars_per_page: int = 40):
        """
        ocr:                run Tesseract on pages whose text layer is (nearly) empty
        min_chars_per_page: below this many characters a PDF page is treated as scanned
        """
        self.ocr = ocr
        self.ocr_dpi = ocr_dpi
        self.ocr_lang = ocr_lang
        self.min_chars_per_page = min_chars_per_page

    @property
    def ocr_available(self) -> bool:
        if not (self.ocr and pytesseract and Image):
            return False
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract(self, path: str) -> ExtractionResult:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self.extract_pdf(path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        if ext in (".html", ".htm") or raw.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
            return self.extract_html(raw)
        return ExtractionResult(pages=[PageText(1, raw, "plain")], file_type="TXT")

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------
    def extract_pdf(self, path: str) -> ExtractionResult:
        if fitz is None:
            raise ImportError("PyMuPDF is required for PDFs. Run `pip install PyMuPDF`.")

        result = ExtractionResult(pages=[], file_type="PDF")
        ocr_ok = self.ocr_available
        warned_no_ocr = False

        with fitz.open(path) as doc:
            meta_title = (doc.metadata or {}).get("title") or None
            result.title = meta_title.strip() if meta_title and meta_title.strip() else None

            for i, page in enumerate(doc, start=1):
                # sort=True orders blocks top-to-bottom, left-to-right, which keeps
                # headings ahead of their body text on multi-column layouts
                text = page.get_text("text", sort=True) or ""
                if len(text.strip()) >= self.min_chars_per_page:
                    result.pages.append(PageText(i, text, "text"))
                    continue

                if ocr_ok:
                    ocr_text = self._ocr_page(page)
                    if len(ocr_text.strip()) > len(text.strip()):
                        result.pages.append(PageText(i, ocr_text, "ocr"))
                        continue
                elif not warned_no_ocr and self.ocr:
                    result.warnings.append(
                        "Scanned pages found but Tesseract is not installed; "
                        "those pages were indexed without OCR.")
                    warned_no_ocr = True

                result.pages.append(PageText(i, text, "text" if text.strip() else "empty"))

        if result.empty_pages:
            result.warnings.append(f"{result.empty_pages} page(s) had no extractable text.")
        return result

    def _ocr_page(self, page) -> str:
        pix = page.get_pixmap(dpi=self.ocr_dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        # psm 6 = assume a uniform block of text; good for ordinance pages
        return pytesseract.image_to_string(img, lang=self.ocr_lang, config="--psm 6")

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------
    def extract_html(self, html: str) -> ExtractionResult:
        if BeautifulSoup is None:
            import re
            text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
            text = re.sub(r"(?s)<[^>]+>", "\n", text)
            return ExtractionResult(pages=[PageText(1, text, "html")], file_type="HTML",
                                    warnings=["beautifulsoup4 not installed; used regex tag stripping."])

        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else None
        for tag in soup(self.HTML_NOISE):
            tag.decompose()

        main = None
        for sel in self.HTML_MAIN_SELECTORS:
            main = soup.select_one(sel)
            if main and len(main.get_text(strip=True)) > 200:
                break
        main = main or soup

        # Put block elements on their own lines so headings stay separate from body text
        for br in main.find_all("br"):
            br.replace_with("\n")
        text = main.get_text(separator="\n", strip=True)
        return ExtractionResult(pages=[PageText(1, text, "html")], file_type="HTML", title=title)
