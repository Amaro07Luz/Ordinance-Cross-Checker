"""
Part B (2/6): Cleaning
Normalizes extracted text so chunking and embeddings see clean ordinance text:
  - Unicode normalization (ligatures, smart quotes, non-breaking spaces)
  - Removes running headers/footers repeated across PDF pages
  - Removes standalone page numbers ("Page 4 of 120", "- 12 -")
  - Re-joins words hyphenated across line breaks ("regu-\\nlation")
  - Re-joins lines that were wrapped mid-sentence by the PDF layout
  - Collapses whitespace
Page boundaries are kept so chunks can cite page numbers.
"""

import re
import unicodedata
from collections import Counter
from typing import List

from processing.extract import PageText

REPLACEMENTS = {
    " ": " ", " ": " ", " ": " ", " ": " ", "​": "",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": " - ", "‐": "-", "‑": "-",
    "•": "-", "·": "-", "": "-", "﻿": "",
}

PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s+)?[-–]?\s*\d{1,4}\s*[-–]?\s*(?:of\s+\d{1,4})?\s*$", re.I)
HYPHEN_BREAK_RE = re.compile(r"([a-z])-\n([a-z])")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")

# Lines that start a new logical block and must never be glued onto the previous line
BLOCK_START_RE = re.compile(
    r"^(?:§|section\b|sec\.|article\b|chapter\b|subchapter\b|division\b|part\b|"
    r"\(?[a-z0-9]{1,4}[.)]\s|\d+(?:\.\d+)+\s|[-*]\s|table\b|figure\b)", re.I)


class TextCleaner:
    def __init__(self, header_footer_ratio: float = 0.5, min_pages_for_hf: int = 3):
        """
        header_footer_ratio: a short line appearing on at least this share of pages
                             (ignoring digits) is treated as a running header/footer
        """
        self.header_footer_ratio = header_footer_ratio
        self.min_pages_for_hf = min_pages_for_hf

    # ------------------------------------------------------------------
    def clean_pages(self, pages: List[PageText]) -> List[PageText]:
        normalized = [PageText(p.page_number, self.normalize(p.text), p.method) for p in pages]
        repeated = self._repeated_lines(normalized)
        out = []
        for p in normalized:
            lines = []
            for line in p.text.split("\n"):
                s = line.strip()
                if not s:
                    lines.append("")
                    continue
                if PAGE_NUMBER_RE.match(s) or self._hf_key(s) in repeated:
                    continue
                lines.append(s)
            text = "\n".join(lines)
            text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
            text = self._join_wrapped_lines(text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            out.append(PageText(p.page_number, text, p.method))
        return out

    def clean_text(self, text: str) -> str:
        return self.clean_pages([PageText(1, text, "plain")])[0].text

    # ------------------------------------------------------------------
    @staticmethod
    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        for a, b in REPLACEMENTS.items():
            text = text.replace(a, b)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
        return "\n".join(MULTISPACE_RE.sub(" ", ln).rstrip() for ln in text.split("\n"))

    @staticmethod
    def _hf_key(line: str) -> str:
        return re.sub(r"\d+", "#", line.lower()).strip()

    def _repeated_lines(self, pages: List[PageText]) -> set:
        if len(pages) < self.min_pages_for_hf:
            return set()
        counts: Counter = Counter()
        for p in pages:
            lines = [ln.strip() for ln in p.text.split("\n") if ln.strip()]
            # headers/footers live in the first/last few lines of a page
            edge = lines[:3] + lines[-3:]
            counts.update({self._hf_key(ln) for ln in edge if len(ln) <= 120})
        threshold = max(2, int(len(pages) * self.header_footer_ratio))
        return {k for k, c in counts.items() if c >= threshold}

    @staticmethod
    def _join_wrapped_lines(text: str) -> str:
        """Glue a line onto the previous one when the PDF wrapped it mid-sentence."""
        out: List[str] = []
        for line in text.split("\n"):
            if (out and out[-1] and line
                    and not out[-1].endswith((".", ":", ";", "?", "!"))
                    and line[0].islower()
                    and not BLOCK_START_RE.match(line)):
                out[-1] = out[-1] + " " + line
            else:
                out.append(line)
        return "\n".join(out)
