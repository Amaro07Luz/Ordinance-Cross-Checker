"""
Smoke test for Part B. Builds SYNTHETIC fixtures (made-up ordinance text, not
real regulations): a multi-page text PDF, a scanned (image-only) PDF, and an
eCode360-style HTML page, then runs the full pipeline.

Run from the repo root:
    python -m processing.test_processing                     # in-memory store
    DATABASE_URL=postgresql://... python -m processing.test_processing   # + pgvector

If Part C's retrieval/ package is on the path, it also runs C's
HybridRetriever over the chunks Part B produced.
"""

import os
import shutil
import tempfile

from ingestion_pipeline_part_a import DocumentChunk, GovernmentSource, SourceRegistry
from processing.chunker import LegalSectionChunker, match_heading
from processing.clean import TextCleaner
from processing.embeddings import HashingEmbedder, RetrievalEmbedderAdapter
from processing.extract import TextExtractor, fitz
from processing.pipeline import ProcessingPipeline, discover_raw_files, slugify
from processing.store import InMemoryStore

FULSHEAR_PAGES = [
    ["ARTICLE 3 ZONING DISTRICTS",
     "Section 3.1 Purpose",
     "The zoning districts established in this article are intended to promote orderly",
     "development of the city.",
     "Section 3.2 Residential Setbacks",
     "In the R-1 district the front yard setback shall be 25 feet. The side yard setback",
     "shall be 7.5 feet and the rear yard setback shall be 20 feet from the property line."],
    ["Section 3.3 Building Height",
     "No building in a residential district shall exceed 35 feet or two and one-half stories",
     "in height, measured from the average finished grade.",
     "ARTICLE 5 SUBDIVISION",
     "Section 5.1 Platting Required",
     "A preliminary plat must be submitted to the Planning and Zoning Commission before",
     "any subdivision of land within the city or its extraterritorial jurisdiction."],
    ["Section 5.2 Final Plat",
     "A final plat shall be approved within two years of preliminary plat approval.",
     "Section 5.3 Parkland Dedication",
     "Each residential subdivision shall dedicate one acre of parkland per 100 dwelling units.",
     "Section 5.4 Sidewalks",
     "Sidewalks of at least five feet in width are required along all public streets."],
]

EVESHAM_HTML = """<!DOCTYPE html><html><head><title>Chapter 160 Zoning</title>
<script>var tracking = 1;</script></head><body>
<nav>Home | Departments | Contact Us</nav>
<div id="codeContent">
<h2>Chapter 160 Zoning</h2>
<h3>Article IV Residential Districts</h3>
<h4>§ 160-12. Height regulations.</h4>
<p>Maximum building height in residential districts is 35 feet or 2.5 stories,
whichever is less. Chimneys and spires are exempt.</p>
<h4>§ 160-13. Minimum lot area.</h4>
<p>The minimum lot area in the R-1 district shall be 20,000 square feet with a
minimum lot width of 100 feet.</p>
<p>See also Section 160-12 of this chapter for height limits.</p>
</div><footer>Copyright eCode360</footer></body></html>"""


def make_text_pdf(path, pages, header="City of Fulshear Unified Development Code"):
    doc = fitz.open()
    for n, lines in enumerate(pages, start=1):
        page = doc.new_page()
        y = 40
        page.insert_text((72, y), header, fontsize=9)
        y += 30
        for ln in lines:
            page.insert_text((72, y), ln, fontsize=10)
            y += 16
        page.insert_text((280, 800), f"Page {n} of {len(pages)}", fontsize=9)
    doc.save(path)


def make_scanned_pdf(path, lines):
    """Renders text to an image and stores only the image: no text layer, needs OCR."""
    src = fitz.open()
    p = src.new_page()
    y = 72
    for ln in lines:
        p.insert_text((72, y), ln, fontsize=13)
        y += 22
    pix = p.get_pixmap(dpi=200)
    out = fitz.open()
    page = out.new_page()
    page.insert_image(page.rect, pixmap=pix)
    out.save(path)


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise AssertionError(msg)


def unit_tests():
    print("Unit checks")
    check(match_heading("§ 220-4. Definitions.") is not None, "detects '§ 220-4.' heading")
    check(match_heading("Section 3.2 Residential Setbacks") is not None, "detects 'Section 3.2'")
    check(match_heading("ARTICLE IV RESIDENTIAL") is not None, "detects 'ARTICLE IV'")
    check(match_heading("Section 160-12 of this chapter applies.") is None,
          "ignores in-text reference 'Section 160-12 of this chapter'")
    cleaned = TextCleaner().clean_text("The regu-\nlation applies to all\nlots in the district.")
    check("regulation applies to all lots" in cleaned, "de-hyphenates and re-joins wrapped lines")


def build_fixtures(raw):
    regs = {s.jurisdiction_name: s for s in SourceRegistry().get_active_sources()}
    ful = os.path.join(raw, slugify("Fulshear"))
    eve = os.path.join(raw, slugify("Evesham Township (Marlton)"))
    mon = os.path.join(raw, slugify("Montgomery County"))
    for d in (ful, eve, mon):
        os.makedirs(d)
    make_text_pdf(os.path.join(ful, "fulshear_udc.pdf"), FULSHEAR_PAGES)
    with open(os.path.join(eve, "chapter_160.html"), "w") as f:
        f.write(EVESHAM_HTML)
    make_scanned_pdf(os.path.join(mon, "septic_permit_scan.pdf"), [
        "Section 4.1 On-Site Sewage Facilities",
        "A permit is required before installing any on-site",
        "sewage facility. Minimum lot size for a septic",
        "system is one acre.",
    ])
    return regs


def pipeline_tests(store, label):
    print(f"\nPipeline checks ({label})")
    raw = tempfile.mkdtemp()
    try:
        build_fixtures(raw)
        files = discover_raw_files(raw)
        check(len(files) == 3, f"discover_raw_files mapped 3 files to Part A sources ({len(files)})")

        pipe = ProcessingPipeline(store, HashingEmbedder(), verbose=True)
        reports = {os.path.basename(r.doc_title): r for r in pipe.run(files)}
        check(all(r.status == "NEW" for r in reports.values()), "first run: all documents NEW")

        chunks = store.load_chunks()
        check(all(isinstance(c, DocumentChunk) for c in chunks), "store returns Part A DocumentChunks")
        titles = {c.section_title for c in chunks}
        check("Section 3.2 Residential Setbacks" in titles, "Fulshear section titles preserved")
        check("§ 160-12. Height regulations." in titles, "Evesham § headings preserved")
        check(not any("Page 1 of 3" in c.content or "Unified Development Code" in c.content
                      for c in chunks), "running header and page footer removed")
        check(not any("tracking" in c.content or "Contact Us" in c.content for c in chunks),
              "HTML script/nav boilerplate removed")
        setback = next(c for c in chunks if c.section_title.startswith("Section 3.2"))
        check("25 feet" in setback.content and "Height" not in setback.content,
              "a chunk holds one section only")
        check(setback.metadata["section_path"][0].startswith("ARTICLE 3"),
              "section_path records the parent Article")
        check(setback.metadata["jurisdiction_type"] == "Municipal",
              "metadata.jurisdiction_type set (needed by Part C's state scope)")
        h512 = next(c for c in chunks if c.section_title.startswith("Section 5.2"))
        check(h512.metadata["page_start"] == 3, "page numbers tracked for citations")

        scan = reports["septic_permit_scan.pdf"]
        if TextExtractor().ocr_available:
            check(scan.ocr_pages == 1, "scanned PDF went through OCR")
            mon = [c for c in chunks if c.jurisdiction == "Montgomery County"]
            check(any("sewage" in c.content.lower() and c.metadata["ocr"] for c in mon),
                  "OCR text indexed and flagged metadata.ocr")
        else:
            print("  [SKIP] Tesseract not installed; OCR checks skipped")

        hits = store.similarity_search(HashingEmbedder().embed_query("front yard setback R-1"),
                                       top_k=3, jurisdictions=["Fulshear"])
        check(hits and hits[0][0].section_title.startswith("Section 3.2"),
              f"vector search finds the setback section (top={hits[0][0].section_title!r})")

        # ---- re-run: nothing changed
        again = pipe.run(files)
        check(all(r.status == "UNCHANGED" for r in again), "second run: all UNCHANGED (skipped)")

        # ---- edit one section of the Fulshear PDF
        edited = [list(p) for p in FULSHEAR_PAGES]
        edited[0][5] = "In the R-1 district the front yard setback shall be 30 feet. The side yard setback"
        ful_pdf = next(p for p in files if p.endswith("fulshear_udc.pdf"))
        make_text_pdf(ful_pdf, edited)
        rep = pipe.process_file(ful_pdf, files[ful_pdf])
        print("  " + str(rep))
        check(rep.status == "CHANGED" and rep.version == 2, "edited PDF detected as CHANGED v2")
        check(rep.sections_modified == ["3.2"], f"only Section 3.2 reported modified ({rep.sections_modified})")
        check(rep.embeddings_computed == 1 and rep.embeddings_reused == rep.chunks_written - 1,
              "only the changed chunk was re-embedded")
        current = [c for c in store.load_chunks(jurisdictions=["Fulshear"])]
        check(all(c.metadata["doc_version"] == 2 for c in current), "only v2 chunks are current")
        check(any("30 feet" in c.content for c in current)
              and not any("be 25 feet" in c.content for c in current), "current text reflects the edit")
        return store.load_chunks()
    finally:
        shutil.rmtree(raw, ignore_errors=True)


def part_c_integration(chunks):
    try:
        from retrieval.query_agent import QueryParser
        from retrieval.search import HybridRetriever
    except ImportError:
        print("\n[SKIP] Part C (retrieval/) not on this branch; integration check skipped")
        return
    print("\nPart C integration (HybridRetriever over Part B chunks)")
    parser = QueryParser()
    retr = HybridRetriever(chunks, embedder=RetrievalEmbedderAdapter(HashingEmbedder()))
    for q, want in [("What are the residential setback rules in Fulshear?", "Section 3.2"),
                    ("maximum building height in Marlton", "§ 160-12"),
                    ("plat requirements for subdividing land in Fulshear", "Section 5.1")]:
        res = retr.retrieve(parser.parse(q), top_k=5)
        check(any(p.section_title.startswith(want) for p in res),
              f"{q!r} -> {[p.section_title for p in res[:3]]}")


if __name__ == "__main__":
    if fitz is None:
        raise SystemExit("PyMuPDF is required: pip install -r processing/requirements.txt")
    unit_tests()
    chunks = pipeline_tests(InMemoryStore(dim=384), "InMemoryStore")
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        from processing.store import PgVectorStore
        store = PgVectorStore(dsn, dim=384)
        with store.conn.cursor() as cur:  # clean slate for the test database
            cur.execute("TRUNCATE documents CASCADE")
        store.conn.commit()
        chunks = pipeline_tests(store, "PgVectorStore / pgvector")
        kw = store.keyword_search("preliminary plat", top_k=2)
        check(kw and kw[0][0].section_title.startswith("Section 5"), "Postgres full-text search works")
    else:
        print("\n[SKIP] DATABASE_URL not set; pgvector checks skipped")
    part_c_integration(chunks)
    print("\nAll Part B checks passed.")
