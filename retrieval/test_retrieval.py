"""
Smoke test + mini eval for Part C. Seed chunks are SYNTHETIC placeholders
(numbers are made up) just to exercise the pipeline. Replace with the real
seed dataset / B's output.

Run from the repo root:  python test_retrieval.py
"""
from ingestion_pipeline_part_a import DocumentChunk
from retrieval.query_agent import QueryParser
from retrieval.search import HybridRetriever
from retrieval.rerank import low_confidence


def mk(cid, jur, state, section, text, jtype="Municipal"):
    return DocumentChunk(chunk_id=cid, doc_id=cid.split("_")[0], jurisdiction=jur, state=state,
                         section_title=section, content=text, chunk_index=0,
                         source_url="https://example.gov/" + cid,
                         metadata={"jurisdiction_type": jtype})


SEED = [
    mk("ful_1", "Fulshear", "TX", "Section 3.2 Residential Setbacks",
       "Section 3.2 Front yard setback shall be 25 feet. Side yard setback 7.5 feet. Rear yard 20 feet."),
    mk("ful_2", "Fulshear", "TX", "Section 5.1 Subdivision Platting",
       "Section 5.1 A preliminary plat must be submitted before any subdivision of land."),
    mk("har_1", "Harris County", "TX", "Section 4.02 Driveway Access",
       "Section 4.02 Driveway access to county roads requires a permit. Minimum spacing 150 feet.",
       "County"),
    mk("har_2", "Harris County", "TX", "Section 6.01 Drainage and Detention",
       "Section 6.01 Developments must provide stormwater detention and drainage plans.", "County"),
    mk("mon_1", "Montgomery County", "TX", "Section 2.4 Permitting",
       "Section 2.4 Subdivision permit applications are reviewed by environmental health.", "County"),
    mk("was_1", "Washington Township (Sewell)", "NJ", "Section 220-4 Minor Subdivision",
       "Section 220-4 Minor subdivision approval requires a plat and application fee."),
    mk("eve_1", "Evesham Township (Marlton)", "NJ", "Section 160-12 Residential Height",
       "Section 160-12 Maximum building height in residential districts is 35 feet or 2.5 stories."),
]

# (query, expected chunk_id that must appear in top 5)
EVAL = [
    ("What are the residential setback rules in Fulshear?", "ful_1"),
    ("driveway access requirements in Cypress, TX", "har_1"),
    ("stormwater detention for a development in zip 77433", "har_2"),
    ("maximum building height in Marlton", "eve_1"),
    ("minor subdivision approval in Sewell 08080", "was_1"),
    ("plat requirements for subdividing land in Fulshear", "ful_2"),
]

if __name__ == "__main__":
    parser, retriever = QueryParser(), HybridRetriever(SEED)
    hits = 0
    for q, expected in EVAL:
        pq = parser.parse(q)
        res = retriever.retrieve(pq, top_k=5)
        ids = [p.chunk_id for p in res]
        ok = expected in ids
        hits += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {q}")
        print(f"   parsed: jur={pq.jurisdictions} parents={pq.parent_jurisdictions} "
              f"types={pq.regulation_types} warnings={pq.warnings}")
        print(f"   top: {[(p.chunk_id, p.score, p.scope) for p in res[:3]]} "
              f"low_conf={low_confidence(res)}")
    print(f"\nTop-5 retrieval accuracy: {hits}/{len(EVAL)} = {hits/len(EVAL):.0%} (target >= 85%)")

    # Edge cases the analysis agent must handle
    for q in ["What are the setback rules?", "zoning rules in Austin"]:
        pq = parser.parse(q)
        res = retriever.retrieve(pq)
        print(f"\nEdge: {q!r} -> warnings={pq.warnings}, low_conf={low_confidence(res)}")
