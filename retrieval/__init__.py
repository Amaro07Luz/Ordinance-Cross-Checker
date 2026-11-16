"""
Workstream C: query understanding and retrieval.

query_agent.py  parse a question into a ParsedQuery
search.py       HybridRetriever: retrieve(parsed_query, top_k=5) -> list[RetrievedPassage]
rerank.py       heuristic reranker and low_confidence() helper

Run the smoke test from the repo root: python -m retrieval.test_retrieval
"""