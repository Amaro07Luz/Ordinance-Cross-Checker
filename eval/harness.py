"""
Eval harness: runs the test query set through the API and reports
response time and query success rate (plus retrieval hit@k, jurisdiction
accuracy and abstain accuracy).

    # in-process (starts the app itself; uses DATABASE_URL / SEED_DIR etc. from env)
    SEED_DIR=eval/seed_corpus python -m eval.harness
    # against a running server
    python -m eval.harness --url http://localhost:8000
    # CI gate: exit 1 if below targets
    python -m eval.harness --min-success 0.85 --max-p95-ms 1500

A query SUCCEEDS when the request returns 200 and:
  - expect_abstain queries: the system abstained
  - all others: it answered (did not abstain), the expected section is in the
    top-k passages, and any expect_answer_contains strings appear in the answer
    (or in the cited passages, for answerers that paraphrase)
Latency is measured with the response cache off (cold), then once more with it
on to report the cached-path latency.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))


def pct(values: List[float], p: int):
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(statistics.quantiles(values, n=100, method="inclusive")[p - 1], 1)


def load_queries(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class InProcessClient:
    def __init__(self):
        from fastapi.testclient import TestClient

        from app.main import create_app
        self.client = TestClient(create_app())
        self.client.__enter__()          # runs startup (index load/seed)
        self.app = self.client.app

    def post(self, path, payload):
        r = self.client.post(path, json=payload, headers={"X-Query-Source": "eval"})
        return r.status_code, r.json()

    def get(self, path):
        r = self.client.get(path)
        return r.status_code, r.json()


class HttpClient:
    def __init__(self, base: str):
        import httpx
        self.c = httpx.Client(base_url=base.rstrip("/"), timeout=60,
                              headers={"X-Query-Source": "eval"})
        self.app = None

    def post(self, path, payload):
        r = self.c.post(path, json=payload)
        return r.status_code, r.json()

    def get(self, path):
        r = self.c.get(path)
        return r.status_code, r.json()


def evaluate_one(q: Dict, status: int, body: Dict, top_k: int) -> Dict:
    res = {"id": q["id"], "question": q["question"], "http_status": status, "checks": {}}
    if status != 200:
        res.update(success=False, error=body.get("detail") or body.get("error"))
        return res
    ans = body["answer"]
    passages = body["passages"][:top_k]
    titles = [p["section_title"] for p in passages]
    c = res["checks"]
    exp_j = q.get("expect_jurisdictions")
    if exp_j is not None:
        c["jurisdiction"] = sorted(body["parsed"]["jurisdictions"]) == sorted(exp_j)
    if q.get("expect_abstain"):
        c["abstained"] = ans["abstained"]
    else:
        c["answered"] = not ans["abstained"]
        if q.get("expect_section"):
            c["hit_at_k"] = any(t.startswith(q["expect_section"]) for t in titles)
            res["hit_rank"] = next((i for i, t in enumerate(titles, 1)
                                    if t.startswith(q["expect_section"])), None)
        if q.get("expect_answer_contains"):
            cited = {cc["chunk_id"] for cc in ans["citations"]}
            haystack = ans["text"] + " ".join(p["content"] for p in passages if p["chunk_id"] in cited)
            c["answer_contains"] = all(s.lower() in haystack.lower() for s in q["expect_answer_contains"])
    gating = {k: v for k, v in c.items() if k != "jurisdiction"}  # jurisdiction reported, not gating
    res.update(success=all(gating.values()), latency_ms=body["timing_ms"]["total"],
               top_sections=titles[:3], abstained=ans["abstained"], answerer=ans["answerer"])
    return res


def run(client, queries: List[Dict], top_k: int) -> Dict:
    results, cold, client_ms = [], [], []
    for q in queries:
        t0 = time.perf_counter()
        status, body = client.post("/api/query", {"question": q["question"], "top_k": top_k,
                                                  "use_cache": False})
        client_ms.append((time.perf_counter() - t0) * 1000)
        r = evaluate_one(q, status, body, top_k)
        results.append(r)
        if "latency_ms" in r:
            cold.append(r["latency_ms"])

    warm = []  # second pass with caching on: first call fills, second hits
    for q in queries:
        client.post("/api/query", {"question": q["question"], "top_k": top_k})
        s, b = client.post("/api/query", {"question": q["question"], "top_k": top_k})
        if s == 200 and b.get("cached"):
            warm.append(b["timing_ms"]["total"])

    def rate(key):
        vals = [r["checks"][key] for r in results if key in r.get("checks", {})]
        return round(sum(vals) / len(vals), 4) if vals else None

    n = len(results)
    return {
        "n_queries": n,
        "success_rate": round(sum(r["success"] for r in results) / n, 4) if n else 0.0,
        "error_rate": round(sum(r["http_status"] != 200 for r in results) / n, 4) if n else 0.0,
        "hit_at_k": rate("hit_at_k"),
        "jurisdiction_accuracy": rate("jurisdiction"),
        "abstain_accuracy": rate("abstained"),
        "answer_contains_rate": rate("answer_contains"),
        "mrr": round(statistics.mean([1 / r["hit_rank"] if r.get("hit_rank") else 0
                                      for r in results if "hit_at_k" in r["checks"]] or [0]), 4),
        "p50_ms": pct(cold, 50), "p95_ms": pct(cold, 95), "max_ms": round(max(cold), 1) if cold else None,
        "client_p95_ms": pct(client_ms, 95),
        "cached_p50_ms": pct(warm, 50),
        "results": results,
    }


def to_markdown(rep: Dict) -> str:
    lines = [f"# Eval report `{rep['run_id']}`", "",
             f"- Date: {rep['created_at']}  ·  commit `{rep.get('git_sha') or 'n/a'}`",
             f"- Answerer: {rep.get('answerer')}  ·  top_k={rep['top_k']}  ·  queries: {rep['n_queries']}",
             "", "| Metric | Value |", "|---|---|"]
    for k, label in [("success_rate", "Query success rate"), ("p50_ms", "Response time p50 (ms)"),
                     ("p95_ms", "Response time p95 (ms)"), ("cached_p50_ms", "Cached p50 (ms)"),
                     ("hit_at_k", f"Retrieval hit@{rep['top_k']}"), ("mrr", "MRR"),
                     ("jurisdiction_accuracy", "Jurisdiction accuracy"),
                     ("abstain_accuracy", "Abstain accuracy (edge cases)"),
                     ("error_rate", "HTTP error rate")]:
        lines.append(f"| {label} | {rep.get(k)} |")
    lines += ["", "| Query | OK | ms | Top section |", "|---|---|---|---|"]
    for r in rep["results"]:
        top = (r.get("top_sections") or ["-"])[0]
        lines.append(f"| {r['id']} | {'✅' if r['success'] else '❌'} | {r.get('latency_ms', '-')} | {top} |")
    if rep.get("gate"):
        lines += ["", f"**Gate:** {rep['gate']}"]
    return "\n".join(lines) + "\n"


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return os.getenv("GITHUB_SHA", "")[:7] or None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=os.path.join(HERE, "queries.jsonl"))
    ap.add_argument("--url", help="evaluate a running server instead of in-process")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "reports"))
    ap.add_argument("--min-success", type=float, default=None)
    ap.add_argument("--max-p95-ms", type=float, default=None)
    args = ap.parse_args(argv)

    queries = load_queries(args.queries)
    client = HttpClient(args.url) if args.url else InProcessClient()
    rep = run(client, queries, args.top_k)
    rep.update(run_id=f"eval-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}",
               created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               git_sha=git_sha(), query_set=os.path.basename(args.queries), top_k=args.top_k,
               answerer=next((r.get("answerer") for r in rep["results"] if r.get("answerer")), None))

    failures = []
    if args.min_success is not None and rep["success_rate"] < args.min_success:
        failures.append(f"success_rate {rep['success_rate']} < {args.min_success}")
    if args.max_p95_ms is not None and (rep["p95_ms"] or 0) > args.max_p95_ms:
        failures.append(f"p95 {rep['p95_ms']} ms > {args.max_p95_ms} ms")
    rep["gate"] = "PASS" if not failures else "FAIL: " + "; ".join(failures)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "latest.json"), "w") as f:
        json.dump(rep, f, indent=2)
    md = to_markdown(rep)
    with open(os.path.join(args.out, "latest.md"), "w") as f:
        f.write(md)
    if client.app is not None:   # in-process: also record the run in the platform DB
        try:
            client.app.state.db.save_eval_run({k: v for k, v in rep.items()})
        except Exception as e:
            print(f"(could not save eval run to DB: {e})")

    for r in rep["results"]:
        if not r["success"]:
            print(f"FAIL {r['id']}: checks={r.get('checks')} top={r.get('top_sections')} "
                  f"error={r.get('error')}")
    print(md)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
