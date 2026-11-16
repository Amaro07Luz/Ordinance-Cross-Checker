import sys
import types

from app.services.answerer import PART_D_CLASS, PART_D_MODULE


def ask(client, question, **kw):
    r = client.post("/api/query", json={"question": question, **kw})
    return r.status_code, r.json()


def test_health_reports_components(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["index"]["chunks"] > 0
    assert body["components"]["D_analysis_agent"].startswith("stand-in")
    assert "X-Request-ID" in r.headers and "X-Response-Time-ms" in r.headers


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Regulatory Intelligence Agent" in r.text


def test_jurisdictions_come_from_part_a_registry(client):
    names = {j["name"] for j in client.get("/api/jurisdictions").json()}
    assert {"Fulshear", "Harris County", "Evesham Township (Marlton)"} <= names


def test_query_answers_with_citation(client):
    status, body = ask(client, "What is the front yard setback in Fulshear?")
    assert status == 200
    assert body["status"] == "answered"
    assert "25 feet" in body["answer"]["text"]
    cite = body["answer"]["citations"][0]
    assert cite["jurisdiction"] == "Fulshear" and cite["section_title"].startswith("Section 3.2")
    assert body["timing_ms"]["total"] >= 0 and body["passages"]


def test_out_of_scope_abstains(client):
    status, body = ask(client, "zoning rules in Austin")
    assert status == 200 and body["status"] == "abstained"


def test_explicit_jurisdiction_filter(client):
    _, body = ask(client, "maximum building height", jurisdiction="Evesham Township (Marlton)")
    assert body["parsed"]["jurisdictions"] == ["Evesham Township (Marlton)"]
    assert body["passages"][0]["jurisdiction"] == "Evesham Township (Marlton)"


def test_unknown_jurisdiction_is_400(client):
    status, body = ask(client, "height limits", jurisdiction="Gotham")
    assert status == 400 and body["error"] == "http_error" and body["request_id"]


def test_validation_error_shape(client):
    status, body = ask(client, "x")
    assert status == 422 and body["error"] == "invalid_request"
    assert body["detail"][0]["field"] == "question"


def test_cache_hit_on_repeat(client):
    q = "driveway access requirements in Cypress, TX"
    _, first = ask(client, q)
    _, second = ask(client, "  Driveway access requirements in Cypress, TX?  ")
    assert first["cached"] is False and second["cached"] is True
    assert second["answer"] == first["answer"]
    _, third = ask(client, q, use_cache=False)
    assert third["cached"] is False


def test_metrics_track_latency_and_success(client):
    for q in ["Septic permit rules in Conroe", "zoning rules in Austin"]:
        ask(client, q, use_cache=False)
    m = client.get("/api/metrics").json()
    assert m["queries"] >= 2 and m["success_rate"] == 1.0
    assert m["latency_ms"]["p95"] is not None


def test_feedback(client):
    _, body = ask(client, "Septic permit rules in Conroe")
    r = client.post("/api/feedback", json={"request_id": body["request_id"], "helpful": True})
    assert r.status_code == 200


def test_reindex_requires_token(client):
    assert client.post("/api/admin/reindex", json={"raw_dir": "eval/seed_corpus"}).status_code == 403
    r = client.post("/api/admin/reindex", json={"raw_dir": "eval/seed_corpus"},
                    headers={"X-Admin-Token": "test-token"})
    assert r.status_code == 200
    assert all("UNCHANGED" in d for d in r.json()["documents"])  # Part B skips unchanged docs


# ---- Part D seam -------------------------------------------------------
def _install_fake_part_d(monkeypatch, behavior):
    pkg = types.ModuleType(PART_D_MODULE.split(".")[0])
    mod = types.ModuleType(PART_D_MODULE)

    class Agent:
        def answer(self, question, parsed, passages):
            return behavior(question, parsed, passages)

    setattr(mod, PART_D_CLASS, Agent)
    monkeypatch.setitem(sys.modules, pkg.__name__, pkg)
    monkeypatch.setitem(sys.modules, PART_D_MODULE, mod)


def test_part_d_is_picked_up_automatically(monkeypatch, make_client):
    _install_fake_part_d(monkeypatch, lambda q, pq, ps: {
        "answer": f"LLM answer for {pq.jurisdictions}", "abstained": False, "confidence": 0.9,
        "citations": [{"chunk_id": ps[0].chunk_id, "quote": "x"}]})
    client = make_client()
    assert client.get("/api/health").json()["components"]["D_analysis_agent"] == "ok"
    _, body = ask(client, "What is the front yard setback in Fulshear?")
    assert body["answer"]["text"] == "LLM answer for ['Fulshear']"
    assert body["answer"]["answerer"].startswith("part-d")
    assert body["answer"]["citations"][0]["section_title"].startswith("Section 3.2")


def test_part_d_failure_falls_back(monkeypatch, make_client):
    def boom(*_):
        raise ConnectionError("LLM API down")
    _install_fake_part_d(monkeypatch, boom)
    client = make_client()
    status, body = ask(client, "What is the front yard setback in Fulshear?")
    assert status == 200 and body["answer"]["answerer"] == "extractive-stub"
    assert any("analysis agent was unavailable" in w for w in body["warnings"])
