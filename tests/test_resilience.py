import time

import pytest

from app.db import summarize
from app.resilience import TTLCache, call_with_timeout, retry


def test_retry_recovers_then_gives_up():
    calls = {"n": 0}

    @retry(attempts=3, base_delay=0.001, retry_on=(ConnectionError,))
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert flaky() == "ok" and calls["n"] == 3

    @retry(attempts=2, base_delay=0.001, retry_on=(ConnectionError,))
    def always():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        always()


def test_retry_does_not_catch_other_errors():
    @retry(attempts=3, base_delay=0.001, retry_on=(ConnectionError,))
    def bad():
        raise ValueError("bug")

    with pytest.raises(ValueError):
        bad()


def test_ttl_cache_expires_and_evicts():
    c = TTLCache(maxsize=2, ttl_s=0.05)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)                 # evicts "a" (LRU)
    assert c.get("a") is None and c.get("c") == 3
    time.sleep(0.06)
    assert c.get("c") is None


def test_call_with_timeout():
    assert call_with_timeout(lambda: 5, 1) == 5
    with pytest.raises(TimeoutError):
        call_with_timeout(time.sleep, 0.05, 0.5)


def test_summarize_metrics():
    rows = [{"status": "answered", "latency_ms": 10, "cached": False},
            {"status": "abstained", "latency_ms": 20, "cached": True},
            {"status": "error", "latency_ms": 30, "cached": False}]
    m = summarize(rows)
    assert m["success_rate"] == round(2 / 3, 4) and m["error_rate"] == round(1 / 3, 4)
    assert m["latency_ms"]["p50"] == 20
