import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture()
def make_client(monkeypatch):
    """Builds a TestClient over the seed corpus. Uses Postgres if DATABASE_URL is set
    (CI), else in-memory. Extra env vars can be passed per test."""
    from fastapi.testclient import TestClient

    clients = []

    def _make(**env):
        defaults = {"SEED_DIR": os.path.join(ROOT, "eval", "seed_corpus"),
                    "EMBEDDING_BACKEND": "hashing", "LOG_JSON": "false",
                    "LOG_LEVEL": "WARNING", "ADMIN_TOKEN": "test-token"}
        for k, v in {**defaults, **env}.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)
        from app.main import create_app
        c = TestClient(create_app())
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture()
def client(make_client):
    return make_client()
