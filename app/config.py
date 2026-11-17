"""Settings, read once from environment variables (see .env.example)."""

import os
from dataclasses import dataclass, field
from typing import Optional


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # Postgres + pgvector. Empty -> in-memory mode (no database needed)
    database_url: Optional[str] = field(default_factory=lambda: os.getenv("DATABASE_URL") or None)
    # Embeddings (Part B). "hashing" needs no model download; use sentence-transformers for demos
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "hashing"))
    embedding_model: Optional[str] = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL") or None)
    # Answer generation: "auto" uses Part D if it is importable, else the extractive stub
    answerer: str = field(default_factory=lambda: os.getenv("ANSWERER", "auto"))
    # Folder of raw documents to index at startup when the index is empty
    # (data/raw in production, eval/seed_corpus for CI and local demos)
    seed_dir: Optional[str] = field(default_factory=lambda: os.getenv("SEED_DIR") or None)

    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    cache_ttl_s: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_S", "600")))
    cache_size: int = field(default_factory=lambda: int(os.getenv("CACHE_SIZE", "512")))
    answer_timeout_s: float = field(default_factory=lambda: float(os.getenv("ANSWER_TIMEOUT_S", "30")))
    max_question_chars: int = 1000

    admin_token: Optional[str] = field(default_factory=lambda: os.getenv("ADMIN_TOKEN") or None)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_json: bool = field(default_factory=lambda: _bool("LOG_JSON", True))


def get_settings() -> Settings:
    return Settings()
