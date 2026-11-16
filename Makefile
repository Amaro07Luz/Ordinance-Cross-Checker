.PHONY: install install-ml db up down run test eval lint reindex

install:          ## runtime + dev dependencies
	pip install -r requirements-dev.txt
install-ml:       ## real embedding model (sentence-transformers)
	pip install -r requirements-ml.txt

db:               ## start only Postgres + pgvector
	docker compose up -d db
up:               ## full stack (db + api) on http://localhost:8000
	docker compose up --build
down:
	docker compose down

run:              ## API with auto-reload (in-memory unless DATABASE_URL is set)
	SEED_DIR=$${SEED_DIR:-eval/seed_corpus} LOG_JSON=false uvicorn app.main:app --reload

test:             ## all workstream smoke tests + Part E tests
	python -m processing.test_processing
	python -m retrieval.test_retrieval
	python -m pytest -q

eval:             ## eval harness with the CI gate
	SEED_DIR=$${SEED_DIR:-eval/seed_corpus} python -m eval.harness --min-success 0.85 --max-p95-ms 1500

lint:
	ruff check app eval tests

reindex:          ## run Part B over data/raw into DATABASE_URL
	python -m processing.pipeline --raw-dir data/raw --embedder $${EMBEDDING_BACKEND:-hashing}
