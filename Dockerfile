# API + processing image.
#   docker build -t regintel .                       # small image, hashing/OpenAI embeddings
#   docker build -t regintel --build-arg WITH_ML=true .   # + sentence-transformers (CPU torch)
FROM python:3.12-slim

ARG WITH_ML=false
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# tesseract-ocr: Part B's OCR fallback for scanned PDFs
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt \
 && if [ "$WITH_ML" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install -r requirements-ml.txt; \
    fi

COPY . .
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=5 \
  CMD curl -fs http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
