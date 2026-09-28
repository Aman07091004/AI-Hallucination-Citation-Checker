FROM python:3.12-slim

# Tesseract is a system binary, not a pip package - pytesseract just shells
# out to it for OCR on scanned PDFs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY frontend/ frontend/
COPY data/raw_corpus/ data/raw_corpus/

ENV PYTHONUNBUFFERED=1

# data/processed (the Chroma vector store + judgment caches) is deliberately
# NOT copied in and has no persistent volume here - it's rebuilt lazily on
# first use per host. Fine for a free-tier deploy; add a persistent disk at
# /app/data/processed later if judgment caching across restarts matters.

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
