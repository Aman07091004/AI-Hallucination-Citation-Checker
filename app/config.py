import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAW_CORPUS_DIR = Path(os.getenv("RAW_CORPUS_DIR", "./data/raw_corpus"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "./data/processed/chroma"))

TESSERACT_CMD = os.getenv("TESSERACT_CMD")  # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
COURTLISTENER_API_TOKEN = os.getenv("COURTLISTENER_API_TOKEN")

# --- API service limits - protect the Perplexity budget and server resources
# on a public deployment, since a single check can trigger several paid
# LLM calls and a scanned PDF can trigger many slow OCR pages. ---
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "200000"))
MAX_CITATIONS = int(os.getenv("MAX_CITATIONS", "40"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "25"))
RATE_LIMIT_JOBS_PER_HOUR = int(os.getenv("RATE_LIMIT_JOBS_PER_HOUR", "20"))

WARM_MODEL_ON_START = os.getenv("WARM_MODEL_ON_START", "true").lower() == "true"
# Only trust X-Forwarded-For for rate limiting if we're actually behind a
# proxy we control - otherwise it's a client-supplied header, trivially
# spoofable to dodge the rate limit entirely.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
