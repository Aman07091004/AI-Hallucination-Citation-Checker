"""
Web service for the Citation Integrity Checker.

    uvicorn app.api.main:app --reload

Serves the JSON API under /api and the interactive frontend (frontend/) at /,
from one process - so a single container is the whole deployment.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config
from app.api import jobs
from app.rag.corpus_index import build_registry

log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}


def _warm_model() -> None:
    """Load the embedding model in the background at startup so the first
    real request doesn't pay the load time. Failure is fine - it just loads
    lazily on first use instead."""
    try:
        from app.rag.embeddings import get_model
        get_model()
    except Exception:
        log.warning("Embedding model warm-up failed; it will load on first use.", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.WARM_MODEL_ON_START:
        threading.Thread(target=_warm_model, daemon=True).start()
    yield


app = FastAPI(title="Citation Integrity Checker", version="1.0.0", lifespan=lifespan)

if config.ALLOWED_ORIGINS:  # only needed if the frontend is hosted on a different origin
    app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


# --- rate limiting: protects the Perplexity budget on a public deployment ---------

_RATE_LOG: dict[str, deque] = defaultdict(deque)
_RATE_LOCK = threading.Lock()


def _client_ip(request: Request) -> str:
    if config.TRUST_PROXY_HEADERS:  # only behind a proxy you control; otherwise the header is spoofable
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request) -> None:
    limit = config.RATE_LIMIT_JOBS_PER_HOUR
    now = time.time()
    ip = _client_ip(request)
    with _RATE_LOCK:
        window = _RATE_LOG[ip]
        while window and window[0] < now - 3600:
            window.popleft()
        if len(window) >= limit:
            raise HTTPException(429, f"Limit of {limit} checks per hour reached. Please try again later.")
        window.append(now)


# --- API -------------------------------------------------------------------------

class TextRequest(BaseModel):
    text: str


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "corpus_documents": len(build_registry()),
        "verification_configured": bool(config.PERPLEXITY_API_KEY),
        "limits": {
            "max_text_chars": config.MAX_TEXT_CHARS,
            "max_citations": config.MAX_CITATIONS,
            "max_upload_mb": config.MAX_UPLOAD_MB,
        },
    }


@app.post("/api/verify/text", status_code=202)
def verify_text(body: TextRequest, request: Request):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Paste some text to check.")
    if len(text) > config.MAX_TEXT_CHARS:
        raise HTTPException(413, f"Text is {len(text):,} characters; the limit is {config.MAX_TEXT_CHARS:,}.")
    _enforce_rate_limit(request)
    return {"job_id": jobs.create_job(jobs.text_reader(text))}


@app.post("/api/verify/file", status_code=202)
async def verify_file(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Unsupported file type. Upload a PDF, Word (.docx) or text (.txt) file.")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    if not data:
        raise HTTPException(400, "That file is empty.")

    _enforce_rate_limit(request)
    return {"job_id": jobs.create_job(jobs.file_reader(data, suffix))}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "That check has expired or doesn't exist. Start a new one.")
    return job


if FRONTEND_DIR.exists():  # mounted last so it never shadows /api
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
