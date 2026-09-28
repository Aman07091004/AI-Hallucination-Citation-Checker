"""
In-memory job store + background runner.

A check can take a minute (each authority may need an LLM call), so the API
doesn't hold a request open. Instead: POST starts a job and returns an id;
the client polls GET /api/jobs/{id} and sees authorities appear, then
verdicts fill in one by one.

Design limits, stated plainly:
  - Jobs live in this process's memory. That's right for a single instance
    (this project's deployment); running several instances would need a
    shared store such as Redis.
  - Jobs are executed one at a time (_PIPELINE_LOCK): the embedding model
    and vector store are shared, and serialising also caps how fast anyone
    can spend the Perplexity budget.
"""

import copy
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from app import config
from app.ingestion.loader import load_document
from app.verify_brief import verify_text_iter

JOB_TTL_SECONDS = 3600

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_PIPELINE_LOCK = threading.Lock()


class InputError(Exception):
    """A problem with the user's document, safe to show to them verbatim."""


def _purge_expired() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    for job_id in [j for j, v in _JOBS.items() if v["created"] < cutoff]:
        del _JOBS[job_id]


def _update(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id].update(fields)


def _run(job_id: str, read_text: Callable[[], tuple[str, list[str]]]) -> None:
    with _PIPELINE_LOCK:
        try:
            _update(job_id, status="running", stage="Reading the document")
            text, warnings = read_text()
            _update(job_id, text=text, warnings=warnings, stage="Finding citations")

            for event in verify_text_iter(text, max_citations=config.MAX_CITATIONS):
                if event["event"] == "extracted":
                    warn = list(warnings)
                    if event["truncated"]:
                        warn.append(
                            f"Found {event['total_found']} authorities; only the first "
                            f"{config.MAX_CITATIONS} were checked."
                        )
                    _update(
                        job_id, citations=event["citations"], total=len(event["citations"]),
                        warnings=warn, stage="Checking authorities",
                    )
                else:
                    result = event["citation"]
                    with _JOBS_LOCK:
                        job = _JOBS[job_id]
                        job["citations"][result["id"]] = result
                        job["done"] += 1

            _update(job_id, status="done", stage="Finished")

        except InputError as exc:
            _update(job_id, status="error", stage="Stopped", error=str(exc))
        except Exception as exc:  # last-resort: never leave a job hanging in "running"
            _update(job_id, status="error", stage="Stopped",
                    error=f"Something went wrong while checking this document ({type(exc).__name__}).")


def create_job(read_text: Callable[[], tuple[str, list[str]]]) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _purge_expired()
        _JOBS[job_id] = {
            "id": job_id, "created": time.time(), "status": "queued", "stage": "Waiting for a free slot",
            "text": None, "citations": [], "total": 0, "done": 0, "warnings": [], "error": None,
        }
    threading.Thread(target=_run, args=(job_id, read_text), daemon=True).start()
    return job_id


def _summarise(citations: list[dict]) -> dict:
    counts = {"verified": 0, "misapplied": 0, "fabricated": 0, "unresolved": 0, "pending": 0}
    for c in citations:
        counts[c["category"] if c["status"] == "done" else "pending"] += 1
    return counts


def get_job(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        snapshot = copy.deepcopy(job)
    snapshot["summary"] = _summarise(snapshot["citations"])
    del snapshot["created"]
    return snapshot


# --- readers: how each kind of input becomes text -------------------------------

def text_reader(text: str) -> Callable[[], tuple[str, list[str]]]:
    def read():
        return _validated(text), []
    return read


def file_reader(data: bytes, suffix: str) -> Callable[[], tuple[str, list[str]]]:
    def read():
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = Path(tmp.name)
        try:
            doc = load_document(path, max_ocr_pages=config.MAX_OCR_PAGES)
        except Exception as exc:
            raise InputError("That file could not be read. Check it isn't corrupted or password-protected.") from exc
        finally:
            path.unlink(missing_ok=True)

        warnings = []
        if doc.ocr_truncated:
            warnings.append(f"This is a scanned document; only the first {config.MAX_OCR_PAGES} pages were read.")
        elif doc.used_ocr:
            warnings.append("This is a scanned document, so text was read with OCR and may contain errors.")
        return _validated(doc.text), warnings
    return read


def _validated(text: str) -> str:
    text = text.strip()
    if not text:
        raise InputError("No readable text was found in this document.")
    if len(text) > config.MAX_TEXT_CHARS:
        raise InputError(
            f"This document is {len(text):,} characters; the limit is {config.MAX_TEXT_CHARS:,}. "
            "Paste or upload the relevant section instead."
        )
    return text
