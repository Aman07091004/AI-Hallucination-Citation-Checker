"""
API behaviour: validation, the job lifecycle, uploads, limits. The real
pipeline (embeddings + paid API calls) is replaced by a fast fake so these
run offline in a couple of seconds.
"""

import time

import pytest
from fastapi.testclient import TestClient

from app import config
from app.api import jobs, main

client = TestClient(main.app)


def _fake_pipeline(text, max_citations=None):
    stubs = [
        {"id": 0, "kind": "case", "name": "Hadley v Baxendale", "citation": "(1854) 9 Ex 341", "spans": [[0, 10]], "status": "pending"},
        {"id": 1, "kind": "case", "name": "Fake v Case", "citation": "[2020] EWHC 1 (Comm)", "spans": [[20, 30]], "status": "pending"},
    ]
    yield {"event": "extracted", "citations": stubs, "truncated": False, "total_found": 2}
    yield {"event": "result", "citation": {**stubs[0], "status": "done", "verdict": "SUPPORTED", "category": "verified",
           "depth": "full", "confidence_pct": 91, "explanation": "ok", "evidence": {}, "scope_note": ""}}
    yield {"event": "result", "citation": {**stubs[1], "status": "done", "verdict": "LIKELY_FABRICATED", "category": "fabricated",
           "depth": "existence_only", "confidence_pct": 88, "explanation": "none found", "evidence": {}, "scope_note": ""}}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(jobs, "verify_text_iter", _fake_pipeline)
    main._RATE_LOG.clear()
    jobs._JOBS.clear()


def _wait(job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_health():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "limits" in body


def test_blank_text_is_rejected():
    assert client.post("/api/verify/text", json={"text": "   "}).status_code == 400


def test_oversized_text_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_TEXT_CHARS", 50)
    assert client.post("/api/verify/text", json={"text": "x" * 51}).status_code == 413


def test_text_job_runs_to_completion_with_summary():
    job_id = client.post("/api/verify/text", json={"text": "Hadley v Baxendale (1854) 9 Ex 341"}).json()["job_id"]
    job = _wait(job_id)
    assert job["status"] == "done"
    assert job["done"] == job["total"] == 2
    assert job["summary"] == {"verified": 1, "misapplied": 0, "fabricated": 1, "unresolved": 0, "pending": 0}
    assert job["text"].startswith("Hadley")


def test_txt_upload_works_end_to_end():
    files = {"file": ("brief.txt", b"Hadley v Baxendale (1854) 9 Ex 341", "text/plain")}
    job_id = client.post("/api/verify/file", files=files).json()["job_id"]
    assert _wait(job_id)["status"] == "done"


def test_unsupported_file_type_is_rejected():
    files = {"file": ("evil.exe", b"MZ", "application/octet-stream")}
    assert client.post("/api/verify/file", files=files).status_code == 415


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    files = {"file": ("a.txt", b"some text", "text/plain")}
    assert client.post("/api/verify/file", files=files).status_code == 413


def test_empty_upload_is_rejected():
    files = {"file": ("a.txt", b"", "text/plain")}
    assert client.post("/api/verify/file", files=files).status_code == 400


def test_unreadable_document_becomes_a_clear_job_error_not_a_crash():
    files = {"file": ("bad.pdf", b"this is not a pdf", "application/pdf")}
    job = _wait(client.post("/api/verify/file", files=files).json()["job_id"])
    assert job["status"] == "error"
    assert "could not be read" in job["error"]


def test_unknown_job_is_404():
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_rate_limit_kicks_in(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_JOBS_PER_HOUR", 2)
    for _ in range(2):
        assert client.post("/api/verify/text", json={"text": "a"}).status_code == 202
    assert client.post("/api/verify/text", json={"text": "a"}).status_code == 429


def test_pipeline_crash_surfaces_as_job_error(monkeypatch):
    def boom(text, max_citations=None):
        raise RuntimeError("kaboom")
        yield
    monkeypatch.setattr(jobs, "verify_text_iter", boom)
    job = _wait(client.post("/api/verify/text", json={"text": "a"}).json()["job_id"])
    assert job["status"] == "error"
    assert "kaboom" not in job["error"]      # internal detail must not leak to users
