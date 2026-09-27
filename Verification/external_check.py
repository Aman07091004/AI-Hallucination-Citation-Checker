import hashlib
import json

import httpx

from app import config

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar"  # web-search-grounded - this is the one place in the pipeline we WANT that

CACHE_PATH = config.PROCESSED_DIR / "external_check_cache.json"

SYSTEM_PROMPT = """You are verifying whether a legal case citation refers to
a real, genuine court case. Search for the case name and citation given below.

IMPORTANT: only answer exists=true if you find direct evidence of an actual
court judgment, case report, or law database entry matching BOTH this
specific case name AND this specific citation together. Finding an
unrelated real company, person, or organization that merely shares a
similar name is NOT evidence the case exists - many fabricated case names
are built from real company names precisely to look plausible. If your
search only turns up unrelated entities with a similar name, and no actual
judgment or case citation, you must answer exists=false, not true.

Respond with ONLY a JSON object, no other text, in this exact shape:
{"exists": true, "confidence": "high", "evidence": "one or two sentences describing what you found", "source": "a URL or publication name if available, else an empty string"}

- exists: true only if you find a specific judgment/case report matching this exact case name and citation.
- exists: false if you can find no such judgment - including cases where you only find unrelated similarly-named entities.
- confidence: "high", "medium", or "low", based on how solid the evidence is.
"""


def _cache_key(case_name: str, citation: str) -> str:
    return hashlib.sha256(f"{case_name}||{citation}".encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {"exists": None, "confidence": "low", "evidence": f"Could not parse model response: {raw[:200]}", "source": ""}

    try:
        parsed = json.loads(raw[start : end + 1])
        return {
            "exists": parsed.get("exists"),
            "confidence": parsed.get("confidence", "low"),
            "evidence": parsed.get("evidence", ""),
            "source": parsed.get("source", ""),
        }
    except json.JSONDecodeError:
        return {"exists": None, "confidence": "low", "evidence": f"Could not parse model response: {raw[:200]}", "source": ""}


def check_citation_exists(case_name: str, citation: str) -> dict:
    """Returns {"exists": True/False/None, "confidence", "evidence", "source", "from_cache"}.
    exists=None means the model's response couldn't be parsed - treat as
    UNCERTAIN, not as a negative result."""
    cache = _load_cache()
    key = _cache_key(case_name, citation)

    if key in cache:
        result = dict(cache[key])
        result["from_cache"] = True
        return result

    if not config.PERPLEXITY_API_KEY:
        raise RuntimeError("PERPLEXITY_API_KEY is not set in .env - required for external verification.")

    user_prompt = f"Case name: {case_name}\nCitation: {citation}\n\nIs this a real, genuine legal case?"

    response = httpx.post(
        PERPLEXITY_API_URL,
        headers={"Authorization": f"Bearer {config.PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    raw_content = response.json()["choices"][0]["message"]["content"]

    result = _parse_json_response(raw_content)
    result["from_cache"] = False

    cache[key] = {k: v for k, v in result.items() if k != "from_cache"}
    _save_cache(cache)

    return result


def classify_external_result(result: dict) -> str:
    """Turns the raw exists/confidence into our final verdict label. A
    citation not in our corpus is never directly called "fabricated" from
    local data alone - this function is the only place that label gets
    applied, and only after an actual external check.

    Safety net: exists=True is only trusted at high confidence. We saw a
    real failure case in testing - a fabricated citation ("Stonegate
    Capital Partners v Redwood Procurement Group") was wrongly marked
    exists=True at MEDIUM confidence, because the model found an unrelated
    real company sharing the same name and treated that as partial
    evidence the CASE existed. Those are different claims. Rather than
    relying solely on prompt wording to prevent this, we also refuse to
    clear a citation as real unless confidence is high - medium/low
    "exists" evidence is downgraded to UNCERTAIN, which is the honest
    state here: we can't confidently say either way, so we shouldn't
    present a confident-looking label."""
    if result["exists"] is True:
        return "REAL_BUT_NOT_IN_LOCAL_CORPUS" if result["confidence"] == "high" else "UNCERTAIN"
    if result["exists"] is False:
        return "LIKELY_FABRICATED"
    return "UNCERTAIN"
