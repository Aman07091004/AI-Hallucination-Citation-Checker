import hashlib
import json
from pathlib import Path

import httpx

from app import config

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar"  # cheapest tier - judging provided text doesn't need Perplexity's web-search capability

CACHE_PATH = config.PROCESSED_DIR / "llm_judgments_cache.json"

SYSTEM_PROMPT = """You are a careful legal fact-checker. You will be given:
1. A claim a legal brief makes about what a case holds, and the case it cites.
2. One or more passages retrieved from that case's actual text.

Judge ONLY using the passages provided. Do not use any outside knowledge of
this case, even if you recognise it. If the passages don't contain enough
information to judge, say so honestly rather than guessing from memory.

Respond with ONLY a JSON object, no other text, in this exact shape:
{"verdict": "SUPPORTED", "confidence": "high", "explanation": "one or two sentences"}

Valid verdict values:
- SUPPORTED: the passages genuinely support the claim as stated.
- MISQUOTED: the case is real, but the passages contradict or don't support the specific claim made.
- UNCLEAR: the passages don't contain enough relevant information to judge either way.

Valid confidence values: "high", "medium", "low".
"""


def _cache_key(case_name: str, citation: str, claim: str, passages: list[str]) -> str:
    raw = case_name + citation + claim + "||".join(passages)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _parse_json_response(raw: str) -> dict:
    """LLMs sometimes wrap JSON in markdown code fences or add stray text
    despite instructions not to - this extracts the JSON object defensively
    rather than assuming a perfectly clean response, and falls back to an
    honest UNCLEAR rather than crashing if parsing still fails."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {"verdict": "UNCLEAR", "confidence": "low", "explanation": f"Could not parse model response: {raw[:200]}"}

    try:
        parsed = json.loads(raw[start : end + 1])
        return {
            "verdict": parsed.get("verdict", "UNCLEAR"),
            "confidence": parsed.get("confidence", "low"),
            "explanation": parsed.get("explanation", ""),
        }
    except json.JSONDecodeError:
        return {"verdict": "UNCLEAR", "confidence": "low", "explanation": f"Could not parse model response: {raw[:200]}"}


def judge_proposition(case_name: str, citation: str, claim: str, passages: list[str]) -> dict:
    """Returns {"verdict", "confidence", "explanation", "from_cache"}.
    verdict is one of SUPPORTED / MISQUOTED / UNCLEAR."""
    cache = _load_cache()
    key = _cache_key(case_name, citation, claim, passages)

    if key in cache:
        result = dict(cache[key])
        result["from_cache"] = True
        return result

    if not config.PERPLEXITY_API_KEY:
        raise RuntimeError("PERPLEXITY_API_KEY is not set in .env - required to call the judgment API.")

    passages_text = "\n\n".join(f"Passage {i + 1}: {p}" for i, p in enumerate(passages))
    user_prompt = (
        f"Case: {case_name} ({citation})\n"
        f'Claim made by the brief: "{claim}"\n\n'
        f"Retrieved passages from the actual case text:\n{passages_text}"
    )

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
