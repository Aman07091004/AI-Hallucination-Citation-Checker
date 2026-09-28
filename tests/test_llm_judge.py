"""
Run with: pytest tests/test_llm_judge.py -v

These test the response-parsing logic only, not the actual API call (which
needs a real PERPLEXITY_API_KEY and costs money) - this is exactly the
kind of thing that should be covered by a fast, free, offline test rather
than only checked by eye when the real thing runs.
"""

from app.verification.llm_judge import _parse_json_response


def test_parses_clean_json():
    raw = '{"verdict": "SUPPORTED", "confidence": "high", "explanation": "Matches."}'
    result = _parse_json_response(raw)
    assert result == {"verdict": "SUPPORTED", "confidence": "high", "explanation": "Matches."}


def test_parses_json_wrapped_in_markdown_fences():
    raw = '```json\n{"verdict": "MISQUOTED", "confidence": "medium", "explanation": "Contradicts."}\n```'
    result = _parse_json_response(raw)
    assert result["verdict"] == "MISQUOTED"
    assert result["confidence"] == "medium"


def test_parses_json_with_preamble_text():
    raw = 'Sure, here is my analysis: {"verdict": "UNCLEAR", "confidence": "low", "explanation": "Not enough info."}'
    result = _parse_json_response(raw)
    assert result["verdict"] == "UNCLEAR"


def test_falls_back_gracefully_on_unparseable_response():
    result = _parse_json_response("I cannot determine this.")
    assert result["verdict"] == "UNCLEAR"
    assert result["confidence"] == "low"
    assert "Could not parse" in result["explanation"]
