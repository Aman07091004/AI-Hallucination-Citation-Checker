from app.verification.external_check import _parse_json_response, classify_external_result


def test_fabricated_case_classification():
    result = _parse_json_response('{"exists": false, "confidence": "high", "evidence": "No record found.", "source": ""}')
    assert classify_external_result(result) == "LIKELY_FABRICATED"


def test_real_but_missing_case_classification():
    result = _parse_json_response('{"exists": true, "confidence": "high", "evidence": "Found on BAILII.", "source": "bailii.org"}')
    assert classify_external_result(result) == "REAL_BUT_NOT_IN_LOCAL_CORPUS"


def test_medium_confidence_exists_true_is_not_trusted_as_real():
    """Regression test for a real failure we hit: 'Stonegate Capital
    Partners v Redwood Procurement Group' (fabricated) was marked
    exists=True at medium confidence because the model found an unrelated
    real company with a similar name. Medium/low confidence 'exists' must
    downgrade to UNCERTAIN, not confidently clear the citation."""
    result = _parse_json_response(
        '{"exists": true, "confidence": "medium", '
        '"evidence": "Found unrelated company with similar name, no direct judgment.", "source": ""}'
    )
    assert classify_external_result(result) == "UNCERTAIN"


def test_unparseable_response_is_uncertain_not_negative():
    """Important: a parsing failure must NOT default to 'fabricated' - that
    would be a false accusation caused by our own bug, not evidence about
    the citation. It must default to UNCERTAIN."""
    result = _parse_json_response("I'm not sure how to answer that.")
    assert result["exists"] is None
    assert classify_external_result(result) == "UNCERTAIN"
