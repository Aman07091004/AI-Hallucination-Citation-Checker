from app.verification.scoring import categorize, parse_confidence_pct, resolve_confidence


def test_verdicts_map_to_the_three_required_categories():
    assert categorize("SUPPORTED") == ("verified", "full")
    assert categorize("MISQUOTED") == ("misapplied", "full")
    assert categorize("LIKELY_FABRICATED")[0] == "fabricated"
    assert categorize("REAL_BUT_NOT_IN_LOCAL_CORPUS") == ("verified", "existence_only")


def test_unknown_verdicts_are_unresolved_never_silently_verified():
    assert categorize("SOMETHING_NEW")[0] == "unresolved"


def test_confidence_parsing_accepts_common_shapes():
    assert parse_confidence_pct(85) == 85
    assert parse_confidence_pct("85%") == 85
    assert parse_confidence_pct(0.85) == 85
    assert parse_confidence_pct(140) == 100       # clamped
    assert parse_confidence_pct("high") is None
    assert parse_confidence_pct(True) is None


def test_falls_back_to_label_when_no_number_given():
    assert resolve_confidence(None, "high", "SUPPORTED") == 90
    assert resolve_confidence(None, "low", "SUPPORTED") == 40


def test_weak_retrieval_caps_confidence():
    assert resolve_confidence(98, "high", "SUPPORTED", retrieval_score=0.70) == 98
    assert resolve_confidence(98, "high", "SUPPORTED", retrieval_score=0.60) == 90
    assert resolve_confidence(98, "high", "SUPPORTED", retrieval_score=0.45) == 75


def test_an_unresolved_verdict_can_never_claim_high_confidence():
    assert resolve_confidence(95, "high", "UNCERTAIN") == 50
    assert resolve_confidence(95, "high", "UNCLEAR") == 50
