"""
Extraction beyond the validated brief: prose-style case names, legislation,
and the character spans the frontend uses to highlight what was found.
These run offline on plain strings.
"""

from app.ingestion.citations import (
    dedupe_citations,
    extract_authorities,
    extract_legislation,
    get_citation_context,
)

SAMPLE = (
    "This rule derives from Hadley v Baxendale (1854) 9 Ex 341. The claimant also relies on "
    "American Cyanamid Co v Ethicon Ltd [1975] AC 396, which held that an injunction requires proof.\n\n"
    "Reliance is further placed on Harrowgate Freight Ltd v Alden Marine plc [2020] EWHC 2231 (Comm). "
    "See Lumley v Gye (1853) 2 E&B 216. The power arises under section 37 of the Senior Courts Act 1981, "
    "and Sections 4 and 5 of the Trade Disputes Act 1906 apply, as does the "
    "Contracts (Rights of Third Parties) Act 1999."
)


def _by_citation(text=SAMPLE):
    return {c.citation: c for c in dedupe_citations(extract_authorities(text))}


def test_case_names_are_not_polluted_by_surrounding_prose():
    found = _by_citation()
    assert found["[1975] AC 396"].case_name == "American Cyanamid Co v Ethicon Ltd"   # not "The claimant also relies on ..."
    assert found["[2020] EWHC 2231 (Comm)"].case_name == "Harrowgate Freight Ltd v Alden Marine plc"  # lowercase "plc" kept
    assert found["(1853) 2 E&B 216"].case_name == "Lumley v Gye"                        # "See" stripped


def test_legislation_is_extracted_with_titles():
    titles = {c.citation for c in extract_legislation(SAMPLE)}
    assert titles == {
        "Senior Courts Act 1981",
        "Trade Disputes Act 1906",
        "Contracts (Rights of Third Parties) Act 1999",
    }


def test_section_reference_is_part_of_the_highlight_but_not_the_title():
    act = _by_citation()["Senior Courts Act 1981"]
    assert SAMPLE[act.span_start:act.span_end] == "section 37 of the Senior Courts Act 1981"
    assert act.kind == "legislation"


def test_spans_cover_the_name_and_the_citation():
    hadley = _by_citation()["(1854) 9 Ex 341"]
    start, end = hadley.spans[0]
    assert SAMPLE[start:end] == "Hadley v Baxendale (1854) 9 Ex 341"


def test_repeat_mentions_are_merged_into_one_authority_with_all_spans():
    text = "Hadley v Baxendale (1854) 9 Ex 341 sets the rule.\n\nAuthorities: Hadley v Baxendale (1854) 9 Ex 341."
    found = dedupe_citations(extract_authorities(text))
    assert len(found) == 1
    assert len(found[0].spans) == 2


def test_bare_act_year_with_no_title_is_ignored():
    assert extract_legislation("The Act 2010 changed things.") == []


def test_a_sentence_boundary_stops_a_legislation_title():
    titles = {c.citation for c in extract_legislation("It was held. Human Rights Act 1998 applies.")}
    assert titles == {"Human Rights Act 1998"}


def test_context_reaches_the_claim_after_the_citation_but_stops_at_the_paragraph():
    offset = SAMPLE.index("[1975] AC 396")
    context = get_citation_context(SAMPLE, offset, forward=260)
    assert "requires proof" in context              # the claim comes AFTER the citation here
    assert "Harrowgate" not in context              # next paragraph is excluded
