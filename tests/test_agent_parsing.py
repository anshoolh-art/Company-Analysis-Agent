"""Unit tests for agent.py's extract_section(), focused on the new
CHALLENGE_TYPE (Challenger) and CONFIDENCE (Judge) fields added by
COUNCIL_FIXES_SPEC.md.

Pure string parsing against canned response text -- no OpenAI client
involved, so these run instantly and catch parsing regressions without
needing a live API key.
"""

from agent import extract_section

CHALLENGER_RESPONSE_FACTUAL = """VERDICT: FAIL

CHALLENGE_TYPE: FACTUAL

CHALLENGE:
The analyst claimed Frencken Group has the lowest debt, but the
dataset shows Frencken's total_debt (133.55) is the highest of all
five companies in FY2024.

REQUIRED_REVISION:
Re-check the debt figures against the dataset and correct the
ranking."""

CHALLENGER_RESPONSE_INTERPRETIVE = """VERDICT: PASS

CHALLENGE_TYPE: INTERPRETIVE

CHALLENGE:
The analyst weighted operating margin more heavily than cash-flow
trajectory. Weighting cash-flow improvement higher would favor AEM
Holdings instead -- both are reasonable given the same correct
figures.

REQUIRED_REVISION:
None."""

CHALLENGER_RESPONSE_NONE = """VERDICT: PASS

CHALLENGE_TYPE: NONE

CHALLENGE:
No issues found. The conclusion is well supported by the dataset.

REQUIRED_REVISION:
None."""

JUDGE_RESPONSE_HIGH = """VERDICT: PASS

CONFIDENCE: HIGH

REASON:
The conclusion answers the question directly, cites the relevant
metrics, and no data is missing for the companies involved."""

JUDGE_RESPONSE_MEDIUM = """VERDICT: PASS

CONFIDENCE: MEDIUM

REASON:
The conclusion is supported, but Micro-Mechanics' missing operating
cash flow means the comparison is incomplete for that company."""

JUDGE_RESPONSE_FAIL = """VERDICT: FAIL

CONFIDENCE: LOW

REASON:
The conclusion ranks Frencken Group first despite it having the
highest debt and lowest margin in the dataset, with no justification
given for why those factors were outweighed.

STATE what specifically must be reconsidered:
The ranking must account for Frencken's debt and margin position or
explain why they don't outweigh its other strengths."""

# Models sometimes wrap labels in markdown despite not being asked to.
JUDGE_RESPONSE_MARKDOWN = """**VERDICT: PASS**

**CONFIDENCE: MEDIUM**

**REASON:**
An INTERPRETIVE disagreement was raised and reasonably resolved."""


def test_challenger_verdict_and_challenge_type_factual():
    verdict = extract_section(CHALLENGER_RESPONSE_FACTUAL, "VERDICT", "CHALLENGE_TYPE")
    challenge_type = extract_section(CHALLENGER_RESPONSE_FACTUAL, "CHALLENGE_TYPE", "CHALLENGE")
    assert verdict.strip().upper() == "FAIL"
    assert challenge_type.strip().upper() == "FACTUAL"


def test_challenger_verdict_and_challenge_type_interpretive():
    verdict = extract_section(CHALLENGER_RESPONSE_INTERPRETIVE, "VERDICT", "CHALLENGE_TYPE")
    challenge_type = extract_section(CHALLENGER_RESPONSE_INTERPRETIVE, "CHALLENGE_TYPE", "CHALLENGE")
    assert verdict.strip().upper() == "PASS"
    assert challenge_type.strip().upper() == "INTERPRETIVE"


def test_challenger_challenge_type_none():
    challenge_type = extract_section(CHALLENGER_RESPONSE_NONE, "CHALLENGE_TYPE", "CHALLENGE")
    assert challenge_type.strip().upper() == "NONE"


def test_challenger_challenge_body_stops_before_required_revision():
    challenge = extract_section(CHALLENGER_RESPONSE_FACTUAL, "CHALLENGE", "REQUIRED_REVISION")
    assert "highest of all" in challenge
    assert "Re-check" not in challenge


def test_judge_verdict_stops_before_confidence_not_reason():
    # Regression guard: CONFIDENCE now sits between VERDICT and REASON.
    # Extracting VERDICT with next_label="REASON" would wrongly swallow
    # the CONFIDENCE line too -- it must use next_label="CONFIDENCE".
    verdict = extract_section(JUDGE_RESPONSE_HIGH, "VERDICT", "CONFIDENCE")
    assert verdict.strip().upper() == "PASS"
    assert "CONFIDENCE" not in verdict.upper()


def test_judge_confidence_high():
    confidence = extract_section(JUDGE_RESPONSE_HIGH, "CONFIDENCE", "REASON")
    assert confidence.strip().upper() == "HIGH"


def test_judge_confidence_medium():
    confidence = extract_section(JUDGE_RESPONSE_MEDIUM, "CONFIDENCE", "REASON")
    assert confidence.strip().upper() == "MEDIUM"


def test_judge_confidence_low_on_fail():
    verdict = extract_section(JUDGE_RESPONSE_FAIL, "VERDICT", "CONFIDENCE")
    confidence = extract_section(JUDGE_RESPONSE_FAIL, "CONFIDENCE", "REASON")
    assert verdict.strip().upper() == "FAIL"
    assert confidence.strip().upper() == "LOW"


def test_judge_markdown_wrapped_labels_still_parse():
    verdict = extract_section(JUDGE_RESPONSE_MARKDOWN, "VERDICT", "CONFIDENCE")
    confidence = extract_section(JUDGE_RESPONSE_MARKDOWN, "CONFIDENCE", "REASON")
    assert verdict.strip().upper() == "PASS"
    assert confidence.strip().upper() == "MEDIUM"


def test_missing_field_falls_back_to_full_text():
    # No CHALLENGE_TYPE label at all in this response.
    response = "VERDICT: PASS\n\nCHALLENGE:\nEverything checks out."
    result = extract_section(response, "CHALLENGE_TYPE", "CHALLENGE")
    # Falls back to the whole (cleaned) text rather than raising or
    # silently returning an empty string.
    assert result == response.strip()


def test_missing_next_label_falls_back_to_full_text():
    # CONFIDENCE is present, but the REASON boundary label is missing,
    # so the label:...next_label: pattern can't match at all -- this
    # falls back to the full cleaned text (not just the CONFIDENCE
    # tail), same fallback behavior as a fully-missing label.
    response = "VERDICT: PASS\n\nCONFIDENCE: HIGH\n\nNo REASON label here."
    result = extract_section(response, "CONFIDENCE", "REASON")
    assert result == response.strip()
