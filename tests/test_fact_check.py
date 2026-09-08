"""Unit tests for fact_check.py.

Every test here corresponds to a real bug found while building this
against actual model output -- either the fabrication it exists to
catch, or a false positive an earlier version of the checker produced
against a genuinely correct answer. None of this needs an OpenAI
client; it's pure text-processing against the real dataset.
"""

import json

import pytest

from fact_check import fact_check_final_answer
from main import load_companies
from agent import build_peer_data


@pytest.fixture(scope="module")
def peer_data_json():
    companies = load_companies("data/semiconductor_peers.json")
    from metrics import calculate_metrics, detect_anomalies

    all_results = []
    for company in companies:
        name = company.get("name") or company.get("company")
        ticker = company.get("ticker") or ""
        metrics = calculate_metrics(company)
        anomalies = detect_anomalies(metrics)
        all_results.append({
            "company": name,
            "ticker": ticker,
            "raw_financials": {
                "FY2024": company.get("FY2024", {}),
                "FY2025": company.get("FY2025", {}),
            },
            "metrics": metrics,
            "anomalies": anomalies,
        })
    return build_peer_data(all_results)


def test_catches_real_fabricated_percentage(peer_data_json):
    # The actual fabricated council answer from a live run: cites
    # Micro-Mechanics' revenue growth as 22.6%. The real figure,
    # confirmed elsewhere in the same run, is 12.64% -- 22.6 doesn't
    # correspond to anything in the dataset for this company.
    answer = (
        "Micro-Mechanics (Holdings): While it shows a revenue growth "
        "of 22.6% (corrected from the previously stated 12.6%), its "
        "operating income growth of 39.9% remains strong."
    )
    warnings = fact_check_final_answer(answer, peer_data_json)
    assert any("22.6" in w and "Micro-Mechanics" in w for w in warnings)


def test_no_false_positive_on_correct_list_style_answer(peer_data_json):
    # Real, correct naive single-agent answer to "who has the most debt".
    answer = (
        "In FY2025, Frencken Group has the most total debt among the "
        "analyzed companies, with a total debt of 90.02 million. The "
        "next highest total debt is from AEM Holdings at 16.37 "
        "million, which is substantially lower than Frencken Group's "
        "figure. Venture Corporation has 19.63 million, UMS "
        "Integration has 1.15 million, and Micro-Mechanics (Holdings) "
        "has 1.00 million."
    )
    assert fact_check_final_answer(answer, peer_data_json) == []


def test_no_false_positive_on_decline_stated_as_positive_percentage(peer_data_json):
    # Regression: Venture Corporation's real revenue_growth is -7.36%.
    # Analyst text routinely states the magnitude of a decline as a
    # plain positive number ("a decline of 7.4%") rather than citing
    # the signed figure -- an earlier version only indexed the signed
    # form and flagged this as unverified.
    answer = "Venture Corporation: a revenue decline of 7.4% from the prior year."
    assert fact_check_final_answer(answer, peer_data_json) == []


def test_percentage_and_raw_pools_dont_cross_contaminate(peer_data_json):
    # Regression: an earlier version checked every cited number
    # against ALL of a company's known values regardless of units,
    # and a percentage (22.6%) coincidentally landed within tolerance
    # of an unrelated raw dollar figure (~23.28, Micro-Mechanics'
    # cash position), masking the fabrication entirely. Percentages
    # must only be checked against percentage-scaled values.
    answer = (
        "Micro-Mechanics (Holdings): revenue growth of 22.6%, cash "
        "position of $23.28 million."
    )
    warnings = fact_check_final_answer(answer, peer_data_json)
    assert any("22.6" in w for w in warnings)
    assert not any("23.28" in w for w in warnings)  # the real cash figure, correctly not flagged


def test_no_false_positive_on_calendar_years(peer_data_json):
    # Regression: "FY2024" / "FY2025" / bare "2024" were matched as
    # plain numbers and flagged as unverified financial figures.
    answer = (
        "AEM Holdings: operating cash flow improved from -17.54 in "
        "FY2024 to 136.00 in FY2025."
    )
    assert fact_check_final_answer(answer, peer_data_json) == []


def test_no_false_positive_on_ticker_codes(peer_data_json):
    # Regression: ticker codes in parentheses right after a company
    # name ("(V03)", "(558)") were read as cited financial figures.
    answer = (
        "Venture Corporation (V03): revenue declined. "
        "UMS Integration (558): revenue grew modestly."
    )
    assert fact_check_final_answer(answer, peer_data_json) == []


def test_no_false_positive_on_numbered_list_markers(peer_data_json):
    # Regression: a numbered-list marker ("2. ", "5. ") immediately
    # before the next bullet's company name fell inside the PREVIOUS
    # company's scan window and was read as a cited figure.
    answer = (
        "1. AEM Holdings: strong operating income growth.\n\n"
        "2. Micro-Mechanics (Holdings): solid revenue growth.\n\n"
        "5. Venture Corporation: declining across key metrics."
    )
    assert fact_check_final_answer(answer, peer_data_json) == []


def test_no_false_positive_on_shortened_company_alias(peer_data_json):
    # Regression: the dataset's full name is "Micro-Mechanics
    # (Holdings)", but prose often drops the suffix after first
    # mention ("Micro-Mechanics"). An earlier version didn't
    # recognize the shortened form as a company boundary at all, so
    # its real, correct numbers were misattributed to whichever OTHER
    # company's window was still open, and flagged as unverified FOR
    # THAT COMPANY.
    answer = (
        "AEM Holdings demonstrated growth, but the absolute increase "
        "was smaller than Micro-Mechanics, which saw operating income "
        "rise from 11.84 to 16.56, an increase of 4.72."
    )
    assert fact_check_final_answer(answer, peer_data_json) == []


def test_empty_answer_returns_no_warnings(peer_data_json):
    assert fact_check_final_answer("", peer_data_json) == []


def test_company_not_mentioned_produces_no_warnings(peer_data_json):
    assert fact_check_final_answer("Nothing about any company here.", peer_data_json) == []
