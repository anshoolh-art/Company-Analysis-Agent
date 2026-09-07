"""Unit tests for metrics.py, checked against hand-calculated values.

The expected numbers below were worked out by hand from Venture
Corporation's real FY2024/FY2025 figures in
data/semiconductor_peers.json (not derived by re-running the same
formula the implementation uses), so a passing test means the code's
arithmetic actually matches an independent calculation:

  FY2024: operating_income=260.5, revenue=2735.93, receivables=704.82,
           total_debt=30.25, operating_cash_flow=482.51
  FY2025: operating_income=243.3, revenue=2534.52, receivables=638.31,
           total_debt=19.63, operating_cash_flow=251.535
"""

import pytest

from metrics import safe_div, calculate_metrics

VENTURE = {
    "name": "Venture Corporation",
    "FY2024": {
        "operating_income": 260.5,
        "cash": 1318.0,
        "inventory": 686.43,
        "receivables": 704.82,
        "operating_cash_flow": 482.51,
        "total_debt": 30.25,
        "revenue": 2735.93,
    },
    "FY2025": {
        "operating_income": 243.3,
        "cash": 1285.0,
        "inventory": 695.84,
        "receivables": 638.31,
        "operating_cash_flow": 251.535,
        "total_debt": 19.63,
        "revenue": 2534.52,
    },
}


@pytest.fixture
def metrics():
    return calculate_metrics(VENTURE)


def test_operating_margin(metrics):
    # 260.5 / 2735.93 and 243.3 / 2534.52, hand-calculated
    assert metrics["operating_margin_2024"] == pytest.approx(0.095215, rel=1e-3)
    assert metrics["operating_margin_2025"] == pytest.approx(0.095995, rel=1e-3)


def test_dso(metrics):
    # (receivables / revenue) * 365, hand-calculated
    assert metrics["dso_2024"] == pytest.approx(94.03, rel=1e-3)
    assert metrics["dso_2025"] == pytest.approx(91.92, rel=1e-3)


def test_cash_conversion(metrics):
    # operating_cash_flow / operating_income, hand-calculated
    assert metrics["cash_conversion_2024"] == pytest.approx(1.852246, rel=1e-3)
    assert metrics["cash_conversion_2025"] == pytest.approx(1.033847, rel=1e-3)


def test_debt_pct_revenue(metrics):
    # total_debt / revenue, hand-calculated
    assert metrics["debt_pct_revenue_2024"] == pytest.approx(0.011057, rel=1e-3)
    assert metrics["debt_pct_revenue_2025"] == pytest.approx(0.007745, rel=1e-3)


def test_revenue_growth(metrics):
    # (revenue_2025 - revenue_2024) / revenue_2024, hand-calculated
    assert metrics["revenue_growth"] == pytest.approx(-0.073617, rel=1e-3)


def test_safe_div_handles_missing_and_zero_inputs():
    assert safe_div(10, 5) == 2
    assert safe_div(None, 5) is None
    assert safe_div(5, None) is None
    assert safe_div(5, 0) is None


def test_calculate_metrics_returns_none_for_missing_fields():
    # Micro-Mechanics has no operating_cash_flow in FY2024/FY2025 in the
    # real dataset -- metrics that depend on it must degrade to None,
    # not raise or silently produce a wrong number.
    company = {
        "FY2024": {"operating_income": 11.84, "revenue": 57.89},
        "FY2025": {"operating_income": 16.56, "revenue": 65.21},
    }
    result = calculate_metrics(company)
    assert result["ocf_margin_2024"] is None
    assert result["cash_conversion_2024"] is None
    assert result["operating_margin_2024"] == pytest.approx(0.204526, rel=1e-3)
