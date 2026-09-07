def safe_div(numer, denom):
    """Return numer/denom or None if inputs are missing/invalid."""
    try:
        if numer is None or denom is None:
            return None
        if denom == 0:
            return None
        return numer / denom
    except Exception:
        return None


def calculate_metrics(company):
    """Extract raw fields (allowing missing) and compute derived metrics.

    If an input is missing, the derived metric will be None.
    """
    fy24 = company.get("FY2024", {})
    fy25 = company.get("FY2025", {})

    # Raw inputs (may be None)
    revenue_2024 = fy24.get("revenue")
    revenue_2025 = fy25.get("revenue")

    operating_income_2024 = fy24.get("operating_income")
    operating_income_2025 = fy25.get("operating_income")

    receivables_2024 = fy24.get("receivables")
    receivables_2025 = fy25.get("receivables")

    inventory_2024 = fy24.get("inventory")
    inventory_2025 = fy25.get("inventory")

    cash_2024 = fy24.get("cash")
    cash_2025 = fy25.get("cash")

    debt_2024 = fy24.get("total_debt")
    debt_2025 = fy25.get("total_debt")

    operating_cash_flow_2024 = fy24.get("operating_cash_flow")
    operating_cash_flow_2025 = fy25.get("operating_cash_flow")

    # Derived metrics (None when inputs missing)
    revenue_growth = None
    if revenue_2024 is not None and revenue_2025 is not None and revenue_2024 != 0:
        revenue_growth = (revenue_2025 - revenue_2024) / revenue_2024

    operating_income_growth = None
    if (
        operating_income_2024 is not None
        and operating_income_2025 is not None
        and operating_income_2024 != 0
    ):
        operating_income_growth = (
            operating_income_2025 - operating_income_2024
        ) / operating_income_2024

    operating_margin_2024 = safe_div(operating_income_2024, revenue_2024)
    operating_margin_2025 = safe_div(operating_income_2025, revenue_2025)

    receivables_growth = None
    if (
        receivables_2024 is not None
        and receivables_2025 is not None
        and receivables_2024 != 0
    ):
        receivables_growth = (
            receivables_2025 - receivables_2024
        ) / receivables_2024

    receivables_pct_revenue_2024 = safe_div(receivables_2024, revenue_2024)
    receivables_pct_revenue_2025 = safe_div(receivables_2025, revenue_2025)

    dso_2024 = None if receivables_pct_revenue_2024 is None else receivables_pct_revenue_2024 * 365
    dso_2025 = None if receivables_pct_revenue_2025 is None else receivables_pct_revenue_2025 * 365

    inventory_growth = None
    if inventory_2024 is not None and inventory_2025 is not None and inventory_2024 != 0:
        inventory_growth = (inventory_2025 - inventory_2024) / inventory_2024

    inventory_pct_revenue_2024 = safe_div(inventory_2024, revenue_2024)
    inventory_pct_revenue_2025 = safe_div(inventory_2025, revenue_2025)

    cash_growth = None
    if cash_2024 is not None and cash_2025 is not None and cash_2024 != 0:
        cash_growth = (cash_2025 - cash_2024) / cash_2024

    debt_growth = None
    if debt_2024 is not None and debt_2025 is not None and debt_2024 != 0:
        debt_growth = (debt_2025 - debt_2024) / debt_2024

    debt_pct_revenue_2024 = safe_div(debt_2024, revenue_2024)
    debt_pct_revenue_2025 = safe_div(debt_2025, revenue_2025)

    ocf_margin_2024 = safe_div(operating_cash_flow_2024, revenue_2024)
    ocf_margin_2025 = safe_div(operating_cash_flow_2025, revenue_2025)

    cash_conversion_2024 = None
    if (
        operating_income_2024 is not None
        and operating_cash_flow_2024 is not None
        and operating_income_2024 != 0
    ):
        cash_conversion_2024 = operating_cash_flow_2024 / operating_income_2024

    cash_conversion_2025 = None
    if (
        operating_income_2025 is not None
        and operating_cash_flow_2025 is not None
        and operating_income_2025 != 0
    ):
        cash_conversion_2025 = operating_cash_flow_2025 / operating_income_2025

    return {
        "revenue_growth": revenue_growth,
        "operating_income_growth": operating_income_growth,
        "operating_margin_2024": operating_margin_2024,
        "operating_margin_2025": operating_margin_2025,
        "receivables_growth": receivables_growth,
        "receivables_pct_revenue_2024": receivables_pct_revenue_2024,
        "receivables_pct_revenue_2025": receivables_pct_revenue_2025,
        "dso_2024": dso_2024,
        "dso_2025": dso_2025,
        "inventory_growth": inventory_growth,
        "inventory_pct_revenue_2024": inventory_pct_revenue_2024,
        "inventory_pct_revenue_2025": inventory_pct_revenue_2025,
        "cash_growth": cash_growth,
        "debt_growth": debt_growth,
        "debt_pct_revenue_2024": debt_pct_revenue_2024,
        "debt_pct_revenue_2025": debt_pct_revenue_2025,
        "ocf_margin_2024": ocf_margin_2024,
        "ocf_margin_2025": ocf_margin_2025,
        "cash_conversion_2024": cash_conversion_2024,
        "cash_conversion_2025": cash_conversion_2025,
    }


def detect_anomalies(metrics):
    anomalies = []

    # Revenue vs receivables
    if metrics.get("receivables_growth") is not None and metrics.get("revenue_growth") is not None:
        if metrics["receivables_growth"] > metrics["revenue_growth"]:
            anomalies.append(
                f"Receivables growth ({metrics['receivables_growth']:.1%}) exceeds revenue growth ({metrics['revenue_growth']:.1%})."
            )

    # Operating margin deterioration
    if metrics.get("operating_margin_2025") is not None and metrics.get("operating_margin_2024") is not None:
        if metrics["operating_margin_2025"] < metrics["operating_margin_2024"]:
            anomalies.append(
                f"Operating margin declined from {metrics['operating_margin_2024']:.1%} to {metrics['operating_margin_2025']:.1%}."
            )

    # DSO deterioration
    if metrics.get("dso_2025") is not None and metrics.get("dso_2024") is not None:
        if metrics["dso_2025"] > metrics["dso_2024"]:
            anomalies.append(
                f"DSO increased from {metrics['dso_2024']:.1f} days to {metrics['dso_2025']:.1f} days."
            )

    # Inventory growing faster than revenue
    if metrics.get("inventory_growth") is not None and metrics.get("revenue_growth") is not None:
        if metrics["inventory_growth"] > metrics["revenue_growth"]:
            anomalies.append(
                f"Inventory growth ({metrics['inventory_growth']:.1%}) exceeds revenue growth ({metrics['revenue_growth']:.1%})."
            )

    # Operating cash flow margin deterioration
    if metrics.get("ocf_margin_2025") is not None and metrics.get("ocf_margin_2024") is not None:
        if metrics["ocf_margin_2025"] < metrics["ocf_margin_2024"]:
            anomalies.append(
                f"Operating cash-flow margin declined from {metrics['ocf_margin_2024']:.1%} to {metrics['ocf_margin_2025']:.1%}."
            )

    # Cash decline
    if metrics.get("cash_growth") is not None and metrics.get("cash_growth") < 0:
        anomalies.append(
            f"Cash declined by {abs(metrics['cash_growth']):.1%}.")

    # Debt increasing
    if metrics.get("debt_growth") is not None and metrics.get("debt_growth") > 0:
        anomalies.append(f"Debt increased by {metrics['debt_growth']:.1%}.")

    return anomalies
