"""Heuristic ground-truth check for numbers an LLM cites about a company.

Built after finding two real fabrication cases in a live run: the
council cited "22.6%" as Micro-Mechanics' revenue growth (the real
figure, confirmed everywhere else in the same run, is 12.64%) and
separately asserted "6.51% is lower than 5.55%" (a false comparison
between two individually-correct numbers).

This catches the FIRST kind of error -- a specific number that
doesn't correspond to anything in the dataset for that company -- by
checking every number mentioned near a company's name against every
number that actually appears for that company: raw financials (both
years), computed metrics (both fraction and percentage form), and
year-over-year differences (analyst text routinely cites an absolute
change, e.g. "an increase of 70.8", which isn't itself a raw field or
a metric).

Two design choices exist because earlier, simpler versions of this
check produced a false negative and a false positive against REAL
answers from an actual run, both fixed here:

1. Percentage-tagged numbers ("22.6%") are checked only against the
   dataset's percentage-scaled values; untagged numbers are checked
   only against raw dollar figures and differences. An earlier
   version compared across both pools and let the real 22.6%
   fabrication slip through because it coincidentally landed within
   tolerance of an unrelated dollar figure for the same company.

2. The text window scanned after a company's name stops at the next
   mention of ANY company name, not a fixed character count. An
   earlier fixed-window version attributed a neighboring company's
   number to the wrong company in list-style answers ("...AEM
   Holdings at 16.37 million. Venture Corporation has 19.63
   million...") -- the window bled past the sentence boundary into
   the next company's own figure.

This does NOT catch a comparison between two numbers that are each
individually correct but reasoned about incorrectly (e.g. "6.51% is
lower than 5.55%"). That needs a different check entirely -- parsing
comparison claims and verifying the arithmetic direction -- and is a
known, explicit limitation, not an oversight.
"""

import json
import re

# Optional leading $, optional trailing % captured separately so we
# know which pool (percentage vs raw) to check the number against.
NUMBER_PATTERN = re.compile(r"\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(%?)")

MAX_WINDOW_CHARS = 250


def _rounded_set(value, ndigits_list):
    return {round(value, n) for n in ndigits_list}


def _collect_known_numbers(all_results):
    """Per company, two separate pools: percentage-scaled values (for
    numbers cited with a %) and raw/absolute values including
    year-over-year differences (for numbers cited without one).
    """
    known_pct = {}
    known_raw = {}

    for result in all_results:
        pct_values = set()
        raw_values = set()

        fy24 = result["raw_financials"].get("FY2024", {})
        fy25 = result["raw_financials"].get("FY2025", {})

        for field, v24 in fy24.items():
            v25 = fy25.get(field)
            if isinstance(v24, (int, float)):
                raw_values |= _rounded_set(v24, (0, 1, 2, 3))
            if isinstance(v25, (int, float)):
                raw_values |= _rounded_set(v25, (0, 1, 2, 3))
            if isinstance(v24, (int, float)) and isinstance(v25, (int, float)):
                diff = v25 - v24
                raw_values |= _rounded_set(diff, (0, 1, 2))
                raw_values |= _rounded_set(abs(diff), (0, 1, 2))

        for v in result["metrics"].values():
            if isinstance(v, (int, float)):
                # Both signed and absolute-value forms: analyst text
                # routinely states a negative metric's magnitude as a
                # plain positive number with a directional word
                # instead ("a decline of 7.4%" for a -7.36% growth
                # rate), so the signed form alone misses it.
                raw_values |= _rounded_set(v, (2, 3, 4))
                raw_values |= _rounded_set(abs(v), (2, 3, 4))
                pct_values |= _rounded_set(v * 100, (0, 1, 2))
                pct_values |= _rounded_set(abs(v) * 100, (0, 1, 2))

        known_pct[result["company"]] = pct_values
        known_raw[result["company"]] = raw_values

    return known_pct, known_raw


def _company_aliases(company):
    """A company's full dataset name, plus its name without a
    parenthetical suffix ("Micro-Mechanics (Holdings)" -> also
    "Micro-Mechanics"). LLM prose frequently drops the suffix after
    the first mention -- without this, a shortened mention isn't
    recognized as a company boundary at all, and its numbers get
    silently absorbed into whichever OTHER company's window happens
    to still be open, which is worse than not checking it.
    """
    aliases = [company]
    if " (" in company:
        aliases.append(company.split(" (")[0].strip())
    return aliases


def _company_mention_spans(final_answer, companies):
    """Every (start, end, company) for every mention of every company
    name or alias in the text, sorted by position, deduplicated so a
    shorter alias occurring at the same start position as the full
    name (which always contains the alias as a prefix) doesn't create
    a second, redundant, shorter span at the same spot -- used to cap
    each company's scan window at the next company mention.
    """
    raw_spans = []
    for company in companies:
        for alias in _company_aliases(company):
            for m in re.finditer(re.escape(alias), final_answer):
                raw_spans.append((m.start(), m.end(), company))

    # Keep only the longest match at each start position.
    best_by_start = {}
    for start, end, company in raw_spans:
        current = best_by_start.get(start)
        if current is None or (end - start) > (current[1] - start):
            best_by_start[start] = (start, end, company)

    spans = sorted(best_by_start.values(), key=lambda s: s[0])
    return spans


# Ticker codes are written as a short all-caps/digit code in
# parentheses right after a company name -- "(V03)", "(558)",
# "(AWX)" -- and would otherwise be misread as financial figures.
_TICKER_PATTERN = re.compile(r"\([A-Z0-9]{2,6}\)")


def _is_calendar_year(raw_str, value):
    """FY2024 / FY2025 (and bare 2024/2025) are dates, not financial
    figures -- a 4-digit whole number in a plausible fiscal-year
    range should never be checked as a cited metric.
    """
    return (
        "." not in raw_str
        and value == int(value)
        and 1990 <= value <= 2099
    )


# A numbered-list marker ("1. ", "2. ") at a line start -- these
# routinely fall inside the *previous* company's scan window, right
# before the next bullet's company name begins, and get misread as a
# cited financial figure ("...ranking.\n\n2. Micro-Mechanics...").
# Treated as a hard window boundary, the same as a company mention.
_LIST_MARKER_PATTERN = re.compile(r"(?:^|\n)\s*\d{1,2}\.\s")


def fact_check_final_answer(final_answer, peer_data_json, tolerance=0.15):
    """Return a list of warning strings, one per number that appears
    near a company's name in `final_answer` but doesn't match any
    number that actually exists for that company in the dataset
    (checked in the matching unit -- percentage vs raw -- based on
    whether the citation carries a % sign).

    `peer_data_json` is the same JSON string every prompt already
    receives -- this needs no new data plumbing.

    Numbers under 0.05 in absolute value are skipped: legitimately
    tiny ratios exist, and matching near-zero against a rounded set
    produces noise either way.
    """
    final_answer = _TICKER_PATTERN.sub("", final_answer)

    all_results = json.loads(peer_data_json)
    known_pct, known_raw = _collect_known_numbers(all_results)
    companies = [r["company"] for r in all_results]
    spans = _company_mention_spans(final_answer, companies)
    boundary_starts = sorted(
        {s[0] for s in spans} | {m.start() for m in _LIST_MARKER_PATTERN.finditer(final_answer)}
    )
    warnings = []

    for i, (start, end, company) in enumerate(spans):
        # Window ends at the next company mention OR the next
        # numbered-list marker (whichever comes first), or the fixed
        # cap -- never bleeds into another company's sentence or
        # swallows the next bullet's list number.
        later_boundaries = [b for b in boundary_starts if b >= end]
        next_boundary = later_boundaries[0] if later_boundaries else len(final_answer)
        window_end = min(end + MAX_WINDOW_CHARS, next_boundary)
        window = final_answer[end:window_end]

        for num_match in NUMBER_PATTERN.finditer(window):
            raw_str, pct_sign = num_match.group(1), num_match.group(2)
            if not raw_str or raw_str in ("-", "."):
                continue
            try:
                value = float(raw_str.replace(",", ""))
            except ValueError:
                continue
            if abs(value) < 0.05:
                continue
            if not pct_sign and _is_calendar_year(raw_str, value):
                continue

            pool = known_pct[company] if pct_sign else known_raw[company]
            if not any(abs(value - k) <= tolerance for k in pool):
                unit = "%" if pct_sign else ""
                warnings.append(
                    f"{company}: cited '{raw_str}{unit}' near this name, but no "
                    f"matching {'percentage' if pct_sign else 'raw/absolute'} "
                    f"figure exists in the dataset for this company "
                    f"(checked within +/-{tolerance})."
                )

    return warnings
