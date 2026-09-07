# Company Analysis Agent

A command-line tool that combines deterministic financial-metric calculations with an LLM "review council" (Analyst → Challenger → Defense → Judge) to answer free-form questions about a peer group of companies.

You ask a question (e.g. *"Which company has the most debt?"*); the tool computes standardized metrics from the raw financial data in Python, then runs those metrics through a multi-agent review loop that drafts an answer, adversarially challenges it, defends/revises it, and independently judges whether the final answer is well-supported — before printing you a single clean conclusion.

## Setup

1. Create/activate the virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   For running the test suite too, use `pip install -r requirements-dev.txt` instead.

2. Add your OpenAI API key. Copy the template and fill it in:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and replace `sk-...` with your real key. `.env` is gitignored and is loaded automatically at startup via `python-dotenv`.

3. Run it:
   ```bash
   python main.py
   ```
   You'll be prompted for a question, then shown a final answer.

## Architecture

```
data/semiconductor_peers.json
        │
        ▼
metrics.py            deterministic layer (no LLM calls)
  calculate_metrics()   growth rates, margins, DSO, cash conversion, debt ratios...
  detect_anomalies()    rule-based flags (e.g. receivables outgrowing revenue)
        │
        ▼
main.py                orchestration + LLM review council
  1. Load data, compute metrics/anomalies per company (metrics.py)
  2. Prompt for a user question
  3. ANALYST      — drafts an initial answer using the Python metrics + raw data
  4. Loop up to MAX_REVIEW_ROUNDS times:
       CHALLENGER — adversarially stress-tests the current conclusion, returns PASS/FAIL
       DEFENSE    — the analyst defends or revises in light of the challenge
       JUDGE      — independently evaluates whether the (revised) conclusion is
                    well-supported, returns PASS/FAIL
       -> exits the loop as soon as the Judge returns PASS
  5. Print only the final answer (DEBUG_MODE=True shows every round's full output)
```

Each of Analyst/Challenger/Defense/Judge is one call to the OpenAI Responses API (`run_agent()` in `main.py`), all using the same peer dataset as context so nothing is invented outside the supplied numbers.

### Settings (top of `main.py`)

| Setting | Purpose |
|---|---|
| `RUN_AI` | Set `False` to only run the Python metrics/anomalies layer, skipping the LLM council entirely. |
| `DEBUG_MODE` | `True` shows every agent's full output each round; `False` (default) shows only the final answer. |
| `MAX_REVIEW_ROUNDS` | Max Challenger→Defense→Judge cycles before settling for the last Defense answer even without a PASS. |
| `MODEL` | OpenAI model used for every agent call. |

## Testing

```bash
pytest tests/
```

`tests/test_metrics.py` checks `calculate_metrics()`'s output against hand-calculated values (operating margin, DSO, cash conversion, debt-to-revenue, revenue growth) worked out independently from Venture Corporation's real FY2024/FY2025 figures in the dataset, plus edge cases for missing/zero inputs.

## Known Limitations

- **No valuation data.** The dataset has revenue, margins, cash flow, and debt, but no share price, market cap, or valuation multiples. Open-ended questions like *"which is the best investment?"* are therefore genuinely underdetermined — different runs may reasonably favor different companies depending on which operating metric they weight most heavily (e.g. margin strength vs. debt reduction vs. cash-flow trajectory), and every agent in the council is prompted to flag this gap explicitly rather than force a ranking the data can't support.
- **Sampling isn't fully deterministic even at `temperature=0`.** OpenAI's serving infrastructure doesn't guarantee bit-identical outputs at temperature 0, so wording (and occasionally which company is favored on ambiguous questions) can still vary slightly between runs.
- **Only two fiscal years of data (FY2024, FY2025).** Trend analysis is a single year-over-year comparison, not a multi-year track record.
- **No macro or industry context.** The agents reason only from the supplied peer dataset — no competitor data outside these five companies, no industry benchmarks, no qualitative context (management changes, product cycles, etc.).
- **Micro-Mechanics has no `operating_cash_flow` in the dataset**, so any metric depending on it (cash conversion, OCF margin) is `None` for that company — this is surfaced to the LLM as a stated data gap, not silently ignored.
