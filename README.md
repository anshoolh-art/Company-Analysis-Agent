# Company Analysis Agent

A command-line tool that combines deterministic financial-metric calculations with an LLM "review council" (Analyst → Challenger → Defense → Judge) to answer free-form questions about a peer group of companies.

You ask a question (e.g. *"Which company has the most debt?"*); the tool computes standardized metrics from the raw financial data in Python, then runs those metrics through a multi-agent review loop that drafts an answer, adversarially challenges it, defends/revises it, and independently judges whether the final answer is well-supported before printing you a single clean conclusion.

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
agent.py               prompt construction + the review council
  build_analyst_prompt() / build_challenger_prompt() / build_defense_prompt() / build_judge_prompt()
  run_single_agent()     Analyst only, one call — no review loop
  run_council()          Analyst -> Challenger -> Defense -> Judge, looping up
                         to MAX_REVIEW_ROUNDS times, exits early on Judge PASS
        │
        ▼
main.py                 CLI entry point
  1. Load data, compute metrics/anomalies per company (metrics.py)
  2. Prompt for a user question
  3. Run run_council() and print the final answer
     (DEBUG_MODE=True also prints every round's full output)
```

`agent.py` holds all the prompt text and the council logic so it has exactly one copy, reusable by both `main.py` (the CLI) and `compare_single_vs_council.py` (the eval script below) without duplicating ~250 lines of prompt text. Each of Analyst/Challenger/Defense/Judge is one call to the OpenAI Responses API (`run_agent()` in `agent.py`), all using the same peer dataset as context so nothing is invented outside the supplied numbers.

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

Note: `agent.py`'s prompt-building and review-loop logic (parsing, verdict handling, round-exit conditions) isn't unit tested — it's harder to test without mocking the OpenAI client, and hasn't been done yet.

## Evaluation: does the council actually help?

The 4-agent council (Analyst → Challenger → Defense → Judge) was a
design hypothesis, not a measured improvement — running it costs up
to ~4x more API calls and latency than a single Analyst call, and
that trade-off is only worth it if the extra review rounds actually
produce better answers.

`compare_single_vs_council.py` tests this with **three** conditions,
not two, on a fixed set of 10 questions (a mix of objective
fact-lookups and genuinely open-ended ones):

- **(a) Naive single-agent** — `run_single_agent()`, one call, no
  self-critique instruction.
- **(b) Self-critique single-agent** — `run_self_critique_agent()`,
  one call, but explicitly told to draft, challenge itself using the
  *same checklist* the real Challenger uses, then revise.
- **(c) Full council** — `run_council()`, the actual pipeline.

(b) exists because (a) vs (c) alone can't separate two different
questions: "does asking for self-critique help" vs. "does splitting
that critique across separate API calls/personas help." Without (b),
a council win over (a) could just mean the council does more
reasoning overall, not that the specific multi-agent structure is
what matters.

`SCORING.md` has the rubric for turning the three-way output into an
actual verdict.

```bash
python compare_single_vs_council.py
```

Status: script is built and smoke-tested (with a mocked API client,
to confirm the plumbing works), but not yet run against the real
dataset with a live API key — the results and conclusion aren't in
yet.

## Known Limitations

- **No valuation data.** The dataset has revenue, margins, cash flow, and debt, but no share price, market cap, or valuation multiples. Open-ended questions like *"which is the best investment?"* are therefore genuinely underdetermined — different runs may reasonably favor different companies depending on which operating metric they weight most heavily (e.g. margin strength vs. debt reduction vs. cash-flow trajectory), and every agent in the council is prompted to flag this gap explicitly rather than force a ranking the data can't support.
- **Sampling isn't fully deterministic even at `temperature=0`.** OpenAI's serving infrastructure doesn't guarantee bit-identical outputs at temperature 0, so wording (and occasionally which company is favored on ambiguous questions) can still vary slightly between runs.
- **Only two fiscal years of data (FY2024, FY2025).** Trend analysis is a single year-over-year comparison, not a multi-year track record.
- **No macro or industry context.** The agents reason only from the supplied peer dataset — no competitor data outside these five companies, no industry benchmarks, no qualitative context (management changes, product cycles, etc.).
- **Micro-Mechanics has no `operating_cash_flow` in the dataset**, so any metric depending on it (cash conversion, OCF margin) is `None` for that company — this is surfaced to the LLM as a stated data gap, not silently ignored.
