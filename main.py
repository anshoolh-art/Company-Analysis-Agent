import json
import os
import re
import sys
import textwrap
from dotenv import load_dotenv
from openai import OpenAI

from metrics import calculate_metrics, detect_anomalies

load_dotenv()

# ============================================================
# SETTINGS
# ============================================================

RUN_AI = True
DEBUG_MODE = False
MAX_REVIEW_ROUNDS = 3
MODEL = "gpt-4o-mini"
DATA_FILE = "data/semiconductor_peers.json"


# ============================================================
# 1. LOAD RAW FINANCIAL DATA
# ============================================================

try:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        companies = data.get("companies")

        if not isinstance(companies, list):
            raise KeyError("'companies' must be a list in the JSON file")

except FileNotFoundError:
    print(f"ERROR: Could not find {DATA_FILE}")
    sys.exit(1)

except json.JSONDecodeError:
    print(f"ERROR: {DATA_FILE} is not valid JSON.")
    sys.exit(1)

except KeyError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


# ============================================================
# 2. HELPERS & CALCULATIONS
# ============================================================
# calculate_metrics() and detect_anomalies() live in metrics.py
# so they can be unit tested without importing this script
# (which blocks on input() at import time).


def run_agent(client, prompt):
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0
    )
    return response.output_text


def extract_section(text, label, next_label=None):
    """Pull the text following 'LABEL:' up to the next known label.

    Strips markdown emphasis (**bold**) and heading markers (###)
    around labels first, since models often add them despite not
    being asked to. Falls back to the full text if the model
    didn't follow the requested format at all, so a parsing miss
    never hides output.
    """
    cleaned = re.sub(r"[*#]+", "", text)
    if next_label:
        pattern = rf"{re.escape(label)}:\s*(.*?)\n\s*{re.escape(next_label)}:"
    else:
        pattern = rf"{re.escape(label)}:\s*(.*)"
    match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else cleaned.strip()
# ============================================================
# ASK FOR USER INPUT
# ============================================================


user_question = input("\nEnter your business or investment question: ")

# ============================================================
# ANALYZE ALL COMPANIES
# ============================================================

all_results = []

for company in companies:
    name = company.get("name") or company.get("company") or "<unknown>"
    ticker = company.get("ticker") or ""

    metrics = calculate_metrics(company)
    anomalies = detect_anomalies(metrics)

    result = {
        "company": name,
        "ticker": ticker,
        "raw_financials": {
            "FY2024": company.get("FY2024", {}),
            "FY2025": company.get("FY2025", {})
        },
        "metrics": metrics,
        "anomalies": anomalies
    }
    all_results.append(result)


# ============================================================
# AI CROSS-COMPANY ANALYSIS
# ============================================================

if not RUN_AI:
    print("\n" + "=" * 60)
    print("AI ANALYSIS — SKIPPED (RUN_AI=False)")
    print("=" * 60)

else:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("\nERROR: OPENAI_API_KEY is not set.")
        print("Set your API key before running AI analysis.")

    else:
        client = OpenAI(api_key=api_key)

        # ----------------------------------------------------
        # Prepare complete peer dataset for the AI
        # ----------------------------------------------------

        peer_data = []

        for result in all_results:
            peer_data.append({
                "company": result["company"],
                "ticker": result["ticker"],
                "raw_financials": result["raw_financials"],
                "metrics": result["metrics"],
                "anomalies": result["anomalies"]
            })

        # Convert Python data into readable JSON
        peer_data_json = json.dumps(peer_data, indent=2)

        # ----------------------------------------------------
        # Agent instructions
        # ----------------------------------------------------

        prompt = textwrap.dedent("""

You are a senior financial analyst performing cross-company
financial analysis.

You are analyzing multiple companies in the same industry using
two layers of financial information:

1. PYTHON METRICS
   These are standardized financial metrics calculated
   deterministically by the Python analysis layer.

2. RAW FINANCIALS
   These are the underlying FY2024 and FY2025 financial figures
   supplied in the dataset.

Your role is not simply to summarize the dataset. You are an
analytical agent responsible for determining what evidence is
needed to answer the user's question, evaluating that evidence,
comparing companies, challenging your own reasoning, and reaching
the strongest conclusion supported by the available data.


============================================================
EVIDENCE AND DATA RULES
============================================================

- Use only information contained in the supplied dataset.
- Do NOT invent financial figures.
- Do NOT assume missing data.
- Do NOT use outside knowledge about the companies, industry,
  management, market conditions, or share prices.
- Treat the Python metrics as authoritative for metrics that have
  already been calculated.
- Do NOT unnecessarily recalculate a metric that already exists
  in the Python metrics.


============================================================
DYNAMIC METRIC GENERATION
============================================================

The predefined Python metrics are NOT necessarily sufficient to
answer every question.

Before reaching a conclusion, determine what financial evidence
is actually relevant to the user's question.

For each metric or measure required:

1. First check whether the metric exists in the Python metrics.
2. If it exists, use the Python metric.
3. If it does not exist, inspect the raw financial data.
4. Determine whether the required metric can be calculated
   from the available raw financial figures.
5. If it can be calculated, derive the metric yourself.
6. Clearly label it as a "Derived Metric".
7. State the formula and relevant inputs used.
8. If it cannot be calculated because required information is
   unavailable, identify the metric as "Unavailable".
9. Explain what additional information would be required.

Only derive metrics that are genuinely relevant to the user's
question. Do not create unnecessary or arbitrary metrics.

A Derived Metric must never be presented as though it were a
predefined Python Metric.


============================================================
ANALYTICAL REASONING
============================================================

When answering the user's question:

1. Determine what the question is actually asking.

2. Identify the financial metrics or evidence most relevant to
   answering it.

3. Compare ALL companies using the relevant evidence.

4. Consider relationships between metrics rather than relying
   on a single metric.

5. Distinguish between:
   - strongest absolute position,
   - strongest improvement or deterioration,
   - and strongest overall evidence relevant to the question.

6. Identify important positive and negative signals.

7. Consider whether apparently positive results may have
   alternative explanations.

8. Do not confuse correlation with proof of causation.

9. Explicitly identify material limitations in the dataset.

10. Reach a clear decision or ranking when the evidence supports
    one.

11. If the evidence is insufficient for a reliable conclusion,
    say so clearly rather than forcing a recommendation.


============================================================
IMPORTANT FINANCIAL DISTINCTIONS
============================================================

Do not equate operating leverage with proven economies of scale.

Economies of scale technically require evidence that average cost
per unit falls as output increases.

If unit-cost, output-volume, fixed-cost, or variable-cost data is
unavailable, do not claim that economies of scale have been
proven.

Instead, describe the evidence as:
- "consistent with economies of scale", or
- "evidence of operating leverage"

when appropriate.

Similarly, do not treat a single favorable financial metric as
conclusive evidence of superior business performance.


============================================================
DATA LIMITATIONS
============================================================

Missing data must remain missing.

Do not estimate, interpolate, or infer missing financial figures
unless the calculation is directly possible from other supplied
figures.

If an important metric cannot be calculated from the available
data, explicitly state:

- what metric is unavailable,
- why it cannot be calculated,
- and what additional data would be needed.

The absence of a metric should itself be considered when judging
the confidence of a conclusion.


============================================================
OUTPUT REQUIREMENTS
============================================================

Answer the user's question directly.

Do not simply describe each company.

Start with a clear conclusion or recommendation whenever the
data supports one.

Then provide the key reasoning supporting that conclusion.

Use approximately ONE sentence per company when comparing the
peer group. Use an additional sentence only when necessary to
explain an important distinction, derived metric, or limitation.

When a Derived Metric materially affects the conclusion, briefly
show:

Derived Metric: [metric name]
Formula: [formula]
Result: [relevant comparison]

Do not overwhelm the user with calculations that are not relevant
to the question.

End with a concise statement of confidence or the most important
limitation when appropriate.


============================================================
USER QUESTION
============================================================

{user_question}


============================================================
TASK
============================================================

Analyze the user's question using the COMPLETE peer dataset,
including both the Python metrics and the underlying raw
financials.

Determine what evidence is required, derive additional relevant
metrics from the raw financials when necessary and possible,
identify limitations where necessary, compare the companies, and
make the strongest conclusion supported by the data.

Do not merely repeat the dataset.


============================================================
PEER DATASET
============================================================

{peer_data_json}
""").format(user_question=user_question, peer_data_json=peer_data_json)

        # ----------------------------------------------------
        # Run AI agent
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("AI CROSS-COMPANY AGENT")
        print("=" * 60)

        # ====================================================
        # 1. ANALYST
        # ====================================================

        if DEBUG_MODE:
            print("\n[ANALYST] analyzing question...")

        analyst_response = run_agent(client, prompt)

        if DEBUG_MODE:
            print("\nANALYST RESPONSE:")
            print(analyst_response)

        # ====================================================
        # 2-4. ITERATIVE REVIEW: CHALLENGER -> DEFENSE -> JUDGE
        # ====================================================
        # current_conclusion starts as the Analyst's answer, and
        # becomes each round's Defense answer thereafter. The loop
        # exits early once the Judge returns VERDICT: PASS, or
        # after MAX_REVIEW_ROUNDS if it never does.

        current_conclusion = analyst_response
        defense_response = analyst_response
        judge_response = None

        for review_round in range(1, MAX_REVIEW_ROUNDS + 1):
            if DEBUG_MODE:
                print(f"\n--- REVIEW ROUND {review_round}/{MAX_REVIEW_ROUNDS} ---")
                print("\n[CHALLENGER] testing conclusion...")

            challenger_prompt = f"""
You are a highly skeptical financial analyst acting as a
CHALLENGER to another analyst.

Your job is NOT to produce an independent answer.

Your job is to aggressively test the analyst's conclusion.

USER'S QUESTION:
{user_question}

ANALYST'S CONCLUSION:
{current_conclusion}

DATASET:
{peer_data_json}

Evaluate the analyst's conclusion.

Look specifically for:

1. Incorrect calculations or interpretation of the provided metrics.
2. Claims not supported by the dataset.
3. Important relevant metrics the analyst ignored.
4. Confusion between absolute and relative metrics.
5. Confusion between current position and improvement/trajectory.
6. Alternative companies that could reasonably be the answer.
7. Missing data that materially weakens the conclusion.
8. Unsupported assumptions.
9. Whether the analyst answered the actual question.
10. Whether the confidence level is justified.

Be adversarial but evidence-based.

If the conclusion is genuinely well supported, say so.

Return:

VERDICT: PASS or FAIL

CHALLENGE:
Explain the strongest issue you found, or explain why the
conclusion survives scrutiny.

REQUIRED_REVISION:
If FAIL, explain exactly what the analyst needs to change.
If PASS, write "None".
"""

            challenger_response = run_agent(client, challenger_prompt)

            if DEBUG_MODE:
                print("\nCHALLENGER RESPONSE:")
                print(challenger_response)
                print("\n[DEFENSE] responding to challenge...")

            defense_prompt = f"""
You are the original financial analyst.

You previously produced this conclusion:

{current_conclusion}

A challenger has now reviewed your conclusion:

{challenger_response}

USER'S QUESTION:
{user_question}

DATASET:
{peer_data_json}

Your task is to defend or revise your conclusion.

You must:

1. Determine whether the challenger's criticism is valid.
2. Correct any genuine errors.
3. Reject criticism that is unsupported by the dataset.
4. Reconsider your conclusion if the challenger identifies
   stronger evidence for another company.
5. Do not invent financial information.
6. Do not assume missing data.
7. Distinguish between absolute performance and improvement.
8. Distinguish between evidence and proof.

Return:

FINAL_STATUS:
DEFENDED or REVISED

FINAL_ANSWER:
Provide the best answer to the user's original question.

DEFENSE:
Briefly explain why the conclusion survives the challenge
or why it was revised.
"""

            defense_response = run_agent(client, defense_prompt)

            if DEBUG_MODE:
                print("\nDEFENSE RESPONSE:")
                print(defense_response)
                print("\n[JUDGE] making final determination...")

            judge_prompt = f"""
You are the FINAL JUDGE in a financial analysis review.

You are independent of both the analyst and challenger.

USER QUESTION:
{user_question}

ORIGINAL ANALYST:
{analyst_response}

CHALLENGER:
{challenger_response}

ANALYST DEFENSE:
{defense_response}

DATASET:
{peer_data_json}

Determine whether the final conclusion is sufficiently
supported by the available financial data.

Do NOT automatically agree with the analyst.

Do NOT automatically agree with the challenger.

Evaluate the evidence yourself.

The analyst should PASS only if:

- The conclusion answers the user's actual question.
- Important relevant metrics were considered.
- No material factual errors remain.
- The conclusion is supported by the dataset.
- Important limitations are acknowledged.
- The reasoning does not rely on unsupported assumptions.

Return exactly:

VERDICT: PASS or FAIL

REASON:
One concise explanation.

If FAIL:
STATE what specifically must be reconsidered.

If PASS:
State why the conclusion is sufficiently supported.
"""

            judge_response = run_agent(client, judge_prompt)
            verdict = extract_section(judge_response, "VERDICT", "REASON").strip().upper()

            if DEBUG_MODE:
                print("\nJUDGE RESPONSE:")
                print(judge_response)
                print(f"\nRound {review_round} verdict: {verdict}")

            current_conclusion = defense_response

            if "PASS" in verdict:
                break

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        final_answer = extract_section(defense_response, "FINAL_ANSWER", "DEFENSE")

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(final_answer)
