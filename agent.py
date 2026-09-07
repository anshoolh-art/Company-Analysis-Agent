"""Shared LLM agent logic: prompt construction, the raw API call, and
the two entry points other code should use:

  run_single_agent()  -- Analyst only, one call, no review loop
  run_council()       -- Analyst -> Challenger -> Defense -> Judge,
                          looping up to max_rounds times

Pulled out of main.py so this logic exists exactly once and can be
imported both by the CLI (main.py) and by
compare_single_vs_council.py without duplicating ~250 lines of prompt
text, and without importing main.py itself (which blocks on input()
as soon as it's loaded).

Every prompt's wording here is unchanged from the original main.py --
this file only restructures *where* the strings live, not what they say.
"""

import json
import re
import textwrap

DEFAULT_MODEL = "gpt-4o-mini"


def run_agent(client, prompt, model=DEFAULT_MODEL):
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
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


def build_peer_data(all_results):
    """Turn analyze_companies()'s output into the JSON blob every prompt shares."""
    peer_data = []
    for result in all_results:
        peer_data.append({
            "company": result["company"],
            "ticker": result["ticker"],
            "raw_financials": result["raw_financials"],
            "metrics": result["metrics"],
            "anomalies": result["anomalies"],
        })
    return json.dumps(peer_data, indent=2)


# ============================================================
# PROMPT TEMPLATES
# Wording is identical to the original main.py -- only pulled into
# functions so each can be called with different arguments from
# different entry points (the CLI vs. the comparison script).
# ============================================================

def build_analyst_prompt(user_question, peer_data_json):
    return textwrap.dedent("""

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
""").format(
        user_question=user_question, peer_data_json=peer_data_json
    )


def build_challenger_prompt(user_question, current_conclusion, peer_data_json):
    return f"""
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

Be adversarial but evidence-based. Any challenge you raise must cite
the specific metric, anomaly, or dataset field it is based on. If you
cannot point to a specific number or a specific gap in the supplied
data, you have not found a genuine issue -- return VERDICT: PASS.

Classify your own challenge as one of:

- FACTUAL: a specific calculation error, an ignored relevant metric,
  or a claim unsupported by the dataset. Only a FACTUAL challenge may
  justify VERDICT: FAIL.
- INTERPRETIVE: a reasonable alternative weighting or framing of the
  same correct facts -- nothing is factually wrong. This must be
  returned as VERDICT: PASS. Still write out the alternative view in
  CHALLENGE so it can be surfaced to the user, not treated as an error.
- NONE: no genuine issue found.

FAIL is only valid when CHALLENGE_TYPE is FACTUAL. If your objection
is interpretive rather than factual, return VERDICT: PASS and
CHALLENGE_TYPE: INTERPRETIVE -- still explain the alternative view in
CHALLENGE.

If the conclusion is genuinely well supported, say so.

Return:

VERDICT: PASS or FAIL

CHALLENGE_TYPE: FACTUAL or INTERPRETIVE or NONE

CHALLENGE:
Explain the strongest issue you found, or explain why the
conclusion survives scrutiny.

REQUIRED_REVISION:
If FAIL, explain exactly what the analyst needs to change.
If PASS, write "None"."""


def build_defense_prompt(user_question, current_conclusion, challenger_response, peer_data_json):
    return f"""
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
Provide the best answer to the user's original question, including
the key reasoning supporting it -- not a bare verdict. Include
approximately one sentence per company of supporting reasoning,
matching the level of detail expected of an initial analysis, not
just a one-line conclusion.

If the challenger's CHALLENGE_TYPE was INTERPRETIVE and you did not
act on it (because it was a reasonable alternative view rather than
an error), note that alternative view in one sentence within
FINAL_ANSWER, e.g. "An alternative reading favoring X's cash-flow
trajectory is also reasonable, but Y is favored here because ..."

DEFENSE:
Briefly explain why the conclusion survives the challenge
or why it was revised."""


def build_judge_prompt(user_question, analyst_response, challenger_response, defense_response, peer_data_json):
    return f"""
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

Also check specifically: does this conclusion contradict any metric,
anomaly, or figure elsewhere in the supplied dataset -- for example, a
company called strongest despite having the worst debt or margin
figures in the same dataset? This is a valid basis for FAIL even if no
other issue is found.

Do not return FAIL solely because the Challenger raised an
INTERPRETIVE point. An interpretive disagreement should be noted
(see confidence guidance below) but is not itself grounds for FAIL --
only a FACTUAL issue (from the Challenger or your own independent
check, including the cross-consistency check above) justifies FAIL.

In addition to your verdict, assess your confidence in the final
conclusion:

- HIGH: PASS reached, no missing data relevant to the question, and
  the question is not one of the dataset's known-underdetermined
  categories (see the project README's Known Limitations -- e.g. no
  valuation data is available).
- MEDIUM: PASS reached, but the question depends on data missing for
  a relevant company (e.g. a company's missing operating cash flow),
  or an INTERPRETIVE disagreement was noted along the way.
- LOW: the question falls in the known-underdetermined category and a
  ranking or pick was forced anyway.

Return exactly:

VERDICT: PASS or FAIL

CONFIDENCE: HIGH or MEDIUM or LOW

REASON:
One concise explanation, covering both the verdict and the confidence
level.

If FAIL:
STATE what specifically must be reconsidered.

If PASS:
State why the conclusion is sufficiently supported."""


def build_self_critique_prompt(user_question, peer_data_json):
    """Same grounding rules as the Analyst prompt, plus an explicit
    self-challenge pass using the *identical* checklist the real
    Challenger agent uses. Exists so run_self_critique_agent() is a
    fair second baseline: it isolates "does asking for self-critique
    help" from "does splitting that critique across separate
    API calls/personas help", which a bare single-agent-vs-council
    comparison conflates.
    """
    return textwrap.dedent("""

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
SELF-CHALLENGE AND OUTPUT REQUIREMENTS
============================================================

Work through this in three passes, all within this single response.

PASS 1 - DRAFT ANSWER
Answer the user's question directly using the rules above. Do not
simply describe each company. Start with a clear conclusion or
recommendation whenever the data supports one, then the key
reasoning. Use approximately one sentence per company when comparing
the peer group, with an additional sentence only when necessary to
explain an important distinction, derived metric, or limitation.

PASS 2 - SELF-CHALLENGE
Now act as a highly skeptical second reviewer of your own PASS 1
answer -- the same way an independent adversarial reviewer would.

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

Be adversarial but evidence-based. If the PASS 1 answer is genuinely
well supported, say so rather than manufacturing a criticism.

PASS 3 - FINAL ANSWER
Revise your PASS 1 answer if PASS 2 identified a genuine issue.
Otherwise keep it as-is. End with a concise statement of confidence
or the most important limitation when appropriate.

Return exactly these three labeled sections and nothing else:

DRAFT:
[your Pass 1 answer]

SELF_CHALLENGE:
[your Pass 2 critique]

FINAL_ANSWER:
[your Pass 3 final answer]


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
""").format(
        user_question=user_question, peer_data_json=peer_data_json
    )


def run_single_agent(client, user_question, peer_data_json, model=DEFAULT_MODEL):
    """Analyst only -- one call, no Challenger/Defense/Judge loop.

    This is the baseline compare_single_vs_council.py measures the
    full council against.
    """
    prompt = build_analyst_prompt(user_question, peer_data_json)
    return run_agent(client, prompt, model=model)


def run_self_critique_agent(client, user_question, peer_data_json, model=DEFAULT_MODEL):
    """Single call, but explicitly told to draft, self-challenge using
    the same checklist the Challenger uses, then revise -- all in one
    response. The second (fairer) baseline for compare_single_vs_council.py,
    alongside the plain run_single_agent().
    """
    prompt = build_self_critique_prompt(user_question, peer_data_json)
    response = run_agent(client, prompt, model=model)
    return extract_section(response, "FINAL_ANSWER")


def run_council(client, user_question, peer_data_json, max_rounds=3, model=DEFAULT_MODEL):
    """Analyst -> Challenger -> Defense -> Judge, up to max_rounds cycles.

    Returns a dict:
      analyst_response   -- the very first, pre-challenge answer
      final_answer       -- the FINAL_ANSWER section of the last Defense
      rounds_run         -- how many Challenger/Defense/Judge cycles ran
      passed             -- whether the Judge ever returned PASS
      confidence         -- "HIGH"/"MEDIUM"/"LOW", the Judge's confidence
                             on the final round used, forced to "LOW" if
                             max_rounds was exhausted without a PASS
      consensus_reached  -- same as `passed`, unless overridden above
      trail              -- per-round challenger/defense/judge/verdict/
                             challenge_type, for DEBUG_MODE-style inspection
    """
    analyst_response = run_agent(client, build_analyst_prompt(
        user_question, peer_data_json), model=model)

    current_conclusion = analyst_response
    defense_response = analyst_response
    trail = []
    passed = False
    confidence = "LOW"
    rounds_run = 0

    for review_round in range(1, max_rounds + 1):
        rounds_run = review_round

        challenger_response = run_agent(
            client,
            build_challenger_prompt(
                user_question, current_conclusion, peer_data_json),
            model=model,
        )
        challenge_type = extract_section(
            challenger_response, "CHALLENGE_TYPE", "CHALLENGE").strip().upper()

        defense_response = run_agent(
            client,
            build_defense_prompt(
                user_question, current_conclusion, challenger_response, peer_data_json),
            model=model,
        )
        judge_response = run_agent(
            client,
            build_judge_prompt(user_question, analyst_response,
                               challenger_response, defense_response, peer_data_json),
            model=model,
        )
        verdict = extract_section(
            judge_response, "VERDICT", "CONFIDENCE").strip().upper()
        confidence = extract_section(
            judge_response, "CONFIDENCE", "REASON").strip().upper()

        trail.append({
            "round": review_round,
            "challenger": challenger_response,
            "challenge_type": challenge_type,
            "defense": defense_response,
            "judge": judge_response,
            "verdict": verdict,
        })

        current_conclusion = defense_response

        if "PASS" in verdict:
            passed = True
            break

    final_answer = extract_section(defense_response, "FINAL_ANSWER", "DEFENSE")

    consensus_reached = passed
    if not passed:
        # Max rounds exhausted without a PASS -- the last proposed
        # answer is unresolved, regardless of what the Judge's last
        # CONFIDENCE line said.
        confidence = "LOW"

    return {
        "analyst_response": analyst_response,
        "final_answer": final_answer,
        "rounds_run": rounds_run,
        "passed": passed,
        "confidence": confidence,
        "consensus_reached": consensus_reached,
        "trail": trail,
    }
