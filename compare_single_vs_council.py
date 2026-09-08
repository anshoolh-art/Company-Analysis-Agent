"""
compare_single_vs_council.py

Runs a fixed set of test questions through THREE conditions:
  (a) naive single-agent    -- just the Analyst, one API call, no
                                instruction to double-check itself
  (b) self-critique single-agent -- one API call, but explicitly told
                                to draft, challenge itself with the
                                same checklist the real Challenger
                                uses, then revise, all in one response
  (c) full council           -- Analyst -> Challenger -> Defense -> Judge

Why three, not two: comparing (a) vs (c) alone conflates two
different things -- "does asking for self-critique help" and "does
splitting that critique across separate API calls/personas help."
(b) isolates the second question. If (b) matches (c), the honest
conclusion is that the 4-agent architecture isn't earning its extra
cost over a single well-prompted call -- and that's a much sharper,
more defensible finding than a two-arm comparison could ever give you.

Writes all three sets of answers side by side so you can score them
against SCORING.md's rubric.

Usage:
    python compare_single_vs_council.py

Writes:
    comparison_results/results_<timestamp>.json  (raw, for scripting)
    comparison_results/results_<timestamp>.md     (for reading/scoring)

Costs real API calls: 10 questions x (1 naive + 1 self-critique + up
to 1 + 3*3 council calls) = up to 140 calls in the worst case (every
question running all 3 council rounds without an early Judge PASS).
Expect it to usually be well under that, but don't run it in a loop.
"""

import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

from metrics import calculate_metrics, detect_anomalies
from agent import build_peer_data, run_single_agent, run_self_critique_agent, run_council
from main import load_companies  # reuses the same loader main.py uses

load_dotenv()

DATA_FILE = "data/semiconductor_peers.json"
OUTPUT_DIR = "comparison_results"
MAX_REVIEW_ROUNDS = 3

# A mix of objective questions (should have one defensible answer --
# if the council doesn't at least match single-agent here, that's
# useful negative signal) and genuinely open-ended ones (per the
# README's Known Limitations, these are underdetermined -- the
# interesting question is *how* each mode handles that, not whether
# they converge on the same company).
TEST_QUESTIONS = [
    # Objective / fact-lookup
    "Which company has the most total debt in FY2025?",
    "Which company had the highest revenue growth from FY2024 to FY2025?",
    "Which company has the lowest operating margin in FY2025?",
    # Multi-metric reasoning -- requires weighing more than one number
    "Which company's financial health improved the most year-over-year?",
    "Which company shows the clearest signs of financial distress?",
    "Is Micro-Mechanics' missing operating cash flow data hiding a problem, or just a data gap?",
    # Genuinely open-ended / underdetermined per the README
    "Which company is the best investment?",
    "Which company would you recommend a long-term investor buy?",
    "Rank all five companies from strongest to weakest.",
    "If you had to short one of these companies, which would it be and why?",
]


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
        return

    client = OpenAI(api_key=api_key)

    companies = load_companies(DATA_FILE)
    all_results = []
    for company in companies:
        name = company.get("name") or company.get("company") or "<unknown>"
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
    peer_data_json = build_peer_data(all_results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    records = []

    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")

        print("  running naive single-agent...")
        naive_answer = run_single_agent(client, question, peer_data_json)

        print("  running self-critique single-agent...")
        self_critique_answer = run_self_critique_agent(client, question, peer_data_json)

        print("  running council (up to {} rounds)...".format(MAX_REVIEW_ROUNDS))
        council_result = run_council(
            client, question, peer_data_json, max_rounds=MAX_REVIEW_ROUNDS
        )

        records.append({
            "question": question,
            "naive_single_agent_answer": naive_answer,
            "self_critique_answer": self_critique_answer,
            "council_final_answer": council_result["final_answer"],
            "council_rounds_run": council_result["rounds_run"],
            "council_judge_passed": council_result["passed"],
            "council_confidence": council_result["confidence"],
            "council_consensus_reached": council_result["consensus_reached"],
            "council_fact_check_warnings": council_result["fact_check_warnings"],
        })

    json_path = os.path.join(OUTPUT_DIR, f"results_{run_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    md_path = os.path.join(OUTPUT_DIR, f"results_{run_id}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Single-agent vs. council comparison -- {run_id}\n\n")
        f.write(
            "Score each pair using SCORING.md's rubric and fill in the "
            "notes line under each question. The raw JSON alongside this "
            "file has the same data if you want to script anything.\n\n"
        )
        for i, r in enumerate(records, start=1):
            f.write(f"## {i}. {r['question']}\n\n")
            f.write(f"**(a) Naive single-agent:**\n\n{r['naive_single_agent_answer']}\n\n")
            f.write(f"**(b) Self-critique single-agent:**\n\n{r['self_critique_answer']}\n\n")
            f.write(
                f"**(c) Full council** "
                f"(rounds run: {r['council_rounds_run']}, "
                f"judge passed: {r['council_judge_passed']}, "
                f"confidence: {r['council_confidence']}, "
                f"consensus reached: {r['council_consensus_reached']}):\n\n"
                f"{r['council_final_answer']}\n\n"
            )
            if r["council_fact_check_warnings"]:
                f.write("**Fact-check warnings (cited figures not found in the dataset):**\n\n")
                for w in r["council_fact_check_warnings"]:
                    f.write(f"- {w}\n")
                f.write("\n")
            f.write("**Scoring notes (a / b / c):**\n\n- \n\n---\n\n")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print("\nOpen the .md file and score each pair against SCORING.md.")


if __name__ == "__main__":
    main()
