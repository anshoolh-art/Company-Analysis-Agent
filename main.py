import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from metrics import calculate_metrics, detect_anomalies
from agent import build_peer_data, run_council, extract_section

load_dotenv()

# ============================================================
# SETTINGS
# ============================================================

RUN_AI = True
DEBUG_MODE = False
MAX_REVIEW_ROUNDS = 3
DATA_FILE = "data/semiconductor_peers.json"


# ============================================================
# LOAD RAW FINANCIAL DATA
# ============================================================
# Pulled into a function (rather than running at import time, like
# the original script did) so compare_single_vs_council.py can reuse
# it without triggering this file's input() prompt below.

def load_companies(data_file):
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            companies = data.get("companies")

            if not isinstance(companies, list):
                raise KeyError("'companies' must be a list in the JSON file")

            return companies

    except FileNotFoundError:
        print(f"ERROR: Could not find {data_file}")
        sys.exit(1)

    except json.JSONDecodeError:
        print(f"ERROR: {data_file} is not valid JSON.")
        sys.exit(1)

    except KeyError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


# ============================================================
# METRICS & ANOMALIES PER COMPANY
# ============================================================
# calculate_metrics() and detect_anomalies() live in metrics.py so
# they can be unit tested in isolation (see tests/test_metrics.py).

def analyze_companies(companies):
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

    return all_results


def main():
    companies = load_companies(DATA_FILE)
    all_results = analyze_companies(companies)

    user_question = input("\nEnter your business or investment question: ")

    if not RUN_AI:
        print("\n" + "=" * 60)
        print("AI ANALYSIS — SKIPPED (RUN_AI=False)")
        print("=" * 60)
        return

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("\nERROR: OPENAI_API_KEY is not set.")
        print("Set your API key before running AI analysis.")
        return

    client = OpenAI(api_key=api_key)
    peer_data_json = build_peer_data(all_results)

    print("\n" + "=" * 60)
    print("AI CROSS-COMPANY AGENT")
    print("=" * 60)

    result = run_council(
        client,
        user_question,
        peer_data_json,
        max_rounds=MAX_REVIEW_ROUNDS,
    )

    if DEBUG_MODE:
        print(f"\n[ANALYST] initial answer:\n{result['analyst_response']}")
        for round_info in result["trail"]:
            print(f"\n--- REVIEW ROUND {round_info['round']}/{MAX_REVIEW_ROUNDS} ---")
            print(f"\nCHALLENGER RESPONSE:\n{round_info['challenger']}")
            print(f"\nChallenge type: {round_info['challenge_type']}")
            print(f"\nDEFENSE RESPONSE:\n{round_info['defense']}")
            print(f"\nJUDGE RESPONSE:\n{round_info['judge']}")
            print(f"\nRound {round_info['round']} verdict: {round_info['verdict']}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["final_answer"])
    print(f"\nConfidence: {result['confidence']}")

    if result["fact_check_warnings"]:
        print("\nFact-check warnings (cited figures not found in the dataset):")
        for w in result["fact_check_warnings"]:
            print(f"  - {w}")
    elif result["confidence"] != "HIGH":
        if not result["consensus_reached"]:
            print(
                f"No consensus reached after {result['rounds_run']} review "
                "round(s) -- showing the last proposed answer."
            )
        elif result["trail"]:
            last_reason = extract_section(result["trail"][-1]["judge"], "REASON")
            print(f"Reason: {last_reason}")


if __name__ == "__main__":
    main()
