# Scoring rubric: single-agent vs. self-critique vs. council

`compare_single_vs_council.py` writes a `results_<timestamp>.md` file
with 10 questions, each answered three ways:

- **(a) Naive single-agent** — one call, just "give your best answer."
- **(b) Self-critique single-agent** — one call, but told to draft,
  challenge itself with the same checklist the real Challenger uses,
  then revise — all in one response.
- **(c) Full council** — the actual Analyst → Challenger → Defense →
  Judge pipeline, up to 3 rounds.

Three arms, not two, because (a) vs (c) alone can't tell you *why*
the council wins if it wins — it could be the adversarial multi-agent
structure, or it could just be that (c) does ~4x more reasoning than
(a). (b) isolates that: it gets the same self-critique instruction
and the same checklist as the council's Challenger, but in one call
instead of four. If (b) matches (c), the honest conclusion is that
prompting for self-critique is what's doing the work — not the
multi-agent architecture — and you could get most of the value for a
quarter of the API calls and latency.

## How to score

For each question, read all three answers and score each one (not
just the set) 0 or 1 on each of these four criteria:

1. **Correct** — every number/claim traces to the real dataset; nothing invented.
2. **Complete** — uses the metrics actually relevant to the question, not just the first one that came to mind.
3. **Appropriately hedged** — flags a real limitation (e.g. no valuation data) instead of forcing false precision on questions that are genuinely underdetermined, but doesn't hedge on questions that do have a clear answer.
4. **On-topic** — answers the question that was actually asked.

Add the four 0/1 scores for a 0-4 total per answer, per question. If
you can avoid knowing which column is which while scoring (e.g. get
someone else to relabel them, or just try not to peek), do — it
removes your own expectation from biasing the read.

At the end, sum each of (a), (b), and (c)'s totals across all 10
questions and compare all three.

## Reading the result honestly

- **(c) beats (b), and (b) beats (a)** → the cleanest possible result:
  both self-critique *and* the specific multi-agent structure are
  each adding something. Worth reporting both gaps separately (a→b
  and b→c), since they're two different claims.
- **(b) matches (c), both beat (a)** → the self-critique instruction
  is doing essentially all the work; the extra API calls of the full
  council aren't earning their cost. This is a genuinely interesting,
  fully defensible finding — "the value was in prompting for
  self-critique, not in the 4-agent architecture" is a *better*
  interview answer than an unexamined "multi-agent review helps."
- **All three roughly tied** → the honest conclusion is that none of
  this extra machinery is earning its cost for this dataset. Say
  that. "I built it, measured it, and it didn't help as much as I
  expected" is a stronger interview answer than silently keeping an
  unproven claim on your resume.
- **(a) or (b) beats (c) on some questions** → note which ones and
  why. This is the most interesting failure mode to be able to
  discuss — it means the Challenger/Defense loop introduced drift
  rather than catching a real error.

## The caveat to say out loud, not just to yourself

This is one scorer (you), on 10 questions, one run, three conditions.
It's enough to move from "no evidence" to "some evidence" — it is not
a rigorous benchmark, there's no inter-rater reliability, and a
different 10 questions could shift the result. Say this plainly if
asked about methodology; overselling a 10-question personal read as a
rigorous eval is exactly the kind of thing that unravels under one
follow-up question.
