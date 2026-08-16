# R4 — Prospective Clean-Sheet Shadow Validation

R4 stops historical optimization and begins a genuinely forward validation stream.

## What is frozen

- R2.1 horizon-specific feature set.
- 167-ticker inception universe.
- R1 point-in-time feature formulas.
- Ridge alpha = 1.0.
- Logistic AVOID C = 1.0 / L2.
- Within-date percentile-rank transform.
- FAVOR and AVOID evidence gates.
- Month-end evaluation cadence.

The initial 2026 model registry was fit using only rows whose outcomes were fully mature
before **2026-01-01**, matching the R3 2026 fold. Its training row/month counts match the
uploaded R3 artifact.

## Prospective inception

R4 inception date: **2026-08-15**.

The first eligible prediction anchor is the last trading observation of August 2026.
It is not issued until market data from September exists. That prevents R4 from pretending
that earlier 2026 observations were prospective.

Predictions are issued once per completed month and are immutable.

## What is tracked

For each 1M / 3M / 6M / 9M horizon:

1. Equal-weight training-oriented feature ranks.
2. Frozen Ridge return-ranking prediction.
3. Frozen Logistic AVOID probability.

R4 records every eligible ticker, not just a hand-selected recommendation.

Outcomes automatically mature after 21 / 63 / 126 / 189 ticker observations and are
appended once.

A monthly anchor is evaluated only after at least 90% of its issued ticker predictions
have matured and at least 30 rows are available.

## Evidence labels

- `INSUFFICIENT`: < 3 evaluated months.
- `EARLY`: 3–11 months.
- `PROMISING`: 12–23 months and a frozen prospective FAVOR/AVOID gate passes.
- `MIXED`: 12–23 months without a passing gate.
- `CONFIRMED_PROSPECTIVE`: >= 24 months with a passing gate.
- `FAILED_TO_CONFIRM`: >= 24 months without a passing gate.

No historical threshold is changed based on prospective results.

## Files written automatically

`results/clean_sheet_r4/`

- `r4_predictions.csv`
- `r4_anchor_manifest.csv`
- `r4_outcomes.csv`
- `r4_monthly_ranking_metrics.csv`
- `r4_monthly_avoid_metrics.csv`
- `r4_assessment.csv`
- `r4_assessment.json`
- `r4_status.json`
- `r4_latest.json`
- `r4_report.md`
- `r4_fetch_failures.csv`
- charts after outcomes begin to mature

## Why there is a separate MAIN scheduler patch

GitHub scheduled workflows run from the repository's default branch. The R4 code is
intentionally isolated on `research/clean-sheet-predictive`, so a workflow that exists only
on that branch would not run automatically.

The separate MAIN patch adds scheduler workflows that **check out the research branch**,
run R4 there, and commit only R4 research data back to the research branch. They do not
alter the production dashboard or production ranking logic.

## Annual model refresh

The feature set and hyperparameters never change. A new calendar-year model is fit once,
using the same frozen universe/features and only outcomes that were fully mature before
January 1 of that model year.

Old model years remain in the registry forever so historical prospective predictions stay
reproducible.
