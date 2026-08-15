# Research Patch R3 — Walk-Forward Multivariate Baseline

R3 is the first clean-sheet phase that combines signals. It remains a research/development phase and does not modify the production dashboard model.

## Why R3 is split into two models

R2.1 found no independently validated FAVOR signal, but it did find multiple AVOID-capable features. R3 therefore does **not** assume BUY and AVOID are symmetric problems.

R3 runs two separate fixed-complexity models for each horizon:

1. **Return Ranking — Ridge regression**
   - predicts future ETF return relative to SPY;
   - tests whether combining the R2.1-selected features improves cross-sectional ranking;
   - compares Ridge against a training-oriented equal-weight feature average and the best single training feature.

2. **AVOID Classifier — L2 logistic regression**
   - predicts whether an ETF will finish in the bottom 20% of SPY-relative returns for that horizon;
   - evaluates AUC, average precision, Brier score, high-risk precision/lift, and the realized return of the high-risk bucket.

There is intentionally **no symmetric FAVOR classifier** in R3. FAVOR has to be earned by the out-of-year Ridge top bucket.

## Frozen candidate inputs

R3 reads `nonredundant_candidate_set.csv` produced by R2.1 and will accept only rows marked `SELECTED`.

Current frozen inputs are:

- 1M: `bars_since_high_252`, `dist_low_252`
- 3M: `dist_low_252`, `sma_200_slope_20`, `dollar_volume_ratio_20_63`
- 6M: `dist_low_126`, `above_sma_200`, `ret_126`
- 9M: `volume_z20`, `dist_low_126`, `dist_low_63`, `range_position_126`, `sma_50_slope_20`

R3 does not reintroduce features that R2.1 excluded as exact or near duplicates.

## Walk-forward protocol

The requested test years are 2020–2026.

For a test year such as 2023:

- the model is trained once at 2023-01-01;
- only observations dated before that cutoff are considered for training;
- a row is eligible only if its future outcome **ended before the cutoff**;
- the fitted model is frozen for the entire 2023 test year;
- late-2026 predictions can exist even when their outcomes have not matured, but they are excluded from evaluation.

This prevents future-outcome leakage into model fitting.

## Feature transform

Each R2.1 input is converted to its percentile rank inside the ETF cross-section on that month-end date. This is point-in-time and matches the cross-sectional evidence discovered in R2.

Rows missing one of the horizon's selected inputs are omitted rather than imputed.

## Fixed models — no tuning

R3 freezes these models before the study runs:

- Ridge: `alpha=1.0`
- Logistic regression: L2 regularization, `C=1.0`, `lbfgs`

No grid search, Bayesian search, tree model, random forest, boosting model, neural network, or AI model is used.

## AVOID target

For every mature month-end cross-section, the realized bottom 20% of future SPY-relative returns are labeled `AVOID=1`.

This avoids tuning an arbitrary fixed loss threshold after seeing the historical results.

## FAVOR evidence gate

Ridge earns `FAVOR_EARNED_DEVELOPMENT_EVIDENCE` only when all four are true:

- mean monthly IC 95% block-bootstrap lower bound > 0;
- top-decile mean SPY-relative return lower bound > 0;
- top-decile SPY-beating-rate lower bound > 50%;
- top-quintile minus bottom-quintile spread lower bound > 0.

Otherwise the result is `FAVOR_NOT_EARNED`.

## AVOID evidence gate

The logistic model earns `AVOID_EARNED_DEVELOPMENT_EVIDENCE` only when all three are true:

- mean monthly AUC 95% block-bootstrap lower bound > 0.50;
- top predicted-risk quintile precision lower bound > 20%;
- top predicted-risk quintile mean SPY-relative-return upper bound < 0.

Otherwise the result is `AVOID_NOT_EARNED`.

## Important research limitation

The year-by-year predictions are point-in-time with respect to model training, but R1, R2 and R2.1 used the same 2016–2026 historical era to discover and select the feature set.

Therefore R3 is **development evidence, not a pristine untouched holdout**. We will still need a later untouched/prospective validation stage before changing the live model.

## Outputs

The workflow creates `clean-sheet-r3-<run id>` containing:

- `r3_predictions.csv.gz`
- `r3_coefficients.csv`
- `r3_fold_manifest.csv`
- `r3_data_coverage.csv`
- `r3_monthly_ranking_metrics.csv`
- `r3_monthly_avoid_metrics.csv`
- `r3_ranking_summary.csv`
- `r3_avoid_summary.csv`
- `r3_ranking_improvement.csv`
- `r3_evidence_gates.csv`
- `r3_yearly_metrics.csv`
- `r3_candidate_inputs.csv`
- `r3_summary.json`
- `r3_spec_frozen.json`
- `r3_report.md`
- four PNG research charts in `charts/`

## Install and run

1. Apply this ZIP on `research/clean-sheet-predictive` only.
2. Commit and push with GitHub Desktop.
3. GitHub → Actions → **10-Year Market Data Backfill**.
4. Start a **new** run on the research branch.
5. Keep 10 backtest years + 1 warm-up year.
6. The workflow will rebuild R1 → R2 → R2.1 → R3 in sequence.
7. Download `clean-sheet-r3-...` and upload that ZIP here.

The R1, R2, and R2.1 artifacts continue to be produced separately in the same run.
