# Research Patch R2 — Univariate Predictive Discovery

R2 is the first actual clean-sheet predictive discovery phase.

It does **not** fit a combined model and it does **not** reuse the production rotation
score, states, Path Risk model, weekly recommendation logic, or Gemini.

## What R2 asks

For each point-in-time OHLCV-derived feature independently:

> Does this feature contain persistent information about an ETF's future outcome at
> 1, 3, 6, or 9 months?

The primary outcome is future ETF return relative to SPY.

Supplementary outcome diagnostics include:

- absolute forward return;
- configured-primary-benchmark relative return;
- maximum forward drawdown;
- maximum adverse relative excursion versus SPY.

## Primary methodology

R2 uses the last available trading observation in each calendar month as the primary
research anchor. This prevents hundreds of highly overlapping daily rows from being treated
as hundreds of independent pieces of evidence.

For ticker-level features, R2 calculates a monthly cross-sectional Spearman information
coefficient (IC) between the feature and future outcome across the ETF universe.

It also measures:

- high-minus-low outcome spread;
- positive-outcome hit-rate spread;
- quintile profiles;
- annual IC stability;
- fixed temporal subperiod ICs;
- 12-month moving-block bootstrap confidence intervals;
- feature redundancy / correlation.

## Market-regime variables

Market-regime features are identical for all ETFs on a given date, so they are **not**
pretended to be ETF-ranking features.

R2 evaluates them separately against:

- SPY's future absolute return;
- SPY's future max drawdown;
- the cross-sectional median ETF excess return versus SPY.

## Frozen discovery labels

The thresholds in `r2_spec.json` are fixed before the R2 study is run.

`ROBUST_CANDIDATE` requires:

- at least 60 monthly observations;
- absolute mean monthly IC >= 0.03;
- 95% 12-month block-bootstrap CI excluding zero;
- annual direction consistency >= 70%;
- high-minus-low spread agreeing with the IC direction.

`PROMISING` requires abs IC >= 0.02 with somewhat lower stability requirements.

`WEAK` requires abs IC >= 0.01 and a bootstrap CI excluding zero.

These labels are **discovery labels only**. They are not probabilities and do not make a
feature production-eligible.

## Outputs

The GitHub Action produces a new `clean-sheet-r2-<run id>` artifact containing:

- `feature_horizon_summary.csv`
- `univariate_outcomes_long.csv`
- `annual_ic.csv`
- `quintile_profiles.csv`
- `market_regime_context.csv`
- `feature_redundancy.csv`
- `family_summary.csv`
- `top_findings.json`
- `r2_validation.json`
- `r2_spec_frozen.json`
- `r2_report.md`

## Installation / run

1. Apply this ZIP on `research/clean-sheet-predictive` only.
2. Commit and push with GitHub Desktop.
3. Start a **new** `10-Year Market Data Backfill` workflow on the research branch.
4. Keep 10 backtest years + 1 warm-up year.
5. The workflow will rebuild the validated R1 dataset and then run R2.
6. Download and upload the `clean-sheet-r2-...` artifact here.

The full R1 artifact is still produced separately in the same run.

## What happens after R2

We will audit R2 before any feature combination.

R3 will only consider features that survive the R2 evidence/stability review, while also
removing redundant features so the same underlying signal is not counted multiple times.
