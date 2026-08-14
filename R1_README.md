# Research Patch R1 — Clean-Sheet Predictive Dataset Engine

R1 is the first clean-sheet research patch. It does **not** create a predictive model.
It creates the point-in-time dataset and validation rules that all later 1M/3M/6M/9M
research must use.

## Important audit finding

The branch ZIP supplied for R1 contains `data/ohlcv_history.csv` with **83,747 rows,
167 tickers, and dates from 2024-08-12 through 2026-08-12** — about two years, not the
10-year backfill used in prior research.

That is not a blocker. The existing `backfill_10y.yml` produces the long-history dataset as
a GitHub Actions artifact rather than committing it to the repo. R1 therefore extends that
existing workflow on the research branch: it downloads 10 research years + one warm-up year
and immediately builds/validates R1 from the downloaded file.

## Files added

- `research/__init__.py`
- `research/clean_sheet_predictive/__init__.py`
- `research/clean_sheet_predictive/feature_definitions.py`
- `research/clean_sheet_predictive/build_features.py`
- `research/clean_sheet_predictive/build_outcomes.py`
- `research/clean_sheet_predictive/build_research_dataset.py`
- `research/clean_sheet_predictive/validate_point_in_time.py`
- `research/clean_sheet_predictive/split_rules.py`
- `research/clean_sheet_predictive/spec.json`
- `research/clean_sheet_predictive/README.md`
- `research/clean_sheet_predictive/tests/test_r1.py`
- `research/clean_sheet_predictive/outputs/.gitkeep`

## Files changed on the research branch

- `.github/workflows/backfill_10y.yml`
- `.gitignore`

No production scoring/dashboard/AI files are changed.

## R1 feature set

70 unweighted OHLCV-derived features across:

- momentum and relative strength;
- trend structure;
- price-cycle location;
- volatility/risk;
- volume/participation;
- SPY market regime, breadth, and cross-sectional dispersion.

R1 contains **no Patch 6 score, state, Path Risk, Gemini, or weekly recommendation input**.

## R1 outcomes

For 21, 63, 126, and 189 trading bars:

- absolute return;
- primary-benchmark relative return;
- SPY-relative return;
- maximum drawdown;
- maximum adverse/favorable excursion;
- benchmark-relative adverse/favorable excursion.

## No-look-ahead controls

R1 includes:

- future-OHLCV mutation leakage test;
- exact horizon end-date alignment checks;
- exact forward-return alignment checks;
- unit tests;
- `training_rows_as_of()` helper that excludes any label whose future horizon had not fully
  matured by the model-training cutoff.

## Local smoke validation performed for this patch

Using the two-year repo dataset:

- rows: 83,747
- tickers: 167
- features: 70
- future-mutation leakage test: PASSED
- leaking features: 0
- unit tests: 3/3 PASSED
- validation status: OK

This smoke run is only an integrity test. It is **not** the predictive study.

## How to run the real R1 study in GitHub

After replacing these files on the research branch and pushing:

1. Go to **Actions**.
2. Open **10-Year Market Data Backfill**.
3. Click **Run workflow**.
4. Select your **research/clean-sheet-predictive** branch.
5. Leave **10** backtest years and **1** warm-up year.
6. Run it.

When it finishes, download the artifact named approximately:

`clean-sheet-r1-<run id>`

It contains:

- `research_dataset.csv.gz`
- `feature_summary.csv`
- `outcome_summary.csv`
- `data_dictionary.csv`
- `validation_report.json`
- `dataset_manifest.json`

Upload that R1 artifact here. R2 will then perform univariate predictive discovery by
horizon before we fit any combined model.
