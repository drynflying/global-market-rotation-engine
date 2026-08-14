# Clean-Sheet Predictive Research — R1

R1 builds a trustworthy point-in-time research dataset before any predictive model is fit.

## Why this exists

This branch intentionally starts over from OHLCV. R1 does not consume the production
rotation score, confirmed states, pair signals, Path Risk model, Gemini output, or weekly
recommendation logic.

The only question R1 prepares us to study is:

> Using information available on date T, which OHLCV-derived variables predict investment
> outcomes 21, 63, 126, and 189 trading days later?

## Feature families

- absolute and benchmark-relative momentum: 21/63/126/189/252 bars;
- trend structure: SMA20/50/100/200, distance and slopes;
- price-cycle location: trailing 63/126/252 highs/lows and 252-bar bars-since extremes;
- risk: realized/downside volatility, volatility acceleration, ATR;
- participation: CMF, dollar-volume trend, volume z-score, signed-volume balance, normalized OBV change;
- market regime: SPY trend/momentum/volatility/drawdown plus cross-sectional breadth and dispersion;
- a small number of same-day cross-sectional percentiles, left unweighted for later testing.

## Outcomes

For 21, 63, 126, and 189 bars R1 attaches:

- absolute forward return;
- configured primary-benchmark relative return;
- SPY-relative return;
- maximum drawdown over the future holding path;
- maximum adverse/favorable excursion;
- primary-benchmark and SPY-relative adverse/favorable excursion.

These are labels only. They are never used to calculate features.

## Leakage protection

R1 includes a future-mutation test: future OHLCV is deliberately changed to absurd values
and all features at/before a historical cutoff must remain identical. It also checks
outcome/end-date alignment.

A later model may use a row for horizon H only after that row's `outcome_end_date_H` has
passed. R2/R3 must enforce that eligibility rule in every walk-forward split.

## Run locally against repo history

```bash
python -m research.clean_sheet_predictive.build_research_dataset \
  --input data/ohlcv_history.csv \
  --config config/rotation_universe.csv \
  --output research/clean_sheet_predictive/outputs
```

The repo history currently serves as a smoke test only if it contains less than the full
research history.

## Full 10-year run

Use GitHub Actions → **10-Year Market Data Backfill** → **Run workflow** and choose the
`research/clean-sheet-predictive` branch (or your actual research-branch name). The workflow
already exists on `main`, so GitHub exposes the Run workflow control; the research-branch
version adds R1 after the backfill. It downloads 10 years plus one warm-up year, builds R1,
validates it, and uploads both the OHLCV backfill artifact and a `clean-sheet-r1-*` artifact.

## Outputs

- `research_dataset.csv.gz`
- `feature_summary.csv`
- `outcome_summary.csv`
- `data_dictionary.csv`
- `validation_report.json`
- `dataset_manifest.json`

The generated outputs are intentionally not committed to the branch; GitHub Actions stores
them as a research artifact.

## Walk-forward split helper

`split_rules.py` provides `training_rows_as_of()`. Later R2/R3 models must use it (or an
equivalent stricter rule) so a 6- or 9-month label cannot enter training before the full
future outcome is actually known.
