# Research Patch R1.2 — Reproducible yfinance Environment

The R1.1 workflow failed for the correct reason: the completeness gate rejected 13
tickers whose latest recovered observation was two years stale.

The GitHub log exposed the upstream cause:

- `requirements.txt` allowed any `yfinance>=0.2.66,<2`;
- the run installed **yfinance 1.6.0**;
- many `repair=True` downloads then failed with
  `ModuleNotFoundError("No module named 'sklearn'")`;
- R1.1 recovery still preserved enough historical segments to produce data for
  **167/167 symbols**, but 13 tickers ended at 2024-08-14 rather than 2026-08-14;
- the R1.1 quality gate correctly blocked R2.

R1.2 fixes the environment rather than weakening the quality gate.

## Files replaced

- `requirements.txt`
- `.github/workflows/backfill_10y.yml`

## Changes

1. Pin `yfinance[repair]==1.5.1` for reproducible research runs.
2. Add explicit `scikit-learn>=1.5,<2`.
3. Keep SciPy as an explicit dependency.
4. Add a workflow dependency-version check before the backfill begins.
5. Keep every R1.1 completeness and staleness gate unchanged.

The goal remains **167/167 usable symbols**. No allowlist is added.

## Why pin the downloader

A historical backtest should not silently change because a new yfinance release is
published between two research runs. R1.2 freezes the acquisition dependency so later R2,
R3, etc. can be reproduced from the same source behavior.

## Run

Apply this ZIP on `research/clean-sheet-predictive`, commit and push, then start a **new**
10-Year Market Data Backfill run with 10 years + 1 warm-up year.

If green, upload the new `clean-sheet-r1-...` artifact.

If red, upload the `market-rotation-10y-ohlcv-...` diagnostics artifact. R1.1 still uploads
diagnostics even when the completeness gate fails.
