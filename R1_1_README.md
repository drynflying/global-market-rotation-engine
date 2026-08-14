# Research Patch R1.1 — Reliable Historical Data Acquisition

R1.1 fixes the source-data completeness weakness discovered after R1.

The R1 point-in-time feature/outcome engine passed its leakage and alignment tests, but
different GitHub runs returned different subsets of the 167 configured Yahoo/yfinance
symbols. R1.1 makes historical-universe completeness a hard research invariant.

## Files replaced

- `.github/workflows/backfill_10y.yml`
- `src/backfill_10y.py`
- `research/clean_sheet_predictive/build_research_dataset.py`
- `research/clean_sheet_predictive/validate_point_in_time.py`
- `research/clean_sheet_predictive/tests/test_r1.py`

## File added

- `research/clean_sheet_predictive/tests/test_r1_1_backfill.py`

## Retrieval strategy

The downloader now preserves every successful result and retries only unresolved symbols:

1. Small batch pass with yfinance threads disabled.
2. Second pass in even smaller batches.
3. Up to three individual full-window attempts with increasing delay.
4. Final segmented historical recovery for any remaining symbols.

Successful data from an earlier pass is never discarded because a later request fails.

## Hard completeness / quality gate

Before R2 is allowed to run, each active `query_ohlcv` symbol must have usable history
unless it is explicitly allowlisted with `BACKFILL_ALLOWED_MISSING`.

Default quality requirements:

- symbol exists in the downloaded data;
- at least 260 daily bars;
- at least 85% weekday coverage between its first and last available observations;
- no internal calendar gap greater than 21 days;
- latest observation no more than 7 days stale versus the newest observation in the dataset.

The thresholds are deliberately broad enough to tolerate exchange holidays while rejecting
materially incomplete chunks.

## Diagnostics

The backfill artifact now contains:

- `ohlcv_10y.csv`
- `backfill_summary.json`
- `fetch_failures.csv`
- `fetch_attempts.csv`
- `ticker_quality.csv`
- `rotation_universe_snapshot.csv`

If completeness still fails, the workflow fails **after writing diagnostics** and uploads
the 10-year artifact anyway so the exact unresolved symbols and request stages can be
reviewed.

## R1 validation change

R1's `validation_report.json` now contains an explicit `universe_completeness` section.
The point-in-time dataset is no longer allowed to report overall `ok` while configured
active symbols are silently missing.

## Install / run

1. Apply this ZIP on `research/clean-sheet-predictive` only.
2. Commit and push with GitHub Desktop.
3. GitHub → Actions → **10-Year Market Data Backfill**.
4. Start a **new** run on the research branch with 10 backtest years and 1 warm-up year.
5. If the run is green, download both artifacts and upload the `clean-sheet-r1-...` artifact here.
6. If it is red, download and upload `market-rotation-10y-ohlcv-...`; the diagnostics will tell us exactly what remains unresolved.

Do not add an allowlist simply to make the workflow green. We should use it only after
confirming that a symbol genuinely cannot support the research dataset.
