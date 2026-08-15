# R3.0.1 — Read-Only NumPy Array Hotfix

The GitHub Actions run failed during the R3 unit tests with:

`ValueError: output array is read-only`

The failure occurred in `_date_weights()` at:

`w *= len(w) / w.sum()`

Some pandas / NumPy combinations can return a read-only Series-backed NumPy array under
copy-on-write semantics. The R3 logic itself was not the problem; the test stopped before
the R3 research run began.

## Fix

`_date_weights()` now:

1. requests an owned NumPy array with `to_numpy(..., copy=True)`;
2. calculates the normalization factor separately; and
3. returns a new normalized array instead of mutating the array in place.

The mathematical weighting rule is unchanged. Each month/date still receives equal total
training weight regardless of how many ETFs are present on that date.

A regression test was added to verify:

- the returned array is writable;
- weights sum to the number of observations; and
- each date receives equal aggregate weight.

## Files replaced

- `research/clean_sheet_predictive/r3_walk_forward.py`
- `research/clean_sheet_predictive/tests/test_r3.py`

No workflow, model, feature, threshold, R1/R2/R2.1, or production-dashboard logic changes.

After applying this patch, commit and push it, then start a **new** 10-Year Market Data
Backfill workflow run. Do not use GitHub's "Re-run failed jobs" on the old run because that
would execute the old commit containing the bug.
