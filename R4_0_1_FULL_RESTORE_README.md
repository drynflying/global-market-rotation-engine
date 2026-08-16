# R4.0.1 — Full Research-Branch Restore

The R4 failure was caused by packaging, not by the R4 model logic.

The first R4 research ZIP contained only the new R4 files under
`research/clean_sheet_predictive/`. If that folder was copied as a replacement rather
than merged, foundational R1 files such as `build_features.py` and `build_outcomes.py`
were removed.

GitHub then failed while importing `test_r4.py` with:

`ModuleNotFoundError: No module named 'research.clean_sheet_predictive.build_features'`

This restore ZIP contains the COMPLETE clean-sheet research tree from R1 through R4,
plus the latest historical-backfill workflow/dependency files.

It includes:
- R1 feature/outcome/dataset engine
- R1.1 completeness fixes
- R1.2 reproducible dependency environment
- R2 univariate discovery
- R2.1 independence/actionability audit
- R3 walk-forward baseline
- R3.0.1 read-only NumPy hotfix
- R4 prospective tracker/model registry

The reconstructed branch was validated using the same command shown in the failing
GitHub Actions step:

`python -m unittest discover -s research/clean_sheet_predictive/tests -v`

All clean-sheet tests passed before this ZIP was produced.

INSTALL THIS ZIP ON `research/clean-sheet-predictive` ONLY.

After copying it into the repository:
1. Commit and push with GitHub Desktop.
2. Start a NEW workflow run.
3. Do not use "Re-run failed jobs" on the old failed commit.

The separate R4 MAIN scheduler ZIP does not need to be changed for this error.
