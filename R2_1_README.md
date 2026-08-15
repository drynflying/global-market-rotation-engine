# Research Patch R2.1 — Signal Independence & Actionability Audit

R2.1 is a cleanup/audit phase between R2 and R3.

R2 found genuine predictive structure, but several "different" robust features were
mathematically equivalent rankings of the same underlying signal. R2.1 removes that false
signal count and asks a stricter investment question:

> Does a feature only separate better from worse ETFs, or does it actually support a
> historically positive FAVOR side and/or a historically negative AVOID side versus SPY?

## What R2.1 does

### 1. Within-date redundancy audit

For all 59 cross-sectional R1 features, R2.1 calculates pairwise Spearman rank correlation
inside each month-end ETF cross-section.

This correctly catches relationships that pooled-across-time correlation can miss.

Example: on any given date, `ret_126` and `rs_spy_126` have identical cross-sectional ranks
because the same SPY return is subtracted from every ETF.

### 2. Exact-equivalence collapse

Features whose absolute within-date rank correlation is effectively 1.0 in at least 95% of
overlapping months are placed in the same exact-equivalence cluster.

Representative preference is frozen:

- raw/interpretable feature before `cs_*` percentile transforms;
- raw return before same-date `rs_spy_*` when rankings are identical;
- broader data coverage;
- then stronger R2 evidence.

### 3. Near-redundancy audit

A pair is considered near-redundant when:

- median within-date absolute Spearman >= 0.90; and
- absolute Spearman >= 0.85 in at least 80% of overlapping months.

R3 candidate selection greedily keeps the stronger evidence feature and documents excluded
near-duplicates.

### 4. Strict FAVOR vs AVOID actionability

For every R2 evidence candidate (ROBUST / PROMISING / WEAK), R2.1 orients the feature using
the R2 IC direction and evaluates the favorable and unfavorable 10% extremes each month.

For discrete features such as `above_sma_200`, the max/min groups are used instead.

`FAVOR_ACTIONABLE` requires BOTH:

- the 95% 12-month block-bootstrap CI for favorable-side mean SPY-relative return > 0; and
- the 95% CI for favorable-side SPY-beating rate > 50%.

`AVOID_ACTIONABLE` requires the symmetric unfavorable-side tests below 0 and below 50%.

If only the FAVOR-minus-AVOID spread is statistically positive, the feature is labeled
`SEPARATION_ONLY`.

This is intentionally stricter than R2's ranking evidence.

## Outputs

The workflow creates `clean-sheet-r2-1-<run id>` with:

- `within_date_redundancy.csv`
- `exact_equivalence_clusters.csv`
- `actionability_audit.csv`
- `actionability_monthly.csv`
- `nonredundant_candidate_set.csv`
- `r2_1_summary.json`
- `r2_1_spec_frozen.json`
- `r2_1_report.md`

## R3 rule

R3 will be allowed to consider only features marked `SELECTED` in
`nonredundant_candidate_set.csv`.

R2.1 still does **not** fit a model and does not make anything production-eligible.

## Install / run

1. Apply this ZIP on `research/clean-sheet-predictive`.
2. Commit and push with GitHub Desktop.
3. Start a new **10-Year Market Data Backfill** run on the research branch.
4. Keep 10 years + 1 warm-up year.
5. Upload the resulting `clean-sheet-r2-1-...` artifact here.

The workflow continues to generate R1 and R2 artifacts in the same run.
