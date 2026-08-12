# Weekly Recommendation Shadow V1

This research specification is frozen for prospective testing beginning in August 2026.

The weekly layer runs only after the final weekly market close (normally Friday after 4:15 PM New York time). It compresses the daily rotation history into a small deterministic finalist set and allows Gemini to act only as a skeptical committee that can reject those finalists.

The output is **shadow-only**. It does not change the visible dashboard, Patch 6 scores/states, daily Gemini analysis, or Patch 7 path-risk shadow research.

## Key rules

- Zero, one, or at most two actions per week.
- At most one `FAVOR` and one `AVOID` action.
- Every recommendation is stamped with the actual market-data recommendation date.
- Friday holidays store the prior Thursday market date plus the Friday `week_ending_date`.
- `Evidence Confidence: HIGH` means strong deterministic evidence agreement; it is **not** a probability of correctness.
- Gemini may reject finalists but cannot invent a ticker, change direction, upgrade confidence, or create unsupported numeric claims.
- Same-date successful recommendations are immutable.
- Outcome tracking at 21/63/84/126 bars is automatic.

## Generated files

- `results/weekly_recommendation_shadow_history.csv`
- `results/weekly_recommendation_shadow_outcomes.csv`
- `results/weekly_recommendation_shadow_status.json`
- `results/weekly_recommendation_shadow_latest.json`
- `results/weekly_recommendation_ai/<recommendation-date>.json`

These files are committed automatically by the existing workflow because it already commits the `results/` directory.
