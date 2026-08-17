# R4.2 — Integrate R4 into the Main Market Rotation Dashboard

R4.1 successfully created the separate `r4.html` research page, but the normal
`docs/index.html` dashboard did not surface R4 at all. R4.2 fixes that.

## What appears on the main dashboard

A new **Clean-Sheet Forward Research / R4 Prospective Evidence** panel is rendered directly
below the main production KPI cards. It shows:

- 1M / 3M / 6M / 9M evidence status
- evaluated prospective months
- Ridge IC
- equal-weight IC
- AVOID AUC
- AVOID precision lift
- latest prospective cohort
- prediction-row and matured-outcome counts
- a button to open the full `r4.html` dashboard

The 6M card is visually marked as the primary research horizon.

## Research / production boundary

The integration is read-only. It reads `docs/r4-data/r4_status.json` and
`docs/r4-data/r4_assessment.json`.

It does NOT change the production rotation score, confirmed states, weekly recommendations,
AI analysis, Cross-Asset signals, or any R4 model.

## Install

Install this ZIP on `main`.

Replace `src/build_dashboard.py`, then commit and push in GitHub Desktop.

After that, run the normal production Market Rotation workflow once. That regenerates
`docs/index.html` using the new generator and publishes the integrated dashboard.

No 10-year backtest or R4 research rerun is required.
