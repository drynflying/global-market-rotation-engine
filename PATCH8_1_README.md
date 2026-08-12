# Patch 8.1 — Visible Weekly High-Conviction Actions

This patch adds the Patch 8 weekly recommendation shadow result to the visible dashboard.

## What changes

- Adds a **Weekly High-Conviction Actions · Prospective Shadow** panel directly below the main AI summary.
- Always shows the actual **Recommendation date** for the latest weekly recommendation.
- Keeps the latest Friday recommendation visible throughout the following week.
- Shows the next scheduled Friday evaluation date.
- Supports:
  - 0 actions: `No high-confidence action this week`
  - 1 action
  - 2 actions
- Shows `FAVOR` / `AVOID`, ticker, exposure, AI committee rationale, and deterministic `Evidence Confidence`.
- Clearly states that Evidence Confidence is **not** a calibrated probability of investment success.
- Before the first Friday recommendation exists, the panel remains visible and shows the next scheduled evaluation date.
- Mobile layout is responsive.

## What does NOT change

- Patch 6 rotation score or confirmed-trend logic.
- Patch 7 prospective Path Risk shadow test.
- Patch 8 weekly finalist/AI committee logic.
- Gemini's normal daily dashboard commentary.
- GitHub Actions workflow.
- The weekly recommendation model, thresholds, confidence rules, or outcome tracking.

## Install

Unzip this package over the root of the repository so:

`src/build_dashboard.py`

replaces the existing file.

Then commit and push with GitHub Desktop and run the normal pipeline.

Because the current market date is Wednesday, August 12, 2026, the dashboard should immediately show the weekly panel with:

`No weekly recommendation has been generated yet.`

and:

`Next scheduled evaluation: Aug 14, 2026 after close`

After the Friday capture succeeds, that placeholder is automatically replaced by the dated weekly recommendation and remains visible on subsequent daily dashboard runs.
