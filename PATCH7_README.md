# Patch 7 — Prospective Shadow Test Infrastructure

This patch is intentionally **research-only**. It does not alter the production dashboard's scoring, states, AI analysis, or visible recommendations.

## Files in this replacement package

### Replace

- `src/common.py`
- `src/run_pipeline.py`

### Add

- `src/research_shadow.py`
- `research/path_risk_v1/model_spec.json`
- `research/path_risk_v1/README.md`
- `PATCH7_README.md`

## What happens after you merge and push

Run the normal **Manual Market Rotation Test** GitHub Action.

The pipeline will continue building the dashboard exactly as before and will additionally create:

- `results/research_shadow_history.csv`
- `results/research_shadow_outcomes.csv`
- `results/research_shadow_status.json`

These files are automatically included by the workflow's existing `git add results/` command.

## Important

The shadow logger is deliberately non-blocking. If the research module has a problem, the normal dashboard pipeline still completes, while `run_summary.json` records the shadow error.

The first captured prediction for each model/date/ticker is immutable. Same-market-date reruns do not replace it.
