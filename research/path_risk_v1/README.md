# Path Risk V1 — Frozen Prospective Shadow Test

**Model version:** `PATH_RISK_V1_2026-08-12`

This folder freezes the first prospective path-risk research models for the Market Rotation Dashboard.

## What this changes

Nothing in the production rotation logic.

Patch 6 remains the control and still owns:

- rotation score;
- raw condition;
- three-day confirmed trend;
- AI input and commentary;
- dashboard rankings and visible recommendations.

The shadow logger records two research candidates after each normal pipeline run:

1. **Universal Path Risk** — one frozen relative-structure model for all eligible cross-sectional groups.
2. **Hybrid Challenger** — the same universal model except `US_INDUSTRY_THEME`, which uses one frozen specialized model.

The models predict the historical **Primary Investable Rotation Outcome** used in the research study:

- median benchmark-relative return across 63 / 84 / 126 trading bars >= +2%; and
- 126-bar benchmark-relative max drawdown no worse than -10%.

The frozen coefficients were fit once for this prospective test using the predefined relative-structure model recipe and all fully matured monthly research anchors available at the freeze date, including the last fully matured partial-month anchor on 2026-02-09. The runtime model is completely frozen. It does **not** retrain or tune itself.

## LOW_TRUST rule

Within each date and `rotation_group`, lower predicted success is worse.

A name is `LOW_TRUST` when:

- the group has at least five currently scoreable members; and
- its predicted-success rank percentile is in the bottom 20% of that group.

`US_MARKET_BREADTH` currently has only three members, so the quintile rule is deliberately disabled there.

The historical research evaluated the veto primarily among Patch 6's top-20% score candidates. The shadow file therefore records both the risk flag and whether it would have vetoed a Patch 6 top-20% candidate.

## Files created by normal pipeline runs

- `results/research_shadow_history.csv` — append-only predictions. The first prediction for a model/date/ticker is never overwritten by same-date reruns.
- `results/research_shadow_outcomes.csv` — realized 21/63/84/126-bar outcomes as they become available. This is separate from the immutable prediction log.
- `results/research_shadow_status.json` — health/status summary for the latest pipeline run.

The existing GitHub Action already commits `results/`, so no workflow change is required.

## Prospective-test discipline

Do not change the features, coefficients, thresholds, or group routing because of early results. The purpose of this patch is to obtain genuinely forward evidence.

Suggested checkpoints:

- ~1 month: data-integrity / 21-bar diagnostics only;
- ~3 months: first directional evidence;
- ~6 months: first fully matured primary-outcome comparison;
- ~12 months: materially stronger prospective evidence.
