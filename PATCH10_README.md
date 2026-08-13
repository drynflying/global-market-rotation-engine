# Patch 10 — Cross-Asset Signals vs SPY

## Purpose

Patch 10 makes the five new reference/watchlist assets visible on the dashboard:

- GLD — Gold
- SLV — Silver
- URA — Uranium
- CPER — Copper
- REMX — Rare Earths

It is intentionally a **display-only dashboard patch**. The five assets remain
`REFERENCE`, `rank_eligible=False`, and outside the production scoring and weekly
recommendation models.

## File replaced

- `src/build_dashboard.py`

## Dashboard behavior

A new **Cross-Asset Signals vs SPY** section appears immediately below the weekly
high-conviction panel.

For every `score_mode=REFERENCE` row whose `universe=CROSS_ASSET`, the dashboard shows:

- ticker and exposure;
- exposure type;
- configured benchmark;
- 20-bar relative strength vs benchmark;
- 63-bar relative strength vs benchmark;
- CMF20;
- simple price-trend posture from close/SMA50/SMA200;
- deterministic relative posture:
  - LEADING: RS20 > 0 and RS63 > 0
  - IMPROVING: RS20 > 0 and RS63 <= 0
  - WEAKENING: RS20 < 0 and RS63 >= 0
  - LAGGING: RS20 < 0 and RS63 < 0
  - MIXED when incomplete/neutral;
- a 63-observation sparkline of 63-bar relative strength with a zero benchmark line.

## Important isolation

Patch 10 does **not**:

- create a cross-asset peer group;
- assign a 0–100 rotation score to these assets;
- alter the 111 scored-signal count;
- change Patch 6 scoring or confirmation rules;
- feed the assets into Patch 7 Path Risk;
- make them eligible for weekly FAVOR/AVOID recommendations;
- change Gemini prompts/validation;
- change Patch 9 scheduling or Pages deployment.

The section is data-driven. Future `REFERENCE` rows added with
`universe=CROSS_ASSET` will automatically appear without another dashboard-code change.

## Install

Unzip this package over the repository root, replace `src/build_dashboard.py`,
commit in GitHub Desktop, push to `main`, then start a **new**
`Market Rotation Daily + Pages` workflow run.

After the run, confirm:

1. Scored signals remains 111.
2. The new Cross-Asset Signals vs SPY section is visible.
3. GLD, SLV, URA, CPER, and REMX appear.
4. No 0–100 peer score/rank is shown for those assets.
5. The published Pages site updates after the deploy job.
