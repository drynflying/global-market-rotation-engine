# Patch 9.1 — AI Pair-Grounding Hardening + Cleaner Fallback

## Purpose

Patch 9.1 fixes the failure mode seen when Gemini generated a finding for a PAIR ticker
(such as IWD) without explicitly naming its configured paired ticker (IWF).

The deterministic validator was correct to reject that response. This patch does **not**
loosen the validator.

## Files replaced

- `src/ai/prompt.py`
- `src/ai/run_analysis.py`
- `src/ai/fallback.py`
- `src/build_dashboard.py`

## Changes

### 1. Exact validator errors are now supplied to the retry

Previously, the retry prompt stripped each validator error down to only the field/category
prefix. For example, Gemini could receive only `accelerating_rotations[1]` instead of the
important requirement that IWD must explicitly name IWF.

Patch 9.1 sends the exact deterministic validation errors to the correction retry.

### 2. PAIR grounding is re-emphasized on the retry

The correction prompt now explicitly requires every PAIR finding, in every finding
category, to:

- read the exact `paired_ticker` from the supplied dataset;
- explicitly name that ticker in the explanation;
- describe the relationship as ticker vs paired ticker;
- avoid generic benchmark language;
- avoid calling a paired security a benchmark unless the configured pair definition
  actually represents a benchmark comparison.

### 3. Deterministic validator remains unchanged

No validator rules are weakened or removed. Invalid AI output is still withheld.

### 4. Cleaner deterministic fallback on the public dashboard

If all requested AI providers fail validation, the dashboard now gives a concise message
that the AI interpretation was withheld while the quantitative calculations remain valid.

The dashboard provider card no longer prints the full Python exception text to users.
Detailed error information remains in the generated AI JSON diagnostics for debugging.

## Not changed

Patch 9.1 does not change:

- rotation scores or weights;
- Patch 6 confirmed-state logic;
- pair calculations;
- Patch 7 prospective Path Risk shadow test;
- Patch 8 weekly recommendation shadow logic;
- Patch 8.1 weekly recommendation UI;
- Patch 9 scheduling or GitHub Pages deployment;
- the newly added cross-asset tickers;
- the deterministic AI validator.

## Installation

Unzip this package over the repository root, replace the four files above, then:

1. Commit the changes in GitHub Desktop.
2. Push to `main`.
3. Start a **new** `Market Rotation Daily + Pages` workflow run.
4. Confirm Gemini either passes validation or, if it fails, the public fallback is concise.

