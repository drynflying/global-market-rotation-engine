# Patch 6 — Confirmed Trend State Layer

## Purpose
Keep daily scores responsive while preventing the dashboard recommendation state from flipping on short-lived daily noise.

## Core rule
- `rotation_state_raw`: today's reactive mathematical condition.
- `rotation_state_confirmed`: canonical trend state.
- A different raw state must occur for **3 consecutive observations** before it replaces the confirmed state.
- `rotation_state` remains the backwards-compatible canonical alias and now equals `rotation_state_confirmed`.
- Pending changes remain visible with `pending_rotation_state`, `pending_state_days`, and `state_confirmation_bars`.

The same 3-observation confirmation layer is applied to PAIR signals:
- `pair_signal_raw`
- `pair_signal_confirmed`
- `pair_signal` = confirmed canonical pair signal
- pending pair fields remain visible.

## Full replacement files
Replace these files completely in GitHub:
1. `src/calculate_trends.py`
2. `src/run_pipeline.py`
3. `src/build_ai_input.py`
4. `src/build_dashboard.py`
5. `src/ai/prompt.py`
6. `src/ai/validator.py`
7. `src/ai/normalizer.py`
8. `src/ai/fallback.py`

Do not edit `src/calculate_metrics.py`, score weights, or configuration weights for this patch.

## Expected behavior on the 2026-08-11 repository snapshot
Raw state counts remain the same as before Patch 6:
- Emerging: 18
- Accelerating: 5
- Reaccelerating: 5
- Persistent leader: 11
- Neutral: 30
- Weakening: 33
- Rotation out: 9

Confirmed state counts using the 3-observation rule become:
- Emerging: 14
- Accelerating: 2
- Reaccelerating: 2
- Persistent leader: 11
- Neutral: 34
- Weakening: 42
- Rotation out: 6

45 of 111 scored signals currently have a raw condition different from the confirmed trend; 30 are on day 1 of confirmation and 15 are on day 2. This is expected on the historical snapshot and is the reason the dashboard now distinguishes current condition from confirmed trend.

Examples:
- IWM: confirmed `WEAKENING`, raw `REACCELERATING`, pending 1/3.
- SMH: confirmed `WEAKENING`, raw `ROTATION_OUT`, pending 2/3.
- IWD: confirmed `ACCELERATING`, raw `EMERGING`, pending 1/3.
- REET: confirmed pair `PAIR_LEADING` and confirmed trend `WEAKENING`.

## Historical audit result
Using 111 scored signals across 242 scored trading days:
- Average raw state changes per ticker: 80.7
- Average confirmed state changes per ticker: 17.4
- State churn reduction: 78.5%
- Median confirmed-state run: 10 trading days
- Only ~9.8% of confirmed runs last 3 days or less

## UI behavior
The dashboard now:
- labels table states as **Confirmed Trend**;
- shows a small `Current: <raw state> · n/3` line when today's raw condition differs;
- adds a **Pending trend confirmations** deterministic panel;
- adds a **Pending pair confirmations** deterministic panel;
- explains the 3-observation confirmation rule near the top of the page;
- sends confirmed trend states to Gemini while keeping raw conditions available as early-warning evidence.

## Acceptance checks after GitHub Actions runs
1. Workflow is green.
2. Gemini validator reports `v1.2.8` (attempt 1 or corrected attempt 2 is acceptable).
3. The top of the dashboard contains the Trend confirmation explanation.
4. Ranking tables say `Confirmed Trend` instead of `State`.
5. At least some rows show a smaller `Current: ... · 1/3` or `2/3` line when raw and confirmed states differ.
6. Deterministic attention flags include `Pending trend confirmations`.
7. Pair table says `Confirmed Pair Signal` and `Confirmed Trend`.
8. Scores, score changes, relative strength, and score formula remain unchanged from the same market-data snapshot.

## Important non-goal
This patch does **not** add a minimum materiality threshold to PAIR_LEADING / PAIR_LAGGING. It stabilizes pair-state changes with persistence, but the raw pair classifier is still sign-based. That can be audited separately later.
