# Patch 8 — Friday Weekly Recommendation Shadow Layer

Patch 8 adds the prospective weekly synthesis layer discussed after the Weekly High-Conviction Recommendation Study.

## Replace

- `src/common.py`
- `src/run_pipeline.py`

## Add

- `src/weekly_recommendation_shadow.py`
- `research/weekly_recommendation_v1/spec.json`
- `research/weekly_recommendation_v1/README.md`
- `PATCH8_README.md`

## Behavior

The existing daily dashboard continues to run exactly as before.

On a normal Friday run after 4:15 PM New York time, Patch 8 additionally:

1. Uses the full week's daily Patch 6 history to create deterministic `FAVOR` and `AVOID` finalists.
2. Restricts selectable finalists to those with frozen high evidence agreement.
3. Sends only those selectable finalists to Gemini as a skeptical investment committee.
4. Allows Gemini to return zero, one, or at most two actions, with at most one per direction.
5. Saves the actual recommendation market date and immutable evidence used at that time.
6. Automatically updates 21/63/84/126-bar forward outcomes as they mature.

`Evidence Confidence: HIGH` is intentionally categorical. It means the deterministic evidence is strongly aligned; it does **not** mean an estimated probability of success.

## Important

This is still a **shadow** layer. It does not add a visible Highly Recommended card yet and does not change Patch 6, Patch 7, the daily AI analysis, or the dashboard ranking. The goal is to accumulate clean prospective weekly recommendations before promotion to the UI.
