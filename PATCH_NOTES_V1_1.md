# Version 1.1 quantitative-engine fixes

This update should be applied before Gemini is enabled.

## Changes

1. Peer-group correction
   - SKYY, CIBR, and OIH moved from `GLOBAL_BROAD` to `US_INDUSTRY_THEME`.
   - Their `universe` field remains unchanged; only their scoring peer group changes.

2. New `score_mode`
   - `CROSS_SECTIONAL`: percentile scoring against true peers; minimum 3 members.
   - `PAIR`: direct relative relationship score.
   - `REFERENCE`: benchmark / confirmation only.

3. Pair signals
   - IWF vs IWD
   - IWD vs IWF
   - RSP vs SPY
   - REET vs XLRE

4. Rotation states
   - EMERGING
   - ACCELERATING
   - PERSISTENT_LEADER
   - REACCELERATING
   - WEAKENING
   - ROTATION_OUT
   - NEUTRAL

5. Formula version
   - `score_formula_version` updated to `v1.1`.

The existing raw OHLCV history does not need to be deleted. On the next manual run,
the engine recalculates the historical analytics using the v1.1 configuration and logic.
