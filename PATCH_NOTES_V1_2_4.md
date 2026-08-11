# v1.2.4 — AI grounding and validation

This patch does not change the quantitative rotation engine.

## New guardrails

- Prohibits unsupported capital/fund-flow claims from OHLCV/CMF data.
- Prohibits unsupported "mega-cap" labels unless supplied by the input data.
- Requires state-specific AI findings to match Python's deterministic state.
- Requires every AI ticker and related ticker to exist in the supplied payload.
- Requires each finding to include explicit numeric/metric evidence.
- Requires PAIR findings to name the exact paired ticker.
- Prevents PAIR signals from being described as peer-group ranks.
- Distinguishes 5-bar score changes from 20/63-bar relative strength.

## Deterministic post-AI validator

`src/ai/validator.py` runs after the provider returns structured JSON and before
the result reaches the dashboard or consensus engine.

If validation fails:

1. the provider result is marked `validation_error`;
2. the invalid AI analysis is excluded from consensus;
3. the dashboard falls back to deterministic analysis if no other provider
   passes validation;
4. validation errors are written to the provider JSON for debugging.

The dashboard provider card now displays the validation status/version.
