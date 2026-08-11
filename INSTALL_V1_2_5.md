# v1.2.5 AI validation retry

Fixes the two validation issues seen in the v1.2.4 run:

- RSP may call SPY a benchmark because SPY is the exact paired ticker and is
  present in benchmark_context.
- A real validation failure now triggers one automatic correction retry before
  the system falls back.

The 5-bar relative-strength rule remains strict. The retry tells Gemini to use
5-bar score change and 20/63-bar relative strength instead.

Install:
1. Copy this patch into the repository root.
2. Merge `src`; replace matching files.
3. Commit: `Fix AI validation retry and pair semantics v1.2.5`
4. Push origin.
5. Rerun Manual Market Rotation Test.
