# v1.2.6 AI validation stability

This patch fixes the remaining false-fallback behavior seen after v1.2.5.

Changes:
- Python now owns the methodology note, so model paraphrasing cannot fail an
  otherwise valid analysis.
- Horizon validation reports the exact field containing an invalid 5-bar
  relative-strength statement.
- Retry feedback no longer repeats the invalid phrase back to Gemini.
- The validator still rejects 5-bar relative-strength claims in actual
  analytical content.
- Quantitative scoring/state logic is unchanged.

Install:
1. Copy this patch into the repository root.
2. Merge `src`; replace matching files.
3. Commit: `Stabilize AI validation and retry v1.2.6`
4. Push origin.
5. Rerun Manual Market Rotation Test.
