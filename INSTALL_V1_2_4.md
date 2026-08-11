# Install v1.2.4 AI-quality patch

1. In GitHub Desktop select your market-rotation repository.
2. Choose **Repository -> Show in Finder**.
3. Unzip this patch.
4. Copy everything inside the patch folder into the repository root.
5. Choose **Merge** for `src` when prompted and **Replace** for matching files.
6. Commit:
   `Add deterministic AI grounding validator v1.2.4`
7. Push origin.
8. Re-run **Manual Market Rotation Test**.

Expected provider card after a valid Gemini response:

- Gemini: SUCCESS
- validation: passed · v1.2.4

No scoring formulas, ETF groups, OHLCV data, or v1.1 rotation states are changed.
