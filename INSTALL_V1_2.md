# Install v1.2

1. In GitHub Desktop, select the market-rotation repository.
2. Choose Repository -> Show in Finder.
3. Unzip this patch.
4. Copy everything inside the patch folder into the repository root.
5. Choose Merge/Replace when macOS asks.
6. Do not delete `data/`, `results/`, or `config/`.
7. In GitHub Desktop commit:
   `Add provider-agnostic AI architecture v1.2`
8. Push origin.
9. Then configure the Gemini GitHub secret/variable as described in
   `AI_INTEGRATION_V1_2.md`.
10. Run `Manual Market Rotation Test`.

The workflow remains safe if Gemini is unavailable: it will generate the
deterministic fallback dashboard and record the AI failure in run_summary.json.
