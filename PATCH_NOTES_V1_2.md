# Version 1.2 — Provider-agnostic AI foundation

- Refactors Gemini-specific analysis into `src/ai/`.
- Adds a shared structured output schema for independent AI analysts.
- Adds Gemini adapter using the Google GenAI SDK structured JSON output.
- Sets Gemini default model to `gemini-3.6-flash`.
- Adds Claude and OpenAI adapter placeholders without activating them.
- Adds independent per-provider result files.
- Adds deterministic model-consensus logic for future 2+ provider runs.
- Adds dated AI result history under `results/ai/history/`.
- Keeps `results/ai_analysis.json` as the dashboard-facing combined bundle.
- Adds AI provider success/failure fields to `run_summary.json`.
- Updates dashboard to show AI provider status and future model consensus.
- Corrects dashboard summary cards to the v1.1 rotation-state names.
- Preserves deterministic fallback behavior when no AI provider succeeds.
- Makes no changes to the v1.1 quantitative scoring engine.
