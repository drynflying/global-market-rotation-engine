# AI Integration v1.2

Version 1.2 makes the AI layer provider-agnostic while activating Gemini first.

## Architecture

The deterministic Python engine remains authoritative. It writes `results/ai_input.json`.
Every enabled AI provider receives the same input and the same output schema.

Current provider status:

- Gemini: implemented
- Claude: adapter placeholder
- OpenAI: adapter placeholder

Provider outputs are written separately:

- `results/ai/gemini.json`
- `results/ai/consensus.json`
- `results/ai/manifest.json`
- `results/ai/history/YYYY-MM-DD/...`

The combined dashboard-facing bundle remains:

- `results/ai_analysis.json`

## GitHub variables and secrets

Repository variable:

- `AI_PROVIDERS=gemini`

Optional repository variable:

- `GEMINI_MODEL=gemini-3.6-flash`

Repository secret:

- `GEMINI_API_KEY=<your Google AI Studio API key>`

Do not put API keys in code, CSV files, commits, issue text, or chat messages.

## Future multi-model activation

When adapters are added later, the same architecture will accept:

`AI_PROVIDERS=gemini,claude,openai`

Each provider will receive the same deterministic payload independently.
The Python consensus layer will compare structured findings after all providers
finish. One provider never sees another provider's answer.

## Failure behavior

AI is enrichment, not a pipeline dependency.

- If Gemini succeeds, the dashboard displays Gemini analysis.
- If Gemini fails, the quantitative pipeline and dashboard still complete with
  a deterministic fallback.
- `results/run_summary.json` records requested, successful, and failed providers.
