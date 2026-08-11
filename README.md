# Market Rotation Dashboard

A personal, longer-horizon ETF rotation monitor.

## Current stage

The included GitHub Action is **manual only**. This is deliberate.

The first run will:

1. Read `config/rotation_universe.csv`.
2. Fetch approximately two years of daily OHLCV for enabled ETFs.
3. Store/merge raw OHLCV in `data/ohlcv_history.csv`.
4. Backfill the historical rotation metrics.
5. Save `results/rotation_history.csv` and `results/rotation_latest.csv`.
6. Build `results/ai_input.json`.
7. Use a deterministic fallback interpretation unless Gemini is enabled.
8. Build `docs/index.html`.
9. Commit the generated data/results/dashboard back to the repository.

This gives you immediate historical trend information on the first successful run rather
than waiting 20–63 trading days to accumulate history.

## Do not enable Gemini yet

The first test should run with no AI key. The deterministic dashboard will still build.

After the market-data and calculation pipeline is verified, Gemini can be enabled by adding:

- GitHub Actions repository secret: `GEMINI_API_KEY`
- GitHub Actions repository variable: `AI_PROVIDER` = `gemini`
- Optional repository variable: `GEMINI_MODEL` = `gemini-2.5-flash`

## Automatic daily scheduling

`.github/workflows/daily_rotation.yml.disabled` is included but intentionally disabled.

Once the manual workflow is proven, it can be renamed to:

`.github/workflows/daily_rotation.yml`

Do not enable it until the manual run is verified.

## Important

This project is a market-rotation research dashboard. It is not a trade execution system,
price-prediction system, or direct measurement of institutional ETF creations/redemptions.


## AI architecture (v1.2)

The AI layer is provider-agnostic. Enable providers with the GitHub Actions
repository variable `AI_PROVIDERS`.

Current implementation:

- `gemini` — active adapter
- `claude` — reserved adapter
- `openai` — reserved adapter

The quantitative engine is always authoritative. AI providers receive the same
curated `results/ai_input.json`, return the same structured schema, and cannot
change rotation scores or states.

See `AI_INTEGRATION_V1_2.md` for setup.
