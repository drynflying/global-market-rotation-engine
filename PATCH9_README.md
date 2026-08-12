# Patch 9 — Automated Daily Run + GitHub Pages Deployment

## What Patch 9 does

Patch 9 changes only the GitHub Actions workflow.

It does **not** change:
- Patch 6 scoring / confirmed-trend logic
- Patch 7 prospective Path Risk shadow test
- Patch 8 weekly recommendation shadow logic
- Patch 8.1 weekly recommendation dashboard UI
- Gemini prompt/model logic

## New daily schedule

The market-rotation pipeline now runs automatically:

**Monday–Friday at 5:30 PM America/New_York**

The workflow uses GitHub's timezone-aware schedule, so Eastern Daylight Time / Eastern Standard Time changes are handled automatically.

The existing **Run workflow** button remains available for manual runs.

## Automatic publishing

Each successful run:

1. checks out the repository;
2. installs dependencies;
3. runs `python -m src.run_pipeline`;
4. uploads a 7-day preview artifact;
5. commits generated `data/`, `results/`, and `docs/` changes;
6. uploads `docs/` as the GitHub Pages artifact;
7. deploys that artifact directly to GitHub Pages.

This means the public dashboard can update automatically after each scheduled run. You do not need to republish it manually.

## One-time GitHub Pages setting

After committing and pushing Patch 9:

1. Open the repository on GitHub.
2. Go to **Settings**.
3. Click **Pages** under Code and automation.
4. Under **Build and deployment → Source**, select **GitHub Actions**.
5. Return to **Actions**.
6. Open **Market Rotation Daily + Pages**.
7. Click **Run workflow** once on `main`.

After that run succeeds, the deployment job will show the published Pages URL.

## Friday weekly recommendation

Because the scheduled pipeline runs at 5:30 PM Eastern, Friday's run occurs after the Patch 8 weekly recommendation gate.

When Friday market data is available, the pipeline can:
- generate the dated weekly recommendation,
- rebuild the dashboard,
- save the shadow history,
- and publish the updated dashboard in the same workflow.

## Market holidays

The workflow still runs Monday–Friday on market holidays. Existing date/immutability protections prevent a same-market-date rerun from creating duplicate Patch 7/8 observations.

## Notes

- The deploy step runs only from the `main` branch.
- The old `.github/workflows/daily_rotation.yml.disabled` file may remain in the repository; GitHub ignores it because it does not have a `.yml` or `.yaml` extension.
- Gemini credentials remain in GitHub Secrets/Variables and are not published into the Pages artifact by this workflow.
