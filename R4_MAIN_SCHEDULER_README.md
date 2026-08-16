# R4 MAIN-Branch Scheduler Patch

Install this ZIP on the repository's **main** branch only.

It adds two GitHub Actions workflow files. Both workflows immediately check out
`research/clean-sheet-predictive`; they do not run the production rotation engine and do
not change the main dashboard.

## Daily prospective tracker

`R4 Prospective Clean-Sheet Shadow`

Runs Monday-Friday at **6:15 PM America/New_York**.

It:

- checks out the research branch;
- fetches current OHLCV for the frozen R4 universe;
- issues a prediction only when a completed month-end becomes newly eligible;
- matures existing 1M/3M/6M/9M outcomes automatically;
- refreshes the prospective assessment;
- commits only `data/ohlcv_history.csv` and `results/clean_sheet_r4/` to the research branch.

## Annual frozen model refresh

Runs January 3 at **6:45 PM America/New_York**.

It does **not** rediscover features. It rebuilds the point-in-time dataset for the frozen
167-ticker universe and appends the next calendar-year Ridge/Logistic model using the same
R2.1 candidate features and hyperparameters.

Old model years are immutable.

## First run

After installing both R4 ZIPs, run `R4 Prospective Clean-Sheet Shadow` manually once.
Because R4 begins on 2026-08-15, the first prediction will not be issued until August 2026
is a completed month and September market data exists. The initial manual run should
therefore mainly verify the pipeline and create the status/report files.
