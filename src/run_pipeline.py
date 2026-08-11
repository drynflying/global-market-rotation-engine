from __future__ import annotations

import json
import sys

import pandas as pd

from src.analyze_with_gemini import analyze
from src.build_ai_input import build_ai_input
from src.build_dashboard import build_dashboard
from src.calculate_metrics import (
    attach_config_and_relative_strength,
    calculate_base_metrics,
    calculate_rotation_scores,
)
from src.calculate_trends import add_rotation_trends
from src.common import (
    FETCH_FAILURES_PATH,
    ROTATION_HISTORY_PATH,
    ROTATION_LATEST_PATH,
    RUN_SUMMARY_PATH,
    ensure_dirs,
    load_config,
)
from src.fetch_market_data import fetch_market_data, update_ohlcv_history


RESULT_COLUMNS = [
    "date", "ticker", "priority", "signal_role", "rank_eligible", "universe",
    "rotation_group", "level", "exposure", "sector", "industry_theme",
    "primary_benchmark", "parent_benchmark", "paired_ticker", "pair_type",
    "data_status", "bars_available", "close", "volume", "dollar_volume",
    "relative_dollar_volume", "return_20", "return_63", "rs20", "rs63",
    "parent_rs20", "parent_rs63", "pair_spread_20", "pair_spread_63",
    "cmf20", "obv_change_20", "sma50", "sma200", "position_52w",
    "trend_score", "rotation_score", "group_rank", "group_size",
    "group_percentile", "score_change_5", "score_change_10",
    "score_change_20", "score_change_63", "rank_change_5", "rank_change_20",
    "rs20_change_5", "rs63_change_20", "days_top_quartile_20",
    "top_quartile_streak", "consecutive_improving_days",
    "rotation_state", "score_formula_version",
]


def save_results(df: pd.DataFrame, cfg: pd.DataFrame, failures: pd.DataFrame):
    available = [c for c in RESULT_COLUMNS if c in df.columns]
    history = df[available].copy()
    history = history[
        history["return_63"].notna()
        | history["rotation_score"].notna()
    ].copy()
    history = history.sort_values(["date", "rotation_group", "group_rank", "ticker"])

    history.to_csv(
        ROTATION_HISTORY_PATH,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.6f",
    )

    latest = (
        df.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
    )
    latest = latest[[c for c in available if c in latest.columns]].copy()
    latest = latest.sort_values(["rotation_group", "group_rank", "ticker"])
    latest.to_csv(
        ROTATION_LATEST_PATH,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.6f",
    )

    failures.to_csv(FETCH_FAILURES_PATH, index=False)

    latest_market_date = (
        pd.to_datetime(latest["date"]).max().strftime("%Y-%m-%d")
        if not latest.empty else None
    )

    summary = {
        "status": "ok" if failures.empty else "partial",
        "latest_market_date": latest_market_date,
        "configured_rows": int(len(cfg)),
        "enabled_ohlcv_rows": int((cfg["enabled"] & cfg["query_ohlcv"]).sum()),
        "latest_result_tickers": int(latest["ticker"].nunique()),
        "failed_symbols": failures["ticker"].tolist() if not failures.empty else [],
        "scored_tickers": int(latest["rotation_score"].notna().sum()),
        "rotation_in_count": int((latest["rotation_state"] == "ROTATION_IN").sum()),
        "accumulating_count": int((latest["rotation_state"] == "ACCUMULATING").sum()),
        "weakening_count": int((latest["rotation_state"] == "WEAKENING").sum()),
        "rotation_out_count": int((latest["rotation_state"] == "ROTATION_OUT").sum()),
    }
    RUN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return history, latest, summary


def validate_required_benchmarks(cfg: pd.DataFrame, latest: pd.DataFrame):
    needed = set(
        cfg.loc[cfg["rank_eligible"], "primary_benchmark"]
        .replace("", pd.NA)
        .dropna()
        .str.upper()
        .tolist()
    )
    available = set(latest["ticker"].astype(str).str.upper())
    missing = sorted(needed - available)
    if missing:
        raise RuntimeError(
            "Required benchmark data is missing. "
            f"The rotation scores cannot be trusted for: {missing}"
        )


def main() -> int:
    print("=== Market Rotation Dashboard Pipeline ===")
    ensure_dirs()

    cfg = load_config()
    print(f"Loaded {len(cfg)} configuration rows.")

    fetched, failures = fetch_market_data(cfg)
    print(f"Fetched {len(fetched):,} OHLCV rows.")
    if not failures.empty:
        print("Some tickers failed. See results/fetch_failures.csv")

    ohlcv = update_ohlcv_history(fetched)
    print(f"Stored {len(ohlcv):,} OHLCV rows.")

    metrics = calculate_base_metrics(ohlcv, cfg)
    metrics = attach_config_and_relative_strength(metrics, cfg)
    metrics = calculate_rotation_scores(metrics)
    metrics = add_rotation_trends(metrics)

    history, latest, summary = save_results(metrics, cfg, failures)
    validate_required_benchmarks(cfg, latest)

    ai_input = build_ai_input(latest, history)
    ai_analysis = analyze(ai_input)
    build_dashboard(latest, history, ai_analysis)

    print(json.dumps(summary, indent=2))
    print("Dashboard written to docs/index.html")
    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
