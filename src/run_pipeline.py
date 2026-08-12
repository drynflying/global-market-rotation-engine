from __future__ import annotations

import json
import sys

import pandas as pd

from src.ai.run_analysis import run_ai_analysis
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
from src.research_shadow import run_shadow_research


RESULT_COLUMNS = [
    "date", "ticker", "priority", "signal_role", "rank_eligible", "score_mode", "universe",
    "rotation_group", "level", "exposure", "sector", "industry_theme",
    "primary_benchmark", "parent_benchmark", "paired_ticker", "pair_type",
    "data_status", "bars_available", "close", "volume", "dollar_volume",
    "relative_dollar_volume", "return_20", "return_63", "rs20", "rs63",
    "parent_rs20", "parent_rs63", "pair_spread_20", "pair_spread_63",
    "signal_rs20", "signal_rs63", "pair_signal_raw", "pair_signal_confirmed", "pair_signal",
    "cmf20", "obv_change_20", "sma50", "sma200", "position_52w",
    "trend_score", "rotation_score", "group_rank", "group_size",
    "group_percentile", "score_change_5", "score_change_10",
    "score_change_20", "score_change_63", "rank_change_5", "rank_change_20",
    "rs20_change_5", "rs63_change_20", "days_leader_zone_20",
    "leader_zone_streak", "days_top_quartile_20", "top_quartile_streak",
    "consecutive_improving_days",
    "rotation_state_raw", "rotation_state_confirmed", "rotation_state",
    "confirmed_state_age", "pending_rotation_state", "pending_state_days",
    "state_confirmation_bars", "confirmed_pair_signal_age",
    "pending_pair_signal", "pending_pair_signal_days", "pair_confirmation_bars",
    "score_formula_version",
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
        "cross_sectional_scored": int(
            ((latest["score_mode"] == "CROSS_SECTIONAL") & latest["rotation_score"].notna()).sum()
        ),
        "pair_scored": int(
            ((latest["score_mode"] == "PAIR") & latest["rotation_score"].notna()).sum()
        ),
        "emerging_count": int((latest["rotation_state"] == "EMERGING").sum()),
        "accelerating_count": int((latest["rotation_state"] == "ACCELERATING").sum()),
        "persistent_leader_count": int(
            (latest["rotation_state"] == "PERSISTENT_LEADER").sum()
        ),
        "reaccelerating_count": int(
            (latest["rotation_state"] == "REACCELERATING").sum()
        ),
        "neutral_count": int((latest["rotation_state"] == "NEUTRAL").sum()),
        "weakening_count": int((latest["rotation_state"] == "WEAKENING").sum()),
        "rotation_out_count": int((latest["rotation_state"] == "ROTATION_OUT").sum()),
        "state_confirmation_bars": int(latest["state_confirmation_bars"].dropna().iloc[0])
        if "state_confirmation_bars" in latest.columns and not latest["state_confirmation_bars"].dropna().empty
        else 3,
        "pending_state_changes": int((latest.get("pending_state_days", 0) > 0).sum())
        if "pending_state_days" in latest.columns else 0,
        "pending_pair_signal_changes": int((latest.get("pending_pair_signal_days", 0) > 0).sum())
        if "pending_pair_signal_days" in latest.columns else 0,
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

    # Patch 7: prospective research shadow test.
    # This is deliberately non-blocking and does not alter any production
    # score, state, AI input, or dashboard output.
    try:
        shadow_status = run_shadow_research(ohlcv, cfg, latest)
        prediction_status = shadow_status.get("prediction", {})
        outcome_status = shadow_status.get("outcomes", {})
        summary.update({
            "shadow_research_status": shadow_status.get("status", "ok"),
            "shadow_model_version": prediction_status.get("model_version"),
            "shadow_market_date": prediction_status.get("market_date"),
            "shadow_appended_rows": prediction_status.get("appended_rows", 0),
            "shadow_skipped_existing_rows": prediction_status.get("skipped_existing_rows", 0),
            "shadow_universal_veto_rows": prediction_status.get("universal_veto_rows", 0),
            "shadow_challenger_veto_rows": prediction_status.get("challenger_veto_rows", 0),
            "shadow_matured_21": outcome_status.get("matured_21", 0),
            "shadow_matured_63": outcome_status.get("matured_63", 0),
            "shadow_matured_84": outcome_status.get("matured_84", 0),
            "shadow_matured_126": outcome_status.get("matured_126", 0),
        })
        print(
            "Shadow research: "
            f"{prediction_status.get('appended_rows', 0)} new predictions, "
            f"{outcome_status.get('matured_126', 0)} fully matured outcomes."
        )
    except Exception as exc:
        summary.update({
            "shadow_research_status": "error",
            "shadow_research_error": f"{type(exc).__name__}: {exc}",
        })
        print(f"WARNING: Shadow research logger failed: {type(exc).__name__}: {exc}")

    ai_input = build_ai_input(latest, history)
    ai_analysis = run_ai_analysis(ai_input)
    summary.update({
        "ai_requested_providers": ai_analysis.get("requested_providers", []),
        "ai_successful_providers": ai_analysis.get("successful_providers", []),
        "ai_failed_providers": ai_analysis.get("failed_providers", []),
        "ai_primary_provider": ai_analysis.get("primary_provider"),
        "ai_provider_status": ai_analysis.get("provider_status"),
        "ai_consensus_provider_count": ai_analysis.get("consensus", {}).get("provider_count", 0),
    })
    RUN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_dashboard(latest, history, ai_analysis)

    print(json.dumps(summary, indent=2))
    print("Dashboard written to docs/index.html")
    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
