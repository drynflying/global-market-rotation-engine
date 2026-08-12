from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "rotation_universe.csv"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"

OHLCV_PATH = DATA_DIR / "ohlcv_history.csv"
ROTATION_HISTORY_PATH = RESULTS_DIR / "rotation_history.csv"
ROTATION_LATEST_PATH = RESULTS_DIR / "rotation_latest.csv"
FETCH_FAILURES_PATH = RESULTS_DIR / "fetch_failures.csv"
RUN_SUMMARY_PATH = RESULTS_DIR / "run_summary.json"
AI_INPUT_PATH = RESULTS_DIR / "ai_input.json"
AI_ANALYSIS_PATH = RESULTS_DIR / "ai_analysis.json"
AI_DIR = RESULTS_DIR / "ai"
AI_HISTORY_DIR = AI_DIR / "history"
AI_CONSENSUS_PATH = AI_DIR / "consensus.json"
AI_MANIFEST_PATH = AI_DIR / "manifest.json"

RESEARCH_DIR = ROOT / "research"
PATH_RISK_MODEL_SPEC_PATH = RESEARCH_DIR / "path_risk_v1" / "model_spec.json"
RESEARCH_SHADOW_HISTORY_PATH = RESULTS_DIR / "research_shadow_history.csv"
RESEARCH_SHADOW_OUTCOMES_PATH = RESULTS_DIR / "research_shadow_outcomes.csv"
RESEARCH_SHADOW_STATUS_PATH = RESULTS_DIR / "research_shadow_status.json"

WEEKLY_RECOMMENDATION_SPEC_PATH = RESEARCH_DIR / "weekly_recommendation_v1" / "spec.json"
WEEKLY_RECOMMENDATION_HISTORY_PATH = RESULTS_DIR / "weekly_recommendation_shadow_history.csv"
WEEKLY_RECOMMENDATION_OUTCOMES_PATH = RESULTS_DIR / "weekly_recommendation_shadow_outcomes.csv"
WEEKLY_RECOMMENDATION_STATUS_PATH = RESULTS_DIR / "weekly_recommendation_shadow_status.json"
WEEKLY_RECOMMENDATION_LATEST_PATH = RESULTS_DIR / "weekly_recommendation_shadow_latest.json"
WEEKLY_RECOMMENDATION_AI_DIR = RESULTS_DIR / "weekly_recommendation_ai"


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_int(value, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ensure_dirs() -> None:
    for directory in [DATA_DIR, RESULTS_DIR, AI_DIR, AI_HISTORY_DIR, WEEKLY_RECOMMENDATION_AI_DIR, DOCS_DIR, DOCS_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_config() -> pd.DataFrame:
    required = {
        "enabled", "query_symbol", "ticker", "priority", "signal_role",
        "rank_eligible", "score_mode", "universe", "rotation_group", "primary_benchmark",
        "parent_benchmark", "paired_ticker", "query_ohlcv",
        "history_bars", "min_history_bars", "rs_short_bars", "rs_long_bars",
        "volume_window_bars", "cmf_window_bars", "sma_fast_bars",
        "sma_slow_bars", "high_lookback_bars", "persistence_bars",
        "score_w_rs20", "score_w_rs63", "score_w_rel_dollar_volume",
        "score_w_cmf20", "score_w_trend", "score_formula_version",
    }

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Missing config/rotation_universe.csv. "
            "Do not rename or move the configuration file."
        )

    cfg = pd.read_csv(CONFIG_PATH, dtype=str, keep_default_na=False)
    missing = sorted(required - set(cfg.columns))
    if missing:
        raise ValueError(
            "rotation_universe.csv is missing required columns: "
            + ", ".join(missing)
        )

    for col in ["enabled", "rank_eligible", "query_ohlcv", "use_adjusted_prices"]:
        if col in cfg.columns:
            cfg[col] = cfg[col].map(as_bool)

    int_defaults = {
        "history_bars": 300,
        "min_history_bars": 260,
        "rs_short_bars": 20,
        "rs_long_bars": 63,
        "volume_window_bars": 20,
        "cmf_window_bars": 20,
        "sma_fast_bars": 50,
        "sma_slow_bars": 200,
        "high_lookback_bars": 252,
        "persistence_bars": 5,
    }
    for col, default in int_defaults.items():
        cfg[col] = cfg[col].map(lambda x: safe_int(x, default))

    float_defaults = {
        "score_w_rs20": 0.30,
        "score_w_rs63": 0.25,
        "score_w_rel_dollar_volume": 0.20,
        "score_w_cmf20": 0.15,
        "score_w_trend": 0.10,
    }
    for col, default in float_defaults.items():
        cfg[col] = cfg[col].map(lambda x: safe_float(x, default))

    cfg["ticker"] = cfg["ticker"].str.strip().str.upper()
    cfg["query_symbol"] = cfg["query_symbol"].str.strip().str.upper()
    cfg["score_mode"] = cfg["score_mode"].str.strip().str.upper()
    cfg = cfg.drop_duplicates(subset=["ticker"], keep="first").copy()

    valid_modes = {"CROSS_SECTIONAL", "PAIR", "REFERENCE"}
    bad_modes = sorted(set(cfg["score_mode"]) - valid_modes)
    if bad_modes:
        raise ValueError(f"Invalid score_mode values: {bad_modes}")

    pair_without_reference = cfg[
        (cfg["score_mode"] == "PAIR") & cfg["paired_ticker"].astype(str).str.strip().eq("")
    ]
    if not pair_without_reference.empty:
        raise ValueError(
            "PAIR rows require paired_ticker. Missing for: "
            + ", ".join(pair_without_reference["ticker"].tolist())
        )

    active = cfg[cfg["enabled"] & cfg["query_ohlcv"]]
    if active.empty:
        raise ValueError("No enabled/query_ohlcv=true tickers exist in the configuration.")

    ticker_set = set(cfg["ticker"])
    broken_refs = []
    for field in ["primary_benchmark", "parent_benchmark", "paired_ticker"]:
        for ticker, ref in cfg[["ticker", field]].itertuples(index=False):
            ref = str(ref).strip().upper()
            if ref and ref not in ticker_set:
                broken_refs.append((ticker, field, ref))

    if broken_refs:
        raise ValueError(
            "Configuration contains benchmark/pair references that are not "
            f"present as tickers. Examples: {broken_refs[:10]}"
        )

    return cfg
