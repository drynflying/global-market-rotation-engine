from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import as_bool


def calculate_base_metrics(ohlcv: pd.DataFrame, cfg: pd.DataFrame) -> pd.DataFrame:
    cfg_by_ticker = cfg.set_index("ticker")
    frames: list[pd.DataFrame] = []

    for ticker, g in ohlcv.groupby("ticker", sort=False):
        if ticker not in cfg_by_ticker.index:
            continue

        row = cfg_by_ticker.loc[ticker]
        g = g.sort_values("date").copy()

        rs_short = int(row["rs_short_bars"])
        rs_long = int(row["rs_long_bars"])
        vol_window = int(row["volume_window_bars"])
        cmf_window = int(row["cmf_window_bars"])
        sma_fast = int(row["sma_fast_bars"])
        sma_slow = int(row["sma_slow_bars"])
        high_window = int(row["high_lookback_bars"])

        g["return_20"] = g["close"].pct_change(rs_short)
        g["return_63"] = g["close"].pct_change(rs_long)

        g["dollar_volume"] = g["close"] * g["volume"]
        avg_dollar_volume = (
            g["dollar_volume"]
            .rolling(vol_window, min_periods=vol_window)
            .mean()
        )
        g["relative_dollar_volume"] = g["dollar_volume"] / avg_dollar_volume

        spread = g["high"] - g["low"]
        money_flow_multiplier = np.where(
            spread.abs() > 1e-12,
            ((2 * g["close"]) - g["high"] - g["low"]) / spread,
            0.0,
        )
        money_flow_volume = money_flow_multiplier * g["volume"]
        g["cmf20"] = (
            pd.Series(money_flow_volume, index=g.index)
            .rolling(cmf_window, min_periods=cmf_window)
            .sum()
            / g["volume"].rolling(cmf_window, min_periods=cmf_window).sum()
        )

        direction = np.sign(g["close"].diff()).fillna(0.0)
        g["obv"] = (direction * g["volume"]).cumsum()
        g["obv_change_20"] = g["obv"] - g["obv"].shift(vol_window)

        g["sma50"] = g["close"].rolling(sma_fast, min_periods=sma_fast).mean()
        g["sma200"] = g["close"].rolling(sma_slow, min_periods=sma_slow).mean()
        g["high_252"] = (
            g["close"].rolling(high_window, min_periods=high_window).max()
        )
        g["position_52w"] = g["close"] / g["high_252"]

        full_stack = (
            (g["close"] > g["sma50"])
            & (g["sma50"] > g["sma200"])
        )
        above_fast_and_slow = (
            (g["close"] > g["sma50"])
            & (g["close"] > g["sma200"])
        )
        above_slow = g["close"] > g["sma200"]

        g["trend_score"] = np.select(
            [full_stack, above_fast_and_slow, above_slow],
            [1.00, 0.75, 0.50],
            default=0.00,
        )
        g.loc[g["sma200"].isna(), "trend_score"] = np.nan
        g["bars_available"] = np.arange(1, len(g) + 1)

        frames.append(g)

    if not frames:
        raise RuntimeError("No base metrics could be calculated.")

    return pd.concat(frames, ignore_index=True)


def attach_config_and_relative_strength(
    metrics: pd.DataFrame,
    cfg: pd.DataFrame,
) -> pd.DataFrame:
    meta_cols = [
        "ticker", "priority", "signal_role", "rank_eligible", "universe",
        "rotation_group", "level", "exposure", "name", "geography", "sector",
        "industry_theme", "primary_benchmark", "parent_benchmark",
        "paired_ticker", "pair_type", "min_history_bars", "persistence_bars",
        "score_w_rs20", "score_w_rs63", "score_w_rel_dollar_volume",
        "score_w_cmf20", "score_w_trend", "score_formula_version",
    ]
    meta_cols = [c for c in meta_cols if c in cfg.columns]
    out = metrics.merge(
        cfg[meta_cols],
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    benchmark_returns = metrics[
        ["date", "ticker", "return_20", "return_63"]
    ].copy()

    def attach(ref_field: str, prefix: str) -> None:
        nonlocal out
        bm = benchmark_returns.rename(
            columns={
                "ticker": ref_field,
                "return_20": f"{prefix}_return_20",
                "return_63": f"{prefix}_return_63",
            }
        )
        out = out.merge(bm, on=["date", ref_field], how="left")

        if prefix == "primary":
            out["rs20"] = out["return_20"] - out["primary_return_20"]
            out["rs63"] = out["return_63"] - out["primary_return_63"]
        elif prefix == "parent":
            out["parent_rs20"] = out["return_20"] - out["parent_return_20"]
            out["parent_rs63"] = out["return_63"] - out["parent_return_63"]
        else:
            out["pair_spread_20"] = out["return_20"] - out["pair_return_20"]
            out["pair_spread_63"] = out["return_63"] - out["pair_return_63"]

    attach("primary_benchmark", "primary")
    attach("parent_benchmark", "parent")
    attach("paired_ticker", "pair")

    out["data_status"] = np.where(
        out["bars_available"] >= out["min_history_bars"],
        "READY",
        "INSUFFICIENT_HISTORY",
    )
    return out


def calculate_rotation_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    eligible_base = (
        out["rank_eligible"].map(as_bool)
        & (out["data_status"] == "READY")
        & out["rs20"].notna()
        & out["rs63"].notna()
        & out["relative_dollar_volume"].notna()
        & out["cmf20"].notna()
        & out["trend_score"].notna()
    )

    # Avoid meaningless percentile ranks in groups with only 1–2 members.
    group_counts = (
        out.loc[eligible_base]
        .groupby(["date", "rotation_group"])["ticker"]
        .transform("count")
    )
    eligible = eligible_base.copy()
    eligible.loc[eligible_base] = group_counts >= 3

    component_map = {
        "rs20": "rs20_pct",
        "rs63": "rs63_pct",
        "relative_dollar_volume": "rel_dollar_volume_pct",
        "cmf20": "cmf20_pct",
    }

    for source, target in component_map.items():
        out[target] = np.nan
        out.loc[eligible, target] = (
            out.loc[eligible]
            .groupby(["date", "rotation_group"])[source]
            .rank(method="average", pct=True)
        )

    out["rotation_score"] = np.nan
    out.loc[eligible, "rotation_score"] = 100 * (
        out.loc[eligible, "score_w_rs20"] * out.loc[eligible, "rs20_pct"]
        + out.loc[eligible, "score_w_rs63"] * out.loc[eligible, "rs63_pct"]
        + out.loc[eligible, "score_w_rel_dollar_volume"]
          * out.loc[eligible, "rel_dollar_volume_pct"]
        + out.loc[eligible, "score_w_cmf20"] * out.loc[eligible, "cmf20_pct"]
        + out.loc[eligible, "score_w_trend"] * out.loc[eligible, "trend_score"]
    )

    out["group_rank"] = np.nan
    out.loc[eligible, "group_rank"] = (
        out.loc[eligible]
        .groupby(["date", "rotation_group"])["rotation_score"]
        .rank(method="min", ascending=False)
    )

    out["group_size"] = np.nan
    out.loc[eligible, "group_size"] = (
        out.loc[eligible]
        .groupby(["date", "rotation_group"])["rotation_score"]
        .transform("count")
    )

    out["group_percentile"] = np.nan
    out.loc[eligible, "group_percentile"] = (
        100
        * out.loc[eligible]
        .groupby(["date", "rotation_group"])["rotation_score"]
        .rank(method="average", pct=True)
    )

    return out
