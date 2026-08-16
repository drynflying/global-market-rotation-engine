from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .feature_definitions import OUTCOME_HORIZONS


def _window_path_metrics(values: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward MDD / adverse excursion / favorable excursion for horizon bars.

    The start close is included as time zero. Metrics are NaN when the complete future
    window is not available or contains non-finite/non-positive values.
    """
    n = len(values)
    mdd = np.full(n, np.nan, dtype=float)
    mae = np.full(n, np.nan, dtype=float)
    mfe = np.full(n, np.nan, dtype=float)
    if n <= horizon:
        return mdd, mae, mfe

    windows = sliding_window_view(values, horizon + 1)
    valid = np.isfinite(windows).all(axis=1) & (windows > 0).all(axis=1)
    if valid.any():
        w = windows[valid]
        norm = w / w[:, [0]]
        peaks = np.maximum.accumulate(norm, axis=1)
        drawdowns = norm / peaks - 1.0
        mdd_vals = np.min(drawdowns, axis=1)
        rel_from_start = norm - 1.0
        mae_vals = np.min(rel_from_start, axis=1)
        mfe_vals = np.max(rel_from_start, axis=1)
        idx = np.flatnonzero(valid)
        mdd[idx] = mdd_vals
        mae[idx] = mae_vals
        mfe[idx] = mfe_vals
    return mdd, mae, mfe


def _relative_excursions(ratio: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(ratio)
    mae = np.full(n, np.nan, dtype=float)
    mfe = np.full(n, np.nan, dtype=float)
    if n <= horizon:
        return mae, mfe
    windows = sliding_window_view(ratio, horizon + 1)
    valid = np.isfinite(windows).all(axis=1) & (windows > 0).all(axis=1)
    if valid.any():
        w = windows[valid]
        path = w / w[:, [0]] - 1.0
        idx = np.flatnonzero(valid)
        mae[idx] = np.min(path, axis=1)
        mfe[idx] = np.max(path, axis=1)
    return mae, mfe


def add_forward_outcomes(features: pd.DataFrame) -> pd.DataFrame:
    df = features.sort_values(["ticker", "date"]).copy()

    # Same-date close lookup for configured primary benchmark and SPY.
    close_lookup = df[["date", "ticker", "close"]].drop_duplicates(["date", "ticker"])
    bench = close_lookup.rename(columns={"ticker": "primary_benchmark", "close": "primary_benchmark_close"})
    df = df.merge(bench, on=["date", "primary_benchmark"], how="left")
    spy = close_lookup[close_lookup["ticker"].eq("SPY")][["date", "close"]].rename(columns={"close": "spy_close"})
    df = df.merge(spy, on="date", how="left", validate="many_to_one")

    out_groups: list[pd.DataFrame] = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"].to_numpy(dtype=float)
        primary_close = g["primary_benchmark_close"].to_numpy(dtype=float)
        spy_close = g["spy_close"].to_numpy(dtype=float)
        dates = g["date"].reset_index(drop=True)

        for h in OUTCOME_HORIZONS:
            g[f"outcome_end_date_{h}"] = dates.shift(-h).to_numpy()
            g[f"fwd_return_{h}"] = g["close"].shift(-h) / g["close"] - 1.0
            g[f"fwd_primary_relative_return_{h}"] = (
                (g["close"].shift(-h) / g["primary_benchmark_close"].shift(-h))
                / (g["close"] / g["primary_benchmark_close"])
                - 1.0
            )
            g[f"fwd_spy_relative_return_{h}"] = (
                (g["close"].shift(-h) / g["spy_close"].shift(-h))
                / (g["close"] / g["spy_close"])
                - 1.0
            )
            mdd, mae, mfe = _window_path_metrics(close, h)
            g[f"fwd_max_drawdown_{h}"] = mdd
            g[f"fwd_max_adverse_excursion_{h}"] = mae
            g[f"fwd_max_favorable_excursion_{h}"] = mfe

            primary_ratio = close / primary_close
            pmae, pmfe = _relative_excursions(primary_ratio, h)
            g[f"fwd_primary_rel_mae_{h}"] = pmae
            g[f"fwd_primary_rel_mfe_{h}"] = pmfe

            spy_ratio = close / spy_close
            smae, smfe = _relative_excursions(spy_ratio, h)
            g[f"fwd_spy_rel_mae_{h}"] = smae
            g[f"fwd_spy_rel_mfe_{h}"] = smfe

            # Convenience labels only; they are not the final R2/R3 investment target.
            g[f"outperformed_primary_{h}"] = (
                g[f"fwd_primary_relative_return_{h}"] > 0
            ).where(g[f"fwd_primary_relative_return_{h}"].notna())
            g[f"outperformed_spy_{h}"] = (
                g[f"fwd_spy_relative_return_{h}"] > 0
            ).where(g[f"fwd_spy_relative_return_{h}"].notna())

        out_groups.append(g)

    return pd.concat(out_groups, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
