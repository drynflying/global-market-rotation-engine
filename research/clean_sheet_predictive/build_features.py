from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .feature_definitions import (
    BASE_OHLCV_COLUMNS,
    RANGE_WINDOWS,
    RETURN_WINDOWS,
    SMA_WINDOWS,
    VOL_WINDOWS,
)


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in BASE_OHLCV_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"OHLCV is missing required columns: {missing}")
    df = raw[BASE_OHLCV_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"])
    df = df[df["close"] > 0]
    df = df[df["volume"].fillna(0) >= 0]
    if df.duplicated(["ticker", "date"]).any():
        dupes = int(df.duplicated(["ticker", "date"]).sum())
        raise ValueError(f"OHLCV contains {dupes} duplicate ticker/date rows")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _bars_since_extreme(values: np.ndarray, window: int, want_max: bool) -> np.ndarray:
    """O(n) bars-since trailing max/min, using only current/past observations."""
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    q: deque[int] = deque()
    for i, val in enumerate(values):
        while q and q[0] <= i - window:
            q.popleft()
        if np.isfinite(val):
            if want_max:
                while q and np.isfinite(values[q[-1]]) and values[q[-1]] <= val:
                    q.pop()
            else:
                while q and np.isfinite(values[q[-1]]) and values[q[-1]] >= val:
                    q.pop()
            q.append(i)
        if i >= window - 1 and q:
            out[i] = float(i - q[0])
    return out


def _cmf(group: pd.DataFrame, window: int) -> pd.Series:
    spread = group["high"] - group["low"]
    multiplier = np.where(
        spread.abs() > 1e-12,
        ((group["close"] - group["low"]) - (group["high"] - group["close"])) / spread,
        0.0,
    )
    mfv = pd.Series(multiplier, index=group.index) * group["volume"]
    numerator = mfv.rolling(window, min_periods=window).sum()
    denominator = group["volume"].rolling(window, min_periods=window).sum()
    return numerator / denominator.replace(0, np.nan)


def _per_ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    close = g["close"]
    daily_ret = close.pct_change(fill_method=None)

    for w in RETURN_WINDOWS:
        g[f"ret_{w}"] = close / close.shift(w) - 1.0

    for w in SMA_WINDOWS:
        sma = close.rolling(w, min_periods=w).mean()
        g[f"sma_{w}"] = sma
        g[f"above_sma_{w}"] = (close > sma).where(sma.notna()).astype(float)
        g[f"dist_sma_{w}"] = close / sma - 1.0
        g[f"sma_{w}_slope_20"] = sma / sma.shift(20) - 1.0
    g["sma_50_over_200"] = g["sma_50"] / g["sma_200"] - 1.0

    for w in RANGE_WINDOWS:
        hi = close.rolling(w, min_periods=w).max()
        lo = close.rolling(w, min_periods=w).min()
        g[f"dist_high_{w}"] = close / hi - 1.0
        g[f"dist_low_{w}"] = close / lo - 1.0
        g[f"range_position_{w}"] = (close - lo) / (hi - lo).replace(0, np.nan)

    arr = close.to_numpy(dtype=float)
    g["bars_since_high_252"] = _bars_since_extreme(arr, 252, True)
    g["bars_since_low_252"] = _bars_since_extreme(arr, 252, False)

    for w in VOL_WINDOWS:
        g[f"vol_{w}"] = daily_ret.rolling(w, min_periods=w).std(ddof=0) * np.sqrt(252.0)
        downside_sq = daily_ret.clip(upper=0).pow(2)
        g[f"downside_vol_{w}"] = (
            downside_sq.rolling(w, min_periods=w).mean().pow(0.5) * np.sqrt(252.0)
        )
    g["vol_accel_21_126"] = g["vol_21"] / g["vol_126"].replace(0, np.nan)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            g["high"] - g["low"],
            (g["high"] - prev_close).abs(),
            (g["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    g["atr20_pct"] = true_range.rolling(20, min_periods=20).mean() / close

    g["cmf20"] = _cmf(g, 20)
    g["cmf63"] = _cmf(g, 63)
    dollar_volume = close * g["volume"]
    dv20 = dollar_volume.rolling(20, min_periods=20).mean()
    dv63 = dollar_volume.rolling(63, min_periods=63).mean()
    g["dollar_volume_ratio_20_63"] = dv20 / dv63.replace(0, np.nan)
    vol_mean20 = g["volume"].rolling(20, min_periods=20).mean()
    vol_std20 = g["volume"].rolling(20, min_periods=20).std(ddof=0)
    g["volume_z20"] = (g["volume"] - vol_mean20) / vol_std20.replace(0, np.nan)

    signed_volume = np.sign(close.diff()).fillna(0.0) * g["volume"]
    for w in (20, 63):
        denom = g["volume"].rolling(w, min_periods=w).sum()
        g[f"signed_volume_balance_{w}"] = (
            signed_volume.rolling(w, min_periods=w).sum() / denom.replace(0, np.nan)
        )

    obv = signed_volume.cumsum()
    for w in (20, 63):
        avg_vol = g["volume"].rolling(w, min_periods=w).mean()
        g[f"obv_change_{w}_norm"] = (obv - obv.shift(w)) / (avg_vol * w).replace(0, np.nan)

    return g


def _prepare_config(config: pd.DataFrame) -> pd.DataFrame:
    cfg = config.copy()
    cfg["ticker"] = cfg["ticker"].astype(str).str.upper().str.strip()
    if "primary_benchmark" not in cfg.columns:
        cfg["primary_benchmark"] = ""
    cfg["primary_benchmark"] = (
        cfg["primary_benchmark"].fillna("").astype(str).str.upper().str.strip()
    )
    keep = [
        c
        for c in [
            "ticker",
            "exposure",
            "universe",
            "rotation_group",
            "level",
            "asset_type",
            "primary_benchmark",
        ]
        if c in cfg.columns
    ]
    return cfg[keep].drop_duplicates("ticker", keep="last")


def build_features(ohlcv: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    df = normalize_ohlcv(ohlcv)
    pieces = [_per_ticker_features(g) for _, g in df.groupby("ticker", sort=False)]
    feat = pd.concat(pieces, ignore_index=True)

    cfg = _prepare_config(config)
    feat = feat.merge(cfg, on="ticker", how="left", validate="many_to_one")

    # Bring benchmark returns onto each row. Missing primary benchmarks remain missing;
    # SPY-relative features are always calculated separately when SPY is available.
    ret_cols = [f"ret_{w}" for w in RETURN_WINDOWS]
    lookup = feat[["date", "ticker", *ret_cols]].copy()
    bench = lookup.rename(
        columns={"ticker": "primary_benchmark", **{c: f"bench_{c}" for c in ret_cols}}
    )
    feat = feat.merge(bench, on=["date", "primary_benchmark"], how="left")

    spy = lookup[lookup["ticker"].eq("SPY")].drop(columns="ticker")
    spy = spy.rename(columns={c: f"spy_{c}" for c in ret_cols})
    feat = feat.merge(spy, on="date", how="left", validate="many_to_one")

    for w in RETURN_WINDOWS:
        feat[f"rs_primary_{w}"] = feat[f"ret_{w}"] - feat[f"bench_ret_{w}"]
        feat[f"rs_spy_{w}"] = feat[f"ret_{w}"] - feat[f"spy_ret_{w}"]

    # Market regime fields come from the SPY row and same-date cross-sectional breadth.
    spy_fields = feat[feat["ticker"].eq("SPY")][
        [
            "date",
            "ret_21",
            "ret_63",
            "ret_126",
            "ret_252",
            "above_sma_200",
            "sma_200_slope_20",
            "vol_21",
            "vol_accel_21_126",
            "dist_high_252",
        ]
    ].copy()
    spy_fields = spy_fields.rename(
        columns={
            "ret_21": "mkt_spy_ret_21",
            "ret_63": "mkt_spy_ret_63",
            "ret_126": "mkt_spy_ret_126",
            "ret_252": "mkt_spy_ret_252",
            "above_sma_200": "mkt_spy_above_sma_200",
            "sma_200_slope_20": "mkt_spy_sma_200_slope_20",
            "vol_21": "mkt_spy_vol_21",
            "vol_accel_21_126": "mkt_spy_vol_accel_21_126",
            "dist_high_252": "mkt_spy_dist_high_252",
        }
    )
    feat = feat.merge(spy_fields, on="date", how="left", validate="many_to_one")

    valid_sma = feat["sma_200"].notna()
    breadth = (
        feat.loc[valid_sma]
        .groupby("date")["above_sma_200"]
        .mean()
        .rename("mkt_breadth_above_sma200")
    )
    dispersion = feat.groupby("date")["ret_63"].std(ddof=0).rename("mkt_dispersion_ret63")
    feat = feat.merge(breadth, on="date", how="left")
    feat = feat.merge(dispersion, on="date", how="left")

    # A small set of same-date cross-sectional percentile features. These are not a
    # score and receive no hand-assigned weight; R2 will determine whether they help.
    feat["cs_ret_63_pct"] = feat.groupby("date")["ret_63"].rank(pct=True, method="average")
    feat["cs_ret_126_pct"] = feat.groupby("date")["ret_126"].rank(pct=True, method="average")
    feat["cs_rs_spy_63_pct"] = feat.groupby("date")["rs_spy_63"].rank(pct=True, method="average")
    feat["cs_rs_spy_126_pct"] = feat.groupby("date")["rs_spy_126"].rank(pct=True, method="average")

    return feat.sort_values(["date", "ticker"]).reset_index(drop=True)
