from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import (
    as_bool,
    PATH_RISK_MODEL_SPEC_PATH,
    RESEARCH_SHADOW_HISTORY_PATH,
    RESEARCH_SHADOW_OUTCOMES_PATH,
    RESEARCH_SHADOW_STATUS_PATH,
)


FEATURE_NAMES = [
    "rel_ret_21",
    "rel_ret_63",
    "rel_ret_126",
    "rel_vs_ema21",
    "rel_vs_ema63",
    "ema21_vs_ema63",
    "ema63_vs_ema126",
    "ema63_slope21",
    "ema126_slope21",
    "dist_low126",
    "drawdown_high126",
    "bars_since_low126",
    "bars_since_high126",
    "rel_vol21",
    "rel_vol63",
]

PREDICTION_COLUMNS = [
    "prediction_id",
    "captured_at_utc",
    "market_date",
    "ticker",
    "rotation_group",
    "primary_benchmark",
    "exposure",
    "patch6_rotation_score",
    "patch6_group_percentile",
    "patch6_raw_state",
    "patch6_confirmed_state",
    "patch6_top20",
    "shadow_feature_complete",
    "universal_path_success_probability",
    "universal_peer_rank",
    "universal_peer_size",
    "universal_peer_percentile",
    "universal_low_trust",
    "universal_veto_on_patch6",
    "challenger_path_success_probability",
    "challenger_model_used",
    "challenger_peer_rank",
    "challenger_peer_size",
    "challenger_peer_percentile",
    "challenger_low_trust",
    "challenger_veto_on_patch6",
    "model_version",
    "target_version",
    "feature_version",
] + [f"feature_{name}" for name in FEATURE_NAMES]

OUTCOME_COLUMNS = [
    "prediction_id",
    "model_version",
    "market_date",
    "ticker",
    "primary_benchmark",
    "outcome_as_of_market_date",
    "matured_21",
    "matured_63",
    "matured_84",
    "matured_126",
    "fwd_relative_return_21",
    "fwd_relative_return_63",
    "fwd_relative_return_84",
    "fwd_relative_return_126",
    "fwd_relative_mdd_126",
    "band_median_relative_return",
    "investable_loose",
    "investable_primary",
    "investable_strict",
]


def _load_spec() -> dict:
    if not PATH_RISK_MODEL_SPEC_PATH.exists():
        raise FileNotFoundError(
            "Missing frozen research model spec: "
            f"{PATH_RISK_MODEL_SPEC_PATH.relative_to(PATH_RISK_MODEL_SPEC_PATH.parents[2])}"
        )
    spec = json.loads(PATH_RISK_MODEL_SPEC_PATH.read_text(encoding="utf-8"))
    expected = spec.get("features", {}).get("ordered_names", [])
    if expected != FEATURE_NAMES:
        raise ValueError(
            "Frozen path-risk model feature order does not match research_shadow.py. "
            "Do not silently reorder or retrain the prospective model."
        )
    return spec


def _bars_since_extreme(values: np.ndarray, find_max: bool) -> np.ndarray:
    out = np.full(len(values), np.nan)
    window = 126
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1 : i + 1]
        if not np.isfinite(sample).all():
            continue
        pos = int(np.argmax(sample) if find_max else np.argmin(sample))
        out[i] = (window - 1) - pos
    return out


def _relative_feature_frame(
    ohlcv: pd.DataFrame,
    cfg: pd.DataFrame,
) -> pd.DataFrame:
    data = ohlcv.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None)
    data["ticker"] = data["ticker"].astype(str).str.upper()

    closes = (
        data.dropna(subset=["date", "ticker", "close"])
        .pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
    )

    cross_cfg = cfg[
        cfg["rank_eligible"]
        & cfg["score_mode"].astype(str).str.upper().eq("CROSS_SECTIONAL")
    ].copy()

    frames: list[pd.DataFrame] = []
    for row in cross_cfg.itertuples(index=False):
        ticker = str(row.ticker).upper()
        benchmark = str(row.primary_benchmark).upper()
        if ticker not in closes.columns or benchmark not in closes.columns:
            continue

        relative_price = (closes[ticker] / closes[benchmark]).dropna()
        if relative_price.empty:
            continue

        g = pd.DataFrame(
            {
                "date": relative_price.index,
                "ticker": ticker,
                "primary_benchmark": benchmark,
                "rotation_group": str(row.rotation_group),
                "relative_price": relative_price.to_numpy(dtype=float),
            }
        )

        g["rel_ret_21"] = g["relative_price"].pct_change(21)
        g["rel_ret_63"] = g["relative_price"].pct_change(63)
        g["rel_ret_126"] = g["relative_price"].pct_change(126)

        ema21 = g["relative_price"].ewm(span=21, adjust=False).mean()
        ema63 = g["relative_price"].ewm(span=63, adjust=False).mean()
        ema126 = g["relative_price"].ewm(span=126, adjust=False).mean()

        g["rel_vs_ema21"] = g["relative_price"] / ema21 - 1.0
        g["rel_vs_ema63"] = g["relative_price"] / ema63 - 1.0
        g["ema21_vs_ema63"] = ema21 / ema63 - 1.0
        g["ema63_vs_ema126"] = ema63 / ema126 - 1.0
        g["ema63_slope21"] = ema63 / ema63.shift(21) - 1.0
        g["ema126_slope21"] = ema126 / ema126.shift(21) - 1.0

        low126 = g["relative_price"].rolling(126, min_periods=80).min()
        high126 = g["relative_price"].rolling(126, min_periods=80).max()
        g["dist_low126"] = g["relative_price"] / low126 - 1.0
        g["drawdown_high126"] = g["relative_price"] / high126 - 1.0

        values = g["relative_price"].to_numpy(dtype=float)
        g["bars_since_low126"] = _bars_since_extreme(values, find_max=False)
        g["bars_since_high126"] = _bars_since_extreme(values, find_max=True)

        log_returns = np.log(g["relative_price"]).diff()
        g["rel_vol21"] = (
            log_returns.rolling(21, min_periods=15).std() * math.sqrt(21)
        )
        g["rel_vol63"] = (
            log_returns.rolling(63, min_periods=40).std() * math.sqrt(63)
        )

        frames.append(
            g[
                [
                    "date",
                    "ticker",
                    "primary_benchmark",
                    "rotation_group",
                ]
                + FEATURE_NAMES
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "primary_benchmark",
                "rotation_group",
            ]
            + FEATURE_NAMES
        )

    return pd.concat(frames, ignore_index=True)


def _logistic_probability(
    frame: pd.DataFrame,
    model: dict,
) -> pd.Series:
    means = pd.Series(model["standardizer_mean"], dtype=float).reindex(FEATURE_NAMES)
    scales = pd.Series(model["standardizer_scale"], dtype=float).reindex(FEATURE_NAMES)
    coefs = pd.Series(model["standardized_coefficients"], dtype=float).reindex(
        FEATURE_NAMES
    )

    if means.isna().any() or scales.isna().any() or coefs.isna().any():
        raise ValueError("Frozen model spec is missing one or more required feature parameters.")
    if (scales <= 0).any():
        raise ValueError("Frozen model contains a non-positive standardization scale.")

    x = frame[FEATURE_NAMES].astype(float)
    z = (x - means) / scales
    linear = float(model["intercept"]) + z.mul(coefs, axis=1).sum(axis=1)
    linear = linear.clip(-40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-linear))
    probability = probability.where(x.notna().all(axis=1))
    return probability


def _rank_within_group(
    frame: pd.DataFrame,
    probability_col: str,
    prefix: str,
    minimum_group_size: int,
) -> pd.DataFrame:
    out = frame.copy()
    out[f"{prefix}_peer_rank"] = np.nan
    out[f"{prefix}_peer_size"] = np.nan
    out[f"{prefix}_peer_percentile"] = np.nan
    out[f"{prefix}_low_trust"] = False

    valid = out[probability_col].notna()
    if not valid.any():
        return out

    ranked = out.loc[valid].copy()
    ranked[f"{prefix}_peer_rank"] = ranked.groupby("rotation_group")[
        probability_col
    ].rank(method="average", ascending=True)
    ranked[f"{prefix}_peer_size"] = ranked.groupby("rotation_group")[
        probability_col
    ].transform("count")
    ranked[f"{prefix}_peer_percentile"] = (
        100.0
        * ranked[f"{prefix}_peer_rank"]
        / ranked[f"{prefix}_peer_size"]
    )
    ranked[f"{prefix}_low_trust"] = (
        (ranked[f"{prefix}_peer_size"] >= minimum_group_size)
        & (ranked[f"{prefix}_peer_percentile"] <= 20.0)
    )

    cols = [
        f"{prefix}_peer_rank",
        f"{prefix}_peer_size",
        f"{prefix}_peer_percentile",
        f"{prefix}_low_trust",
    ]
    out.loc[ranked.index, cols] = ranked[cols]
    return out


def _existing_prediction_ids() -> set[str]:
    if not RESEARCH_SHADOW_HISTORY_PATH.exists():
        return set()
    try:
        ids = pd.read_csv(
            RESEARCH_SHADOW_HISTORY_PATH,
            usecols=["prediction_id"],
            dtype=str,
        )["prediction_id"]
        return set(ids.dropna())
    except (ValueError, pd.errors.EmptyDataError):
        return set()


def record_shadow_predictions(
    ohlcv: pd.DataFrame,
    cfg: pd.DataFrame,
    latest: pd.DataFrame,
) -> dict:
    spec = _load_spec()
    model_version = str(spec["model_version"])
    minimum_group_size = int(spec["shadow_rules"].get("minimum_group_size", 5))

    if latest.empty:
        return {
            "model_version": model_version,
            "market_date": None,
            "eligible_rows": 0,
            "appended_rows": 0,
            "skipped_existing_rows": 0,
        }

    latest_work = latest.copy()
    latest_work["date"] = pd.to_datetime(latest_work["date"], errors="coerce").dt.tz_localize(None)
    scored_cross = latest_work[
        latest_work["rank_eligible"].map(as_bool)
        & latest_work["score_mode"].astype(str).str.upper().eq("CROSS_SECTIONAL")
        & latest_work["rotation_score"].notna()
    ].copy()
    market_date = scored_cross["date"].max() if not scored_cross.empty else pd.NaT

    current = scored_cross[scored_cross["date"].eq(market_date)].copy()

    if current.empty:
        return {
            "model_version": model_version,
            "market_date": market_date.strftime("%Y-%m-%d") if pd.notna(market_date) else None,
            "eligible_rows": 0,
            "appended_rows": 0,
            "skipped_existing_rows": 0,
        }

    features = _relative_feature_frame(ohlcv, cfg)
    current_features = features[features["date"].eq(market_date)].copy()
    current = current.merge(
        current_features,
        on=["date", "ticker", "primary_benchmark", "rotation_group"],
        how="left",
        validate="one_to_one",
    )

    current["shadow_feature_complete"] = current[FEATURE_NAMES].notna().all(axis=1)
    current["universal_path_success_probability"] = _logistic_probability(
        current,
        spec["universal_model"],
    )
    current["industry_theme_path_success_probability"] = _logistic_probability(
        current,
        spec["industry_theme_model"],
    )

    current = _rank_within_group(
        current,
        "universal_path_success_probability",
        "universal",
        minimum_group_size,
    )

    use_specialized = current["rotation_group"].eq("US_INDUSTRY_THEME")
    current["challenger_path_success_probability"] = current[
        "universal_path_success_probability"
    ]
    current.loc[use_specialized, "challenger_path_success_probability"] = current.loc[
        use_specialized,
        "industry_theme_path_success_probability",
    ]
    current["challenger_model_used"] = np.where(
        use_specialized,
        "US_INDUSTRY_THEME_SPECIALIZED_V1",
        "UNIVERSAL_PATH_RISK_V1",
    )

    current = _rank_within_group(
        current,
        "challenger_path_success_probability",
        "challenger",
        minimum_group_size,
    )

    current["patch6_top20"] = (
        pd.to_numeric(current["group_percentile"], errors="coerce") >= 80.0
    )
    current["universal_veto_on_patch6"] = (
        current["patch6_top20"] & current["universal_low_trust"].map(bool)
    )
    current["challenger_veto_on_patch6"] = (
        current["patch6_top20"] & current["challenger_low_trust"].map(bool)
    )

    captured_at = pd.Timestamp.now(tz="UTC").isoformat()
    market_date_text = market_date.strftime("%Y-%m-%d")

    rows = pd.DataFrame(
        {
            "prediction_id": [
                f"{model_version}|{market_date_text}|{ticker}"
                for ticker in current["ticker"].astype(str)
            ],
            "captured_at_utc": captured_at,
            "market_date": market_date_text,
            "ticker": current["ticker"].astype(str).to_numpy(),
            "rotation_group": current["rotation_group"].astype(str).to_numpy(),
            "primary_benchmark": current["primary_benchmark"].astype(str).to_numpy(),
            "exposure": current.get("exposure", pd.Series("", index=current.index)).astype(str).to_numpy(),
            "patch6_rotation_score": pd.to_numeric(current["rotation_score"], errors="coerce").to_numpy(),
            "patch6_group_percentile": pd.to_numeric(current["group_percentile"], errors="coerce").to_numpy(),
            "patch6_raw_state": current.get("rotation_state_raw", pd.Series("", index=current.index)).astype(str).to_numpy(),
            "patch6_confirmed_state": current.get("rotation_state_confirmed", pd.Series("", index=current.index)).astype(str).to_numpy(),
            "patch6_top20": current["patch6_top20"].map(bool).to_numpy(),
            "shadow_feature_complete": current["shadow_feature_complete"].map(bool).to_numpy(),
            "universal_path_success_probability": current["universal_path_success_probability"].to_numpy(),
            "universal_peer_rank": current["universal_peer_rank"].to_numpy(),
            "universal_peer_size": current["universal_peer_size"].to_numpy(),
            "universal_peer_percentile": current["universal_peer_percentile"].to_numpy(),
            "universal_low_trust": current["universal_low_trust"].map(bool).to_numpy(),
            "universal_veto_on_patch6": current["universal_veto_on_patch6"].map(bool).to_numpy(),
            "challenger_path_success_probability": current["challenger_path_success_probability"].to_numpy(),
            "challenger_model_used": current["challenger_model_used"].astype(str).to_numpy(),
            "challenger_peer_rank": current["challenger_peer_rank"].to_numpy(),
            "challenger_peer_size": current["challenger_peer_size"].to_numpy(),
            "challenger_peer_percentile": current["challenger_peer_percentile"].to_numpy(),
            "challenger_low_trust": current["challenger_low_trust"].map(bool).to_numpy(),
            "challenger_veto_on_patch6": current["challenger_veto_on_patch6"].map(bool).to_numpy(),
            "model_version": model_version,
            "target_version": str(spec["target_version"]),
            "feature_version": str(spec["feature_version"]),
        }
    )

    for feature in FEATURE_NAMES:
        rows[f"feature_{feature}"] = current[feature].to_numpy()

    rows = rows[PREDICTION_COLUMNS].sort_values(["rotation_group", "ticker"])
    existing_ids = _existing_prediction_ids()
    is_new = ~rows["prediction_id"].isin(existing_ids)
    new_rows = rows[is_new].copy()

    RESEARCH_SHADOW_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not new_rows.empty:
        new_rows.to_csv(
            RESEARCH_SHADOW_HISTORY_PATH,
            mode="a",
            header=not RESEARCH_SHADOW_HISTORY_PATH.exists(),
            index=False,
            float_format="%.10f",
        )

    return {
        "model_version": model_version,
        "market_date": market_date_text,
        "eligible_rows": int(len(rows)),
        "feature_complete_rows": int(rows["shadow_feature_complete"].sum()),
        "appended_rows": int(len(new_rows)),
        "skipped_existing_rows": int((~is_new).sum()),
        "patch6_top20_rows": int(rows["patch6_top20"].sum()),
        "universal_low_trust_rows": int(rows["universal_low_trust"].sum()),
        "universal_veto_rows": int(rows["universal_veto_on_patch6"].sum()),
        "challenger_low_trust_rows": int(rows["challenger_low_trust"].sum()),
        "challenger_veto_rows": int(rows["challenger_veto_on_patch6"].sum()),
    }


def _relative_series(
    closes: pd.DataFrame,
    ticker: str,
    benchmark: str,
) -> pd.Series:
    if ticker not in closes.columns or benchmark not in closes.columns:
        return pd.Series(dtype=float)
    return (closes[ticker] / closes[benchmark]).dropna().sort_index()


def update_shadow_outcomes(ohlcv: pd.DataFrame) -> dict:
    if not RESEARCH_SHADOW_HISTORY_PATH.exists():
        return {
            "prediction_rows": 0,
            "matured_21": 0,
            "matured_63": 0,
            "matured_84": 0,
            "matured_126": 0,
        }

    history = pd.read_csv(RESEARCH_SHADOW_HISTORY_PATH, dtype={"ticker": str})
    if history.empty:
        return {
            "prediction_rows": 0,
            "matured_21": 0,
            "matured_63": 0,
            "matured_84": 0,
            "matured_126": 0,
        }

    data = ohlcv.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None)
    data["ticker"] = data["ticker"].astype(str).str.upper()
    closes = (
        data.dropna(subset=["date", "ticker", "close"])
        .pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
    )
    as_of = closes.index.max() if not closes.empty else pd.NaT

    rows: list[dict] = []
    for prediction in history.itertuples(index=False):
        ticker = str(prediction.ticker).upper()
        benchmark = str(prediction.primary_benchmark).upper()
        market_date = pd.Timestamp(prediction.market_date)
        relative = _relative_series(closes, ticker, benchmark)

        outcome = {
            "prediction_id": str(prediction.prediction_id),
            "model_version": str(prediction.model_version),
            "market_date": market_date.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "primary_benchmark": benchmark,
            "outcome_as_of_market_date": as_of.strftime("%Y-%m-%d") if pd.notna(as_of) else None,
            "matured_21": False,
            "matured_63": False,
            "matured_84": False,
            "matured_126": False,
            "fwd_relative_return_21": np.nan,
            "fwd_relative_return_63": np.nan,
            "fwd_relative_return_84": np.nan,
            "fwd_relative_return_126": np.nan,
            "fwd_relative_mdd_126": np.nan,
            "band_median_relative_return": np.nan,
            "investable_loose": np.nan,
            "investable_primary": np.nan,
            "investable_strict": np.nan,
        }

        if relative.empty or market_date not in relative.index:
            rows.append(outcome)
            continue

        pos = int(relative.index.get_loc(market_date))
        base = float(relative.iloc[pos])
        available_forward = len(relative) - 1 - pos

        for horizon in [21, 63, 84, 126]:
            matured = available_forward >= horizon
            outcome[f"matured_{horizon}"] = matured
            if matured:
                outcome[f"fwd_relative_return_{horizon}"] = (
                    float(relative.iloc[pos + horizon]) / base - 1.0
                )

        if outcome["matured_126"]:
            path = relative.iloc[pos + 1 : pos + 127].astype(float) / base - 1.0
            outcome["fwd_relative_mdd_126"] = float(path.min())
            band = np.median(
                [
                    outcome["fwd_relative_return_63"],
                    outcome["fwd_relative_return_84"],
                    outcome["fwd_relative_return_126"],
                ]
            )
            outcome["band_median_relative_return"] = float(band)
            mdd = float(outcome["fwd_relative_mdd_126"])
            outcome["investable_loose"] = bool((band >= 0.00) and (mdd >= -0.12))
            outcome["investable_primary"] = bool((band >= 0.02) and (mdd >= -0.10))
            outcome["investable_strict"] = bool((band >= 0.04) and (mdd >= -0.08))

        rows.append(outcome)

    outcomes = pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
    outcomes = outcomes.sort_values(["market_date", "ticker"])
    RESEARCH_SHADOW_OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(
        RESEARCH_SHADOW_OUTCOMES_PATH,
        index=False,
        float_format="%.10f",
    )

    return {
        "prediction_rows": int(len(outcomes)),
        "matured_21": int(outcomes["matured_21"].sum()),
        "matured_63": int(outcomes["matured_63"].sum()),
        "matured_84": int(outcomes["matured_84"].sum()),
        "matured_126": int(outcomes["matured_126"].sum()),
    }


def run_shadow_research(
    ohlcv: pd.DataFrame,
    cfg: pd.DataFrame,
    latest: pd.DataFrame,
) -> dict:
    prediction_summary = record_shadow_predictions(ohlcv, cfg, latest)
    outcome_summary = update_shadow_outcomes(ohlcv)

    status = {
        "status": "ok",
        "prediction": prediction_summary,
        "outcomes": outcome_summary,
        "history_file": str(RESEARCH_SHADOW_HISTORY_PATH.name),
        "outcomes_file": str(RESEARCH_SHADOW_OUTCOMES_PATH.name),
        "dashboard_effect": "none",
    }
    RESEARCH_SHADOW_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_SHADOW_STATUS_PATH.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    return status
