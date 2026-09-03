from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import average_precision_score, roc_auc_score

from research.clean_sheet_predictive.build_features import build_features
from research.clean_sheet_predictive.build_outcomes import add_forward_outcomes
from src.common import load_config
from src.fetch_market_data import fetch_market_data, update_ohlcv_history


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = Path(__file__).with_name("r4_spec.json")
DEFAULT_REGISTRY = Path(__file__).with_name("r4_model_registry.json")
RESULT_DIR = ROOT / "results" / "clean_sheet_r4"
PREDICTIONS_PATH = RESULT_DIR / "r4_predictions.csv"
ANCHORS_PATH = RESULT_DIR / "r4_anchor_manifest.csv"
OUTCOMES_PATH = RESULT_DIR / "r4_outcomes.csv"
RANKING_MONTHLY_PATH = RESULT_DIR / "r4_monthly_ranking_metrics.csv"
AVOID_MONTHLY_PATH = RESULT_DIR / "r4_monthly_avoid_metrics.csv"
ASSESSMENT_CSV_PATH = RESULT_DIR / "r4_assessment.csv"
ASSESSMENT_JSON_PATH = RESULT_DIR / "r4_assessment.json"
STATUS_PATH = RESULT_DIR / "r4_status.json"
LATEST_PATH = RESULT_DIR / "r4_latest.json"
REPORT_PATH = RESULT_DIR / "r4_report.md"
FETCH_FAILURES_PATH = RESULT_DIR / "r4_fetch_failures.csv"
CHART_DIR = RESULT_DIR / "charts"


PREDICTION_COLUMNS = [
    "prediction_id",
    "model_version",
    "model_year",
    "issued_at_utc",
    "anchor_date",
    "ticker",
    "horizon_bars",
    "horizon_label",
    "feature_count",
    "ridge_prediction",
    "ridge_rank_pct",
    "ridge_top_quintile",
    "ridge_top_decile",
    "equal_weight_score",
    "equal_weight_rank_pct",
    "equal_weight_top_quintile",
    "equal_weight_top_decile",
    "avoid_probability",
    "avoid_risk_rank_pct",
    "avoid_top_risk_quintile",
    "avoid_top_risk_decile",
    "anchor_close",
    "anchor_spy_close",
]

ANCHOR_COLUMNS = [
    "anchor_date",
    "horizon_bars",
    "horizon_label",
    "model_version",
    "issued_at_utc",
    "eligible_rows",
    "frozen_universe_count",
    "coverage_fraction",
]

OUTCOME_COLUMNS = [
    "prediction_id",
    "recorded_at_utc",
    "anchor_date",
    "ticker",
    "horizon_bars",
    "horizon_label",
    "outcome_end_date",
    "fwd_return",
    "fwd_spy_relative_return",
    "fwd_max_drawdown",
    "fwd_spy_rel_mae",
    "fwd_spy_rel_mfe",
    "beat_spy",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return _empty(columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns].copy()


def _write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    out = out[columns]
    out.to_csv(path, index=False)


def _runtime_integrity(spec: dict, registry: dict) -> None:
    if registry.get("feature_version") != spec.get("feature_version"):
        raise RuntimeError("R4 feature-version mismatch.")

    expected_yf = registry["data_source"]["yfinance_version"]
    if yf.__version__ != expected_yf:
        raise RuntimeError(
            f"R4 requires yfinance {expected_yf}; installed {yf.__version__}."
        )

    paths = {
        "build_features.py": ROOT / "research/clean_sheet_predictive/build_features.py",
        "feature_definitions.py": ROOT / "research/clean_sheet_predictive/feature_definitions.py",
        "build_outcomes.py": ROOT / "research/clean_sheet_predictive/build_outcomes.py",
    }
    failures = []
    for name, path in paths.items():
        actual = _sha256(path)
        expected = registry["source_code_hashes"][name]
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")
    if failures:
        raise RuntimeError(
            "R4 frozen research code changed. Prospective run blocked: "
            + "; ".join(failures)
        )


def _frozen_config(cfg: pd.DataFrame, frozen_tickers: set[str]) -> pd.DataFrame:
    available = set(cfg["ticker"].astype(str).str.upper())
    missing = sorted(frozen_tickers - available)
    if missing:
        raise RuntimeError(
            "Current config is missing frozen R4 tickers: " + ", ".join(missing)
        )
    out = cfg[cfg["ticker"].isin(frozen_tickers)].copy()
    out["enabled"] = True
    out["query_ohlcv"] = True
    return out


def _completed_anchor_dates(
    dates: pd.Series,
    first_eligible_anchor_date: str,
) -> list[pd.Timestamp]:
    d = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    if d.empty:
        return []
    latest_period = d.iloc[-1].to_period("M")
    frame = pd.DataFrame({"date": d})
    frame["month"] = frame["date"].dt.to_period("M")
    monthly_last = frame.groupby("month", observed=True)["date"].max()
    first = pd.Timestamp(first_eligible_anchor_date)
    return [
        pd.Timestamp(date)
        for period, date in monthly_last.items()
        if period < latest_period and pd.Timestamp(date) >= first
    ]


def _prediction_id(
    model_version: str,
    anchor_date: pd.Timestamp,
    horizon: int,
    ticker: str,
) -> str:
    raw = f"{model_version}|{pd.Timestamp(anchor_date).strftime('%Y-%m-%d')}|{horizon}|{ticker}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _score_anchor_horizon(
    anchor: pd.DataFrame,
    *,
    model_year_record: dict,
    horizon: int,
    spec: dict,
    registry: dict,
    issued_at: str,
) -> tuple[pd.DataFrame, dict]:
    model = model_year_record["horizons"][str(horizon)]
    features = list(model["features"])

    p = anchor[["date", "ticker", "close"] + features].copy()
    for feature in features:
        p[feature] = p[feature].rank(
            method="average", pct=True, na_option="keep"
        )
    p = p.replace([np.inf, -np.inf], np.nan).dropna(subset=features).copy()

    min_rows = int(spec["minimum_prediction_cross_section_rows"])
    if len(p) < min_rows:
        raise RuntimeError(
            f"R4 anchor {anchor['date'].iloc[0].date()} / {horizon} bars has only "
            f"{len(p)} feature-complete rows (< {min_rows})."
        )

    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    scale = np.where(np.abs(scale) < 1e-15, 1.0, scale)
    x = p[features].to_numpy(dtype=float)
    xs = (x - mean) / scale

    ridge_coef = np.asarray(model["ridge"]["coefficients"], dtype=float)
    ridge = xs @ ridge_coef + float(model["ridge"]["intercept"])

    logit_coef = np.asarray(model["logistic"]["coefficients"], dtype=float)
    logit_z = xs @ logit_coef + float(model["logistic"]["intercept"])
    avoid = _sigmoid(logit_z)

    orientation = model["equal_weight_orientation"]
    equal = np.mean(
        np.column_stack(
            [p[f].to_numpy(dtype=float) * int(orientation[f]) for f in features]
        ),
        axis=1,
    )

    p["ridge_prediction"] = ridge
    p["equal_weight_score"] = equal
    p["avoid_probability"] = avoid

    p["ridge_rank_pct"] = p["ridge_prediction"].rank(method="average", pct=True)
    p["equal_weight_rank_pct"] = p["equal_weight_score"].rank(method="average", pct=True)
    p["avoid_risk_rank_pct"] = p["avoid_probability"].rank(method="average", pct=True)

    spy = anchor.loc[anchor["ticker"].eq("SPY"), "close"]
    if spy.empty:
        raise RuntimeError("SPY close missing on prospective anchor.")
    spy_close = float(spy.iloc[0])

    anchor_date = pd.Timestamp(p["date"].iloc[0])
    model_version = str(model_year_record["model_version"])
    out = pd.DataFrame(
        {
            "prediction_id": [
                _prediction_id(model_version, anchor_date, horizon, t)
                for t in p["ticker"]
            ],
            "model_version": model_version,
            "model_year": int(model_year_record["model_year"]),
            "issued_at_utc": issued_at,
            "anchor_date": anchor_date.strftime("%Y-%m-%d"),
            "ticker": p["ticker"].astype(str).to_numpy(),
            "horizon_bars": int(horizon),
            "horizon_label": spec["horizon_labels"][str(horizon)],
            "feature_count": len(features),
            "ridge_prediction": p["ridge_prediction"].to_numpy(),
            "ridge_rank_pct": p["ridge_rank_pct"].to_numpy(),
            "ridge_top_quintile": p["ridge_rank_pct"].gt(0.80).to_numpy(),
            "ridge_top_decile": p["ridge_rank_pct"].gt(0.90).to_numpy(),
            "equal_weight_score": p["equal_weight_score"].to_numpy(),
            "equal_weight_rank_pct": p["equal_weight_rank_pct"].to_numpy(),
            "equal_weight_top_quintile": p["equal_weight_rank_pct"].gt(0.80).to_numpy(),
            "equal_weight_top_decile": p["equal_weight_rank_pct"].gt(0.90).to_numpy(),
            "avoid_probability": p["avoid_probability"].to_numpy(),
            "avoid_risk_rank_pct": p["avoid_risk_rank_pct"].to_numpy(),
            "avoid_top_risk_quintile": p["avoid_risk_rank_pct"].gt(0.80).to_numpy(),
            "avoid_top_risk_decile": p["avoid_risk_rank_pct"].gt(0.90).to_numpy(),
            "anchor_close": p["close"].to_numpy(dtype=float),
            "anchor_spy_close": spy_close,
        }
    )

    manifest = {
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "horizon_bars": int(horizon),
        "horizon_label": spec["horizon_labels"][str(horizon)],
        "model_version": model_version,
        "issued_at_utc": issued_at,
        "eligible_rows": int(len(out)),
        "frozen_universe_count": int(registry["frozen_universe_count"]),
        "coverage_fraction": float(len(out) / registry["frozen_universe_count"]),
    }
    return out, manifest


def _append_new_anchors(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    manifests: pd.DataFrame,
    *,
    spec: dict,
    registry: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    completed = _completed_anchor_dates(
        features["date"], spec["first_eligible_anchor_date"]
    )
    existing_anchor_dates = set(
        pd.to_datetime(manifests["anchor_date"], errors="coerce")
        .dropna()
        .dt.strftime("%Y-%m-%d")
    ) if not manifests.empty else set()

    new_prediction_parts = []
    new_manifests = []
    issued_anchors = 0

    frozen = set(registry["frozen_universe_tickers"])
    features = features[features["ticker"].isin(frozen)].copy()

    for anchor_date in completed:
        key = anchor_date.strftime("%Y-%m-%d")
        if key in existing_anchor_dates:
            counts = manifests[manifests["anchor_date"].eq(key)]["horizon_bars"].nunique()
            if counts != len(spec["horizons_bars"]):
                raise RuntimeError(
                    f"R4 anchor {key} is partially issued ({counts} horizons). "
                    "Manual audit required; R4 will not fill an old partial anchor."
                )
            continue

        model_year = str(anchor_date.year)
        if model_year not in registry["models_by_year"]:
            raise RuntimeError(
                f"No frozen R4 model registry exists for anchor year {model_year}. "
                "Run the annual R4 model refresh before issuing this anchor."
            )

        anchor = features[features["date"].eq(anchor_date)].copy()
        if anchor.empty:
            continue

        issued_at = _utc_now()
        staged_parts = []
        staged_manifests = []
        for h in map(int, spec["horizons_bars"]):
            pred, manifest = _score_anchor_horizon(
                anchor,
                model_year_record=registry["models_by_year"][model_year],
                horizon=h,
                spec=spec,
                registry=registry,
                issued_at=issued_at,
            )
            staged_parts.append(pred)
            staged_manifests.append(manifest)

        new_prediction_parts.extend(staged_parts)
        new_manifests.extend(staged_manifests)
        issued_anchors += 1

    if new_prediction_parts:
        new_preds = pd.concat(new_prediction_parts, ignore_index=True)
        overlap = set(predictions["prediction_id"]).intersection(new_preds["prediction_id"])
        if overlap:
            raise RuntimeError(
                "R4 immutable prediction IDs unexpectedly overlap new rows: "
                + ", ".join(sorted(list(overlap))[:10])
            )
        predictions = pd.concat([predictions, new_preds], ignore_index=True)

    if new_manifests:
        manifests = pd.concat([manifests, pd.DataFrame(new_manifests)], ignore_index=True)

    return predictions, manifests, issued_anchors


def _append_matured_outcomes(
    feature_outcomes: pd.DataFrame,
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    spec: dict,
) -> tuple[pd.DataFrame, int]:
    if predictions.empty:
        return outcomes, 0

    existing = set(outcomes["prediction_id"].astype(str)) if not outcomes.empty else set()
    new_rows = []
    recorded_at = _utc_now()

    for h in map(int, spec["horizons_bars"]):
        pred = predictions[predictions["horizon_bars"].eq(h)].copy()
        if pred.empty:
            continue
        pred["anchor_date_dt"] = pd.to_datetime(pred["anchor_date"])

        cols = [
            "date",
            "ticker",
            f"outcome_end_date_{h}",
            f"fwd_return_{h}",
            f"fwd_spy_relative_return_{h}",
            f"fwd_max_drawdown_{h}",
            f"fwd_spy_rel_mae_{h}",
            f"fwd_spy_rel_mfe_{h}",
        ]
        source = feature_outcomes[cols].rename(columns={"date": "anchor_date_dt"})
        merged = pred.merge(source, on=["anchor_date_dt", "ticker"], how="left")
        matured = merged[
            merged[f"outcome_end_date_{h}"].notna()
            & ~merged["prediction_id"].astype(str).isin(existing)
        ].copy()

        for row in matured.itertuples(index=False):
            rel = float(getattr(row, f"fwd_spy_relative_return_{h}"))
            new_rows.append(
                {
                    "prediction_id": row.prediction_id,
                    "recorded_at_utc": recorded_at,
                    "anchor_date": pd.Timestamp(row.anchor_date_dt).strftime("%Y-%m-%d"),
                    "ticker": row.ticker,
                    "horizon_bars": h,
                    "horizon_label": spec["horizon_labels"][str(h)],
                    "outcome_end_date": pd.Timestamp(
                        getattr(row, f"outcome_end_date_{h}")
                    ).strftime("%Y-%m-%d"),
                    "fwd_return": float(getattr(row, f"fwd_return_{h}")),
                    "fwd_spy_relative_return": rel,
                    "fwd_max_drawdown": float(getattr(row, f"fwd_max_drawdown_{h}")),
                    "fwd_spy_rel_mae": float(getattr(row, f"fwd_spy_rel_mae_{h}")),
                    "fwd_spy_rel_mfe": float(getattr(row, f"fwd_spy_rel_mfe_{h}")),
                    "beat_spy": bool(rel > 0),
                }
            )
            existing.add(str(row.prediction_id))

    if new_rows:
        outcomes = pd.concat([outcomes, pd.DataFrame(new_rows)], ignore_index=True)

    return outcomes, len(new_rows)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    return float(pair["x"].rank().corr(pair["y"].rank()))


def _quantile_masks(score: pd.Series) -> dict[str, pd.Series]:
    rank = score.rank(method="average", pct=True)
    return {
        "top_q": rank.gt(0.80),
        "bottom_q": rank.le(0.20),
        "top_d": rank.gt(0.90),
        "bottom_d": rank.le(0.10),
    }


def _evaluate_months(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    spec: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty or outcomes.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = predictions.merge(
        outcomes[
            [
                "prediction_id",
                "fwd_spy_relative_return",
            ]
        ],
        on="prediction_id",
        how="left",
    )

    ranking_rows = []
    avoid_rows = []
    min_rows = int(spec["minimum_evaluation_cross_section_rows"])
    maturity_fraction = float(spec["minimum_maturity_fraction"])

    for (h, anchor_date), g in merged.groupby(
        ["horizon_bars", "anchor_date"], observed=True, sort=True
    ):
        issued = len(g)
        matured = g.dropna(subset=["fwd_spy_relative_return"]).copy()
        required = max(min_rows, int(math.ceil(issued * maturity_fraction)))
        if len(matured) < required:
            continue

        valid_returns = matured["fwd_spy_relative_return"]
        actual_rank = valid_returns.rank(method="first", pct=True)
        matured["avoid_actual"] = (
            actual_rank <= float(spec["avoid_target"]["bottom_fraction"])
        ).astype(int)

        for method, score_col in {
            "ridge": "ridge_prediction",
            "equal_weight": "equal_weight_score",
        }.items():
            p = matured.dropna(subset=[score_col, "fwd_spy_relative_return"]).copy()
            if len(p) < min_rows or p[score_col].nunique() < 2:
                continue
            masks = _quantile_masks(p[score_col])
            tq, bq = p[masks["top_q"]], p[masks["bottom_q"]]
            td, bd = p[masks["top_d"]], p[masks["bottom_d"]]
            ranking_rows.append(
                {
                    "horizon_bars": int(h),
                    "horizon_label": spec["horizon_labels"][str(int(h))],
                    "anchor_date": anchor_date,
                    "method": method,
                    "issued_rows": issued,
                    "matured_rows": int(len(p)),
                    "maturity_fraction": float(len(p) / issued),
                    "ic": _safe_spearman(p[score_col], p["fwd_spy_relative_return"]),
                    "top_quintile_mean_rel_return": float(tq["fwd_spy_relative_return"].mean()),
                    "bottom_quintile_mean_rel_return": float(bq["fwd_spy_relative_return"].mean()),
                    "q5_minus_q1_spread": float(
                        tq["fwd_spy_relative_return"].mean()
                        - bq["fwd_spy_relative_return"].mean()
                    ),
                    "top_quintile_beat_spy_rate": float(
                        (tq["fwd_spy_relative_return"] > 0).mean()
                    ),
                    "top_decile_mean_rel_return": float(td["fwd_spy_relative_return"].mean()),
                    "bottom_decile_mean_rel_return": float(bd["fwd_spy_relative_return"].mean()),
                    "top_decile_beat_spy_rate": float(
                        (td["fwd_spy_relative_return"] > 0).mean()
                    ),
                }
            )

        p = matured.dropna(
            subset=["avoid_probability", "avoid_actual", "fwd_spy_relative_return"]
        ).copy()
        if (
            len(p) >= min_rows
            and p["avoid_actual"].nunique() > 1
            and p["avoid_probability"].nunique() > 1
        ):
            labels = p["avoid_actual"].astype(int)
            prob = p["avoid_probability"].astype(float)
            base = float(labels.mean())
            risk_rank = prob.rank(method="average", pct=True)
            q = p[risk_rank.gt(0.80)]
            d = p[risk_rank.gt(0.90)]
            avoid_rows.append(
                {
                    "horizon_bars": int(h),
                    "horizon_label": spec["horizon_labels"][str(int(h))],
                    "anchor_date": anchor_date,
                    "issued_rows": issued,
                    "matured_rows": int(len(p)),
                    "maturity_fraction": float(len(p) / issued),
                    "base_rate": base,
                    "roc_auc": float(roc_auc_score(labels, prob)),
                    "average_precision": float(average_precision_score(labels, prob)),
                    "top_risk_quintile_precision": float(q["avoid_actual"].mean()),
                    "top_risk_quintile_lift": (
                        float(q["avoid_actual"].mean() / base) if base > 0 else np.nan
                    ),
                    "top_risk_quintile_mean_rel_return": float(
                        q["fwd_spy_relative_return"].mean()
                    ),
                    "top_risk_quintile_beat_spy_rate": float(
                        (q["fwd_spy_relative_return"] > 0).mean()
                    ),
                    "top_risk_decile_precision": float(d["avoid_actual"].mean()),
                    "top_risk_decile_lift": (
                        float(d["avoid_actual"].mean() / base) if base > 0 else np.nan
                    ),
                    "top_risk_decile_mean_rel_return": float(
                        d["fwd_spy_relative_return"].mean()
                    ),
                }
            )

    return pd.DataFrame(ranking_rows), pd.DataFrame(avoid_rows)


def _block_bootstrap_mean_ci(
    values: pd.Series,
    *,
    block_months: int,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = len(arr)
    if n < max(6, block_months * 2):
        return float("nan"), float("nan")
    block = min(block_months, n)
    starts_max = n - block
    blocks_needed = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        starts = rng.integers(0, starts_max + 1, size=blocks_needed)
        sample = np.concatenate([arr[s : s + block] for s in starts])[:n]
        means[i] = float(np.mean(sample))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _metric_summary(
    frame: pd.DataFrame,
    metrics: list[str],
    *,
    key_prefix: str,
    spec: dict,
) -> dict:
    out = {}
    if frame.empty:
        for metric in metrics:
            out[f"mean_{metric}"] = None
            out[f"{metric}_ci_low"] = None
            out[f"{metric}_ci_high"] = None
        return out

    cfg = spec["prospective_bootstrap"]
    for metric in metrics:
        values = frame[metric]
        mean_value = float(values.mean())
        out[f"mean_{metric}"] = mean_value if np.isfinite(mean_value) else None
        seed = (
            int(cfg["seed"])
            + zlib.crc32(f"{key_prefix}|{metric}".encode("utf-8"))
        ) % (2**32 - 1)
        lo, hi = _block_bootstrap_mean_ci(
            values,
            block_months=int(cfg["block_months"]),
            repetitions=int(cfg["repetitions"]),
            seed=seed,
        )
        out[f"{metric}_ci_low"] = lo if np.isfinite(lo) else None
        out[f"{metric}_ci_high"] = hi if np.isfinite(hi) else None
    return out


def _favor_pass(summary: dict, months: int, spec: dict) -> bool:
    if months < int(spec["favor_gate"]["minimum_evaluated_months"]):
        return False
    required = [
        summary.get("ic_ci_low"),
        summary.get("top_decile_mean_rel_return_ci_low"),
        summary.get("top_decile_beat_spy_rate_ci_low"),
        summary.get("q5_minus_q1_spread_ci_low"),
    ]
    if any(x is None for x in required):
        return False
    return bool(
        summary["ic_ci_low"] > 0
        and summary["top_decile_mean_rel_return_ci_low"] > 0
        and summary["top_decile_beat_spy_rate_ci_low"] > 0.50
        and summary["q5_minus_q1_spread_ci_low"] > 0
    )


def _avoid_pass(summary: dict, months: int, spec: dict) -> bool:
    if months < int(spec["avoid_gate"]["minimum_evaluated_months"]):
        return False
    required = [
        summary.get("roc_auc_ci_low"),
        summary.get("top_risk_quintile_precision_ci_low"),
        summary.get("top_risk_quintile_mean_rel_return_ci_high"),
    ]
    if any(x is None for x in required):
        return False
    return bool(
        summary["roc_auc_ci_low"] > 0.50
        and summary["top_risk_quintile_precision_ci_low"] > 0.20
        and summary["top_risk_quintile_mean_rel_return_ci_high"] < 0
    )


def _status_for(months: int, any_gate: bool) -> str:
    if months < 3:
        return "INSUFFICIENT"
    if months < 12:
        return "EARLY"
    if months < 24:
        return "PROMISING" if any_gate else "MIXED"
    return "CONFIRMED_PROSPECTIVE" if any_gate else "FAILED_TO_CONFIRM"


def _build_assessment(
    ranking: pd.DataFrame,
    avoid: pd.DataFrame,
    spec: dict,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    details = {}

    ranking_metrics = [
        "ic",
        "top_decile_mean_rel_return",
        "top_decile_beat_spy_rate",
        "q5_minus_q1_spread",
        "top_quintile_mean_rel_return",
    ]
    avoid_metrics = [
        "base_rate",
        "roc_auc",
        "average_precision",
        "top_risk_quintile_precision",
        "top_risk_quintile_lift",
        "top_risk_quintile_mean_rel_return",
        "top_risk_decile_precision",
        "top_risk_decile_lift",
        "top_risk_decile_mean_rel_return",
    ]

    for h in map(int, spec["horizons_bars"]):
        r_h = ranking[ranking["horizon_bars"].eq(h)] if not ranking.empty else pd.DataFrame()
        a_h = avoid[avoid["horizon_bars"].eq(h)] if not avoid.empty else pd.DataFrame()
        months = int(
            max(
                r_h["anchor_date"].nunique() if not r_h.empty else 0,
                a_h["anchor_date"].nunique() if not a_h.empty else 0,
            )
        )

        method_details = {}
        favor_any = False
        for method in ["ridge", "equal_weight"]:
            g = r_h[r_h["method"].eq(method)] if not r_h.empty else pd.DataFrame()
            s = _metric_summary(
                g,
                ranking_metrics,
                key_prefix=f"r4|{h}|{method}",
                spec=spec,
            )
            s = {
                k.replace("mean_", "", 1) if k.startswith("mean_") else k: v
                for k, v in s.items()
            }
            # Keep explicit mean names as well for easier CSV/reporting.
            normalized = {}
            for metric in ranking_metrics:
                normalized[f"mean_{metric}"] = s.get(metric)
                normalized[f"{metric}_ci_low"] = s.get(f"{metric}_ci_low")
                normalized[f"{metric}_ci_high"] = s.get(f"{metric}_ci_high")
            # Gate helper expects non-mean point names + CIs.
            gate_view = {
                "ic_ci_low": normalized["ic_ci_low"],
                "top_decile_mean_rel_return_ci_low": normalized["top_decile_mean_rel_return_ci_low"],
                "top_decile_beat_spy_rate_ci_low": normalized["top_decile_beat_spy_rate_ci_low"],
                "q5_minus_q1_spread_ci_low": normalized["q5_minus_q1_spread_ci_low"],
            }
            passed = _favor_pass(gate_view, months, spec)
            normalized["favor_gate_pass"] = passed
            favor_any = favor_any or passed
            method_details[method] = normalized

        a_summary_raw = _metric_summary(
            a_h,
            avoid_metrics,
            key_prefix=f"r4|{h}|avoid",
            spec=spec,
        )
        a_norm = {}
        for metric in avoid_metrics:
            a_norm[f"mean_{metric}"] = a_summary_raw[f"mean_{metric}"]
            a_norm[f"{metric}_ci_low"] = a_summary_raw[f"{metric}_ci_low"]
            a_norm[f"{metric}_ci_high"] = a_summary_raw[f"{metric}_ci_high"]
        a_gate_view = {
            "roc_auc_ci_low": a_norm["roc_auc_ci_low"],
            "top_risk_quintile_precision_ci_low": a_norm["top_risk_quintile_precision_ci_low"],
            "top_risk_quintile_mean_rel_return_ci_high": a_norm[
                "top_risk_quintile_mean_rel_return_ci_high"
            ],
        }
        avoid_pass = _avoid_pass(a_gate_view, months, spec)
        a_norm["avoid_gate_pass"] = avoid_pass

        evidence_status = _status_for(months, favor_any or avoid_pass)

        rows.append(
            {
                "horizon_bars": h,
                "horizon_label": spec["horizon_labels"][str(h)],
                "primary_research_horizon": h == int(spec["primary_research_horizon_bars"]),
                "evaluated_months": months,
                "evidence_status": evidence_status,
                "ridge_favor_gate_pass": bool(method_details["ridge"]["favor_gate_pass"]),
                "equal_weight_favor_gate_pass": bool(method_details["equal_weight"]["favor_gate_pass"]),
                "avoid_gate_pass": bool(avoid_pass),
                "ridge_mean_ic": method_details["ridge"]["mean_ic"],
                "equal_weight_mean_ic": method_details["equal_weight"]["mean_ic"],
                "avoid_mean_roc_auc": a_norm["mean_roc_auc"],
                "avoid_mean_precision": a_norm["mean_top_risk_quintile_precision"],
                "avoid_mean_lift": a_norm["mean_top_risk_quintile_lift"],
                "avoid_mean_rel_return": a_norm["mean_top_risk_quintile_mean_rel_return"],
            }
        )
        details[str(h)] = {
            "horizon_label": spec["horizon_labels"][str(h)],
            "evaluated_months": months,
            "evidence_status": evidence_status,
            "ranking": method_details,
            "avoid": a_norm,
        }

    return pd.DataFrame(rows), {
        "research_patch": "R4",
        "generated_at_utc": _utc_now(),
        "prospective_only": True,
        "historical_reoptimization": False,
        "by_horizon": details,
    }


def _make_latest(
    predictions: pd.DataFrame,
    spec: dict,
) -> dict:
    if predictions.empty:
        return {
            "research_patch": "R4",
            "status": "waiting_for_first_completed_month",
            "latest_anchor_date": None,
            "by_horizon": {},
        }

    latest = predictions["anchor_date"].max()
    p = predictions[predictions["anchor_date"].eq(latest)].copy()

    # CSV-loaded columns and columns concatenated onto an initially empty
    # DataFrame can retain object dtype even when every valid value is numeric.
    # Normalize all values used by the latest-ranking payload and discard
    # non-finite values from each ranking instead of treating them as zero.
    ranking_columns = (
        "ridge_prediction",
        "ridge_rank_pct",
        "equal_weight_score",
        "equal_weight_rank_pct",
        "avoid_probability",
        "avoid_risk_rank_pct",
    )
    for column in ranking_columns:
        if column in p.columns:
            p[column] = pd.to_numeric(p[column], errors="coerce")

    result = {}
    for h, g in p.groupby("horizon_bars", sort=True):
        ridge_rows = g[
            np.isfinite(g["ridge_prediction"].to_numpy(dtype=float))
        ]
        equal_weight_rows = g[
            np.isfinite(g["equal_weight_score"].to_numpy(dtype=float))
        ]
        avoid_rows = g[
            np.isfinite(g["avoid_probability"].to_numpy(dtype=float))
        ]

        result[str(int(h))] = {
            "horizon_label": spec["horizon_labels"][str(int(h))],
            "ridge_top_10": ridge_rows.nlargest(10, "ridge_prediction")[
                ["ticker", "ridge_prediction", "ridge_rank_pct"]
            ].to_dict(orient="records"),
            "equal_weight_top_10": equal_weight_rows.nlargest(
                10, "equal_weight_score"
            )[
                ["ticker", "equal_weight_score", "equal_weight_rank_pct"]
            ].to_dict(orient="records"),
            "avoid_highest_risk_10": avoid_rows.nlargest(
                10, "avoid_probability"
            )[
                ["ticker", "avoid_probability", "avoid_risk_rank_pct"]
            ].to_dict(orient="records"),
        }
    return {
        "research_patch": "R4",
        "status": "prospective_shadow",
        "latest_anchor_date": latest,
        "by_horizon": result,
        "important_note": (
            "These are immutable prospective shadow rankings, not production recommendations."
        ),
    }


def _make_charts(
    ranking: pd.DataFrame,
    avoid: pd.DataFrame,
) -> list[str]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    if not ranking.empty:
        for method in ["ridge", "equal_weight"]:
            g = ranking[ranking["method"].eq(method)].copy()
            if g.empty:
                continue
            pivot = g.pivot(index="anchor_date", columns="horizon_label", values="ic")
            ax = pivot.plot(figsize=(9, 5), marker="o")
            ax.set_title(f"R4 Prospective Monthly Ranking IC — {method}")
            ax.set_xlabel("Prospective anchor")
            ax.set_ylabel("Spearman IC")
            ax.axhline(0, linewidth=1)
            plt.tight_layout()
            path = CHART_DIR / f"ranking_ic_{method}.png"
            plt.savefig(path, dpi=160)
            plt.close()
            made.append(path.name)

    if not avoid.empty:
        pivot = avoid.pivot(
            index="anchor_date",
            columns="horizon_label",
            values="top_risk_quintile_lift",
        )
        ax = pivot.plot(figsize=(9, 5), marker="o")
        ax.set_title("R4 Prospective AVOID Precision Lift")
        ax.set_xlabel("Prospective anchor")
        ax.set_ylabel("Precision lift vs base rate")
        ax.axhline(1.0, linewidth=1)
        plt.tight_layout()
        path = CHART_DIR / "avoid_precision_lift.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(path.name)
    return made


def _write_report(
    status: dict,
    assessment: pd.DataFrame,
) -> None:
    lines = [
        "# R4 — Prospective Clean-Sheet Shadow Validation",
        "",
        "**This report contains only predictions issued after the R4 inception date.**",
        "",
        f"- Latest market date: {status.get('latest_market_date')}",
        f"- Latest completed prospective anchor: {status.get('latest_issued_anchor_date')}",
        f"- Immutable prediction rows: {status.get('prediction_rows', 0):,}",
        f"- Immutable matured outcome rows: {status.get('matured_outcome_rows', 0):,}",
        "",
        "## Evidence status",
        "",
    ]
    if assessment.empty:
        lines.append("- No prospective anchor has matured enough for evaluation yet.")
    else:
        for row in assessment.itertuples(index=False):
            lines.append(
                f"- **{row.horizon_label}**: {row.evidence_status}; "
                f"{row.evaluated_months} evaluated month(s); "
                f"Ridge IC {row.ridge_mean_ic if pd.notna(row.ridge_mean_ic) else 'n/a'}; "
                f"AVOID AUC {row.avoid_mean_roc_auc if pd.notna(row.avoid_mean_roc_auc) else 'n/a'}."
            )

    lines.extend(
        [
            "",
            "## Frozen rules",
            "",
            "- R4 does not alter the R2.1 feature set, thresholds, or model hyperparameters.",
            "- Predictions are month-end only and immutable once issued.",
            "- Outcomes are appended once when mature and never rewritten.",
            "- No production dashboard recommendation is changed by R4.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_r4(*, fetch: bool = True) -> dict:
    spec = _read_json(DEFAULT_SPEC)
    registry = _read_json(DEFAULT_REGISTRY)
    _runtime_integrity(spec, registry)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    frozen = set(registry["frozen_universe_tickers"])
    cfg_frozen = _frozen_config(cfg, frozen)

    if fetch:
        fetched, failures = fetch_market_data(cfg_frozen)
        history = update_ohlcv_history(fetched)
        failures.to_csv(FETCH_FAILURES_PATH, index=False)
    else:
        from src.common import OHLCV_PATH
        if not OHLCV_PATH.exists():
            raise FileNotFoundError("data/ohlcv_history.csv does not exist.")
        history = pd.read_csv(OHLCV_PATH, parse_dates=["date"])
        pd.DataFrame(columns=["ticker", "reason"]).to_csv(
            FETCH_FAILURES_PATH, index=False
        )

    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["ticker"] = history["ticker"].astype(str).str.upper().str.strip()
    history = history[history["ticker"].isin(frozen)].copy()

    features = build_features(history, cfg_frozen)
    predictions = _load_csv(PREDICTIONS_PATH, PREDICTION_COLUMNS)
    manifests = _load_csv(ANCHORS_PATH, ANCHOR_COLUMNS)
    outcomes = _load_csv(OUTCOMES_PATH, OUTCOME_COLUMNS)

    if not manifests.empty:
        manifests["anchor_date"] = manifests["anchor_date"].astype(str)
    if not predictions.empty:
        predictions["anchor_date"] = predictions["anchor_date"].astype(str)
        predictions["horizon_bars"] = pd.to_numeric(
            predictions["horizon_bars"], errors="coerce"
        ).astype("Int64")
    if not outcomes.empty:
        outcomes["anchor_date"] = outcomes["anchor_date"].astype(str)
        outcomes["horizon_bars"] = pd.to_numeric(
            outcomes["horizon_bars"], errors="coerce"
        ).astype("Int64")

    predictions, manifests, new_anchor_count = _append_new_anchors(
        features,
        predictions,
        manifests,
        spec=spec,
        registry=registry,
    )

    feature_outcomes = add_forward_outcomes(features)
    outcomes, new_outcome_count = _append_matured_outcomes(
        feature_outcomes,
        predictions,
        outcomes,
        spec,
    )

    _write_csv(predictions, PREDICTIONS_PATH, PREDICTION_COLUMNS)
    _write_csv(manifests, ANCHORS_PATH, ANCHOR_COLUMNS)
    _write_csv(outcomes, OUTCOMES_PATH, OUTCOME_COLUMNS)

    ranking, avoid = _evaluate_months(predictions, outcomes, spec)
    ranking.to_csv(RANKING_MONTHLY_PATH, index=False)
    avoid.to_csv(AVOID_MONTHLY_PATH, index=False)

    assessment_df, assessment_json = _build_assessment(ranking, avoid, spec)
    assessment_df.to_csv(ASSESSMENT_CSV_PATH, index=False)
    ASSESSMENT_JSON_PATH.write_text(
        json.dumps(assessment_json, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    latest_payload = _make_latest(predictions, spec)
    LATEST_PATH.write_text(
        json.dumps(latest_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    charts = _make_charts(ranking, avoid)

    latest_market_date = (
        pd.Timestamp(history["date"].max()).strftime("%Y-%m-%d")
        if not history.empty else None
    )
    latest_anchor = (
        str(manifests["anchor_date"].max()) if not manifests.empty else None
    )
    by_horizon_matured = {
        str(h): int(outcomes[outcomes["horizon_bars"].eq(h)].shape[0])
        for h in map(int, spec["horizons_bars"])
    }
    by_horizon_eval_months = {
        str(h): int(
            ranking[
                ranking["horizon_bars"].eq(h)
                & ranking["method"].eq("ridge")
            ]["anchor_date"].nunique()
        ) if not ranking.empty else 0
        for h in map(int, spec["horizons_bars"])
    }

    status = {
        "research_patch": "R4",
        "status": "ok",
        "prospective_inception_date": spec["inception_date"],
        "first_eligible_anchor_date": spec["first_eligible_anchor_date"],
        "latest_market_date": latest_market_date,
        "latest_issued_anchor_date": latest_anchor,
        "new_anchors_issued_this_run": int(new_anchor_count),
        "new_outcomes_matured_this_run": int(new_outcome_count),
        "prediction_rows": int(len(predictions)),
        "anchor_manifest_rows": int(len(manifests)),
        "matured_outcome_rows": int(len(outcomes)),
        "matured_outcomes_by_horizon": by_horizon_matured,
        "evaluated_months_by_horizon": by_horizon_eval_months,
        "frozen_universe_count": int(registry["frozen_universe_count"]),
        "current_registry_model_years": sorted(
            int(y) for y in registry["models_by_year"]
        ),
        "charts": charts,
        "production_logic_changed": False,
        "historical_reoptimization_performed": False,
    }
    STATUS_PATH.write_text(
        json.dumps(status, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_report(status, assessment_df)

    print(json.dumps(status, indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run R4 prospective clean-sheet shadow validation."
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Use the repository data/ohlcv_history.csv without internet fetching.",
    )
    args = parser.parse_args()
    run_r4(fetch=not args.no_fetch)


if __name__ == "__main__":
    main()
