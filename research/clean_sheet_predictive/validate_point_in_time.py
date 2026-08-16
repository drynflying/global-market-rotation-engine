from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .build_features import build_features
from .feature_definitions import FEATURE_COLUMNS, OUTCOME_HORIZONS


def _safe_float(x: Any) -> float | None:
    if pd.isna(x):
        return None
    return float(x)


def future_mutation_test(ohlcv: pd.DataFrame, config: pd.DataFrame) -> dict:
    """Prove feature values at a cutoff do not change when future OHLCV is mutated.

    The test uses a compact but cross-sectional sample so it remains cheap even when the
    full 10-year dataset is large.
    """
    raw = ohlcv.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    all_dates = sorted(raw["date"].unique())
    if len(all_dates) < 420:
        return {"status": "skipped", "reason": "fewer than 420 unique dates"}

    cutoff = pd.Timestamp(all_dates[-126])
    wanted = ["SPY", "ITOT", "IWM", "QQQ", "XLE", "XLK", "XLV", "EFA", "EEM", "GLD", "URA", "CPER"]
    available = set(raw["ticker"].astype(str).str.upper())
    tickers = [t for t in wanted if t in available]
    if len(tickers) < 5:
        tickers = sorted(available)[:12]

    # Include configured benchmarks for the sampled assets so primary-relative
    # features are also exercised by the mutation test.
    cfg_all = config.copy()
    cfg_all["ticker"] = cfg_all["ticker"].astype(str).str.upper()
    if "primary_benchmark" in cfg_all.columns:
        bench = cfg_all[cfg_all["ticker"].isin(tickers)]["primary_benchmark"]
        bench = bench.fillna("").astype(str).str.upper().str.strip()
        for b in bench:
            if b and b in available and b not in tickers:
                tickers.append(b)

    sample = raw[raw["ticker"].isin(tickers)].copy()
    cfg = cfg_all[cfg_all["ticker"].isin(tickers)].copy()
    baseline = build_features(sample, cfg)

    mutated = sample.copy()
    future = mutated["date"] > cutoff
    # Intentionally absurd future values. If any feature at/before cutoff moves, there is leakage.
    mutated.loc[future, "close"] *= 7.0
    mutated.loc[future, "open"] *= 6.0
    mutated.loc[future, "high"] *= 8.0
    mutated.loc[future, "low"] *= 5.0
    mutated.loc[future, "volume"] *= 11.0
    changed = build_features(mutated, cfg)

    cols = [c for c in FEATURE_COLUMNS if c in baseline.columns and c in changed.columns]
    keys = ["date", "ticker"]
    a = baseline[baseline["date"] <= cutoff][keys + cols].sort_values(keys).reset_index(drop=True)
    b = changed[changed["date"] <= cutoff][keys + cols].sort_values(keys).reset_index(drop=True)
    if len(a) != len(b) or not a[keys].equals(b[keys]):
        return {"status": "failed", "reason": "row/key mismatch after mutation"}

    bad: list[str] = []
    max_abs_diff = 0.0
    for c in cols:
        av = pd.to_numeric(a[c], errors="coerce").to_numpy(dtype=float)
        bv = pd.to_numeric(b[c], errors="coerce").to_numpy(dtype=float)
        equal = np.isclose(av, bv, rtol=1e-10, atol=1e-12, equal_nan=True)
        if not equal.all():
            bad.append(c)
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any():
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(av[finite] - bv[finite]))))
    return {
        "status": "passed" if not bad else "failed",
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "tickers": tickers,
        "features_checked": len(cols),
        "leaking_features": bad,
        "max_abs_diff": max_abs_diff,
    }


def outcome_alignment_checks(dataset: pd.DataFrame) -> dict:
    checks: dict[str, Any] = {}
    ordered = dataset.sort_values(["ticker", "date"]).copy()
    for h in OUTCOME_HORIZONS:
        end_col = f"outcome_end_date_{h}"
        ret_col = f"fwd_return_{h}"
        expected_end = ordered.groupby("ticker")["date"].shift(-h)
        expected_ret = ordered.groupby("ticker")["close"].shift(-h) / ordered["close"] - 1.0
        actual_end = pd.to_datetime(ordered[end_col], errors="coerce")
        matured = ordered[ret_col].notna()
        bad_date = matured & actual_end.isna()
        end_mismatch = ~((actual_end == expected_end) | (actual_end.isna() & expected_end.isna()))
        av = pd.to_numeric(ordered[ret_col], errors="coerce").to_numpy(dtype=float)
        ev = pd.to_numeric(expected_ret, errors="coerce").to_numpy(dtype=float)
        ret_equal = np.isclose(av, ev, rtol=1e-10, atol=1e-12, equal_nan=True)
        finite = np.isfinite(av) & np.isfinite(ev)
        max_return_error = float(np.max(np.abs(av[finite] - ev[finite]))) if finite.any() else 0.0
        checks[str(h)] = {
            "matured_rows": int(matured.sum()),
            "missing_end_date_on_matured": int(bad_date.sum()),
            "end_date_offset_mismatches": int(end_mismatch.sum()),
            "forward_return_mismatches": int((~ret_equal).sum()),
            "max_forward_return_abs_error": max_return_error,
            "immature_rows": int((~matured).sum()),
        }
    return checks



def universe_completeness_check(
    raw_ohlcv: pd.DataFrame,
    config: pd.DataFrame,
) -> dict:
    cfg = config.copy()
    if "enabled" in cfg.columns:
        enabled = cfg["enabled"].astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes", "y"}
        )
    else:
        enabled = pd.Series(True, index=cfg.index)

    if "query_ohlcv" in cfg.columns:
        query_ohlcv = cfg["query_ohlcv"].astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes", "y"}
        )
    else:
        query_ohlcv = pd.Series(True, index=cfg.index)

    symbol_col = "query_symbol" if "query_symbol" in cfg.columns else "ticker"
    expected = set(
        cfg.loc[enabled & query_ohlcv, symbol_col]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )
    expected.discard("")

    actual = set(
        raw_ohlcv["ticker"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )
    actual.discard("")

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return {
        "status": "passed" if not missing else "failed",
        "configured_active_symbols": len(expected),
        "symbols_with_data": len(actual & expected),
        "completeness_ratio": round(
            len(actual & expected) / max(len(expected), 1),
            6,
        ),
        "missing_symbols": missing,
        "unexpected_symbols": unexpected,
    }


def build_validation_report(
    raw_ohlcv: pd.DataFrame,
    config: pd.DataFrame,
    dataset: pd.DataFrame,
) -> dict:
    raw = raw_ohlcv.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    feature_columns_present = [c for c in FEATURE_COLUMNS if c in dataset.columns]
    feature_missing = {
        c: round(float(dataset[c].isna().mean()), 6) for c in feature_columns_present
    }
    latest = pd.Timestamp(dataset["date"].max())
    report = {
        "status": "ok",
        "raw_data": {
            "rows": int(len(raw)),
            "tickers": int(raw["ticker"].nunique()),
            "earliest_date": pd.Timestamp(raw["date"].min()).strftime("%Y-%m-%d"),
            "latest_date": pd.Timestamp(raw["date"].max()).strftime("%Y-%m-%d"),
            "duplicate_ticker_date_rows": int(raw.duplicated(["ticker", "date"]).sum()),
        },
        "research_dataset": {
            "rows": int(len(dataset)),
            "tickers": int(dataset["ticker"].nunique()),
            "earliest_date": pd.Timestamp(dataset["date"].min()).strftime("%Y-%m-%d"),
            "latest_date": latest.strftime("%Y-%m-%d"),
            "feature_columns": len(feature_columns_present),
        },
        "universe_completeness": universe_completeness_check(raw_ohlcv, config),
        "future_mutation_leakage_test": future_mutation_test(raw_ohlcv, config),
        "outcome_alignment": outcome_alignment_checks(dataset),
        "feature_missing_fraction": feature_missing,
    }
    failures = []
    if report["raw_data"]["duplicate_ticker_date_rows"]:
        failures.append("duplicate ticker/date OHLCV rows")
    if report["universe_completeness"].get("status") == "failed":
        failures.append(
            "active universe incomplete: "
            + ", ".join(report["universe_completeness"].get("missing_symbols", []))
        )
    if report["future_mutation_leakage_test"].get("status") == "failed":
        failures.append("future mutation leakage test failed")
    if any(v["missing_end_date_on_matured"] for v in report["outcome_alignment"].values()):
        failures.append("matured outcomes without an end date")
    if any(v["end_date_offset_mismatches"] for v in report["outcome_alignment"].values()):
        failures.append("horizon outcome end-date alignment mismatch")
    if any(v["forward_return_mismatches"] for v in report["outcome_alignment"].values()):
        failures.append("forward-return alignment mismatch")
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    return report


def write_validation_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
