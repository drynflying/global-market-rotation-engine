from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .build_features import build_features
from .build_outcomes import add_forward_outcomes
from .feature_definitions import FEATURE_COLUMNS, FEATURE_FAMILIES, FEATURE_VERSION, OUTCOME_HORIZONS
from .validate_point_in_time import build_validation_report, write_validation_report


def _feature_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    family_lookup = {
        feature: family for family, features in FEATURE_FAMILIES.items() for feature in features
    }
    for feature in FEATURE_COLUMNS:
        if feature not in dataset.columns:
            continue
        s = pd.to_numeric(dataset[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "family": family_lookup.get(feature, "other"),
                "non_null_rows": int(s.notna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100.0), 3),
                "mean": float(s.mean()) if s.notna().any() else None,
                "std": float(s.std(ddof=0)) if s.notna().any() else None,
                "p05": float(s.quantile(0.05)) if s.notna().any() else None,
                "median": float(s.median()) if s.notna().any() else None,
                "p95": float(s.quantile(0.95)) if s.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def _outcome_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in OUTCOME_HORIZONS:
        for target in [
            f"fwd_return_{h}",
            f"fwd_primary_relative_return_{h}",
            f"fwd_spy_relative_return_{h}",
            f"fwd_max_drawdown_{h}",
            f"fwd_primary_rel_mae_{h}",
            f"fwd_spy_rel_mae_{h}",
        ]:
            s = pd.to_numeric(dataset[target], errors="coerce")
            rows.append(
                {
                    "horizon_bars": h,
                    "target": target,
                    "matured_rows": int(s.notna().sum()),
                    "missing_pct": round(float(s.isna().mean() * 100.0), 3),
                    "mean": float(s.mean()) if s.notna().any() else None,
                    "median": float(s.median()) if s.notna().any() else None,
                    "p10": float(s.quantile(0.10)) if s.notna().any() else None,
                    "p90": float(s.quantile(0.90)) if s.notna().any() else None,
                }
            )
    return pd.DataFrame(rows)


def _data_dictionary(dataset: pd.DataFrame) -> pd.DataFrame:
    family_lookup = {
        feature: family for family, features in FEATURE_FAMILIES.items() for feature in features
    }
    rows = []
    for col in dataset.columns:
        if col in family_lookup:
            kind = "feature"
            family = family_lookup[col]
        elif col.startswith("fwd_") or col.startswith("outperformed_") or col.startswith("outcome_end_date_"):
            kind = "future_outcome"
            family = "outcome"
        elif col in {"date", "ticker", "exposure", "universe", "rotation_group", "level", "asset_type", "primary_benchmark"}:
            kind = "identifier_metadata"
            family = "metadata"
        else:
            kind = "intermediate"
            family = "support"
        rows.append({"column": col, "kind": kind, "family": family, "dtype": str(dataset[col].dtype)})
    return pd.DataFrame(rows)


def build_dataset(
    input_path: Path,
    config_path: Path,
    output_dir: Path,
    analysis_years: int | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path)
    config = pd.read_csv(config_path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")

    features = build_features(raw, config)
    dataset = add_forward_outcomes(features)

    raw_start = pd.Timestamp(raw["date"].min())
    raw_end = pd.Timestamp(raw["date"].max())
    analysis_start = raw_start
    if analysis_years:
        analysis_start = raw_end - pd.DateOffset(years=analysis_years)
        dataset = dataset[dataset["date"] >= analysis_start].copy()

    # Store dates as YYYY-MM-DD strings in the portable compressed CSV.
    export = dataset.copy()
    for col in export.columns:
        if col == "date" or col.startswith("outcome_end_date_"):
            if pd.api.types.is_datetime64_any_dtype(export[col]):
                export[col] = export[col].dt.strftime("%Y-%m-%d")
    dataset_path = output_dir / "research_dataset.csv.gz"
    export.to_csv(dataset_path, index=False, compression="gzip")

    _feature_summary(dataset).to_csv(output_dir / "feature_summary.csv", index=False)
    _outcome_summary(dataset).to_csv(output_dir / "outcome_summary.csv", index=False)
    _data_dictionary(dataset).to_csv(output_dir / "data_dictionary.csv", index=False)

    report = build_validation_report(raw, config, dataset)
    write_validation_report(report, output_dir / "validation_report.json")

    manifest = {
        "dataset_version": FEATURE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "config_file": str(config_path),
        "raw_rows": int(len(raw)),
        "raw_tickers": int(raw["ticker"].nunique()),
        "raw_earliest_date": raw_start.strftime("%Y-%m-%d"),
        "raw_latest_date": raw_end.strftime("%Y-%m-%d"),
        "analysis_years": analysis_years,
        "analysis_start_date": pd.Timestamp(dataset["date"].min()).strftime("%Y-%m-%d"),
        "analysis_latest_date": pd.Timestamp(dataset["date"].max()).strftime("%Y-%m-%d"),
        "dataset_rows": int(len(dataset)),
        "dataset_tickers": int(dataset["ticker"].nunique()),
        "feature_count": len([c for c in FEATURE_COLUMNS if c in dataset.columns]),
        "outcome_horizons_bars": list(OUTCOME_HORIZONS),
        "validation_status": report["status"],
        "universe_completeness_status": report["universe_completeness"]["status"],
        "configured_active_symbols": report["universe_completeness"]["configured_active_symbols"],
        "symbols_with_data": report["universe_completeness"]["symbols_with_data"],
        "missing_symbols": report["universe_completeness"]["missing_symbols"],
        "important_note": (
            "Future outcomes are attached for research, but a row is eligible for model training "
            "only after its horizon-specific outcome_end_date has passed. R2/R3 must enforce this "
            "in every walk-forward split."
        ),
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if report["status"] != "ok":
        raise RuntimeError(f"R1/R1.1 research dataset validation failed: {report.get('failures', [])}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the clean-sheet R1 point-in-time predictive research dataset.")
    parser.add_argument("--input", required=True, type=Path, help="OHLCV CSV path")
    parser.add_argument("--config", required=True, type=Path, help="rotation_universe.csv path")
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    parser.add_argument("--analysis-years", type=int, default=None, help="Keep only the trailing N calendar years after features are built using all warm-up data")
    args = parser.parse_args()
    manifest = build_dataset(args.input, args.config, args.output, args.analysis_years)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
