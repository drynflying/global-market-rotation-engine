from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = Path(__file__).with_name("r4_spec.json")
DEFAULT_REGISTRY = Path(__file__).with_name("r4_model_registry.json")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _month_end_anchor_dates(dates: pd.Series) -> pd.DatetimeIndex:
    d = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    frame = pd.DataFrame({"date": d})
    frame["month"] = frame["date"].dt.to_period("M")
    return pd.DatetimeIndex(
        frame.groupby("month", observed=True)["date"].max().sort_values().to_numpy()
    )


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    return float(pair["x"].rank().corr(pair["y"].rank()))


def _make_avoid_label(frame: pd.DataFrame, target: str, bottom_fraction: float) -> pd.Series:
    labels = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, idx in frame.groupby("date", observed=True).groups.items():
        valid = frame.loc[idx, target].dropna()
        if len(valid) < 5:
            continue
        ranks = valid.rank(method="first", pct=True)
        labels.loc[valid.index] = (ranks <= bottom_fraction).astype(float)
    return labels


def _date_weights(dates: pd.Series) -> np.ndarray:
    counts = dates.value_counts()
    w = dates.map(lambda d: 1.0 / counts[d]).to_numpy(dtype=float, copy=True)
    total = float(w.sum())
    if not np.isfinite(w).all() or total <= 0:
        raise RuntimeError("Invalid model-training date weights.")
    return w * (len(w) / total)


def _training_orientation(
    train: pd.DataFrame, features: list[str], target: str
) -> tuple[dict[str, int], str, dict[str, float]]:
    mean_ics: dict[str, float] = {}
    for feature in features:
        vals = []
        for _, g in train[["date", feature, target]].groupby("date", observed=True):
            ic = _safe_spearman(g[feature], g[target])
            if np.isfinite(ic):
                vals.append(ic)
        mean_ics[feature] = float(np.mean(vals)) if vals else 0.0

    orientation = {f: (1 if mean_ics[f] >= 0 else -1) for f in features}
    best = max(features, key=lambda f: abs(mean_ics[f]))
    return orientation, best, mean_ics


def _source_integrity(registry: dict) -> None:
    expected = registry["source_code_hashes"]
    paths = {
        "build_features.py": ROOT / "research/clean_sheet_predictive/build_features.py",
        "feature_definitions.py": ROOT / "research/clean_sheet_predictive/feature_definitions.py",
        "build_outcomes.py": ROOT / "research/clean_sheet_predictive/build_outcomes.py",
    }
    failures = []
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != expected[name]:
            failures.append(f"{name}: expected {expected[name]}, got {actual}")
    if failures:
        raise RuntimeError(
            "R4 frozen feature/outcome code changed. Model refresh blocked: "
            + "; ".join(failures)
        )


def build_model_year(
    *,
    dataset_path: Path,
    model_year: int,
    registry_path: Path,
    spec_path: Path,
) -> dict:
    spec = _read_json(spec_path)
    registry = _read_json(registry_path)
    _source_integrity(registry)

    frozen = set(map(str, registry["frozen_universe_tickers"]))
    candidates = {
        int(h): list(v) for h, v in spec["candidate_features_by_horizon"].items()
    }
    all_features = sorted(set(sum(candidates.values(), [])))
    horizons = list(map(int, spec["horizons_bars"]))

    needed = {"date", "ticker"} | set(all_features)
    for h in horizons:
        needed.update(
            {
                f"fwd_spy_relative_return_{h}",
                f"outcome_end_date_{h}",
            }
        )

    header = pd.read_csv(dataset_path, nrows=0).columns
    missing = sorted(needed - set(header))
    if missing:
        raise RuntimeError("R4 model build dataset missing columns: " + ", ".join(missing))

    data = pd.read_csv(dataset_path, usecols=sorted(needed), low_memory=False)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data = data[data["ticker"].isin(frozen)].copy()
    for h in horizons:
        data[f"outcome_end_date_{h}"] = pd.to_datetime(
            data[f"outcome_end_date_{h}"], errors="coerce"
        )

    actual = set(data["ticker"].unique())
    missing_universe = sorted(frozen - actual)
    if missing_universe:
        raise RuntimeError(
            "R4 model refresh is missing frozen-universe tickers: "
            + ", ".join(missing_universe)
        )

    anchors = data[data["date"].isin(_month_end_anchor_dates(data["date"]))].copy()
    anchors = anchors[
        anchors["date"] >= pd.Timestamp(spec["development_start_date"])
    ].copy()

    for feature in all_features:
        anchors[feature] = anchors.groupby("date", observed=True)[feature].rank(
            method="average", pct=True, na_option="keep"
        )

    cutoff = pd.Timestamp(year=model_year, month=1, day=1)
    model_year_record = {
        "model_version": f"CLEAN_SHEET_R4_{model_year}_V1",
        "model_year": int(model_year),
        "training_cutoff_exclusive": cutoff.strftime("%Y-%m-%d"),
        "created_from_r3_run_id": None,
        "validation": {
            "frozen_feature_set": True,
            "frozen_universe": True,
            "no_hyperparameter_search": True,
        },
        "horizons": {},
    }

    for h in horizons:
        features = candidates[h]
        target = f"fwd_spy_relative_return_{h}"
        end_col = f"outcome_end_date_{h}"

        a = anchors.copy()
        a["avoid_label"] = _make_avoid_label(
            a, target, float(spec["avoid_target"]["bottom_fraction"])
        )
        mask = (
            a["date"].lt(cutoff)
            & a[end_col].lt(cutoff)
            & a[target].notna()
        )
        cols = ["date", "ticker", target, end_col, "avoid_label"] + features
        train = (
            a.loc[mask, cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=features + [target, "avoid_label"])
            .copy()
        )

        if train["date"].nunique() < 24 or len(train) < 1500:
            raise RuntimeError(
                f"Insufficient frozen R4 training data for {h} bars in {model_year}: "
                f"{len(train)} rows / {train['date'].nunique()} months"
            )
        if train["avoid_label"].nunique() < 2:
            raise RuntimeError(f"AVOID class is degenerate for {h} bars.")

        scaler = StandardScaler()
        x = train[features].to_numpy(dtype=float)
        xs = scaler.fit_transform(x)
        weights = _date_weights(train["date"])

        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(xs, train[target].to_numpy(dtype=float), sample_weight=weights)

        logit = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
        )
        logit.fit(
            xs,
            train["avoid_label"].astype(int).to_numpy(),
            sample_weight=weights,
        )

        orientation, best, mean_ics = _training_orientation(
            train, features, target
        )

        model_year_record["horizons"][str(h)] = {
            "horizon_bars": h,
            "horizon_label": spec["horizon_labels"][str(h)],
            "features": features,
            "training_cutoff_exclusive": cutoff.strftime("%Y-%m-%d"),
            "training_start_inclusive": spec["development_start_date"],
            "training_rows": int(len(train)),
            "training_months": int(train["date"].nunique()),
            "training_avoid_base_rate": float(train["avoid_label"].mean()),
            "scaler_mean": [float(x) for x in scaler.mean_],
            "scaler_scale": [float(x) for x in scaler.scale_],
            "ridge": {
                "alpha": 1.0,
                "intercept": float(ridge.intercept_),
                "coefficients": [float(x) for x in ridge.coef_],
            },
            "logistic": {
                "C": 1.0,
                "intercept": float(logit.intercept_[0]),
                "coefficients": [float(x) for x in logit.coef_[0]],
            },
            "equal_weight_orientation": {
                f: int(orientation[f]) for f in features
            },
            "training_mean_univariate_ic": mean_ics,
            "best_univariate_feature": best,
        }

    key = str(model_year)
    if key in registry["models_by_year"]:
        existing = registry["models_by_year"][key]
        # Idempotent annual workflow: an exact same model year is okay; a drift is not.
        old = json.dumps(existing, sort_keys=True)
        new = json.dumps(model_year_record, sort_keys=True)
        if old != new:
            raise RuntimeError(
                f"R4 model year {model_year} already exists and would change. "
                "Registry immutability blocked the rewrite."
            )
        return existing

    registry["models_by_year"][key] = model_year_record
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return model_year_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append a frozen-feature annual R4 model year."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-year", required=True, type=int)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args()

    result = build_model_year(
        dataset_path=args.dataset,
        model_year=args.model_year,
        registry_path=args.registry,
        spec_path=args.spec,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
