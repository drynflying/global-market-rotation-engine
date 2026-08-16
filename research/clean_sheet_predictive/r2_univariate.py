from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SPEC_PATH = Path(__file__).with_name("r2_spec.json")
RETURN_OUTCOMES = {
    "spy_relative_return": "fwd_spy_relative_return_{h}",
    "absolute_return": "fwd_return_{h}",
    "primary_relative_return": "fwd_primary_relative_return_{h}",
    "max_drawdown": "fwd_max_drawdown_{h}",
    "spy_relative_mae": "fwd_spy_rel_mae_{h}",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_spec(path: Path | None = None) -> dict:
    return _read_json(path or DEFAULT_SPEC_PATH)


def _require_valid_r1(manifest: dict, validation: dict, spec: dict) -> None:
    failures: list[str] = []
    if validation.get("status") != spec["dataset_requirement"]["r1_validation_status"]:
        failures.append(
            f"R1 validation status={validation.get('status')!r}, "
            f"expected {spec['dataset_requirement']['r1_validation_status']!r}"
        )
    completeness = validation.get("universe_completeness", {})
    if completeness.get("status") != spec["dataset_requirement"]["universe_completeness_status"]:
        failures.append(
            f"universe completeness={completeness.get('status')!r}, "
            f"missing={completeness.get('missing_symbols', [])}"
        )
    expected = spec["dataset_requirement"].get("expected_tickers")
    if expected is not None and int(manifest.get("dataset_tickers", -1)) != int(expected):
        failures.append(
            f"dataset_tickers={manifest.get('dataset_tickers')} expected={expected}"
        )
    if failures:
        raise RuntimeError("R2 blocked by R1 integrity gate: " + "; ".join(failures))


def _feature_table(dictionary: pd.DataFrame) -> pd.DataFrame:
    features = dictionary[dictionary["kind"].eq("feature")][["column", "family"]].copy()
    if features.empty:
        raise RuntimeError("No feature rows found in R1 data_dictionary.csv")
    features["scope"] = np.where(
        features["family"].eq("market_regime"),
        "date_level",
        "cross_sectional",
    )
    return features.reset_index(drop=True)


def _month_end_anchor_dates(dates: pd.Series) -> pd.DatetimeIndex:
    d = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    frame = pd.DataFrame({"date": d})
    frame["month"] = frame["date"].dt.to_period("M")
    anchors = frame.groupby("month", observed=True)["date"].max().sort_values()
    return pd.DatetimeIndex(anchors.to_numpy())


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    xr = pair["x"].rank(method="average")
    yr = pair["y"].rank(method="average")
    return float(xr.corr(yr))


def _high_low_masks(x: pd.Series) -> tuple[pd.Series, pd.Series, str]:
    clean = x.replace([np.inf, -np.inf], np.nan)
    unique = clean.dropna().nunique()

    if unique < 2:
        empty = pd.Series(False, index=x.index)
        return empty, empty, "insufficient"

    if unique <= 5:
        lo = clean.min()
        hi = clean.max()
        return clean.eq(hi), clean.eq(lo), "discrete_extremes"

    ranks = clean.rank(method="average", pct=True)
    return ranks.gt(0.80), ranks.le(0.20), "quintile_extremes"


def _cross_section_month_stats(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    min_rows: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    for date, g in frame[["date", "ticker", feature, target]].groupby("date", sort=True):
        p = g.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, target])
        if len(p) < min_rows or p[feature].nunique() < 2 or p[target].nunique() < 2:
            continue

        ic = _safe_spearman(p[feature], p[target])
        high_mask, low_mask, grouping = _high_low_masks(p[feature])
        high = p.loc[high_mask, target]
        low = p.loc[low_mask, target]
        if high.empty or low.empty:
            spread = np.nan
            median_spread = np.nan
            high_hit = np.nan
            low_hit = np.nan
        else:
            spread = float(high.mean() - low.mean())
            median_spread = float(high.median() - low.median())
            high_hit = float((high > 0).mean())
            low_hit = float((low > 0).mean())

        rows.append(
            {
                "date": pd.Timestamp(date),
                "n": int(len(p)),
                "ic": ic,
                "high_minus_low_mean": spread,
                "high_minus_low_median": median_spread,
                "high_positive_rate": high_hit,
                "low_positive_rate": low_hit,
                "positive_rate_spread": (
                    high_hit - low_hit
                    if pd.notna(high_hit) and pd.notna(low_hit)
                    else np.nan
                ),
                "grouping": grouping,
            }
        )

    return pd.DataFrame(rows)


def _monthly_quintiles(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    min_rows: int,
) -> pd.DataFrame:
    per_month: list[pd.DataFrame] = []

    for date, g in frame[["date", feature, target]].groupby("date", sort=True):
        p = g.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, target]).copy()
        if len(p) < min_rows or p[feature].nunique() < 5:
            continue

        ranks = p[feature].rank(method="first", pct=True)
        p["quintile"] = np.minimum((np.ceil(ranks * 5)).astype(int), 5)
        agg = p.groupby("quintile", observed=True)[target].agg(["mean", "median", "count"])
        agg["positive_rate"] = p.groupby("quintile", observed=True)[target].apply(
            lambda s: float((s > 0).mean())
        )
        agg = agg.reset_index()
        agg["date"] = pd.Timestamp(date)
        per_month.append(agg)

    if not per_month:
        return pd.DataFrame(
            columns=["quintile", "mean", "median", "count", "positive_rate", "date"]
        )
    return pd.concat(per_month, ignore_index=True)


def _block_bootstrap_mean_ci(
    values: pd.Series,
    *,
    block_months: int,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = len(arr)
    if n < max(12, block_months * 2):
        return float("nan"), float("nan")

    block = min(block_months, n)
    starts_max = n - block
    blocks_needed = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)

    for i in range(repetitions):
        starts = rng.integers(0, starts_max + 1, size=blocks_needed)
        sample = np.concatenate([arr[s : s + block] for s in starts])[:n]
        means[i] = np.mean(sample)

    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _annual_ic_rows(
    month_stats: pd.DataFrame,
    feature: str,
    family: str,
    horizon: int,
) -> list[dict]:
    if month_stats.empty:
        return []
    temp = month_stats.dropna(subset=["ic"]).copy()
    temp["year"] = temp["date"].dt.year
    rows = []
    for year, g in temp.groupby("year"):
        rows.append(
            {
                "feature": feature,
                "family": family,
                "horizon_bars": horizon,
                "year": int(year),
                "months": int(len(g)),
                "mean_ic": float(g["ic"].mean()),
                "median_ic": float(g["ic"].median()),
            }
        )
    return rows


def _subperiod_stats(
    month_stats: pd.DataFrame,
    subperiods: list[dict],
) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for period in subperiods:
        name = period["name"]
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        g = month_stats[
            month_stats["date"].between(start, end)
        ].dropna(subset=["ic"])
        result[f"{name}_months"] = int(len(g))
        result[f"{name}_mean_ic"] = float(g["ic"].mean()) if len(g) else None
    return result


def _annual_direction_consistency(
    annual_rows: list[dict],
    full_mean_ic: float,
) -> float:
    vals = [
        r["mean_ic"]
        for r in annual_rows
        if r["months"] >= 4 and pd.notna(r["mean_ic"]) and r["mean_ic"] != 0
    ]
    if not vals or not np.isfinite(full_mean_ic) or full_mean_ic == 0:
        return float("nan")
    sign = np.sign(full_mean_ic)
    return float(np.mean([np.sign(v) == sign for v in vals]))


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    return bool(
        np.isfinite(lo)
        and np.isfinite(hi)
        and ((lo > 0 and hi > 0) or (lo < 0 and hi < 0))
    )


def _spread_agrees(mean_ic: float, spread: float) -> bool:
    if not np.isfinite(mean_ic) or not np.isfinite(spread) or mean_ic == 0 or spread == 0:
        return False
    return bool(np.sign(mean_ic) == np.sign(spread))


def _evidence_label(
    *,
    mean_ic: float,
    ci_lo: float,
    ci_hi: float,
    annual_consistency: float,
    spread: float,
    months: int,
    spec: dict,
) -> str:
    abs_ic = abs(mean_ic) if np.isfinite(mean_ic) else 0.0
    ci_ok = _ci_excludes_zero(ci_lo, ci_hi)
    spread_ok = _spread_agrees(mean_ic, spread)

    robust = spec["evidence_labels"]["ROBUST_CANDIDATE"]
    if (
        months >= robust["minimum_months"]
        and abs_ic >= robust["abs_mean_monthly_ic_min"]
        and ci_ok
        and annual_consistency >= robust["annual_direction_consistency_min"]
        and spread_ok
    ):
        return "ROBUST_CANDIDATE"

    promising = spec["evidence_labels"]["PROMISING"]
    if (
        months >= promising["minimum_months"]
        and abs_ic >= promising["abs_mean_monthly_ic_min"]
        and ci_ok
        and annual_consistency >= promising["annual_direction_consistency_min"]
        and spread_ok
    ):
        return "PROMISING"

    weak = spec["evidence_labels"]["WEAK"]
    if (
        months >= weak["minimum_months"]
        and abs_ic >= weak["abs_mean_monthly_ic_min"]
        and ci_ok
    ):
        return "WEAK"

    return "INCONSISTENT_OR_NONE"


def _summary_row(
    month_stats: pd.DataFrame,
    *,
    feature: str,
    family: str,
    horizon: int,
    outcome_kind: str,
    bootstrap: bool,
    spec: dict,
) -> tuple[dict, list[dict]]:
    clean = month_stats.dropna(subset=["ic"]).copy()
    annual = (
        _annual_ic_rows(clean, feature, family, horizon)
        if outcome_kind == "spy_relative_return"
        else []
    )

    if clean.empty:
        return (
            {
                "feature": feature,
                "family": family,
                "horizon_bars": horizon,
                "horizon_label": spec["horizon_labels"][str(horizon)],
                "outcome_kind": outcome_kind,
                "months": 0,
                "mean_monthly_ic": np.nan,
                "median_monthly_ic": np.nan,
                "ic_std": np.nan,
                "bootstrap_ci_low": np.nan,
                "bootstrap_ci_high": np.nan,
                "annual_direction_consistency": np.nan,
                "mean_high_minus_low": np.nan,
                "median_high_minus_low": np.nan,
                "mean_positive_rate_spread": np.nan,
                "evidence_label": "INCONSISTENT_OR_NONE",
            },
            annual,
        )

    mean_ic = float(clean["ic"].mean())
    median_ic = float(clean["ic"].median())
    ic_std = float(clean["ic"].std(ddof=1))
    spread = float(clean["high_minus_low_mean"].mean())
    median_spread = float(clean["high_minus_low_median"].mean())
    hit_spread = float(clean["positive_rate_spread"].mean())

    if bootstrap:
        seed = (
            int(spec["bootstrap"]["seed"])
            + zlib.crc32(f"{feature}|{horizon}|{outcome_kind}".encode("utf-8"))
        ) % (2**32 - 1)
        ci_lo, ci_hi = _block_bootstrap_mean_ci(
            clean["ic"],
            block_months=int(spec["bootstrap"]["block_months"]),
            repetitions=int(spec["bootstrap"]["repetitions"]),
            seed=seed,
        )
    else:
        ci_lo, ci_hi = np.nan, np.nan

    consistency = (
        _annual_direction_consistency(annual, mean_ic)
        if outcome_kind == "spy_relative_return"
        else np.nan
    )

    label = (
        _evidence_label(
            mean_ic=mean_ic,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            annual_consistency=consistency,
            spread=spread,
            months=len(clean),
            spec=spec,
        )
        if outcome_kind == "spy_relative_return"
        else ""
    )

    row = {
        "feature": feature,
        "family": family,
        "horizon_bars": horizon,
        "horizon_label": spec["horizon_labels"][str(horizon)],
        "outcome_kind": outcome_kind,
        "months": int(len(clean)),
        "mean_monthly_ic": mean_ic,
        "median_monthly_ic": median_ic,
        "ic_std": ic_std,
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "annual_direction_consistency": consistency,
        "mean_high_minus_low": spread,
        "median_high_minus_low": median_spread,
        "mean_positive_rate_spread": hit_spread,
        "evidence_label": label,
        "direction_if_used": (
            "HIGHER_FEATURE_FAVORABLE"
            if mean_ic > 0
            else "LOWER_FEATURE_FAVORABLE"
            if mean_ic < 0
            else "NONE"
        ),
    }
    if outcome_kind == "spy_relative_return":
        row.update(_subperiod_stats(clean, spec["temporal_subperiods"]))
    return row, annual


def _quintile_profile_rows(
    quintiles: pd.DataFrame,
    *,
    feature: str,
    family: str,
    horizon: int,
    spec: dict,
) -> list[dict]:
    if quintiles.empty:
        return []

    rows: list[dict] = []
    for q, g in quintiles.groupby("quintile", sort=True):
        rows.append(
            {
                "feature": feature,
                "family": family,
                "horizon_bars": horizon,
                "horizon_label": spec["horizon_labels"][str(horizon)],
                "quintile": int(q),
                "months": int(g["date"].nunique()),
                "mean_monthly_mean_outcome": float(g["mean"].mean()),
                "mean_monthly_median_outcome": float(g["median"].mean()),
                "mean_positive_rate": float(g["positive_rate"].mean()),
                "mean_constituent_count": float(g["count"].mean()),
            }
        )
    return rows


def _market_regime_evaluation(
    anchors: pd.DataFrame,
    features: pd.DataFrame,
    horizons: list[int],
    spec: dict,
) -> pd.DataFrame:
    market_features = features.loc[
        features["scope"].eq("date_level"), "column"
    ].tolist()
    if not market_features:
        return pd.DataFrame()

    rows: list[dict] = []
    spy = anchors[anchors["ticker"].eq("SPY")].copy()
    med_excess_by_h = {
        h: anchors.groupby("date")[f"fwd_spy_relative_return_{h}"].median()
        for h in horizons
    }

    for feature in market_features:
        for h in horizons:
            targets = {
                "spy_absolute_return": f"fwd_return_{h}",
                "spy_max_drawdown": f"fwd_max_drawdown_{h}",
            }

            for kind, target in targets.items():
                p = spy[["date", feature, target]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                ic = _safe_spearman(p[feature], p[target])
                high_mask, low_mask, grouping = _high_low_masks(p[feature])
                high = p.loc[high_mask, target]
                low = p.loc[low_mask, target]
                spread = (
                    float(high.mean() - low.mean())
                    if len(high) and len(low)
                    else np.nan
                )
                seed = (
                    int(spec["bootstrap"]["seed"])
                    + zlib.crc32(f"market|{feature}|{h}|{kind}".encode("utf-8"))
                ) % (2**32 - 1)
                # Date-level data has one observation per month. Bootstrap the
                # feature/target Spearman indirectly by block resampling monthly
                # observations and recomputing the correlation.
                ci_lo, ci_hi = _block_bootstrap_spearman_ci(
                    p[feature],
                    p[target],
                    block_months=int(spec["bootstrap"]["block_months"]),
                    repetitions=int(spec["bootstrap"]["repetitions"]),
                    seed=seed,
                )
                rows.append(
                    {
                        "feature": feature,
                        "family": "market_regime",
                        "horizon_bars": h,
                        "horizon_label": spec["horizon_labels"][str(h)],
                        "context_target": kind,
                        "months": int(len(p)),
                        "spearman": ic,
                        "bootstrap_ci_low": ci_lo,
                        "bootstrap_ci_high": ci_hi,
                        "high_minus_low": spread,
                        "grouping": grouping,
                    }
                )

            med = med_excess_by_h[h].rename("median_excess").reset_index()
            p = spy[["date", feature]].merge(med, on="date", how="inner").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            ic = _safe_spearman(p[feature], p["median_excess"])
            high_mask, low_mask, grouping = _high_low_masks(p[feature])
            high = p.loc[high_mask, "median_excess"]
            low = p.loc[low_mask, "median_excess"]
            spread = (
                float(high.mean() - low.mean())
                if len(high) and len(low)
                else np.nan
            )
            seed = (
                int(spec["bootstrap"]["seed"])
                + zlib.crc32(f"market|{feature}|{h}|median_excess".encode("utf-8"))
            ) % (2**32 - 1)
            ci_lo, ci_hi = _block_bootstrap_spearman_ci(
                p[feature],
                p["median_excess"],
                block_months=int(spec["bootstrap"]["block_months"]),
                repetitions=int(spec["bootstrap"]["repetitions"]),
                seed=seed,
            )
            rows.append(
                {
                    "feature": feature,
                    "family": "market_regime",
                    "horizon_bars": h,
                    "horizon_label": spec["horizon_labels"][str(h)],
                    "context_target": "cross_sectional_median_excess_vs_spy",
                    "months": int(len(p)),
                    "spearman": ic,
                    "bootstrap_ci_low": ci_lo,
                    "bootstrap_ci_high": ci_hi,
                    "high_minus_low": spread,
                    "grouping": grouping,
                }
            )

    return pd.DataFrame(rows)


def _block_bootstrap_spearman_ci(
    x: pd.Series,
    y: pd.Series,
    *,
    block_months: int,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    p = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    n = len(p)
    if n < max(12, block_months * 2):
        return float("nan"), float("nan")

    block = min(block_months, n)
    starts_max = n - block
    blocks_needed = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    vals = np.empty(repetitions, dtype=float)

    xa = p["x"].to_numpy()
    ya = p["y"].to_numpy()
    for i in range(repetitions):
        starts = rng.integers(0, starts_max + 1, size=blocks_needed)
        idx = np.concatenate(
            [np.arange(s, s + block) for s in starts]
        )[:n]
        vals[i] = _safe_spearman(
            pd.Series(xa[idx]),
            pd.Series(ya[idx]),
        )

    vals = vals[np.isfinite(vals)]
    if len(vals) < max(20, repetitions // 10):
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def _redundancy_table(
    anchors: pd.DataFrame,
    features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    cols = features.loc[
        features["scope"].eq("cross_sectional"), "column"
    ].tolist()
    if len(cols) < 2:
        return pd.DataFrame(
            columns=["feature_a", "feature_b", "spearman", "abs_spearman"]
        )

    sample = anchors[cols].replace([np.inf, -np.inf], np.nan)
    corr = sample.corr(method="spearman", min_periods=500)
    rows: list[dict] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "spearman": float(value),
                        "abs_spearman": float(abs(value)),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        "abs_spearman", ascending=False
    ) if rows else pd.DataFrame(
        columns=["feature_a", "feature_b", "spearman", "abs_spearman"]
    )


def _family_summary(primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame()
    return (
        primary.groupby(["family", "horizon_bars", "horizon_label"], observed=True)
        .agg(
            feature_count=("feature", "nunique"),
            median_abs_ic=("mean_monthly_ic", lambda s: float(np.nanmedian(np.abs(s)))),
            max_abs_ic=("mean_monthly_ic", lambda s: float(np.nanmax(np.abs(s)))),
            robust_candidates=("evidence_label", lambda s: int((s == "ROBUST_CANDIDATE").sum())),
            promising=("evidence_label", lambda s: int((s == "PROMISING").sum())),
            weak=("evidence_label", lambda s: int((s == "WEAK").sum())),
        )
        .reset_index()
    )


def _top_findings(primary: pd.DataFrame, spec: dict) -> dict:
    rank = {
        "ROBUST_CANDIDATE": 0,
        "PROMISING": 1,
        "WEAK": 2,
        "INCONSISTENT_OR_NONE": 3,
    }
    p = primary.copy()
    p["_label_rank"] = p["evidence_label"].map(rank).fillna(9)
    p["_abs_ic"] = p["mean_monthly_ic"].abs()
    p = p.sort_values(
        ["horizon_bars", "_label_rank", "_abs_ic"],
        ascending=[True, True, False],
    )

    by_horizon = {}
    for h, g in p.groupby("horizon_bars", sort=True):
        selected = g.head(15)
        by_horizon[str(int(h))] = [
            {
                "feature": row.feature,
                "family": row.family,
                "evidence_label": row.evidence_label,
                "mean_monthly_ic": row.mean_monthly_ic,
                "bootstrap_ci": [row.bootstrap_ci_low, row.bootstrap_ci_high],
                "annual_direction_consistency": row.annual_direction_consistency,
                "high_minus_low_mean": row.mean_high_minus_low,
                "direction_if_used": row.direction_if_used,
            }
            for row in selected.itertuples(index=False)
        ]

    return {
        "research_patch": "R2",
        "status": "discovery_only_not_production_validated",
        "primary_target": spec["primary_target"],
        "by_horizon": by_horizon,
    }


def _write_markdown_report(
    output: Path,
    primary: pd.DataFrame,
    family: pd.DataFrame,
    market: pd.DataFrame,
    anchors: pd.DataFrame,
    spec: dict,
) -> None:
    lines = [
        "# R2 — Univariate Predictive Discovery",
        "",
        "**Status: discovery evidence only. No combined model was fitted.**",
        "",
        f"- Month-end anchor observations: {anchors['date'].nunique()} months",
        f"- Tickers represented: {anchors['ticker'].nunique()}",
        f"- Primary target: `{spec['primary_target']}`",
        "- Evidence labels were frozen in `r2_spec.json` before this run.",
        "",
    ]

    for h in spec["horizons_bars"]:
        g = primary[primary["horizon_bars"].eq(h)].copy()
        if g.empty:
            continue
        g["abs_ic"] = g["mean_monthly_ic"].abs()
        g = g.sort_values(
            ["evidence_label", "abs_ic"],
            key=lambda s: (
                s.map({
                    "ROBUST_CANDIDATE": 0,
                    "PROMISING": 1,
                    "WEAK": 2,
                    "INCONSISTENT_OR_NONE": 3,
                }).fillna(9)
                if s.name == "evidence_label"
                else -s
            ),
        )
        lines.extend([
            f"## {spec['horizon_labels'][str(h)]} / {h} bars",
            "",
        ])
        for row in g.head(10).itertuples(index=False):
            lines.append(
                f"- **{row.feature}** ({row.family}) — {row.evidence_label}; "
                f"mean monthly IC {row.mean_monthly_ic:+.4f}; "
                f"95% block-bootstrap CI [{row.bootstrap_ci_low:+.4f}, "
                f"{row.bootstrap_ci_high:+.4f}]; "
                f"high-minus-low {row.mean_high_minus_low:+.4f}."
            )
        lines.append("")

    lines.extend([
        "## Interpretation constraints",
        "",
        "- A strong R2 feature is a candidate for R3, not a trading rule.",
        "- Direction is not flipped to make results look favorable; high-minus-low is reported raw.",
        "- Market-regime variables are contextual date-level predictors and are not eligible to rank ETFs cross-sectionally.",
        "- Correlated/redundant features are reported separately so R3 does not double-count the same information.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def run_r2(
    *,
    dataset_path: Path,
    dictionary_path: Path,
    manifest_path: Path,
    validation_path: Path,
    output_dir: Path,
    spec_path: Path | None = None,
) -> dict:
    spec = _load_spec(spec_path)
    manifest = _read_json(manifest_path)
    validation = _read_json(validation_path)
    _require_valid_r1(manifest, validation, spec)

    dictionary = pd.read_csv(dictionary_path)
    features = _feature_table(dictionary)
    horizons = [int(h) for h in spec["horizons_bars"]]

    needed = {"date", "ticker"}
    needed.update(features["column"].tolist())
    for h in horizons:
        for pattern in RETURN_OUTCOMES.values():
            needed.add(pattern.format(h=h))
    existing_columns = pd.read_csv(dataset_path, nrows=0).columns
    missing_needed = sorted(set(needed) - set(existing_columns))
    if missing_needed:
        raise RuntimeError(
            "R2 input dataset is missing required columns: " + ", ".join(missing_needed)
        )

    data = pd.read_csv(
        dataset_path,
        usecols=sorted(needed),
        parse_dates=["date"],
        low_memory=False,
    )
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()

    anchors_dates = _month_end_anchor_dates(data["date"])
    anchors = data[data["date"].isin(anchors_dates)].copy()
    anchors = anchors.sort_values(["date", "ticker"]).reset_index(drop=True)

    min_rows = int(spec["minimum_cross_section_rows"])
    primary_rows: list[dict] = []
    long_rows: list[dict] = []
    annual_rows: list[dict] = []
    quintile_rows: list[dict] = []

    cross_features = features[features["scope"].eq("cross_sectional")]

    for item in cross_features.itertuples(index=False):
        feature = item.column
        family = item.family

        for h in horizons:
            for outcome_kind, pattern in RETURN_OUTCOMES.items():
                target = pattern.format(h=h)
                month_stats = _cross_section_month_stats(
                    anchors,
                    feature,
                    target,
                    min_rows,
                )
                row, annual = _summary_row(
                    month_stats,
                    feature=feature,
                    family=family,
                    horizon=h,
                    outcome_kind=outcome_kind,
                    bootstrap=(outcome_kind == "spy_relative_return"),
                    spec=spec,
                )
                long_rows.append(row)

                if outcome_kind == "spy_relative_return":
                    primary_rows.append(row.copy())
                    annual_rows.extend(annual)
                    quintiles = _monthly_quintiles(
                        anchors,
                        feature,
                        target,
                        min_rows,
                    )
                    quintile_rows.extend(
                        _quintile_profile_rows(
                            quintiles,
                            feature=feature,
                            family=family,
                            horizon=h,
                            spec=spec,
                        )
                    )

    primary = pd.DataFrame(primary_rows)
    long = pd.DataFrame(long_rows)
    annual = pd.DataFrame(annual_rows)
    quintiles = pd.DataFrame(quintile_rows)
    market = _market_regime_evaluation(
        anchors,
        features,
        horizons,
        spec,
    )
    redundancy = _redundancy_table(
        anchors,
        features,
        float(spec["redundancy_threshold_abs_spearman"]),
    )
    family = _family_summary(primary)
    findings = _top_findings(primary, spec)

    output_dir.mkdir(parents=True, exist_ok=True)
    primary.to_csv(output_dir / "feature_horizon_summary.csv", index=False)
    long.to_csv(output_dir / "univariate_outcomes_long.csv", index=False)
    annual.to_csv(output_dir / "annual_ic.csv", index=False)
    quintiles.to_csv(output_dir / "quintile_profiles.csv", index=False)
    market.to_csv(output_dir / "market_regime_context.csv", index=False)
    redundancy.to_csv(output_dir / "feature_redundancy.csv", index=False)
    family.to_csv(output_dir / "family_summary.csv", index=False)
    (output_dir / "top_findings.json").write_text(
        json.dumps(findings, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    validation_out = {
        "research_patch": "R2",
        "status": "ok",
        "r1_dataset_version": manifest.get("dataset_version"),
        "r1_validation_status": validation.get("status"),
        "r1_universe_completeness": validation.get("universe_completeness"),
        "dataset_rows": int(manifest.get("dataset_rows", len(data))),
        "dataset_tickers": int(manifest.get("dataset_tickers", data["ticker"].nunique())),
        "month_end_anchor_months": int(anchors["date"].nunique()),
        "month_end_anchor_rows": int(len(anchors)),
        "cross_sectional_feature_count": int(len(cross_features)),
        "market_regime_feature_count": int((features["scope"] == "date_level").sum()),
        "horizons_bars": horizons,
        "combined_model_fitted": False,
        "production_logic_used": False,
        "evidence_labels_are_discovery_only": True,
        "bootstrap_repetitions": int(spec["bootstrap"]["repetitions"]),
        "bootstrap_block_months": int(spec["bootstrap"]["block_months"]),
    }
    (output_dir / "r2_validation.json").write_text(
        json.dumps(validation_out, indent=2),
        encoding="utf-8",
    )
    (output_dir / "r2_spec_frozen.json").write_text(
        json.dumps(spec, indent=2),
        encoding="utf-8",
    )
    _write_markdown_report(
        output_dir / "r2_report.md",
        primary,
        family,
        market,
        anchors,
        spec,
    )
    return validation_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run clean-sheet R2 univariate predictive discovery."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dictionary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spec", type=Path, default=None)
    args = parser.parse_args()

    result = run_r2(
        dataset_path=args.dataset,
        dictionary_path=args.dictionary,
        manifest_path=args.manifest,
        validation_path=args.validation,
        output_dir=args.output,
        spec_path=args.spec,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
