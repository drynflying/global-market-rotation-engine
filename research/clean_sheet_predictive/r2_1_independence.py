from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SPEC_PATH = Path(__file__).with_name("r2_1_spec.json")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_spec(path: Path | None = None) -> dict:
    return _read_json(path or DEFAULT_SPEC_PATH)


def _feature_scope(dictionary: pd.DataFrame) -> pd.DataFrame:
    f = dictionary[dictionary["kind"].eq("feature")][["column", "family"]].copy()
    f["scope"] = np.where(f["family"].eq("market_regime"), "date_level", "cross_sectional")
    return f.reset_index(drop=True)


def _month_end_anchor_dates(dates: pd.Series) -> pd.DatetimeIndex:
    d = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    f = pd.DataFrame({"date": d})
    f["month"] = f["date"].dt.to_period("M")
    return pd.DatetimeIndex(
        f.groupby("month", observed=True)["date"].max().sort_values().to_numpy()
    )


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
    out = np.empty(repetitions, dtype=float)

    for i in range(repetitions):
        starts = rng.integers(0, starts_max + 1, size=blocks_needed)
        sample = np.concatenate([arr[s : s + block] for s in starts])[:n]
        out[i] = float(np.mean(sample))

    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi)


def _rank_redundancy(
    anchors: pd.DataFrame,
    features: list[str],
    *,
    min_rows: int,
    spec: dict,
) -> pd.DataFrame:
    monthly_rows: list[pd.DataFrame] = []

    for date, g in anchors.groupby("date", sort=True):
        x = g[features].replace([np.inf, -np.inf], np.nan)
        corr = x.corr(method="spearman", min_periods=min_rows)
        if corr.empty:
            continue

        rows = []
        for i, a in enumerate(features):
            for b in features[i + 1 :]:
                rho = corr.loc[a, b]
                if pd.notna(rho):
                    rows.append(
                        {
                            "date": pd.Timestamp(date),
                            "feature_a": a,
                            "feature_b": b,
                            "rho": float(rho),
                            "abs_rho": float(abs(rho)),
                        }
                    )
        if rows:
            monthly_rows.append(pd.DataFrame(rows))

    if not monthly_rows:
        return pd.DataFrame()

    m = pd.concat(monthly_rows, ignore_index=True)
    floor = float(spec["redundancy"]["near_redundancy_abs_rho_floor"])

    summary = (
        m.groupby(["feature_a", "feature_b"], observed=True)
        .agg(
            months=("rho", "size"),
            mean_rho=("rho", "mean"),
            median_rho=("rho", "median"),
            mean_abs_rho=("abs_rho", "mean"),
            median_abs_rho=("abs_rho", "median"),
            pct_abs_ge_0999=("abs_rho", lambda s: float((s >= 0.9999).mean())),
            pct_abs_ge_095=("abs_rho", lambda s: float((s >= 0.95).mean())),
            pct_abs_ge_090=("abs_rho", lambda s: float((s >= 0.90).mean())),
            pct_abs_ge_floor=("abs_rho", lambda s: float((s >= floor).mean())),
        )
        .reset_index()
    )

    exact_rho = float(spec["redundancy"]["exact_rank_equivalence_abs_rho"])
    exact_frac = float(spec["redundancy"]["exact_rank_equivalence_month_fraction"])
    near_med = float(spec["redundancy"]["near_redundancy_median_abs_rho"])
    near_frac = float(spec["redundancy"]["near_redundancy_month_fraction"])

    def relation(row: pd.Series) -> str:
        if (
            row["median_abs_rho"] >= exact_rho
            and row["pct_abs_ge_0999"] >= exact_frac
        ):
            return "EXACT_RANK_EQUIVALENT"
        if (
            row["median_abs_rho"] >= near_med
            and row["pct_abs_ge_floor"] >= near_frac
        ):
            return "NEAR_REDUNDANT"
        return "DISTINCT"

    summary["relation"] = summary.apply(relation, axis=1)
    return summary.sort_values(
        ["relation", "median_abs_rho"],
        ascending=[True, False],
    ).reset_index(drop=True)


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> list[list[str]]:
        out: dict[str, list[str]] = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return [sorted(v) for v in out.values()]


def _interpretability_penalty(feature: str) -> tuple[int, int, str]:
    # Lower is preferred.
    if feature.startswith("cs_"):
        return (3, 0, feature)
    if feature.startswith("rs_primary_"):
        return (2, 0, feature)
    if feature.startswith("rs_spy_"):
        return (1, 0, feature)
    if feature.startswith("ret_"):
        return (0, 0, feature)
    return (0, 1, feature)


def _evidence_rank(label: str) -> int:
    return {
        "ROBUST_CANDIDATE": 0,
        "PROMISING": 1,
        "WEAK": 2,
        "INCONSISTENT_OR_NONE": 9,
    }.get(str(label), 9)


def _action_rank(label: str) -> int:
    return {
        "BOTH_SIDES_ACTIONABLE": 0,
        "FAVOR_ACTIONABLE": 1,
        "AVOID_ACTIONABLE": 1,
        "SEPARATION_ONLY": 2,
        "NO_ACTIONABLE_EDGE": 3,
    }.get(str(label), 9)


def _choose_exact_representative(
    members: list[str],
    anchors: pd.DataFrame,
    r2_primary: pd.DataFrame,
) -> str:
    coverage = {
        f: int(anchors[f].replace([np.inf, -np.inf], np.nan).notna().sum())
        for f in members
    }

    def strength(f: str) -> float:
        g = r2_primary[r2_primary["feature"].eq(f)]
        if g.empty:
            return 0.0
        return float(g["mean_monthly_ic"].abs().max())

    return sorted(
        members,
        key=lambda f: (
            _interpretability_penalty(f),
            -coverage.get(f, 0),
            -strength(f),
            f,
        ),
    )[0]


def _exact_clusters(
    features: list[str],
    redundancy: pd.DataFrame,
    anchors: pd.DataFrame,
    r2_primary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    uf = _UnionFind(features)
    exact = redundancy[redundancy["relation"].eq("EXACT_RANK_EQUIVALENT")]
    for row in exact.itertuples(index=False):
        uf.union(row.feature_a, row.feature_b)

    rows: list[dict] = []
    mapping: dict[str, str] = {}

    cluster_id = 0
    for members in sorted(uf.groups(), key=lambda x: (len(x), x), reverse=True):
        rep = _choose_exact_representative(members, anchors, r2_primary)
        for m in members:
            mapping[m] = rep
        if len(members) <= 1:
            continue
        cluster_id += 1
        rows.append(
            {
                "cluster_id": f"EQ{cluster_id:02d}",
                "cluster_size": len(members),
                "representative": rep,
                "members": "|".join(members),
                "reason": "Within-date cross-sectional ranks are effectively identical in >=95% of overlapping months.",
            }
        )

    return pd.DataFrame(rows), mapping


def _extreme_monthly_metrics(
    anchors: pd.DataFrame,
    *,
    feature: str,
    horizon: int,
    direction: str,
    extreme_fraction: float,
    min_rows: int,
) -> pd.DataFrame:
    ret_col = f"fwd_spy_relative_return_{horizon}"
    mdd_col = f"fwd_max_drawdown_{horizon}"
    mae_col = f"fwd_spy_rel_mae_{horizon}"
    cols = ["date", "ticker", feature, ret_col, mdd_col, mae_col]
    rows: list[dict] = []

    for date, g in anchors[cols].groupby("date", sort=True):
        p = g.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, ret_col]).copy()
        if len(p) < min_rows or p[feature].nunique() < 2:
            continue

        unique = int(p[feature].nunique())
        if unique <= 5:
            high_mask = p[feature].eq(p[feature].max())
            low_mask = p[feature].eq(p[feature].min())
            grouping = "discrete_extremes"
        else:
            rank = p[feature].rank(method="average", pct=True)
            high_mask = rank.gt(1.0 - extreme_fraction)
            low_mask = rank.le(extreme_fraction)
            grouping = "decile_extremes"

        if direction == "HIGHER_FEATURE_FAVORABLE":
            favor = p[high_mask]
            avoid = p[low_mask]
        else:
            favor = p[low_mask]
            avoid = p[high_mask]

        if favor.empty or avoid.empty:
            continue

        rows.append(
            {
                "date": pd.Timestamp(date),
                "grouping": grouping,
                "favor_n": int(len(favor)),
                "avoid_n": int(len(avoid)),
                "favor_mean_rel_return": float(favor[ret_col].mean()),
                "avoid_mean_rel_return": float(avoid[ret_col].mean()),
                "favor_median_rel_return": float(favor[ret_col].median()),
                "avoid_median_rel_return": float(avoid[ret_col].median()),
                "favor_beat_spy_rate": float((favor[ret_col] > 0).mean()),
                "avoid_beat_spy_rate": float((avoid[ret_col] > 0).mean()),
                "oriented_return_spread": float(favor[ret_col].mean() - avoid[ret_col].mean()),
                "favor_mean_max_drawdown": float(favor[mdd_col].mean()) if favor[mdd_col].notna().any() else np.nan,
                "avoid_mean_max_drawdown": float(avoid[mdd_col].mean()) if avoid[mdd_col].notna().any() else np.nan,
                "favor_mean_spy_rel_mae": float(favor[mae_col].mean()) if favor[mae_col].notna().any() else np.nan,
                "avoid_mean_spy_rel_mae": float(avoid[mae_col].mean()) if avoid[mae_col].notna().any() else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _metric_ci(
    monthly: pd.DataFrame,
    col: str,
    *,
    feature: str,
    horizon: int,
    spec: dict,
) -> tuple[float, float]:
    seed = (
        int(spec["bootstrap"]["seed"])
        + zlib.crc32(f"{feature}|{horizon}|{col}".encode("utf-8"))
    ) % (2**32 - 1)
    return _block_bootstrap_mean_ci(
        monthly[col],
        block_months=int(spec["bootstrap"]["block_months"]),
        repetitions=int(spec["bootstrap"]["repetitions"]),
        seed=seed,
    )


def _actionability_row(
    anchors: pd.DataFrame,
    r2_row: pd.Series,
    spec: dict,
) -> tuple[dict, pd.DataFrame]:
    feature = str(r2_row["feature"])
    horizon = int(r2_row["horizon_bars"])
    direction = str(r2_row["direction_if_used"])
    monthly = _extreme_monthly_metrics(
        anchors,
        feature=feature,
        horizon=horizon,
        direction=direction,
        extreme_fraction=float(spec["extreme_fraction"]),
        min_rows=int(spec["minimum_cross_section_rows"]),
    )

    out = {
        "feature": feature,
        "family": r2_row["family"],
        "horizon_bars": horizon,
        "horizon_label": r2_row["horizon_label"],
        "r2_evidence_label": r2_row["evidence_label"],
        "r2_mean_monthly_ic": r2_row["mean_monthly_ic"],
        "direction_if_used": direction,
        "months": int(len(monthly)),
    }

    metric_cols = [
        "favor_mean_rel_return",
        "avoid_mean_rel_return",
        "favor_beat_spy_rate",
        "avoid_beat_spy_rate",
        "oriented_return_spread",
    ]
    for col in metric_cols:
        out[col] = float(monthly[col].mean()) if len(monthly) else np.nan
        lo, hi = _metric_ci(monthly, col, feature=feature, horizon=horizon, spec=spec) if len(monthly) else (np.nan, np.nan)
        out[f"{col}_ci_low"] = lo
        out[f"{col}_ci_high"] = hi

    for col in [
        "favor_mean_max_drawdown",
        "avoid_mean_max_drawdown",
        "favor_mean_spy_rel_mae",
        "avoid_mean_spy_rel_mae",
    ]:
        out[col] = float(monthly[col].mean()) if len(monthly) else np.nan

    favor_actionable = (
        np.isfinite(out["favor_mean_rel_return_ci_low"])
        and out["favor_mean_rel_return_ci_low"] > 0
        and np.isfinite(out["favor_beat_spy_rate_ci_low"])
        and out["favor_beat_spy_rate_ci_low"] > 0.50
    )
    avoid_actionable = (
        np.isfinite(out["avoid_mean_rel_return_ci_high"])
        and out["avoid_mean_rel_return_ci_high"] < 0
        and np.isfinite(out["avoid_beat_spy_rate_ci_high"])
        and out["avoid_beat_spy_rate_ci_high"] < 0.50
    )
    separation = (
        np.isfinite(out["oriented_return_spread_ci_low"])
        and out["oriented_return_spread_ci_low"] > 0
    )

    if favor_actionable and avoid_actionable:
        label = "BOTH_SIDES_ACTIONABLE"
    elif favor_actionable:
        label = "FAVOR_ACTIONABLE"
    elif avoid_actionable:
        label = "AVOID_ACTIONABLE"
    elif separation:
        label = "SEPARATION_ONLY"
    else:
        label = "NO_ACTIONABLE_EDGE"

    out["favor_actionable"] = favor_actionable
    out["avoid_actionable"] = avoid_actionable
    out["separation_significant"] = separation
    out["actionability_label"] = label
    return out, monthly


def _near_pair_lookup(redundancy: pd.DataFrame) -> dict[frozenset[str], pd.Series]:
    lookup: dict[frozenset[str], pd.Series] = {}
    for _, row in redundancy[
        redundancy["relation"].isin(["EXACT_RANK_EQUIVALENT", "NEAR_REDUNDANT"])
    ].iterrows():
        lookup[frozenset([row["feature_a"], row["feature_b"]])] = row
    return lookup


def _candidate_set(
    r2_primary: pd.DataFrame,
    actionability: pd.DataFrame,
    exact_map: dict[str, str],
    redundancy: pd.DataFrame,
    spec: dict,
) -> pd.DataFrame:
    eligible = r2_primary[
        r2_primary["evidence_label"].isin(spec["eligible_r2_labels"])
    ].copy()
    eligible = eligible.merge(
        actionability[
            ["feature", "horizon_bars", "actionability_label"]
        ],
        on=["feature", "horizon_bars"],
        how="left",
    )

    lookup = _near_pair_lookup(redundancy)
    rows: list[dict] = []

    for horizon, g in eligible.groupby("horizon_bars", sort=True):
        g = g.copy()
        g["exact_representative"] = g["feature"].map(exact_map).fillna(g["feature"])
        g["_evidence_rank"] = g["evidence_label"].map(_evidence_rank)
        g["_action_rank"] = g["actionability_label"].map(_action_rank).fillna(9)
        g["_abs_ic"] = g["mean_monthly_ic"].abs()
        g["_interp"] = g["feature"].map(lambda x: _interpretability_penalty(str(x))[0])

        # Members that are not the chosen exact-equivalence representative are
        # automatically excluded before near-redundancy selection.
        exact_nonrep = g[g["feature"] != g["exact_representative"]]
        for row in exact_nonrep.itertuples(index=False):
            rows.append(
                {
                    "horizon_bars": int(horizon),
                    "horizon_label": row.horizon_label,
                    "feature": row.feature,
                    "family": row.family,
                    "r2_evidence_label": row.evidence_label,
                    "actionability_label": row.actionability_label,
                    "mean_monthly_ic": row.mean_monthly_ic,
                    "status": "EXCLUDED",
                    "reason": "EXACT_RANK_EQUIVALENT",
                    "kept_feature": row.exact_representative,
                    "within_date_median_abs_rho": 1.0,
                }
            )

        reps = g[g["feature"] == g["exact_representative"]].copy()
        reps = reps.sort_values(
            ["_evidence_rank", "_action_rank", "_abs_ic", "_interp", "feature"],
            ascending=[True, True, False, True, True],
        )

        selected: list[str] = []

        for row in reps.itertuples(index=False):
            feature = row.feature
            blocking = None
            blocking_pair = None
            for kept in selected:
                pair = lookup.get(frozenset([feature, kept]))
                if pair is not None and pair["relation"] == "NEAR_REDUNDANT":
                    blocking = kept
                    blocking_pair = pair
                    break

            if blocking is None:
                selected.append(feature)
                rows.append(
                    {
                        "horizon_bars": int(horizon),
                        "horizon_label": row.horizon_label,
                        "feature": feature,
                        "family": row.family,
                        "r2_evidence_label": row.evidence_label,
                        "actionability_label": row.actionability_label,
                        "mean_monthly_ic": row.mean_monthly_ic,
                        "status": "SELECTED",
                        "reason": "NONREDUNDANT_R2_CANDIDATE",
                        "kept_feature": feature,
                        "within_date_median_abs_rho": np.nan,
                    }
                )
            else:
                rows.append(
                    {
                        "horizon_bars": int(horizon),
                        "horizon_label": row.horizon_label,
                        "feature": feature,
                        "family": row.family,
                        "r2_evidence_label": row.evidence_label,
                        "actionability_label": row.actionability_label,
                        "mean_monthly_ic": row.mean_monthly_ic,
                        "status": "EXCLUDED",
                        "reason": "NEAR_REDUNDANT_WITH_STRONGER_SELECTED_FEATURE",
                        "kept_feature": blocking,
                        "within_date_median_abs_rho": float(blocking_pair["median_abs_rho"]),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["horizon_bars", "status", "r2_evidence_label", "feature"]
    ).reset_index(drop=True)


def _report(
    path: Path,
    r2_primary: pd.DataFrame,
    redundancy: pd.DataFrame,
    clusters: pd.DataFrame,
    actionability: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    lines = [
        "# R2.1 — Signal Independence & Actionability Audit",
        "",
        "**No combined model is fitted in R2.1.**",
        "",
        "## Independence audit",
        "",
        f"- Feature pairs audited: {len(redundancy):,}",
        f"- Exact rank-equivalent pairs: {(redundancy['relation'] == 'EXACT_RANK_EQUIVALENT').sum():,}",
        f"- Near-redundant pairs: {(redundancy['relation'] == 'NEAR_REDUNDANT').sum():,}",
        f"- Exact-equivalence clusters with 2+ members: {len(clusters):,}",
        "",
        "## Actionability",
        "",
    ]

    if actionability.empty:
        lines.append("- No R2 evidence candidates were available.")
    else:
        counts = actionability["actionability_label"].value_counts()
        for label, count in counts.items():
            lines.append(f"- {label}: {int(count)} feature/horizon combinations")

    lines.extend(["", "## R3-permitted nonredundant candidates", ""])
    if candidates.empty:
        lines.append("- None.")
    else:
        selected = candidates[candidates["status"].eq("SELECTED")]
        for h, g in selected.groupby("horizon_bars", sort=True):
            lines.append(f"### {g['horizon_label'].iloc[0]} / {int(h)} bars")
            for row in g.sort_values(
                ["r2_evidence_label", "mean_monthly_ic"],
                ascending=[True, False]
            ).itertuples(index=False):
                lines.append(
                    f"- **{row.feature}** ({row.family}) — "
                    f"{row.r2_evidence_label}; {row.actionability_label}; "
                    f"IC {row.mean_monthly_ic:+.4f}."
                )
            lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `FAVOR_ACTIONABLE` means the historically favorable extreme group itself had positive SPY-relative return and a >50% SPY-beating rate with both 95% block-bootstrap intervals on the favorable side of those thresholds.",
            "- `AVOID_ACTIONABLE` applies the symmetric requirement to the historically unfavorable extreme group.",
            "- `SEPARATION_ONLY` means the feature reliably ranked better versus worse ETFs, but did not establish an independently positive FAVOR or independently negative AVOID side under the strict actionability test.",
            "- R3 is allowed to consume only rows marked `SELECTED`; excluded rows remain documented for auditability.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r2_1(
    *,
    dataset_path: Path,
    dictionary_path: Path,
    r2_summary_path: Path,
    r2_validation_path: Path,
    output_dir: Path,
    spec_path: Path | None = None,
) -> dict:
    spec = _load_spec(spec_path)
    r2_validation = _read_json(r2_validation_path)
    if r2_validation.get("status") != "ok":
        raise RuntimeError(
            f"R2.1 requires successful R2 output; got {r2_validation.get('status')!r}"
        )

    dictionary = pd.read_csv(dictionary_path)
    feature_meta = _feature_scope(dictionary)
    cross_features = feature_meta.loc[
        feature_meta["scope"].eq("cross_sectional"), "column"
    ].tolist()

    r2_primary = pd.read_csv(r2_summary_path)
    eligible_r2 = r2_primary[
        r2_primary["evidence_label"].isin(spec["eligible_r2_labels"])
    ].copy()

    needed = {"date", "ticker"}
    needed.update(cross_features)
    for h in spec["horizons_bars"]:
        needed.update(
            {
                f"fwd_spy_relative_return_{h}",
                f"fwd_max_drawdown_{h}",
                f"fwd_spy_rel_mae_{h}",
            }
        )

    header = pd.read_csv(dataset_path, nrows=0).columns
    missing = sorted(set(needed) - set(header))
    if missing:
        raise RuntimeError(
            "R2.1 dataset missing required columns: " + ", ".join(missing)
        )

    date_only = pd.read_csv(dataset_path, usecols=["date"], parse_dates=["date"])
    anchor_dates = _month_end_anchor_dates(date_only["date"])
    del date_only

    data = pd.read_csv(
        dataset_path,
        usecols=sorted(needed),
        parse_dates=["date"],
        low_memory=False,
    )
    anchors = data[data["date"].isin(anchor_dates)].copy()
    anchors["ticker"] = anchors["ticker"].astype(str).str.upper().str.strip()
    anchors = anchors.sort_values(["date", "ticker"]).reset_index(drop=True)
    del data

    redundancy = _rank_redundancy(
        anchors,
        cross_features,
        min_rows=int(spec["minimum_cross_section_rows"]),
        spec=spec,
    )
    clusters, exact_map = _exact_clusters(
        cross_features,
        redundancy,
        anchors,
        r2_primary,
    )

    action_rows: list[dict] = []
    monthly_action_rows: list[pd.DataFrame] = []
    for _, row in eligible_r2.iterrows():
        result, monthly = _actionability_row(anchors, row, spec)
        action_rows.append(result)
        if not monthly.empty:
            monthly = monthly.copy()
            monthly.insert(0, "feature", row["feature"])
            monthly.insert(1, "horizon_bars", int(row["horizon_bars"]))
            monthly_action_rows.append(monthly)

    actionability = pd.DataFrame(action_rows)
    monthly_action = (
        pd.concat(monthly_action_rows, ignore_index=True)
        if monthly_action_rows
        else pd.DataFrame()
    )

    candidates = _candidate_set(
        r2_primary,
        actionability,
        exact_map,
        redundancy,
        spec,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    redundancy.to_csv(output_dir / "within_date_redundancy.csv", index=False)
    clusters.to_csv(output_dir / "exact_equivalence_clusters.csv", index=False)
    actionability.to_csv(output_dir / "actionability_audit.csv", index=False)
    monthly_action.to_csv(output_dir / "actionability_monthly.csv", index=False)
    candidates.to_csv(output_dir / "nonredundant_candidate_set.csv", index=False)

    selected = candidates[candidates["status"].eq("SELECTED")].copy()
    summary = {
        "research_patch": "R2.1",
        "status": "ok",
        "combined_model_fitted": False,
        "r2_evidence_feature_horizon_count": int(len(eligible_r2)),
        "cross_sectional_feature_count": int(len(cross_features)),
        "month_end_anchor_months": int(anchors["date"].nunique()),
        "exact_rank_equivalent_pairs": int(
            (redundancy["relation"] == "EXACT_RANK_EQUIVALENT").sum()
        ),
        "near_redundant_pairs": int(
            (redundancy["relation"] == "NEAR_REDUNDANT").sum()
        ),
        "exact_equivalence_clusters": int(len(clusters)),
        "selected_r3_candidate_count": int(len(selected)),
        "selected_by_horizon": {
            str(int(h)): g["feature"].tolist()
            for h, g in selected.groupby("horizon_bars", sort=True)
        },
        "actionability_counts": (
            actionability["actionability_label"].value_counts().to_dict()
            if not actionability.empty
            else {}
        ),
        "important_note": (
            "R2.1 labels are historical discovery/audit evidence only. "
            "No candidate is production-validated."
        ),
    }
    (output_dir / "r2_1_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "r2_1_spec_frozen.json").write_text(
        json.dumps(spec, indent=2),
        encoding="utf-8",
    )
    _report(
        output_dir / "r2_1_report.md",
        r2_primary,
        redundancy,
        clusters,
        actionability,
        candidates,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run R2.1 signal independence and actionability audit."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dictionary", required=True, type=Path)
    parser.add_argument("--r2-summary", required=True, type=Path)
    parser.add_argument("--r2-validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spec", type=Path, default=None)
    args = parser.parse_args()

    result = run_r2_1(
        dataset_path=args.dataset,
        dictionary_path=args.dictionary,
        r2_summary_path=args.r2_summary,
        r2_validation_path=args.r2_validation,
        output_dir=args.output,
        spec_path=args.spec,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
