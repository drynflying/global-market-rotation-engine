from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


DEFAULT_SPEC_PATH = Path(__file__).with_name("r3_spec.json")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_spec(path: Path | None = None) -> dict:
    return _read_json(path or DEFAULT_SPEC_PATH)


def _month_end_anchor_dates(dates: pd.Series) -> pd.DatetimeIndex:
    d = pd.to_datetime(dates, errors="coerce").dropna().drop_duplicates().sort_values()
    f = pd.DataFrame({"date": d})
    f["month"] = f["date"].dt.to_period("M")
    return pd.DatetimeIndex(
        f.groupby("month", observed=True)["date"].max().sort_values().to_numpy()
    )


def _candidate_features_by_horizon(candidate_set: pd.DataFrame, spec: dict) -> dict[int, list[str]]:
    selected = candidate_set[candidate_set["status"].eq("SELECTED")].copy()
    result: dict[int, list[str]] = {}
    for h in map(int, spec["horizons_bars"]):
        feats = selected.loc[selected["horizon_bars"].eq(h), "feature"].astype(str).tolist()
        if not feats:
            raise RuntimeError(f"R3 has no R2.1 SELECTED features for horizon {h}")
        result[h] = feats
    return result


def _cross_sectional_rank_transform(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for feature in features:
        out[feature] = out.groupby("date", observed=True)[feature].rank(
            method="average", pct=True, na_option="keep"
        )
    return out


def _make_avoid_label(frame: pd.DataFrame, target: str, bottom_fraction: float) -> pd.Series:
    labels = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, idx in frame.groupby("date", observed=True).groups.items():
        s = frame.loc[idx, target]
        valid = s.dropna()
        if len(valid) < 5:
            continue
        ranks = valid.rank(method="first", pct=True)
        labels.loc[valid.index] = (ranks <= bottom_fraction).astype(float)
    return labels


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    p = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(p) < 3 or p["x"].nunique() < 2 or p["y"].nunique() < 2:
        return float("nan")
    return float(p["x"].rank().corr(p["y"].rank()))


def _date_weights(dates: pd.Series) -> np.ndarray:
    counts = dates.value_counts()

    # Pandas / NumPy may expose Series-backed arrays as read-only under
    # copy-on-write semantics. Request an owned array explicitly and avoid
    # in-place mutation so this remains compatible across pandas versions.
    w = dates.map(lambda d: 1.0 / counts[d]).to_numpy(dtype=float, copy=True)

    total = float(w.sum())
    if np.isfinite(w).all() and total > 0:
        w = w * (len(w) / total)

    return w


def _training_feature_orientation(train: pd.DataFrame, features: list[str], target: str) -> tuple[dict[str, int], str]:
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
    return orientation, best


def _eligible_training_rows(
    anchors: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    outcome_end_col: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    mask = (
        anchors["date"].lt(cutoff)
        & pd.to_datetime(anchors[outcome_end_col], errors="coerce").lt(cutoff)
        & anchors[target].notna()
    )
    cols = ["date", "ticker", target, outcome_end_col, "avoid_label"] + features
    train = anchors.loc[mask, cols].replace([np.inf, -np.inf], np.nan)
    return train.dropna(subset=features + [target, "avoid_label"]).copy()


def _test_rows(
    anchors: pd.DataFrame,
    *,
    year: int,
    features: list[str],
    target: str,
    outcome_end_col: str,
) -> pd.DataFrame:
    cols = ["date", "ticker", target, outcome_end_col, "avoid_label"] + features
    test = anchors.loc[anchors["date"].dt.year.eq(year), cols].replace([np.inf, -np.inf], np.nan)
    return test.dropna(subset=features).copy()


def _fit_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    horizon: int,
    year: int,
    spec: dict,
) -> tuple[pd.DataFrame, list[dict], dict]:
    scaler = StandardScaler()
    x_train = train[features].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    weights = _date_weights(train["date"])

    ridge_cfg = spec["models"]["return_ranking"]
    ridge = Ridge(alpha=float(ridge_cfg["alpha"]), fit_intercept=bool(ridge_cfg["fit_intercept"]))
    ridge.fit(x_train_scaled, train[target].to_numpy(dtype=float), sample_weight=weights)

    logistic_cfg = spec["models"]["avoid_classifier"]
    logit = LogisticRegression(
        C=float(logistic_cfg["C"]),
        penalty=str(logistic_cfg["penalty"]),
        solver=str(logistic_cfg["solver"]),
        max_iter=int(logistic_cfg["max_iter"]),
    )
    logit.fit(
        x_train_scaled,
        train["avoid_label"].astype(int).to_numpy(),
        sample_weight=weights,
    )

    orientation, best_feature = _training_feature_orientation(train, features, target)
    equal_weight = np.mean(
        np.column_stack([test[f].to_numpy(dtype=float) * orientation[f] for f in features]),
        axis=1,
    )
    best_uni = test[best_feature].to_numpy(dtype=float) * orientation[best_feature]

    pred = test[["date", "ticker", target, "avoid_label"]].copy()
    pred["horizon_bars"] = horizon
    pred["test_year"] = year
    pred["ridge_prediction"] = ridge.predict(x_test_scaled)
    pred["equal_weight_score"] = equal_weight
    pred["best_univariate_score"] = best_uni
    pred["best_univariate_feature"] = best_feature
    pred["avoid_probability"] = logit.predict_proba(x_test_scaled)[:, 1]
    pred["outcome_matured"] = pred[target].notna()

    coefficients: list[dict] = []
    for feature, ridge_coef, logit_coef in zip(features, ridge.coef_, logit.coef_[0]):
        coefficients.append(
            {
                "horizon_bars": horizon,
                "test_year": year,
                "feature": feature,
                "ridge_standardized_coefficient": float(ridge_coef),
                "logistic_standardized_coefficient": float(logit_coef),
                "training_orientation": int(orientation[feature]),
                "best_univariate_feature": feature == best_feature,
            }
        )

    fold_info = {
        "horizon_bars": horizon,
        "test_year": year,
        "train_rows": int(len(train)),
        "train_months": int(train["date"].nunique()),
        "test_prediction_rows": int(len(test)),
        "test_months": int(test["date"].nunique()),
        "matured_test_rows": int(test[target].notna().sum()),
        "matured_test_months": int(test.loc[test[target].notna(), "date"].nunique()),
        "best_univariate_feature": best_feature,
        "training_avoid_base_rate": float(train["avoid_label"].mean()),
    }
    return pred, coefficients, fold_info


def _quantile_masks(score: pd.Series) -> dict[str, pd.Series]:
    r = score.rank(method="average", pct=True)
    return {
        "top_q": r.gt(0.80),
        "bottom_q": r.le(0.20),
        "top_d": r.gt(0.90),
        "bottom_d": r.le(0.10),
    }


def _monthly_ranking_metrics(predictions: pd.DataFrame, target: str, min_rows: int) -> pd.DataFrame:
    methods = {
        "ridge": "ridge_prediction",
        "equal_weight": "equal_weight_score",
        "best_univariate": "best_univariate_score",
    }
    rows: list[dict] = []
    matured = predictions[predictions["outcome_matured"]].copy()

    for (h, date), g in matured.groupby(["horizon_bars", "date"], observed=True, sort=True):
        g = g.dropna(subset=[target])
        if len(g) < min_rows:
            continue
        for method, score_col in methods.items():
            p = g.dropna(subset=[score_col, target])
            if len(p) < min_rows or p[score_col].nunique() < 2:
                continue
            masks = _quantile_masks(p[score_col])
            top_q, bottom_q = p[masks["top_q"]], p[masks["bottom_q"]]
            top_d, bottom_d = p[masks["top_d"]], p[masks["bottom_d"]]
            rows.append(
                {
                    "horizon_bars": int(h),
                    "date": pd.Timestamp(date),
                    "method": method,
                    "n": int(len(p)),
                    "ic": _safe_spearman(p[score_col], p[target]),
                    "top_quintile_mean_rel_return": float(top_q[target].mean()),
                    "bottom_quintile_mean_rel_return": float(bottom_q[target].mean()),
                    "q5_minus_q1_spread": float(top_q[target].mean() - bottom_q[target].mean()),
                    "top_quintile_beat_spy_rate": float((top_q[target] > 0).mean()),
                    "bottom_quintile_beat_spy_rate": float((bottom_q[target] > 0).mean()),
                    "top_decile_mean_rel_return": float(top_d[target].mean()),
                    "bottom_decile_mean_rel_return": float(bottom_d[target].mean()),
                    "top_decile_beat_spy_rate": float((top_d[target] > 0).mean()),
                    "bottom_decile_beat_spy_rate": float((bottom_d[target] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _monthly_avoid_metrics(predictions: pd.DataFrame, target: str, min_rows: int) -> pd.DataFrame:
    rows: list[dict] = []
    matured = predictions[predictions["outcome_matured"]].copy()

    for (h, date), g in matured.groupby(["horizon_bars", "date"], observed=True, sort=True):
        p = g.dropna(subset=["avoid_probability", "avoid_label", target]).copy()
        if len(p) < min_rows or p["avoid_label"].nunique() < 2:
            continue
        labels = p["avoid_label"].astype(int)
        prob = p["avoid_probability"].astype(float)
        base = float(labels.mean())
        r = prob.rank(method="average", pct=True)
        q = p[r.gt(0.80)]
        d = p[r.gt(0.90)]
        q_labels = q["avoid_label"].astype(int)
        d_labels = d["avoid_label"].astype(int)
        rows.append(
            {
                "horizon_bars": int(h),
                "date": pd.Timestamp(date),
                "n": int(len(p)),
                "base_rate": base,
                "roc_auc": float(roc_auc_score(labels, prob)),
                "average_precision": float(average_precision_score(labels, prob)),
                "brier_score": float(brier_score_loss(labels, prob)),
                "top_risk_quintile_precision": float(q_labels.mean()),
                "top_risk_quintile_recall": float(q_labels.sum() / max(labels.sum(), 1)),
                "top_risk_quintile_lift": float(q_labels.mean() / base) if base > 0 else np.nan,
                "top_risk_quintile_mean_rel_return": float(q[target].mean()),
                "top_risk_quintile_beat_spy_rate": float((q[target] > 0).mean()),
                "top_risk_decile_precision": float(d_labels.mean()),
                "top_risk_decile_lift": float(d_labels.mean() / base) if base > 0 else np.nan,
                "top_risk_decile_mean_rel_return": float(d[target].mean()),
            }
        )
    return pd.DataFrame(rows)


def _block_bootstrap_mean_ci(values: pd.Series, *, block_months: int, repetitions: int, seed: int) -> tuple[float, float]:
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
    return tuple(map(float, np.quantile(out, [0.025, 0.975])))


def _ci(series: pd.Series, key: str, spec: dict) -> tuple[float, float]:
    seed = (int(spec["bootstrap"]["seed"]) + zlib.crc32(key.encode("utf-8"))) % (2**32 - 1)
    return _block_bootstrap_mean_ci(
        series,
        block_months=int(spec["bootstrap"]["block_months"]),
        repetitions=int(spec["bootstrap"]["repetitions"]),
        seed=seed,
    )


def _ranking_summary(monthly: pd.DataFrame, spec: dict) -> pd.DataFrame:
    metrics = [
        "ic",
        "q5_minus_q1_spread",
        "top_decile_mean_rel_return",
        "top_decile_beat_spy_rate",
        "top_quintile_mean_rel_return",
        "top_quintile_beat_spy_rate",
    ]
    rows = []
    for (h, method), g in monthly.groupby(["horizon_bars", "method"], observed=True, sort=True):
        row = {
            "horizon_bars": int(h),
            "horizon_label": spec["horizon_labels"][str(int(h))],
            "method": method,
            "months": int(g["date"].nunique()),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(g[metric].mean())
            lo, hi = _ci(g[metric], f"ranking|{h}|{method}|{metric}", spec)
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _avoid_summary(monthly: pd.DataFrame, spec: dict) -> pd.DataFrame:
    metrics = [
        "base_rate",
        "roc_auc",
        "average_precision",
        "brier_score",
        "top_risk_quintile_precision",
        "top_risk_quintile_lift",
        "top_risk_quintile_mean_rel_return",
        "top_risk_quintile_beat_spy_rate",
        "top_risk_decile_precision",
        "top_risk_decile_lift",
        "top_risk_decile_mean_rel_return",
    ]
    rows = []
    for h, g in monthly.groupby("horizon_bars", observed=True, sort=True):
        row = {
            "horizon_bars": int(h),
            "horizon_label": spec["horizon_labels"][str(int(h))],
            "months": int(g["date"].nunique()),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(g[metric].mean())
            lo, hi = _ci(g[metric], f"avoid|{h}|{metric}", spec)
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_ranking_improvement(monthly: pd.DataFrame, spec: dict) -> pd.DataFrame:
    rows = []
    for h, g in monthly.groupby("horizon_bars", observed=True):
        pivot = g.pivot(index="date", columns="method", values="ic")
        for baseline in ["equal_weight", "best_univariate"]:
            if "ridge" not in pivot or baseline not in pivot:
                continue
            diff = (pivot["ridge"] - pivot[baseline]).dropna()
            lo, hi = _ci(diff, f"ridge_improvement|{h}|{baseline}", spec)
            rows.append(
                {
                    "horizon_bars": int(h),
                    "horizon_label": spec["horizon_labels"][str(int(h))],
                    "baseline": baseline,
                    "months": int(len(diff)),
                    "mean_ic_difference": float(diff.mean()),
                    "ic_difference_ci_low": lo,
                    "ic_difference_ci_high": hi,
                    "ridge_significantly_better": bool(np.isfinite(lo) and lo > 0),
                }
            )
    return pd.DataFrame(rows)


def _evidence_gates(ranking_summary: pd.DataFrame, avoid_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ridge = ranking_summary[ranking_summary["method"].eq("ridge")].set_index("horizon_bars")
    avoid = avoid_summary.set_index("horizon_bars")
    horizons = sorted(set(ridge.index).union(avoid.index))
    for h in horizons:
        r = ridge.loc[h] if h in ridge.index else None
        a = avoid.loc[h] if h in avoid.index else None
        favor_pass = bool(
            r is not None
            and r["ic_ci_low"] > 0
            and r["top_decile_mean_rel_return_ci_low"] > 0
            and r["top_decile_beat_spy_rate_ci_low"] > 0.50
            and r["q5_minus_q1_spread_ci_low"] > 0
        )
        avoid_pass = bool(
            a is not None
            and a["roc_auc_ci_low"] > 0.50
            and a["top_risk_quintile_precision_ci_low"] > 0.20
            and a["top_risk_quintile_mean_rel_return_ci_high"] < 0
        )
        rows.append(
            {
                "horizon_bars": int(h),
                "horizon_label": (r["horizon_label"] if r is not None else a["horizon_label"]),
                "favor_gate": "FAVOR_EARNED_DEVELOPMENT_EVIDENCE" if favor_pass else "FAVOR_NOT_EARNED",
                "avoid_gate": "AVOID_EARNED_DEVELOPMENT_EVIDENCE" if avoid_pass else "AVOID_NOT_EARNED",
                "favor_pass": favor_pass,
                "avoid_pass": avoid_pass,
            }
        )
    return pd.DataFrame(rows)


def _yearly_metrics(ranking_monthly: pd.DataFrame, avoid_monthly: pd.DataFrame, spec: dict) -> pd.DataFrame:
    rows = []
    ridge = ranking_monthly[ranking_monthly["method"].eq("ridge")].copy()
    if not ridge.empty:
        ridge["year"] = ridge["date"].dt.year
    av = avoid_monthly.copy()
    if not av.empty:
        av["year"] = av["date"].dt.year
    for h in map(int, spec["horizons_bars"]):
        years = sorted(set(ridge.loc[ridge["horizon_bars"].eq(h), "year"].tolist()) | set(av.loc[av["horizon_bars"].eq(h), "year"].tolist()))
        for year in years:
            rg = ridge[(ridge["horizon_bars"] == h) & (ridge["year"] == year)]
            ag = av[(av["horizon_bars"] == h) & (av["year"] == year)]
            rows.append(
                {
                    "horizon_bars": h,
                    "horizon_label": spec["horizon_labels"][str(h)],
                    "year": int(year),
                    "ranking_months": int(len(rg)),
                    "ridge_mean_ic": float(rg["ic"].mean()) if len(rg) else np.nan,
                    "ridge_q5_minus_q1": float(rg["q5_minus_q1_spread"].mean()) if len(rg) else np.nan,
                    "ridge_top_decile_rel_return": float(rg["top_decile_mean_rel_return"].mean()) if len(rg) else np.nan,
                    "avoid_months": int(len(ag)),
                    "avoid_mean_auc": float(ag["roc_auc"].mean()) if len(ag) else np.nan,
                    "avoid_top_risk_q_precision": float(ag["top_risk_quintile_precision"].mean()) if len(ag) else np.nan,
                    "avoid_top_risk_q_rel_return": float(ag["top_risk_quintile_mean_rel_return"].mean()) if len(ag) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _make_charts(output_dir: Path, ranking_summary: pd.DataFrame, avoid_summary: pd.DataFrame) -> list[str]:
    charts = output_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    if not ranking_summary.empty:
        p = ranking_summary.pivot(index="horizon_label", columns="method", values="mean_ic")
        ax = p.plot(kind="bar", figsize=(9, 5))
        ax.set_title("R3 Walk-Forward Mean Monthly Ranking IC")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Mean monthly Spearman IC")
        ax.axhline(0, linewidth=1)
        plt.tight_layout()
        path = charts / "ranking_mean_ic_by_horizon.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(str(path.name))

        ridge = ranking_summary[ranking_summary["method"].eq("ridge")]
        if not ridge.empty:
            ax = ridge.plot(x="horizon_label", y="mean_top_decile_mean_rel_return", kind="bar", legend=False, figsize=(8, 5))
            ax.set_title("R3 Ridge Top-Decile Mean SPY-Relative Return")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Mean relative return")
            ax.axhline(0, linewidth=1)
            plt.tight_layout()
            path = charts / "ridge_top_decile_relative_return.png"
            plt.savefig(path, dpi=160)
            plt.close()
            made.append(str(path.name))

    if not avoid_summary.empty:
        ax = avoid_summary.plot(x="horizon_label", y="mean_top_risk_quintile_lift", kind="bar", legend=False, figsize=(8, 5))
        ax.set_title("R3 AVOID Classifier: Top-Risk Quintile Precision Lift")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Precision / base rate")
        ax.axhline(1, linewidth=1)
        plt.tight_layout()
        path = charts / "avoid_precision_lift_by_horizon.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(str(path.name))

        ax = avoid_summary.plot(x="horizon_label", y="mean_top_risk_quintile_mean_rel_return", kind="bar", legend=False, figsize=(8, 5))
        ax.set_title("R3 AVOID Classifier: Top-Risk Quintile Future Relative Return")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Mean SPY-relative return")
        ax.axhline(0, linewidth=1)
        plt.tight_layout()
        path = charts / "avoid_top_risk_relative_return.png"
        plt.savefig(path, dpi=160)
        plt.close()
        made.append(str(path.name))

    return made


def _write_report(
    path: Path,
    candidates: dict[int, list[str]],
    ranking: pd.DataFrame,
    avoid: pd.DataFrame,
    gates: pd.DataFrame,
    improvement: pd.DataFrame,
    spec: dict,
) -> None:
    lines = [
        "# R3 — Walk-Forward Multivariate Baseline",
        "",
        "**Development evidence only. R3 is not a pristine holdout because the feature set was discovered using the same historical era in R1-R2.1.**",
        "",
        "## Frozen candidate inputs",
        "",
    ]
    for h, feats in candidates.items():
        lines.append(f"- {spec['horizon_labels'][str(h)]} / {h} bars: " + ", ".join(f"`{f}`" for f in feats))

    lines.extend(["", "## Evidence gates", ""])
    for row in gates.itertuples(index=False):
        lines.append(f"- **{row.horizon_label}**: {row.favor_gate}; {row.avoid_gate}.")

    lines.extend(["", "## Ridge ranking summary", ""])
    ridge = ranking[ranking["method"].eq("ridge")]
    for row in ridge.itertuples(index=False):
        lines.append(
            f"- **{row.horizon_label}**: mean IC {row.mean_ic:+.4f} "
            f"(95% CI {row.ic_ci_low:+.4f} to {row.ic_ci_high:+.4f}); "
            f"top-decile relative return {row.mean_top_decile_mean_rel_return:+.2%}; "
            f"top-decile beat-SPY rate {row.mean_top_decile_beat_spy_rate:.1%}."
        )

    lines.extend(["", "## AVOID classifier summary", ""])
    for row in avoid.itertuples(index=False):
        lines.append(
            f"- **{row.horizon_label}**: mean monthly AUC {row.mean_roc_auc:.3f}; "
            f"top-risk quintile precision {row.mean_top_risk_quintile_precision:.1%}; "
            f"lift {row.mean_top_risk_quintile_lift:.2f}x; "
            f"future relative return {row.mean_top_risk_quintile_mean_rel_return:+.2%}."
        )

    if not improvement.empty:
        lines.extend(["", "## Does Ridge beat simpler ranking baselines?", ""])
        for row in improvement.itertuples(index=False):
            verdict = "YES" if row.ridge_significantly_better else "NO"
            lines.append(
                f"- **{row.horizon_label} vs {row.baseline}**: mean IC difference "
                f"{row.mean_ic_difference:+.4f}; CI {row.ic_difference_ci_low:+.4f} to "
                f"{row.ic_difference_ci_high:+.4f}; significant improvement: {verdict}."
            )

    lines.extend(
        [
            "",
            "## Constraints",
            "",
            "- No model hyperparameter search was performed.",
            "- A model can fail the FAVOR gate even when it has positive ranking IC; that is intentional.",
            "- The AVOID classifier predicts future bottom-quintile SPY-relative performers, not a symmetric BUY class.",
            "- A later untouched/prospective validation phase is still required before any production change.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r3(
    *,
    dataset_path: Path,
    r2_1_candidates_path: Path,
    r2_1_summary_path: Path,
    output_dir: Path,
    spec_path: Path | None = None,
) -> dict:
    spec = _load_spec(spec_path)
    r21_summary = _read_json(r2_1_summary_path)
    if r21_summary.get("status") != "ok":
        raise RuntimeError(f"R3 requires successful R2.1 output; got {r21_summary.get('status')!r}")

    candidate_set = pd.read_csv(r2_1_candidates_path)
    candidates = _candidate_features_by_horizon(candidate_set, spec)
    all_features = sorted(set(sum(candidates.values(), [])))

    needed = {"date", "ticker"} | set(all_features)
    for h in map(int, spec["horizons_bars"]):
        needed.update(
            {
                f"fwd_spy_relative_return_{h}",
                f"outcome_end_date_{h}",
            }
        )
    header = pd.read_csv(dataset_path, nrows=0).columns
    missing = sorted(set(needed) - set(header))
    if missing:
        raise RuntimeError("R3 dataset missing required columns: " + ", ".join(missing))

    raw = pd.read_csv(dataset_path, usecols=sorted(needed), low_memory=False)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["ticker"] = raw["ticker"].astype(str).str.upper().str.strip()
    for h in map(int, spec["horizons_bars"]):
        raw[f"outcome_end_date_{h}"] = pd.to_datetime(raw[f"outcome_end_date_{h}"], errors="coerce")

    anchor_dates = _month_end_anchor_dates(raw["date"])
    anchors_base = raw[raw["date"].isin(anchor_dates)].copy()
    anchors_base = _cross_sectional_rank_transform(anchors_base, all_features)
    del raw

    predictions_all: list[pd.DataFrame] = []
    coefficient_rows: list[dict] = []
    fold_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for h in map(int, spec["horizons_bars"]):
        features = candidates[h]
        target = f"fwd_spy_relative_return_{h}"
        end_col = f"outcome_end_date_{h}"
        anchors = anchors_base.copy()
        anchors["avoid_label"] = _make_avoid_label(
            anchors, target, float(spec["avoid_target"]["bottom_fraction"])
        )

        coverage_rows.append(
            {
                "horizon_bars": h,
                "horizon_label": spec["horizon_labels"][str(h)],
                "candidate_features": "|".join(features),
                "anchor_months": int(anchors["date"].nunique()),
                "anchor_rows": int(len(anchors)),
                "complete_feature_rows": int(anchors[features].notna().all(axis=1).sum()),
                "matured_outcome_rows": int(anchors[target].notna().sum()),
            }
        )

        for year in map(int, spec["test_years"]):
            cutoff = pd.Timestamp(year=year, month=1, day=1)
            train = _eligible_training_rows(
                anchors,
                features=features,
                target=target,
                outcome_end_col=end_col,
                cutoff=cutoff,
            )
            test = _test_rows(
                anchors,
                year=year,
                features=features,
                target=target,
                outcome_end_col=end_col,
            )
            if test.empty:
                continue
            if train["date"].nunique() < int(spec["minimum_training_months"]):
                continue
            if len(train) < int(spec["minimum_training_rows"]):
                continue
            if train["avoid_label"].nunique() < 2:
                continue

            pred, coefs, fold_info = _fit_fold(
                train,
                test,
                features=features,
                target=target,
                horizon=h,
                year=year,
                spec=spec,
            )
            pred = pred.rename(columns={target: "realized_spy_relative_return"})
            predictions_all.append(pred)
            coefficient_rows.extend(coefs)
            fold_rows.append(fold_info)

    if not predictions_all:
        raise RuntimeError("R3 generated no walk-forward predictions.")

    predictions = pd.concat(predictions_all, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    folds = pd.DataFrame(fold_rows)
    coverage = pd.DataFrame(coverage_rows)

    ranking_monthly = _monthly_ranking_metrics(
        predictions,
        "realized_spy_relative_return",
        int(spec["minimum_test_cross_section_rows"]),
    )
    avoid_monthly = _monthly_avoid_metrics(
        predictions,
        "realized_spy_relative_return",
        int(spec["minimum_test_cross_section_rows"]),
    )
    ranking_summary = _ranking_summary(ranking_monthly, spec)
    avoid_summary = _avoid_summary(avoid_monthly, spec)
    improvement = _paired_ranking_improvement(ranking_monthly, spec)
    gates = _evidence_gates(ranking_summary, avoid_summary)
    yearly = _yearly_metrics(ranking_monthly, avoid_monthly, spec)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "r3_predictions.csv.gz", index=False, compression="gzip")
    coefficients.to_csv(output_dir / "r3_coefficients.csv", index=False)
    folds.to_csv(output_dir / "r3_fold_manifest.csv", index=False)
    coverage.to_csv(output_dir / "r3_data_coverage.csv", index=False)
    ranking_monthly.to_csv(output_dir / "r3_monthly_ranking_metrics.csv", index=False)
    avoid_monthly.to_csv(output_dir / "r3_monthly_avoid_metrics.csv", index=False)
    ranking_summary.to_csv(output_dir / "r3_ranking_summary.csv", index=False)
    avoid_summary.to_csv(output_dir / "r3_avoid_summary.csv", index=False)
    improvement.to_csv(output_dir / "r3_ranking_improvement.csv", index=False)
    gates.to_csv(output_dir / "r3_evidence_gates.csv", index=False)
    yearly.to_csv(output_dir / "r3_yearly_metrics.csv", index=False)
    candidate_set[candidate_set["status"].eq("SELECTED")].to_csv(
        output_dir / "r3_candidate_inputs.csv", index=False
    )
    (output_dir / "r3_spec_frozen.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    charts = _make_charts(output_dir, ranking_summary, avoid_summary)
    summary = {
        "research_patch": "R3",
        "status": "ok",
        "development_evidence_only": True,
        "pristine_holdout": False,
        "combined_models_fitted": ["Ridge", "LogisticRegression"],
        "hyperparameter_search_performed": False,
        "candidate_features_by_horizon": {str(k): v for k, v in candidates.items()},
        "walk_forward_test_years_requested": spec["test_years"],
        "walk_forward_folds_run": int(len(folds)),
        "prediction_rows": int(len(predictions)),
        "matured_prediction_rows": int(predictions["outcome_matured"].sum()),
        "ranking_evaluation_months_by_horizon": {
            str(int(h)): int(g["date"].nunique())
            for h, g in ranking_monthly[ranking_monthly["method"].eq("ridge")].groupby("horizon_bars")
        },
        "avoid_evaluation_months_by_horizon": {
            str(int(h)): int(g["date"].nunique())
            for h, g in avoid_monthly.groupby("horizon_bars")
        },
        "evidence_gates": gates.to_dict(orient="records"),
        "charts": charts,
        "important_note": (
            "R3 walk-forward predictions are point-in-time with respect to model training, "
            "but R1-R2.1 used the same historical era for feature discovery. These are development "
            "results, not untouched holdout validation."
        ),
    }
    (output_dir / "r3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(
        output_dir / "r3_report.md",
        candidates,
        ranking_summary,
        avoid_summary,
        gates,
        improvement,
        spec,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R3 walk-forward multivariate baseline.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--r2-1-candidates", required=True, type=Path)
    parser.add_argument("--r2-1-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spec", type=Path, default=None)
    args = parser.parse_args()

    result = run_r3(
        dataset_path=args.dataset,
        r2_1_candidates_path=args.r2_1_candidates,
        r2_1_summary_path=args.r2_1_summary,
        output_dir=args.output,
        spec_path=args.spec,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
