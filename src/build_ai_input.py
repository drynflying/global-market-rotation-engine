from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import AI_INPUT_PATH


def _clean_number(value, digits=4):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _clean_text(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _row_payload(r: pd.Series) -> dict:
    return {
        "ticker": _clean_text(r.get("ticker")),
        "score_mode": _clean_text(r.get("score_mode")),
        "rotation_group": _clean_text(r.get("rotation_group")),
        "exposure": _clean_text(r.get("exposure")),
        "sector": _clean_text(r.get("sector")),
        "state": _clean_text(r.get("rotation_state")),
        "score": _clean_number(r.get("rotation_score"), 2),
        "rank": _clean_number(r.get("group_rank"), 0),
        "group_size": _clean_number(r.get("group_size"), 0),
        "score_change_5": _clean_number(r.get("score_change_5"), 2),
        "score_change_20": _clean_number(r.get("score_change_20"), 2),
        "rank_change_5": _clean_number(r.get("rank_change_5"), 0),
        "rank_change_20": _clean_number(r.get("rank_change_20"), 0),
        "signal_rs20_pct_points": (
            _clean_number(100 * r.get("signal_rs20"), 2)
            if pd.notna(r.get("signal_rs20")) else None
        ),
        "signal_rs63_pct_points": (
            _clean_number(100 * r.get("signal_rs63"), 2)
            if pd.notna(r.get("signal_rs63")) else None
        ),
        "relative_dollar_volume": _clean_number(r.get("relative_dollar_volume"), 2),
        "cmf20": _clean_number(r.get("cmf20"), 3),
        "trend_score": _clean_number(r.get("trend_score"), 2),
        "leader_zone_streak": int(r.get("leader_zone_streak", 0) or 0),
        "days_leader_zone_20": int(r.get("days_leader_zone_20", 0) or 0),
        "primary_benchmark": _clean_text(r.get("primary_benchmark")),
        "parent_benchmark": _clean_text(r.get("parent_benchmark")),
        "parent_rs20_pct_points": (
            _clean_number(100 * r.get("parent_rs20"), 2)
            if pd.notna(r.get("parent_rs20")) else None
        ),
        "paired_ticker": _clean_text(r.get("paired_ticker")),
        "pair_type": _clean_text(r.get("pair_type")),
        "pair_signal": _clean_text(r.get("pair_signal")),
        "pair_spread_20_pct_points": (
            _clean_number(100 * r.get("pair_spread_20"), 2)
            if pd.notna(r.get("pair_spread_20")) else None
        ),
        "pair_spread_63_pct_points": (
            _clean_number(100 * r.get("pair_spread_63"), 2)
            if pd.notna(r.get("pair_spread_63")) else None
        ),
    }


def _evidence_payload(r: pd.Series) -> dict:
    """Compact row for global rankings/attention context."""
    return {
        "ticker": _clean_text(r.get("ticker")),
        "exposure": _clean_text(r.get("exposure")),
        "sector": _clean_text(r.get("sector")),
        "state": _clean_text(r.get("rotation_state")),
        "score": _clean_number(r.get("rotation_score"), 2),
        "rank": _clean_number(r.get("group_rank"), 0),
        "group_size": _clean_number(r.get("group_size"), 0),
        "score_change_5": _clean_number(r.get("score_change_5"), 2),
        "score_change_20": _clean_number(r.get("score_change_20"), 2),
        "signal_rs20_pct_points": (
            _clean_number(100 * r.get("signal_rs20"), 2)
            if pd.notna(r.get("signal_rs20")) else None
        ),
        "signal_rs63_pct_points": (
            _clean_number(100 * r.get("signal_rs63"), 2)
            if pd.notna(r.get("signal_rs63")) else None
        ),
        "cmf20": _clean_number(r.get("cmf20"), 3),
    }


def _pair_attention_payload(r: pd.Series) -> dict:
    payload = _evidence_payload(r)
    payload.update(
        {
            "score_mode": _clean_text(r.get("score_mode")),
            "paired_ticker": _clean_text(r.get("paired_ticker")),
            "pair_type": _clean_text(r.get("pair_type")),
            "pair_signal": _clean_text(r.get("pair_signal")),
            "pair_spread_20_pct_points": (
                _clean_number(100 * r.get("pair_spread_20"), 2)
                if pd.notna(r.get("pair_spread_20")) else None
            ),
            "pair_spread_63_pct_points": (
                _clean_number(100 * r.get("pair_spread_63"), 2)
                if pd.notna(r.get("pair_spread_63")) else None
            ),
        }
    )
    return payload


def _build_deterministic_attention(
    scored: pd.DataFrame,
    cross: pd.DataFrame,
    pairs: pd.DataFrame,
    history: pd.DataFrame,
) -> dict:
    """
    Build deterministic context that the AI must treat as ground truth.

    This mirrors the dashboard's attention concepts but is generated before
    the AI call so material conflicts and 63-bar score extrema cannot be
    omitted simply because the model did not notice them in a large payload.
    """
    lowest_scores = [
        _evidence_payload(r)
        for _, r in cross.nsmallest(5, "rotation_score").iterrows()
    ]

    extreme_cmf = cross[cross["cmf20"].notna()].copy()
    extreme_cmf = extreme_cmf[extreme_cmf["cmf20"].abs() >= 0.40].copy()
    if not extreme_cmf.empty:
        extreme_cmf["_abs_cmf"] = extreme_cmf["cmf20"].abs()
        extreme_cmf = extreme_cmf.nlargest(5, "_abs_cmf")
    extreme_cmf_rows = [_evidence_payload(r) for _, r in extreme_cmf.iterrows()]

    score_extremes_63 = []
    for ticker, g in history[history["rotation_score"].notna()].groupby("ticker"):
        g = g.sort_values("date").tail(63)
        if len(g) < 63:
            continue

        current_score = float(g.iloc[-1]["rotation_score"])
        low_score = float(g["rotation_score"].min())
        high_score = float(g["rotation_score"].max())
        if high_score - low_score < 1.0:
            continue

        kind = None
        if np.isclose(current_score, high_score, atol=0.01):
            kind = "HIGH"
        elif np.isclose(current_score, low_score, atol=0.01):
            kind = "LOW"

        if kind:
            score_extremes_63.append(
                {
                    "ticker": str(ticker),
                    "kind": kind,
                    "latest_score": _clean_number(current_score, 2),
                    "low_score": _clean_number(low_score, 2),
                    "high_score": _clean_number(high_score, 2),
                }
            )

    highs_63 = sorted(
        (x for x in score_extremes_63 if x["kind"] == "HIGH"),
        key=lambda x: x["latest_score"],
        reverse=True,
    )[:8]
    lows_63 = sorted(
        (x for x in score_extremes_63 if x["kind"] == "LOW"),
        key=lambda x: x["latest_score"],
    )[:8]

    sector_divergences = []
    sector_rows = cross[
        cross["sector"].notna() & cross["score_change_20"].notna()
    ].copy()
    for sector, g in sector_rows.groupby("sector"):
        sector_name = str(sector).strip()
        if not sector_name or len(g) < 2:
            continue

        improver = g.loc[g["score_change_20"].idxmax()]
        deteriorator = g.loc[g["score_change_20"].idxmin()]
        improvement = float(improver["score_change_20"])
        deterioration = float(deteriorator["score_change_20"])
        divergence = improvement - deterioration

        if improvement < 15 or deterioration > -15 or divergence < 40:
            continue

        other_weak = g[
            g["ticker"].ne(deteriorator["ticker"])
            & g["rotation_state"].isin(["ROTATION_OUT", "WEAKENING"])
        ].sort_values("rotation_score").head(1)

        other_weak_payload = None
        if not other_weak.empty:
            w = other_weak.iloc[0]
            other_weak_payload = {
                "ticker": str(w["ticker"]),
                "state": str(w["rotation_state"]),
                "score": _clean_number(w["rotation_score"], 2),
            }

        sector_divergences.append(
            {
                "sector": sector_name,
                "improver": str(improver["ticker"]),
                "improver_score_change_20": _clean_number(improvement, 2),
                "deteriorator": str(deteriorator["ticker"]),
                "deteriorator_score_change_20": _clean_number(deterioration, 2),
                "divergence_points": _clean_number(divergence, 2),
                "other_weak": other_weak_payload,
            }
        )

    sector_divergences.sort(
        key=lambda x: x["divergence_points"], reverse=True
    )
    sector_divergences = sector_divergences[:6]

    strong_states = {
        "EMERGING", "ACCELERATING", "PERSISTENT_LEADER", "REACCELERATING"
    }
    weak_states = {"WEAKENING", "ROTATION_OUT"}
    pair_state_tensions = []
    for _, r in pairs.sort_values("ticker").iterrows():
        pair_signal = str(r.get("pair_signal") or "")
        rotation_state = str(r.get("rotation_state") or "")
        conflict = (
            pair_signal == "PAIR_LEADING" and rotation_state in weak_states
        ) or (
            pair_signal == "PAIR_LAGGING" and rotation_state in strong_states
        )
        if conflict:
            pair_state_tensions.append(_pair_attention_payload(r))

    mixed_horizon_rs = []
    mixed = scored[
        scored["signal_rs20"].notna() & scored["signal_rs63"].notna()
    ].copy()
    mixed = mixed[
        ((mixed["signal_rs20"] > 0) & (mixed["signal_rs63"] < 0))
        | ((mixed["signal_rs20"] < 0) & (mixed["signal_rs63"] > 0))
    ].copy()
    if not mixed.empty:
        mixed["_horizon_gap"] = (
            mixed["signal_rs20"] - mixed["signal_rs63"]
        ).abs()
        mixed = mixed.sort_values("_horizon_gap", ascending=False)
        mixed_horizon_rs = [
            _evidence_payload(r) for _, r in mixed.iterrows()
        ]

    return {
        "lowest_current_scores": lowest_scores,
        "extreme_cmf20": extreme_cmf_rows,
        "score_63_bar_highs": highs_63,
        "score_63_bar_lows": lows_63,
        "sector_divergences": sector_divergences,
        "pair_state_tensions": pair_state_tensions,
        "mixed_horizon_relative_strength": mixed_horizon_rs,
    }


def _ordered_add(target: list[str], seen: set[str], values) -> None:
    for value in values:
        ticker = str(value or "").strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            target.append(ticker)


def build_ai_input(
    latest: pd.DataFrame,
    history: pd.DataFrame,
    output_path: Path = AI_INPUT_PATH,
) -> dict:
    latest = latest.copy()
    latest["date"] = pd.to_datetime(latest["date"])
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])

    as_of = latest["date"].max().strftime("%Y-%m-%d")
    scored = latest[
        latest["rotation_score"].notna()
        & latest["rank_eligible"].astype(bool)
    ].copy()

    cross = scored[scored["score_mode"].eq("CROSS_SECTIONAL")].copy()
    pairs = scored[scored["score_mode"].eq("PAIR")].copy()

    top_scores = cross.nlargest(15, "rotation_score")
    top_improvers = cross[cross["score_change_20"].notna()].nlargest(
        15, "score_change_20"
    )
    top_deteriorators = cross[cross["score_change_20"].notna()].nsmallest(
        10, "score_change_20"
    )
    lowest_scores = cross.nsmallest(10, "rotation_score")

    global_rankings = {
        "highest_current_scores": [
            _evidence_payload(r) for _, r in top_scores.iterrows()
        ],
        "biggest_20_bar_improvements": [
            _evidence_payload(r) for _, r in top_improvers.iterrows()
        ],
        "biggest_20_bar_deteriorations": [
            _evidence_payload(r) for _, r in top_deteriorators.iterrows()
        ],
        "lowest_current_scores": [
            _evidence_payload(r) for _, r in lowest_scores.iterrows()
        ],
    }

    deterministic_attention = _build_deterministic_attention(
        scored, cross, pairs, history
    )

    group_summaries = []
    group_focus_candidates: list[str] = []

    for group, g in cross.groupby("rotation_group"):
        leaders = g.nlargest(5, "rotation_score")
        improvers = g[g["score_change_20"].notna()].nlargest(5, "score_change_20")
        weakeners = g[g["score_change_20"].notna()].nsmallest(5, "score_change_20")

        for frame in [leaders, improvers, weakeners]:
            group_focus_candidates.extend(frame["ticker"].tolist())

        group_summaries.append(
            {
                "rotation_group": group,
                "member_count": int(len(g)),
                "leaders": [_row_payload(r) for _, r in leaders.iterrows()],
                "biggest_20d_improvers": [_row_payload(r) for _, r in improvers.iterrows()],
                "biggest_20d_weakeners": [_row_payload(r) for _, r in weakeners.iterrows()],
            }
        )

    pair_signals = [_row_payload(r) for _, r in pairs.sort_values("ticker").iterrows()]

    # Build focus context in deterministic priority order. The previous
    # alphabetic truncation could drop the most important exceptions.
    focus_tickers: list[str] = []
    focus_seen: set[str] = set()

    attention_tickers = []
    for item in deterministic_attention["lowest_current_scores"]:
        attention_tickers.append(item.get("ticker"))
    for item in deterministic_attention["extreme_cmf20"]:
        attention_tickers.append(item.get("ticker"))
    for item in deterministic_attention["score_63_bar_highs"]:
        attention_tickers.append(item.get("ticker"))
    for item in deterministic_attention["score_63_bar_lows"]:
        attention_tickers.append(item.get("ticker"))
    for item in deterministic_attention["sector_divergences"]:
        attention_tickers.extend([item.get("improver"), item.get("deteriorator")])
        if item.get("other_weak"):
            attention_tickers.append(item["other_weak"].get("ticker"))
    for item in deterministic_attention["pair_state_tensions"]:
        attention_tickers.extend([item.get("ticker"), item.get("paired_ticker")])

    _ordered_add(focus_tickers, focus_seen, attention_tickers)
    _ordered_add(focus_tickers, focus_seen, top_scores["ticker"].tolist())
    _ordered_add(focus_tickers, focus_seen, top_improvers["ticker"].tolist())
    _ordered_add(focus_tickers, focus_seen, top_deteriorators["ticker"].tolist())
    _ordered_add(focus_tickers, focus_seen, pairs["ticker"].tolist())
    _ordered_add(focus_tickers, focus_seen, group_focus_candidates)
    focus_tickers = focus_tickers[:90]

    snapshots = {}
    for ticker in focus_tickers:
        g = history[
            (history["ticker"] == ticker)
            & history["rotation_score"].notna()
        ].sort_values("date")

        if g.empty:
            continue

        current = g.iloc[-1]
        points = {}
        for lag in [0, 5, 10, 20, 63]:
            if len(g) > lag:
                row = g.iloc[-1 - lag]
                points[str(lag)] = {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "score": _clean_number(row.get("rotation_score"), 2),
                    "rank": _clean_number(row.get("group_rank"), 0),
                    "signal_rs20_pct_points": (
                        _clean_number(100 * row.get("signal_rs20"), 2)
                        if pd.notna(row.get("signal_rs20")) else None
                    ),
                    "signal_rs63_pct_points": (
                        _clean_number(100 * row.get("signal_rs63"), 2)
                        if pd.notna(row.get("signal_rs63")) else None
                    ),
                }

        snapshots[ticker] = {
            "current": _row_payload(current),
            "snapshots_bars_ago": points,
        }

    references = latest[latest["signal_role"] == "BENCHMARK"].copy()
    benchmark_context = []
    for _, r in references.iterrows():
        benchmark_context.append(
            {
                "ticker": _clean_text(r.get("ticker")),
                "exposure": _clean_text(r.get("exposure")),
                "return_20_pct": (
                    _clean_number(100 * r.get("return_20"), 2)
                    if pd.notna(r.get("return_20")) else None
                ),
                "return_63_pct": (
                    _clean_number(100 * r.get("return_63"), 2)
                    if pd.notna(r.get("return_63")) else None
                ),
                "trend_score": _clean_number(r.get("trend_score"), 2),
            }
        )

    payload = {
        "as_of": as_of,
        "purpose": (
            "Longer-horizon capital-rotation monitoring. "
            "This is not a daily trading signal."
        ),
        "score_formula_version": (
            str(latest["score_formula_version"].dropna().iloc[0])
            if "score_formula_version" in latest.columns
            and not latest["score_formula_version"].dropna().empty
            else "unknown"
        ),
        "state_definitions": {
            "EMERGING": "Strong recent improvement before full long-horizon confirmation.",
            "ACCELERATING": "Positive relative strength on both horizons with rising score.",
            "PERSISTENT_LEADER": "Sustained high relative leadership with persistence.",
            "REACCELERATING": "Longer-term strength remains and recent deterioration is turning upward.",
            "WEAKENING": "Leadership or relative strength is deteriorating.",
            "ROTATION_OUT": "Low score with negative relative strength on both horizons.",
            "NEUTRAL": "Mixed evidence or no clear directional rotation.",
        },
        "interpretation_rules": [
            "Focus on direction and persistence, not only today's score.",
            "A high but falling score is different from a high and rising score.",
            "Prefer rotations confirmed by the parent sector or geographic pair.",
            "PAIR scores are direct relationship signals and should not be ranked against cross-sectional groups.",
            "If 20-bar and 63-bar signal relative strength have opposite signs, disclose both horizons and describe the evidence as mixed rather than citing only the favorable horizon.",
            "Use deterministic_attention sector divergences and pair-state tensions as mandatory conflicts to acknowledge in risks_or_conflicts.",
            "63-bar score extrema describe the rotation SCORE history, not a price high or low.",
            "For broad sector or regional theses, prefer the strongest supplied quantitative examples first; broad benchmarks can be used as confirmation rather than replacing stronger leaders.",
            "Do not make price targets or claim that ETF volume proves institutional net flows.",
        ],
        "benchmark_context": benchmark_context,
        "global_rankings": global_rankings,
        "deterministic_attention": deterministic_attention,
        "group_summaries": group_summaries,
        "pair_signals": pair_signals,
        "focus_securities": snapshots,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload
