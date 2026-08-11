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


def _row_payload(r: pd.Series) -> dict:
    return {
        "ticker": r["ticker"],
        "rotation_group": r["rotation_group"],
        "exposure": r.get("exposure"),
        "state": r.get("rotation_state"),
        "score": _clean_number(r.get("rotation_score"), 2),
        "rank": _clean_number(r.get("group_rank"), 0),
        "group_size": _clean_number(r.get("group_size"), 0),
        "score_change_5": _clean_number(r.get("score_change_5"), 2),
        "score_change_20": _clean_number(r.get("score_change_20"), 2),
        "rank_change_5": _clean_number(r.get("rank_change_5"), 0),
        "rank_change_20": _clean_number(r.get("rank_change_20"), 0),
        "rs20_pct_points": _clean_number(100 * r.get("rs20"), 2)
        if pd.notna(r.get("rs20")) else None,
        "rs63_pct_points": _clean_number(100 * r.get("rs63"), 2)
        if pd.notna(r.get("rs63")) else None,
        "relative_dollar_volume": _clean_number(r.get("relative_dollar_volume"), 2),
        "cmf20": _clean_number(r.get("cmf20"), 3),
        "trend_score": _clean_number(r.get("trend_score"), 2),
        "top_quartile_streak": int(r.get("top_quartile_streak", 0) or 0),
        "days_top_quartile_20": int(r.get("days_top_quartile_20", 0) or 0),
        "primary_benchmark": r.get("primary_benchmark") or None,
        "parent_benchmark": r.get("parent_benchmark") or None,
        "parent_rs20_pct_points": _clean_number(100 * r.get("parent_rs20"), 2)
        if pd.notna(r.get("parent_rs20")) else None,
        "paired_ticker": r.get("paired_ticker") or None,
        "pair_spread_20_pct_points": _clean_number(100 * r.get("pair_spread_20"), 2)
        if pd.notna(r.get("pair_spread_20")) else None,
    }


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

    group_summaries = []
    focus_tickers: set[str] = set()

    for group, g in scored.groupby("rotation_group"):
        leaders = g.nlargest(5, "rotation_score")
        improvers = g[g["score_change_20"].notna()].nlargest(5, "score_change_20")
        weakeners = g[g["score_change_20"].notna()].nsmallest(5, "score_change_20")

        for frame in [leaders, improvers, weakeners]:
            focus_tickers.update(frame["ticker"].tolist())

        group_summaries.append(
            {
                "rotation_group": group,
                "member_count": int(len(g)),
                "leaders": [_row_payload(r) for _, r in leaders.iterrows()],
                "biggest_20d_improvers": [_row_payload(r) for _, r in improvers.iterrows()],
                "biggest_20d_weakeners": [_row_payload(r) for _, r in weakeners.iterrows()],
            }
        )

    # Limit AI context while preserving the most decision-useful securities.
    focus_tickers = set(list(sorted(focus_tickers))[:80])

    snapshots = {}
    for ticker in sorted(focus_tickers):
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
                    "rs20_pct_points": _clean_number(100 * row.get("rs20"), 2)
                    if pd.notna(row.get("rs20")) else None,
                    "rs63_pct_points": _clean_number(100 * row.get("rs63"), 2)
                    if pd.notna(row.get("rs63")) else None,
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
                "ticker": r["ticker"],
                "exposure": r.get("exposure"),
                "return_20_pct": _clean_number(100 * r.get("return_20"), 2)
                if pd.notna(r.get("return_20")) else None,
                "return_63_pct": _clean_number(100 * r.get("return_63"), 2)
                if pd.notna(r.get("return_63")) else None,
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
        "interpretation_rules": [
            "Focus on direction and persistence, not only today's score.",
            "A high but falling score is different from a high and rising score.",
            "Prefer rotations confirmed by the parent sector or geographic pair.",
            "Distinguish emerging rotation, persistent leadership, weakening, and rotation out.",
            "Do not make price targets or claim that ETF volume proves institutional net flows.",
        ],
        "benchmark_context": benchmark_context,
        "group_summaries": group_summaries,
        "focus_securities": snapshots,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
