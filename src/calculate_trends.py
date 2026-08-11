from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import as_bool, safe_int


def _consecutive_true(values: pd.Series) -> pd.Series:
    out = []
    streak = 0
    for value in values.fillna(False).astype(bool):
        streak = streak + 1 if value else 0
        out.append(streak)
    return pd.Series(out, index=values.index, dtype="int64")


def add_rotation_trends(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()

    for lag in [5, 10, 20, 63]:
        out[f"score_change_{lag}"] = (
            out["rotation_score"]
            - out.groupby("ticker")["rotation_score"].shift(lag)
        )

    for lag in [5, 20]:
        out[f"rank_change_{lag}"] = (
            out.groupby("ticker")["group_rank"].shift(lag)
            - out["group_rank"]
        )

    out["rs20_change_5"] = (
        out["rs20"] - out.groupby("ticker")["rs20"].shift(5)
    )
    out["rs63_change_20"] = (
        out["rs63"] - out.groupby("ticker")["rs63"].shift(20)
    )

    pieces = []
    for _, g in out.groupby("ticker", sort=False):
        g = g.copy()
        top_quartile = g["group_percentile"] >= 75
        improving = g["rotation_score"].diff() > 0

        g["days_top_quartile_20"] = (
            top_quartile.astype(int)
            .rolling(20, min_periods=1)
            .sum()
            .astype(int)
        )
        g["top_quartile_streak"] = _consecutive_true(top_quartile)
        g["consecutive_improving_days"] = _consecutive_true(improving)
        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)

    def classify(r) -> str:
        if not as_bool(r.get("rank_eligible")):
            return "REFERENCE"
        if r.get("data_status") != "READY" or pd.isna(r.get("rotation_score")):
            return "NOT_SCORED"

        score = r["rotation_score"]
        rs20 = r["rs20"]
        rs63 = r["rs63"]
        ch5 = r["score_change_5"]
        ch20 = r["score_change_20"]
        streak = safe_int(r.get("top_quartile_streak"), 0)
        persistence = safe_int(r.get("persistence_bars"), 5)

        if (
            score >= 75
            and rs20 > 0
            and rs63 > 0
            and streak >= persistence
        ):
            return "ROTATION_IN"

        if score >= 60 and pd.notna(ch5) and ch5 > 0:
            return "ACCUMULATING"

        if (
            score <= 25
            and rs20 < 0
            and rs63 < 0
            and (pd.isna(ch5) or ch5 <= 0)
        ):
            return "ROTATION_OUT"

        if pd.notna(ch5) and pd.notna(ch20) and ch5 < 0 and ch20 < 0:
            return "WEAKENING"

        return "NEUTRAL"

    out["rotation_state"] = out.apply(classify, axis=1)
    return out
