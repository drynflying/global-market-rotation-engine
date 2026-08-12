from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import as_bool, safe_int


STATE_CONFIRMATION_BARS = 3
PAIR_CONFIRMATION_BARS = 3


def _consecutive_true(values: pd.Series) -> pd.Series:
    out = []
    streak = 0
    for value in values.fillna(False).astype(bool):
        streak = streak + 1 if value else 0
        out.append(streak)
    return pd.Series(out, index=values.index, dtype="int64")


def _confirm_categorical_signal(
    values: pd.Series,
    confirmation_bars: int,
    passthrough_values: set[str] | None = None,
) -> pd.DataFrame:
    """
    Convert a reactive daily categorical condition into a confirmed trend.

    A new state must appear on `confirmation_bars` consecutive observations
    before it replaces the currently confirmed state. Until then, the existing
    confirmed state remains in force and the candidate is exposed as pending.

    Passthrough states such as REFERENCE / NOT_SCORED are applied immediately
    because they are data-availability states rather than market judgments.
    """
    passthrough_values = passthrough_values or set()
    confirmation_bars = max(int(confirmation_bars), 1)

    confirmed = None
    confirmed_age = 0
    pending = None
    pending_days = 0

    rows: list[dict] = []

    for raw_value in values:
        raw = "" if pd.isna(raw_value) else str(raw_value).strip()

        if not raw:
            confirmed = ""
            confirmed_age = 0
            pending = None
            pending_days = 0
        elif raw in passthrough_values:
            confirmed = raw
            confirmed_age = 1
            pending = None
            pending_days = 0
        elif confirmed is None or confirmed == "" or confirmed in passthrough_values:
            # Initialize the first scored/usable observation immediately.
            confirmed = raw
            confirmed_age = 1
            pending = None
            pending_days = 0
        elif raw == confirmed:
            confirmed_age += 1
            pending = None
            pending_days = 0
        else:
            if raw == pending:
                pending_days += 1
            else:
                pending = raw
                pending_days = 1

            if pending_days >= confirmation_bars:
                confirmed = pending
                confirmed_age = 1
                pending = None
                pending_days = 0
            else:
                confirmed_age += 1

        rows.append(
            {
                "confirmed": confirmed or "",
                "confirmed_age": int(confirmed_age),
                "pending": pending or "",
                "pending_days": int(pending_days),
            }
        )

    return pd.DataFrame(rows, index=values.index)


def add_rotation_trends(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()

    for lag in [5, 10, 20, 63]:
        out[f"score_change_{lag}"] = (
            out["rotation_score"]
            - out.groupby("ticker")["rotation_score"].shift(lag)
        )

    # Only cross-sectional signals have meaningful group ranks.
    for lag in [5, 20]:
        out[f"rank_change_{lag}"] = (
            out.groupby("ticker")["group_rank"].shift(lag)
            - out["group_rank"]
        )

    out["rs20_change_5"] = (
        out["signal_rs20"] - out.groupby("ticker")["signal_rs20"].shift(5)
    )
    out["rs63_change_20"] = (
        out["signal_rs63"] - out.groupby("ticker")["signal_rs63"].shift(20)
    )

    pieces = []
    for _, g in out.groupby("ticker", sort=False):
        g = g.copy()

        # "Leader zone" has one consistent meaning across score modes:
        # - Cross-sectional: top quartile of peers
        # - Pair: fixed score >= 75
        leader_zone = np.where(
            g["score_mode"].eq("PAIR"),
            g["rotation_score"] >= 75,
            g["group_percentile"] >= 75,
        )
        leader_zone = pd.Series(leader_zone, index=g.index).fillna(False)

        improving = g["rotation_score"].diff() > 0

        g["days_leader_zone_20"] = (
            leader_zone.astype(int)
            .rolling(20, min_periods=1)
            .sum()
            .astype(int)
        )
        g["leader_zone_streak"] = _consecutive_true(leader_zone)
        g["consecutive_improving_days"] = _consecutive_true(improving)

        # Backward-compatible aliases for existing dashboard/history consumers.
        # For PAIR rows these aliases mean "leader zone", not literal quartile.
        g["days_top_quartile_20"] = g["days_leader_zone_20"]
        g["top_quartile_streak"] = g["leader_zone_streak"]

        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)

    def classify_raw(r) -> str:
        if not as_bool(r.get("rank_eligible")):
            return "REFERENCE"

        if r.get("data_status") != "READY" or pd.isna(r.get("rotation_score")):
            return "NOT_SCORED"

        score = float(r["rotation_score"])
        rs20 = r.get("signal_rs20")
        rs63 = r.get("signal_rs63")
        ch5 = r.get("score_change_5")
        ch20 = r.get("score_change_20")
        streak = safe_int(r.get("leader_zone_streak"), 0)
        persistence = safe_int(r.get("persistence_bars"), 5)

        # Strongly negative on both horizons and already in the bottom score zone.
        if (
            score <= 25
            and pd.notna(rs20) and rs20 < 0
            and pd.notna(rs63) and rs63 < 0
            and (
                (pd.notna(ch20) and ch20 <= -5)
                or (pd.notna(ch5) and ch5 <= 0)
            )
        ):
            return "ROTATION_OUT"

        # Longer-term strength remains, but a prior 20-bar deterioration has
        # turned upward again over the latest 5 bars.
        if (
            score >= 55
            and pd.notna(rs63) and rs63 > 0
            and pd.notna(ch20) and ch20 <= -10
            and pd.notna(ch5) and ch5 > 0
        ):
            return "REACCELERATING"

        # Meaningful 20-bar deterioration gets priority over a stale high score.
        if (
            pd.notna(ch20) and ch20 <= -10
            and (
                (pd.notna(ch5) and ch5 <= 0)
                or (pd.notna(rs20) and rs20 < 0)
            )
        ):
            return "WEAKENING"

        # Sustained relative leadership with persistence.
        if (
            score >= 70
            and pd.notna(rs20) and rs20 > 0
            and pd.notna(rs63) and rs63 > 0
            and streak >= persistence
            and (pd.isna(ch20) or ch20 >= -10)
        ):
            return "PERSISTENT_LEADER"

        # Sharp 20-bar improvement before full longer-horizon confirmation.
        if (
            score >= 60
            and pd.notna(ch20) and ch20 >= 15
            and pd.notna(rs20) and rs20 > 0
            and (
                (pd.notna(rs63) and rs63 <= 0)
                or streak < persistence
            )
        ):
            return "EMERGING"

        # Positive relative strength on both horizons with rising score.
        if (
            score >= 60
            and pd.notna(rs20) and rs20 > 0
            and pd.notna(rs63) and rs63 > 0
            and pd.notna(ch5) and ch5 > 0
            and pd.notna(ch20) and ch20 > 0
        ):
            return "ACCELERATING"

        if (
            pd.notna(ch5) and ch5 < 0
            and pd.notna(ch20) and ch20 < 0
        ):
            return "WEAKENING"

        return "NEUTRAL"

    # Preserve today's fully reactive mathematical condition.
    out["rotation_state_raw"] = out.apply(classify_raw, axis=1)

    # Preserve the direct, sign-based pair relationship as the raw condition.
    # calculate_rotation_scores() creates pair_signal before this function runs.
    out["pair_signal_raw"] = out.get("pair_signal", "").fillna("").astype(str)

    stabilized_pieces = []
    for _, g in out.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        g = g.copy()

        state_confirmation = _confirm_categorical_signal(
            g["rotation_state_raw"],
            confirmation_bars=STATE_CONFIRMATION_BARS,
            passthrough_values={"REFERENCE", "NOT_SCORED"},
        )
        g["rotation_state_confirmed"] = state_confirmation["confirmed"]
        g["confirmed_state_age"] = state_confirmation["confirmed_age"]
        g["pending_rotation_state"] = state_confirmation["pending"]
        g["pending_state_days"] = state_confirmation["pending_days"]
        g["state_confirmation_bars"] = STATE_CONFIRMATION_BARS

        if g["score_mode"].eq("PAIR").any():
            pair_confirmation = _confirm_categorical_signal(
                g["pair_signal_raw"],
                confirmation_bars=PAIR_CONFIRMATION_BARS,
            )
            g["pair_signal_confirmed"] = pair_confirmation["confirmed"]
            g["confirmed_pair_signal_age"] = pair_confirmation["confirmed_age"]
            g["pending_pair_signal"] = pair_confirmation["pending"]
            g["pending_pair_signal_days"] = pair_confirmation["pending_days"]
            g["pair_confirmation_bars"] = PAIR_CONFIRMATION_BARS
        else:
            g["pair_signal_confirmed"] = ""
            g["confirmed_pair_signal_age"] = 0
            g["pending_pair_signal"] = ""
            g["pending_pair_signal_days"] = 0
            g["pair_confirmation_bars"] = PAIR_CONFIRMATION_BARS

        stabilized_pieces.append(g)

    out = pd.concat(stabilized_pieces, ignore_index=True)

    # Canonical downstream fields are now the confirmed trend signals.
    # Existing consumers can keep using rotation_state / pair_signal without
    # silently reverting to the noisy raw daily condition. The raw fields remain
    # available for transparency and early-warning display.
    out["rotation_state"] = out["rotation_state_confirmed"]
    pair_mask = out["score_mode"].eq("PAIR")
    out.loc[pair_mask, "pair_signal"] = out.loc[pair_mask, "pair_signal_confirmed"]
    out.loc[~pair_mask, "pair_signal"] = ""

    return out
