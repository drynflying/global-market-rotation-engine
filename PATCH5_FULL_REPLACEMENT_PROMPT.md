# Patch 5 — Full-file replacement package

Replace each repository file below in full. Do not modify `src/build_dashboard.py`; Patches 2–4 remain intact.
## Acceptance criteria

- Validator version becomes `v1.2.7`.
- Mixed-sign 20/63-bar relative strength must disclose both exact supplied values and describe the evidence as mixed/divergent/conflicting.
- `risks_or_conflicts` must cover every deterministic sector divergence and pair/state tension.
- Gemini receives global rankings, 63-bar rotation-score highs/lows, extreme weakness/CMF, and deterministic conflicts.
- Important attention tickers are prioritized in `focus_securities`.
- Missing text values serialize as JSON `null`, not `NaN`.

## `src/build_ai_input.py`

```python
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

```

## `src/ai/prompt.py`

```python
from __future__ import annotations

import json


SYSTEM_INSTRUCTION = """
You are one independent AI analyst interpreting a deterministic ETF
rotation dataset.

The arithmetic, relative strength, pair spreads, rankings, persistence,
rotation states, deterministic attention flags, and 63-bar score extrema
have already been calculated by Python. Do not recalculate, replace, or
invent those values.

Your role is interpretation:
- identify emerging rotations;
- identify accelerating and persistent leadership;
- identify reacceleration after a pullback in leadership;
- identify weakening and rotation-out conditions;
- identify geographic rotation;
- identify direct pair relationships such as Growth vs Value;
- connect evidence across sectors, industries, countries, regions, parents,
  benchmarks, and paired markets;
- highlight conflicts where short and long horizons disagree;
- explicitly acknowledge material deterministic conflicts supplied in
  deterministic_attention.

Hard grounding rules:
1. Use only the supplied JSON.
2. Every RotationFinding.ticker must be present in the supplied data.
3. Every related_ticker must be present in the supplied data.
4. Do not change a Python state. If a finding is placed in a state-specific
   category, its ticker must have that exact deterministic state.
5. A high score that is falling is different from a high score that is rising.
6. CROSS_SECTIONAL scores are peer-group comparisons.
7. PAIR scores are direct relationships. For a PAIR signal:
   - name the exact paired ticker in the explanation;
   - describe leading/lagging versus that paired ticker;
   - never describe the PAIR score as a peer-group rank;
   - do not call the paired security a benchmark unless the supplied pair
     definition itself is explicitly a benchmark comparison.
8. Do not claim or imply actual money/fund/capital flows. The dataset does not
   contain ETF creations/redemptions or institutional net-flow data.
   Avoid phrases such as:
   - capital inflow / outflow;
   - capital shift;
   - capital rotates / capital rotation;
   - money flowing into / out of;
   - fund inflow / outflow;
   - institutional buying / selling.
   Use precise alternatives such as:
   - relative leadership is shifting toward;
   - relative strength is improving;
   - rotation signal is strengthening;
   - relative leadership is deteriorating.
9. Do not use "mega-cap" or "megacap" unless the supplied exposure or label
   explicitly uses that term.
10. ETF trading volume and CMF are market-participation indicators, not proof
    of institutional buying, selling, or net fund flow.
11. For each RotationFinding.explanation, cite concrete supplied evidence.
    Include at least one numeric value and identify the relevant metric/horizon.
    Prefer two or more metrics when useful.
12. Be precise about horizons:
    - score_change_5 is a 5-bar SCORE change;
    - signal_rs20 is 20-bar relative strength;
    - signal_rs63 is 63-bar relative strength.
    Never describe a 5-bar score change as 5-bar relative strength.
13. MIXED-HORIZON DISCLOSURE IS MANDATORY. If a discussed ticker has
    signal_rs20_pct_points and signal_rs63_pct_points with opposite signs,
    then any sentence or finding that uses relative-strength evidence for that
    ticker must cite BOTH supplied values and explicitly describe the horizons
    as mixed, divergent, conflicting, or split. Never cite only the favorable
    horizon. This applies even when one or both values are small.
14. For pair signals, pair_spread_20 and pair_spread_63 are the direct
    20- and 63-bar relative-performance spreads versus paired_ticker.
15. deterministic_attention is authoritative context calculated by Python:
    - sector_divergences are material within-sector disagreements;
    - pair_state_tensions are pair-signal/state disagreements;
    - score_63_bar_highs/lows are extrema of the ROTATION SCORE, not price;
    - lowest_current_scores and extreme_cmf20 identify quantitative extremes.
16. risks_or_conflicts MUST acknowledge every supplied sector_divergence by
    naming at least the improver and deteriorator in the same risk/conflict
    statement. It MUST also acknowledge every supplied pair_state_tension by
    naming the ticker and its paired_ticker and explaining the distinction.
17. When discussing a 63-bar score extreme, say "63-bar score high/low" or
    equivalent. Never call it a 63-bar price high/low.
18. global_rankings are deterministic evidence-selection aids. For a broad
    regional, sector, or style thesis, prefer the strongest supporting examples
    from highest_current_scores and biggest_20_bar_improvements. A broad ETF or
    benchmark can confirm breadth, but do not choose a materially weaker example
    while ignoring stronger supplied examples that directly support the thesis.
19. Do not give personalized investment advice, trade instructions,
    allocations, buy/sell commands, or price targets.
20. Prefer a small number of high-information findings rather than simply
    repeating the highest scores.
21. Write probabilistically when drawing a higher-level inference. Prefer
    "consistent with broadening" over "proves broadening" or
    "indicates capital is flowing."

Methodology-note requirement:
State explicitly that the interpretation uses 5-bar score changes,
20- and 63-bar relative strength/pair spreads, persistence metrics, and
Python-calculated deterministic attention flags including 63-bar score
extrema. Do not imply that direct 5-bar relative strength is calculated.

The output must follow the supplied structured schema exactly.
""".strip()


def build_prompt(payload: dict, correction_instructions: str | None = None) -> str:
    correction = ""
    if correction_instructions:
        correction = (
            "\n\nCORRECTION REQUIRED:\n"
            "A prior structured response failed deterministic validation. "
            "Rewrite the affected content using only supported metrics and "
            "deterministic attention context. "
            "For any 5-bar observation, use score_change_5 only. "
            "For direct relative-strength evidence, use only the 20-bar or "
            "63-bar horizons supplied in the dataset. "
            "If those two horizons have opposite signs for a discussed ticker, "
            "cite BOTH exact supplied values and describe the evidence as mixed. "
            "Ensure risks_or_conflicts covers every supplied sector_divergence "
            "and pair_state_tension. "
            "Do not repeat validator wording verbatim.\n"
            + correction_instructions.strip()
        )

    return (
        SYSTEM_INSTRUCTION
        + correction
        + "\n\nToday's deterministic rotation dataset follows:\n"
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    )

```

## `src/ai/validator.py`

```python
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field


VALIDATOR_VERSION = "v1.2.7"

CATEGORY_TO_STATE = {
    "emerging_rotations": "EMERGING",
    "accelerating_rotations": "ACCELERATING",
    "persistent_leaders": "PERSISTENT_LEADER",
    "reaccelerating_rotations": "REACCELERATING",
    "weakening_rotations": "WEAKENING",
    "rotation_out": "ROTATION_OUT",
}

FINDING_CATEGORIES = [
    "emerging_rotations",
    "accelerating_rotations",
    "persistent_leaders",
    "reaccelerating_rotations",
    "weakening_rotations",
    "rotation_out",
    "geographic_rotations",
    "pair_relationships",
]

PROHIBITED_LANGUAGE = [
    ("capital-flow language", re.compile(
        r"\bcapital\s+(?:inflows?|outflows?|flows?|shift(?:s|ed|ing)?|"
        r"rotat(?:e|es|ed|ing|ion|ions))\b", re.I
    )),
    ("money-flow language", re.compile(
        r"\bmoney\s+(?:flows?|flowing|moves?|moving)\s+(?:into|out\s+of)\b", re.I
    )),
    ("fund-flow language", re.compile(
        r"\bfund\s+(?:inflows?|outflows?|flows?)\b", re.I
    )),
    ("institutional transaction claim", re.compile(
        r"\binstitutional\s+(?:buying|selling|inflows?|outflows?)\b", re.I
    )),
]

NUMERIC_RE = re.compile(r"[-+−]?\d+(?:\.\d+)?")
EVIDENCE_TERM_RE = re.compile(
    r"\b(?:score|bar|relative strength|rs20|rs63|spread|cmf|streak|rank|"
    r"percent|percentage|leader zone|pair)\b",
    re.I,
)
RS_TERM_RE = re.compile(r"\b(?:relative strength|rs20|rs63|signal_rs20|signal_rs63)\b", re.I)
MIXED_WORD_RE = re.compile(r"\b(?:mixed|divergent|divergence|conflict(?:ing)?|split)\b", re.I)


class AIOutputValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        preview = "; ".join(errors[:8])
        if len(errors) > 8:
            preview += f"; ... +{len(errors) - 8} more"
        super().__init__(f"AI output failed deterministic validation: {preview}")


@dataclass
class PayloadIndex:
    known_tickers: set[str]
    states: dict[str, str]
    score_modes: dict[str, str]
    paired_tickers: dict[str, str]
    pair_types: dict[str, str]
    benchmark_tickers: set[str]
    metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)
    supplied_text: str = ""


def _norm_ticker(value) -> str:
    return str(value or "").strip().upper()


def _as_float(value):
    try:
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _index_payload(payload: dict) -> PayloadIndex:
    known: set[str] = set()
    states: dict[str, str] = {}
    score_modes: dict[str, str] = {}
    paired_tickers: dict[str, str] = {}
    pair_types: dict[str, str] = {}
    benchmark_tickers: set[str] = set()
    metrics: dict[str, dict[str, float | None]] = {}

    for item in payload.get("benchmark_context", []) or []:
        ticker = _norm_ticker(item.get("ticker"))
        if ticker:
            benchmark_tickers.add(ticker)
            known.add(ticker)

    def visit(node):
        if isinstance(node, dict):
            ticker = _norm_ticker(node.get("ticker"))
            if ticker:
                known.add(ticker)

                state_value = str(node.get("state") or "").strip().upper()
                if state_value:
                    states[ticker] = state_value

                mode = str(node.get("score_mode") or "").strip().upper()
                if mode:
                    score_modes[ticker] = mode

                paired = _norm_ticker(node.get("paired_ticker"))
                if paired:
                    paired_tickers[ticker] = paired
                    known.add(paired)

                pair_type = str(node.get("pair_type") or "").strip().upper()
                if pair_type:
                    pair_types[ticker] = pair_type

                ticker_metrics = metrics.setdefault(ticker, {})
                for key in (
                    "score",
                    "score_change_5",
                    "score_change_20",
                    "signal_rs20_pct_points",
                    "signal_rs63_pct_points",
                    "pair_spread_20_pct_points",
                    "pair_spread_63_pct_points",
                    "cmf20",
                ):
                    value = _as_float(node.get(key))
                    if value is not None:
                        ticker_metrics[key] = value

            for key in ("primary_benchmark", "parent_benchmark", "paired_ticker"):
                value = _norm_ticker(node.get(key))
                if value:
                    known.add(value)

            for value in node.values():
                visit(value)

        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(payload)

    return PayloadIndex(
        known_tickers=known,
        states=states,
        score_modes=score_modes,
        paired_tickers=paired_tickers,
        pair_types=pair_types,
        benchmark_tickers=benchmark_tickers,
        metrics=metrics,
        supplied_text=json.dumps(payload, ensure_ascii=False).lower(),
    )


def _numeric_values(text: str) -> list[float]:
    normalized = str(text).replace("−", "-")
    values = []
    for match in NUMERIC_RE.findall(normalized):
        try:
            values.append(float(match))
        except ValueError:
            pass
    return values


def _contains_value(text: str, expected: float, tolerance: float = 0.021) -> bool:
    return any(abs(value - expected) <= tolerance for value in _numeric_values(text))


def _mixed_horizon(metrics: dict[str, float | None]) -> bool:
    rs20 = metrics.get("signal_rs20_pct_points")
    rs63 = metrics.get("signal_rs63_pct_points")
    if rs20 is None or rs63 is None:
        return False
    return (rs20 > 0 > rs63) or (rs63 > 0 > rs20)


def _validate_mixed_horizon_text(
    field_path: str,
    ticker: str,
    text: str,
    idx: PayloadIndex,
    errors: list[str],
) -> None:
    metrics = idx.metrics.get(ticker, {})
    if not _mixed_horizon(metrics):
        return
    if not RS_TERM_RE.search(text):
        return

    rs20 = metrics["signal_rs20_pct_points"]
    rs63 = metrics["signal_rs63_pct_points"]
    lower = text.lower()

    if "20" not in lower or "63" not in lower:
        errors.append(
            f"mixed-horizon disclosure: {field_path} for {ticker} must name both 20-bar and 63-bar relative-strength horizons"
        )
        return

    if not _contains_value(text, rs20) or not _contains_value(text, rs63):
        errors.append(
            f"mixed-horizon disclosure: {field_path} for {ticker} must cite both supplied relative-strength values ({rs20:.2f} and {rs63:.2f} percentage points)"
        )

    if not MIXED_WORD_RE.search(text):
        errors.append(
            f"mixed-horizon disclosure: {field_path} for {ticker} must describe opposite-sign relative-strength horizons as mixed/divergent/conflicting"
        )


def _contains_ticker(text: str, ticker: str) -> bool:
    return bool(re.search(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", text, re.I))


def validate_analysis_against_payload(analysis: dict, payload: dict) -> dict:
    idx = _index_payload(payload)
    errors: list[str] = []
    prose = json.dumps(analysis, ensure_ascii=False)

    for label, pattern in PROHIBITED_LANGUAGE:
        match = pattern.search(prose)
        if match:
            errors.append(f"{label}: prohibited phrase '{match.group(0)}'")

    if re.search(r"\bmega[- ]?cap\b", prose, re.I):
        if not re.search(r"\bmega[- ]?cap\b", idx.supplied_text, re.I):
            errors.append(
                "unsupported mega-cap characterization: source payload does not use that label"
            )

    for category in FINDING_CATEGORIES:
        findings = analysis.get(category, []) or []
        required_state = CATEGORY_TO_STATE.get(category)

        for pos, finding in enumerate(findings):
            prefix = f"{category}[{pos}]"
            ticker = _norm_ticker(finding.get("ticker"))

            if not ticker:
                errors.append(f"{prefix}: missing ticker")
                continue

            if ticker not in idx.known_tickers:
                errors.append(f"{prefix}: ticker {ticker} was not supplied")

            for related in finding.get("related_tickers", []) or []:
                related_ticker = _norm_ticker(related)
                if related_ticker and related_ticker not in idx.known_tickers:
                    errors.append(
                        f"{prefix}: related ticker {related_ticker} was not supplied"
                    )

            if required_state:
                actual_state = idx.states.get(ticker)
                if not actual_state:
                    errors.append(
                        f"{prefix}: deterministic state for {ticker} is unavailable"
                    )
                elif actual_state != required_state:
                    errors.append(
                        f"{prefix}: {ticker} is {actual_state}, not {required_state}"
                    )

            title = str(finding.get("title") or "")
            explanation = str(finding.get("explanation") or "")
            text = f"{title} {explanation}"

            if not NUMERIC_RE.search(explanation):
                errors.append(f"{prefix}: explanation lacks numeric evidence")
            if not EVIDENCE_TERM_RE.search(explanation):
                errors.append(
                    f"{prefix}: explanation does not name a quantitative metric/horizon"
                )

            _validate_mixed_horizon_text(prefix, ticker, text, idx, errors)

            mode = idx.score_modes.get(ticker)
            if category == "pair_relationships" and mode != "PAIR":
                errors.append(f"{prefix}: {ticker} is not a PAIR signal")

            if mode == "PAIR":
                paired = idx.paired_tickers.get(ticker)

                if paired and not re.search(
                    rf"\b{re.escape(paired)}\b", text, re.I
                ):
                    errors.append(
                        f"{prefix}: PAIR finding for {ticker} must name paired ticker {paired}"
                    )

                if re.search(r"\b(?:rank|ranked|peer[- ]?group)\b", text, re.I):
                    errors.append(
                        f"{prefix}: PAIR finding for {ticker} must not use peer-group rank language"
                    )

                paired_is_benchmark = paired in idx.benchmark_tickers if paired else False
                pair_type = idx.pair_types.get(ticker, "")
                if (
                    "BENCHMARK" not in pair_type
                    and not paired_is_benchmark
                    and re.search(r"\bbenchmark\b", text, re.I)
                ):
                    errors.append(
                        f"{prefix}: PAIR finding for {ticker} uses generic benchmark language "
                        f"but paired ticker is {paired}"
                    )

    for pos, ticker_value in enumerate(
        analysis.get("dashboard_focus_tickers", []) or []
    ):
        ticker = _norm_ticker(ticker_value)
        if ticker and ticker not in idx.known_tickers:
            errors.append(
                f"dashboard_focus_tickers[{pos}]: {ticker} was not supplied"
            )

    # Horizon-language validation is applied to analytical model-authored fields.
    # methodology_note is canonicalized by Python before validation.
    horizon_fields = [
        ("headline", str(analysis.get("headline") or "")),
        ("market_regime", str(analysis.get("market_regime") or "")),
        ("executive_summary", str(analysis.get("executive_summary") or "")),
    ]
    for category in FINDING_CATEGORIES:
        for pos, finding in enumerate(analysis.get(category, []) or []):
            horizon_fields.append(
                (f"{category}[{pos}].title", str(finding.get("title") or ""))
            )
            horizon_fields.append(
                (
                    f"{category}[{pos}].explanation",
                    str(finding.get("explanation") or ""),
                )
            )
    for pos, value in enumerate(analysis.get("cross_market_confirmations", []) or []):
        horizon_fields.append((f"cross_market_confirmations[{pos}]", str(value)))
    for pos, value in enumerate(analysis.get("risks_or_conflicts", []) or []):
        horizon_fields.append((f"risks_or_conflicts[{pos}]", str(value)))

    bad_horizon = re.compile(
        r"\b5[- ]?(?:bar|day|trading[- ]?bar)?\s+relative[- ]?strength\b",
        re.I,
    )
    for field_path, field_text in horizon_fields:
        if bad_horizon.search(field_text):
            errors.append(
                f"{field_path}: 5-bar observations must be described as score changes; "
                "relative-strength evidence is limited to 20/63 bars"
            )

    # Apply mixed-horizon disclosure to the free-form confirmation/risk lines,
    # which are normally ticker-specific. Avoid applying this to broad summary
    # paragraphs where one ticker mention and an unrelated RS phrase could
    # otherwise create a false positive. Finding explanations were already
    # checked above against their primary ticker.
    freeform_mixed_fields = []
    for pos, value in enumerate(analysis.get("cross_market_confirmations", []) or []):
        freeform_mixed_fields.append((f"cross_market_confirmations[{pos}]", str(value)))
    for pos, value in enumerate(analysis.get("risks_or_conflicts", []) or []):
        freeform_mixed_fields.append((f"risks_or_conflicts[{pos}]", str(value)))

    for field_path, field_text in freeform_mixed_fields:
        for ticker, ticker_metrics in idx.metrics.items():
            if not _mixed_horizon(ticker_metrics):
                continue
            if _contains_ticker(field_text, ticker) and RS_TERM_RE.search(field_text):
                _validate_mixed_horizon_text(
                    field_path, ticker, field_text, idx, errors
                )

    # Deterministic conflict coverage: the AI may add more risks, but it may
    # not omit the material conflicts Python has already identified.
    risks = [str(x) for x in analysis.get("risks_or_conflicts", []) or []]
    attention = payload.get("deterministic_attention", {}) or {}

    for pos, conflict in enumerate(attention.get("sector_divergences", []) or []):
        improver = _norm_ticker(conflict.get("improver"))
        deteriorator = _norm_ticker(conflict.get("deteriorator"))
        if improver and deteriorator:
            covered = any(
                _contains_ticker(text, improver)
                and _contains_ticker(text, deteriorator)
                for text in risks
            )
            if not covered:
                errors.append(
                    "deterministic conflict coverage: "
                    f"risks_or_conflicts must acknowledge sector divergence "
                    f"{improver} vs {deteriorator}"
                )

    for pos, tension in enumerate(attention.get("pair_state_tensions", []) or []):
        ticker = _norm_ticker(tension.get("ticker"))
        paired = _norm_ticker(tension.get("paired_ticker"))
        if ticker and paired:
            covered = any(
                _contains_ticker(text, ticker)
                and _contains_ticker(text, paired)
                for text in risks
            )
            if not covered:
                errors.append(
                    "deterministic conflict coverage: "
                    f"risks_or_conflicts must acknowledge pair/state tension "
                    f"{ticker} vs {paired}"
                )

    if errors:
        # Preserve order while removing duplicate messages caused by a finding
        # appearing in both the finding-level and free-form horizon passes.
        errors = list(dict.fromkeys(errors))
        raise AIOutputValidationError(errors)

    return {
        "status": "passed",
        "validator_version": VALIDATOR_VERSION,
        "checks": {
            "ticker_grounding": True,
            "state_category_consistency": True,
            "pair_semantics": True,
            "language_guardrails": True,
            "numeric_evidence": True,
            "horizon_precision": True,
            "mixed_horizon_disclosure": True,
            "deterministic_conflict_coverage": True,
        },
    }

```

## `src/ai/normalizer.py`

```python
from __future__ import annotations

import copy


CANONICAL_METHODOLOGY_NOTE = (
    "Interpretation uses deterministic 5-bar score changes, 20- and 63-bar "
    "relative strength or pair spreads, persistence metrics, and Python-"
    "calculated attention flags including 63-bar rotation-score extrema and "
    "material cross-market conflicts. ETF volume and CMF are treated as "
    "market-participation indicators, not fund-flow data."
)


def normalize_analysis(analysis: dict) -> dict:
    """
    Normalize fields that should be deterministic rather than model-authored.

    The methodology note is application metadata, not an analytical judgment.
    Owning it in Python prevents a valid analysis from failing because a model
    paraphrases the metric horizons imprecisely.
    """
    out = copy.deepcopy(analysis)
    out["methodology_note"] = CANONICAL_METHODOLOGY_NOTE
    return out

```
