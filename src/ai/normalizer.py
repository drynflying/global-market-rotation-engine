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
