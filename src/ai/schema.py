from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


Confidence = Literal["low", "medium", "high"]


class RotationFinding(BaseModel):
    ticker: str = Field(description="Primary ETF ticker from the supplied dataset.")
    title: str = Field(description="Short finding title.")
    explanation: str = Field(
        description="Concise interpretation grounded only in supplied quantitative evidence."
    )
    confidence: Confidence
    related_tickers: list[str] = Field(
        default_factory=list,
        description="Other supplied ETF tickers that confirm, conflict with, or contextualize the finding.",
    )


class AIAnalysis(BaseModel):
    headline: str
    market_regime: str
    executive_summary: str

    emerging_rotations: list[RotationFinding] = Field(default_factory=list)
    accelerating_rotations: list[RotationFinding] = Field(default_factory=list)
    persistent_leaders: list[RotationFinding] = Field(default_factory=list)
    reaccelerating_rotations: list[RotationFinding] = Field(default_factory=list)
    weakening_rotations: list[RotationFinding] = Field(default_factory=list)
    rotation_out: list[RotationFinding] = Field(default_factory=list)
    geographic_rotations: list[RotationFinding] = Field(default_factory=list)
    pair_relationships: list[RotationFinding] = Field(default_factory=list)

    cross_market_confirmations: list[str] = Field(default_factory=list)
    risks_or_conflicts: list[str] = Field(default_factory=list)
    dashboard_focus_tickers: list[str] = Field(default_factory=list)
    methodology_note: str


CATEGORY_FIELDS = [
    "emerging_rotations",
    "accelerating_rotations",
    "persistent_leaders",
    "reaccelerating_rotations",
    "weakening_rotations",
    "rotation_out",
    "geographic_rotations",
    "pair_relationships",
]
