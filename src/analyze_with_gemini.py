from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from src.common import AI_ANALYSIS_PATH


class RotationItem(BaseModel):
    ticker: str
    title: str
    explanation: str
    confidence: Literal["low", "medium", "high"]


class AIAnalysis(BaseModel):
    headline: str
    market_regime: str
    executive_summary: str
    emerging_rotations: list[RotationItem] = Field(default_factory=list)
    persistent_leaders: list[RotationItem] = Field(default_factory=list)
    weakening_rotations: list[RotationItem] = Field(default_factory=list)
    geographic_rotations: list[RotationItem] = Field(default_factory=list)
    risks_or_conflicts: list[str] = Field(default_factory=list)
    dashboard_focus_tickers: list[str] = Field(default_factory=list)
    methodology_note: str


SYSTEM_INSTRUCTION = """
You are interpreting a quantitative ETF capital-rotation dataset.

The arithmetic and rankings have already been calculated by deterministic Python.
Do not recalculate or invent values. Do not give personalized investment advice,
price targets, or trade instructions.

Your job is to identify observed rotation patterns over time:
- emerging leadership
- accelerating leadership
- persistent leadership
- reaccelerating leadership
- weakening leadership
- geographic shifts
- breadth or confirmation/conflict across parent sectors and paired markets

Use only the data supplied in the JSON. Treat ETF OHLCV as market participation
evidence, not proof of institutional net fund flow. Emphasize 5-, 20-, and
63-trading-bar trends rather than one-day moves. CROSS_SECTIONAL scores are
peer-group ranks; PAIR scores are direct relationship signals and must not be
ranked against cross-sectional groups.
""".strip()


def _fallback(payload: dict, reason: str) -> dict:
    groups = payload.get("group_summaries", [])
    leaders = []
    improvers = []
    weakeners = []

    for group in groups:
        leaders.extend(group.get("leaders", []))
        improvers.extend(group.get("biggest_20d_improvers", []))
        weakeners.extend(group.get("biggest_20d_weakeners", []))

    leaders = sorted(
        [x for x in leaders if x.get("score") is not None],
        key=lambda x: x["score"],
        reverse=True,
    )[:5]
    improvers = sorted(
        [x for x in improvers if x.get("score_change_20") is not None],
        key=lambda x: x["score_change_20"],
        reverse=True,
    )[:5]
    weakeners = sorted(
        [x for x in weakeners if x.get("score_change_20") is not None],
        key=lambda x: x["score_change_20"],
    )[:5]

    def convert(item, prefix):
        return {
            "ticker": item["ticker"],
            "title": f"{prefix}: {item.get('exposure') or item['ticker']}",
            "explanation": (
                f"Rotation score {item.get('score')}; "
                f"20-bar change {item.get('score_change_20')}."
            ),
            "confidence": "medium",
        }

    return {
        "headline": "Quantitative rotation dashboard",
        "market_regime": "AI interpretation unavailable",
        "executive_summary": (
            "The dashboard is using deterministic rotation calculations only. "
            f"AI was not used for this run: {reason}"
        ),
        "emerging_rotations": [convert(x, "Improving") for x in improvers[:4]],
        "persistent_leaders": [convert(x, "Leader") for x in leaders[:4]],
        "weakening_rotations": [convert(x, "Weakening") for x in weakeners[:4]],
        "geographic_rotations": [],
        "risks_or_conflicts": [
            "Rotation scores are comparative signals, not forecasts.",
            "ETF trading volume is not the same as ETF net creations/redemptions.",
        ],
        "dashboard_focus_tickers": list(
            dict.fromkeys(
                [x["ticker"] for x in improvers[:4]]
                + [x["ticker"] for x in leaders[:4]]
            )
        )[:8],
        "methodology_note": "Deterministic fallback summary; no AI API call was used.",
        "provider_status": "fallback",
    }


def analyze(payload: dict) -> dict:
    provider = os.getenv("AI_PROVIDER", "").strip().lower()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip() or "gemini-2.5-flash"

    if provider != "gemini":
        result = _fallback(payload, "AI_PROVIDER is not set to gemini")
        AI_ANALYSIS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    if not api_key:
        result = _fallback(payload, "GEMINI_API_KEY secret is missing")
        AI_ANALYSIS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=(
                SYSTEM_INSTRUCTION
                + "\n\nHere is today's structured rotation dataset:\n"
                + json.dumps(payload, separators=(",", ":"))
            ),
            config={
                "response_mime_type": "application/json",
                "response_schema": AIAnalysis,
            },
        )

        if getattr(response, "parsed", None) is not None:
            parsed = response.parsed
            result = (
                parsed.model_dump()
                if hasattr(parsed, "model_dump")
                else dict(parsed)
            )
        else:
            result = AIAnalysis.model_validate_json(response.text).model_dump()

        result["provider_status"] = "gemini"
        result["provider_model"] = model
    except Exception as exc:
        result = _fallback(payload, f"Gemini call failed: {type(exc).__name__}: {exc}")

    AI_ANALYSIS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
