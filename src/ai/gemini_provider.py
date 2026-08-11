from __future__ import annotations

import os

from src.ai.prompt import build_prompt
from src.ai.schema import AIAnalysis


PROVIDER_NAME = "gemini"
DEFAULT_MODEL = "gemini-3.6-flash"


def analyze(payload: dict) -> tuple[dict, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY secret is missing")

    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(payload),
        config={
            "response_format": {
                "text": {
                    "mime_type": "application/json",
                    "schema": AIAnalysis.model_json_schema(),
                }
            }
        },
    )

    if not getattr(response, "text", None):
        raise RuntimeError("Gemini returned no text response")

    analysis = AIAnalysis.model_validate_json(response.text).model_dump()
    return analysis, model
