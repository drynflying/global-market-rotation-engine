from __future__ import annotations

import importlib
import json
import os
from datetime import datetime
from pathlib import Path

from src.ai.consensus import build_consensus
from src.ai.fallback import build_fallback_analysis
from src.ai.schema import AIAnalysis
from src.common import (
    AI_ANALYSIS_PATH,
    AI_CONSENSUS_PATH,
    AI_DIR,
    AI_HISTORY_DIR,
    AI_MANIFEST_PATH,
)


PROVIDER_MODULES = {
    "gemini": "src.ai.gemini_provider",
    "claude": "src.ai.claude_provider",
    "openai": "src.ai.openai_provider",
}


def _requested_providers() -> list[str]:
    raw = os.getenv("AI_PROVIDERS", "").strip()
    if not raw:
        # Backward compatibility with the v1.0/v1.1 variable.
        raw = os.getenv("AI_PROVIDER", "").strip()

    providers = []
    for item in raw.split(","):
        name = item.strip().lower()
        if name and name not in providers:
            providers.append(name)
    return providers


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_ai_analysis(payload: dict) -> dict:
    AI_DIR.mkdir(parents=True, exist_ok=True)
    AI_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    as_of = str(payload.get("as_of") or datetime.utcnow().date().isoformat())
    requested = _requested_providers()
    provider_results: dict[str, dict] = {}

    for provider in requested:
        module_name = PROVIDER_MODULES.get(provider)
        if not module_name:
            provider_results[provider] = {
                "provider": provider,
                "status": "error",
                "model": None,
                "error": f"Unknown AI provider: {provider}",
                "analysis": None,
            }
            continue

        try:
            module = importlib.import_module(module_name)
            analysis, model = module.analyze(payload)
            analysis = AIAnalysis.model_validate(analysis).model_dump()
            provider_results[provider] = {
                "provider": provider,
                "status": "success",
                "model": model,
                "error": None,
                "analysis": analysis,
            }
        except Exception as exc:
            provider_results[provider] = {
                "provider": provider,
                "status": "error",
                "model": None,
                "error": f"{type(exc).__name__}: {exc}",
                "analysis": None,
            }

        _write_json(AI_DIR / f"{provider}.json", provider_results[provider])
        _write_json(
            AI_HISTORY_DIR / as_of / f"{provider}.json",
            provider_results[provider],
        )

    successful = [
        p for p in requested
        if provider_results.get(p, {}).get("status") == "success"
    ]
    failed = [
        p for p in requested
        if provider_results.get(p, {}).get("status") != "success"
    ]

    consensus = build_consensus(provider_results)
    _write_json(AI_CONSENSUS_PATH, consensus)
    _write_json(AI_HISTORY_DIR / as_of / "consensus.json", consensus)

    if successful:
        primary_provider = successful[0]
        primary_analysis = provider_results[primary_provider]["analysis"]
        provider_status = "ai"
    else:
        primary_provider = "deterministic_fallback"
        reason = (
            "no AI providers were enabled"
            if not requested
            else "all requested AI providers failed: "
            + "; ".join(
                f"{p}: {provider_results[p].get('error')}"
                for p in failed
            )
        )
        primary_analysis = build_fallback_analysis(payload, reason)
        provider_status = "fallback"

    manifest = {
        "as_of": as_of,
        "requested_providers": requested,
        "successful_providers": successful,
        "failed_providers": failed,
        "primary_provider": primary_provider,
        "provider_status": provider_status,
        "provider_models": {
            p: provider_results[p].get("model")
            for p in successful
        },
    }
    _write_json(AI_MANIFEST_PATH, manifest)
    _write_json(AI_HISTORY_DIR / as_of / "manifest.json", manifest)

    # Keep the top-level primary-analysis fields for backward compatibility
    # while adding the multi-provider bundle underneath.
    bundle = dict(primary_analysis)
    bundle.update(
        {
            "as_of": as_of,
            "provider_status": provider_status,
            "primary_provider": primary_provider,
            "requested_providers": requested,
            "successful_providers": successful,
            "failed_providers": failed,
            "provider_results": provider_results,
            "consensus": consensus,
        }
    )
    _write_json(AI_ANALYSIS_PATH, bundle)
    return bundle
