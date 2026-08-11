from __future__ import annotations

import importlib
import json
import os
from datetime import datetime
from pathlib import Path

from src.ai.consensus import build_consensus
from src.ai.fallback import build_fallback_analysis
from src.ai.schema import AIAnalysis
from src.ai.normalizer import normalize_analysis
from src.ai.validator import (
    AIOutputValidationError,
    VALIDATOR_VERSION,
    validate_analysis_against_payload,
)
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
            analysis = normalize_analysis(AIAnalysis.model_validate(analysis).model_dump())

            try:
                validation = validate_analysis_against_payload(analysis, payload)
                validation["attempt_count"] = 1
                validation["retry_used"] = False
            except AIOutputValidationError as first_exc:
                correction = (
                    "Revise the response so it passes these categories of checks:\n"
                    + "\n".join(
                        f"- {error.split(':', 1)[0]}"
                        if ":" in error else "- metric horizon precision"
                        for error in first_exc.errors
                    )
                )
                analysis, model = module.analyze(
                    payload,
                    correction_instructions=correction,
                )
                analysis = normalize_analysis(AIAnalysis.model_validate(analysis).model_dump())
                validation = validate_analysis_against_payload(analysis, payload)
                validation["attempt_count"] = 2
                validation["retry_used"] = True
                validation["first_attempt_errors"] = first_exc.errors

            provider_results[provider] = {
                "provider": provider,
                "status": "success",
                "model": model,
                "error": None,
                "validation": validation,
                "analysis": analysis,
            }
        except AIOutputValidationError as exc:
            provider_results[provider] = {
                "provider": provider,
                "status": "validation_error",
                "model": locals().get("model"),
                "error": f"{type(exc).__name__}: {exc}",
                "validation": {
                    "status": "failed",
                    "validator_version": VALIDATOR_VERSION,
                    "attempt_count": 2,
                    "retry_used": True,
                    "errors": exc.errors,
                },
                "analysis": None,
            }
        except Exception as exc:
            provider_results[provider] = {
                "provider": provider,
                "status": "error",
                "model": None,
                "error": f"{type(exc).__name__}: {exc}",
                "validation": {
                    "status": "not_run",
                    "validator_version": VALIDATOR_VERSION,
                },
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
        "ai_validator_version": VALIDATOR_VERSION,
        "provider_validation": {
            p: provider_results.get(p, {}).get("validation")
            for p in requested
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
