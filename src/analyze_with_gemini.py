"""
Backward-compatibility shim.

The application is provider-agnostic starting in v1.2. New code should import
run_ai_analysis from src.ai.run_analysis.
"""

from src.ai.run_analysis import run_ai_analysis


def analyze(payload: dict) -> dict:
    return run_ai_analysis(payload)
