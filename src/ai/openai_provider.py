"""
OpenAI provider placeholder.

Version 1.2 establishes the provider interface but intentionally does not
activate OpenAI yet. A future patch will implement this adapter with structured
outputs while preserving the same AIAnalysis schema and input payload used by
the other providers.
"""

PROVIDER_NAME = "openai"


def analyze(payload: dict):
    raise NotImplementedError(
        "OpenAI adapter is reserved for a future provider integration patch."
    )
