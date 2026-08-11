"""
Claude provider placeholder.

Version 1.2 establishes the provider interface but intentionally does not
activate Claude yet. A future patch will implement this adapter against
Anthropic's Messages API structured outputs while preserving the same
AIAnalysis schema and input payload used by Gemini.
"""

PROVIDER_NAME = "claude"


def analyze(payload: dict, correction_instructions: str | None = None):
    raise NotImplementedError(
        "Claude adapter is reserved for the next provider integration patch."
    )
