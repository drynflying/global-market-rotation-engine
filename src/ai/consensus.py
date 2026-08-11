from __future__ import annotations

from collections import Counter, defaultdict

from src.ai.schema import CATEGORY_FIELDS


def _assessment_name(category_field: str) -> str:
    return {
        "emerging_rotations": "EMERGING",
        "accelerating_rotations": "ACCELERATING",
        "persistent_leaders": "PERSISTENT_LEADER",
        "reaccelerating_rotations": "REACCELERATING",
        "weakening_rotations": "WEAKENING",
        "rotation_out": "ROTATION_OUT",
        "geographic_rotations": "GEOGRAPHIC",
        "pair_relationships": "PAIR",
    }[category_field]


def build_consensus(provider_results: dict[str, dict]) -> dict:
    successful = {
        name: result
        for name, result in provider_results.items()
        if result.get("status") == "success" and result.get("analysis")
    }
    provider_count = len(successful)

    votes = defaultdict(list)
    for provider, result in successful.items():
        analysis = result["analysis"]
        for category in CATEGORY_FIELDS:
            assessment = _assessment_name(category)
            for item in analysis.get(category, []):
                ticker = str(item.get("ticker", "")).upper().strip()
                if ticker:
                    votes[ticker].append(
                        {
                            "provider": provider,
                            "assessment": assessment,
                            "title": item.get("title", ""),
                            "confidence": item.get("confidence", ""),
                        }
                    )

    items = []
    for ticker, ticker_votes in votes.items():
        counts = Counter(v["assessment"] for v in ticker_votes)
        dominant, dominant_count = counts.most_common(1)[0]
        provider_names = sorted({v["provider"] for v in ticker_votes})
        conflict = len(counts) > 1
        items.append(
            {
                "ticker": ticker,
                "dominant_assessment": dominant,
                "agreement_count": dominant_count,
                "provider_count": provider_count,
                "agreement_ratio": (
                    round(dominant_count / provider_count, 3)
                    if provider_count else 0.0
                ),
                "providers_mentioning": provider_names,
                "assessment_counts": dict(counts),
                "conflict": conflict,
                "votes": ticker_votes,
            }
        )

    items.sort(
        key=lambda x: (
            x["agreement_count"],
            not x["conflict"],
            x["agreement_ratio"],
            x["ticker"],
        ),
        reverse=True,
    )

    return {
        "provider_count": provider_count,
        "mode": "multi_provider" if provider_count >= 2 else "single_provider",
        "items": items,
        "strong_consensus": [
            x for x in items
            if provider_count >= 2
            and x["agreement_count"] == provider_count
            and not x["conflict"]
        ][:12],
        "disagreements": [
            x for x in items
            if provider_count >= 2 and x["conflict"]
        ][:12],
    }
