from __future__ import annotations

import json
import re
from dataclasses import dataclass


VALIDATOR_VERSION = "v1.2.5"

CATEGORY_TO_STATE = {
    "emerging_rotations": "EMERGING",
    "accelerating_rotations": "ACCELERATING",
    "persistent_leaders": "PERSISTENT_LEADER",
    "reaccelerating_rotations": "REACCELERATING",
    "weakening_rotations": "WEAKENING",
    "rotation_out": "ROTATION_OUT",
}

FINDING_CATEGORIES = [
    "emerging_rotations",
    "accelerating_rotations",
    "persistent_leaders",
    "reaccelerating_rotations",
    "weakening_rotations",
    "rotation_out",
    "geographic_rotations",
    "pair_relationships",
]

PROHIBITED_LANGUAGE = [
    ("capital-flow language", re.compile(
        r"\bcapital\s+(?:inflows?|outflows?|flows?|shift(?:s|ed|ing)?|"
        r"rotat(?:e|es|ed|ing|ion|ions))\b", re.I
    )),
    ("money-flow language", re.compile(
        r"\bmoney\s+(?:flows?|flowing|moves?|moving)\s+(?:into|out\s+of)\b", re.I
    )),
    ("fund-flow language", re.compile(
        r"\bfund\s+(?:inflows?|outflows?|flows?)\b", re.I
    )),
    ("institutional transaction claim", re.compile(
        r"\binstitutional\s+(?:buying|selling|inflows?|outflows?)\b", re.I
    )),
]

NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
EVIDENCE_TERM_RE = re.compile(
    r"\b(?:score|bar|relative strength|rs20|rs63|spread|cmf|streak|rank|"
    r"percent|percentage|leader zone|pair)\b",
    re.I,
)


class AIOutputValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        preview = "; ".join(errors[:8])
        if len(errors) > 8:
            preview += f"; ... +{len(errors) - 8} more"
        super().__init__(f"AI output failed deterministic validation: {preview}")


@dataclass
class PayloadIndex:
    known_tickers: set[str]
    states: dict[str, str]
    score_modes: dict[str, str]
    paired_tickers: dict[str, str]
    pair_types: dict[str, str]
    benchmark_tickers: set[str]
    supplied_text: str


def _norm_ticker(value) -> str:
    return str(value or "").strip().upper()


def _index_payload(payload: dict) -> PayloadIndex:
    known: set[str] = set()
    states: dict[str, str] = {}
    score_modes: dict[str, str] = {}
    paired: dict[str, str] = {}
    pair_types: dict[str, str] = {}
    benchmark_tickers: set[str] = set()

    for item in payload.get("benchmark_context", []) or []:
        ticker = _norm_ticker(item.get("ticker"))
        if ticker:
            benchmark_tickers.add(ticker)
            known.add(ticker)

    def visit(node):
        if isinstance(node, dict):
            ticker = _norm_ticker(node.get("ticker"))
            if ticker:
                known.add(ticker)

                state = str(node.get("state") or "").strip().upper()
                if state:
                    states[ticker] = state

                mode = str(node.get("score_mode") or "").strip().upper()
                if mode:
                    score_modes[ticker] = mode

                paired = _norm_ticker(node.get("paired_ticker"))
                if paired:
                    paired_tickers[ticker] = paired
                    known.add(paired)

                pair_type = str(node.get("pair_type") or "").strip().upper()
                if pair_type:
                    pair_types[ticker] = pair_type

            for key in ("primary_benchmark", "parent_benchmark", "paired_ticker"):
                value = _norm_ticker(node.get(key))
                if value:
                    known.add(value)

            for value in node.values():
                visit(value)

        elif isinstance(node, list):
            for value in node:
                visit(value)

    paired_tickers = paired
    visit(payload)

    return PayloadIndex(
        known_tickers=known,
        states=states,
        score_modes=score_modes,
        paired_tickers=paired_tickers,
        pair_types=pair_types,
        benchmark_tickers=benchmark_tickers,
        supplied_text=json.dumps(payload, ensure_ascii=False).lower(),
    )


def validate_analysis_against_payload(analysis: dict, payload: dict) -> dict:
    idx = _index_payload(payload)
    errors: list[str] = []
    prose = json.dumps(analysis, ensure_ascii=False)

    for label, pattern in PROHIBITED_LANGUAGE:
        match = pattern.search(prose)
        if match:
            errors.append(f"{label}: prohibited phrase '{match.group(0)}'")

    if re.search(r"\bmega[- ]?cap\b", prose, re.I):
        if not re.search(r"\bmega[- ]?cap\b", idx.supplied_text, re.I):
            errors.append(
                "unsupported mega-cap characterization: source payload does not use that label"
            )

    for category in FINDING_CATEGORIES:
        findings = analysis.get(category, []) or []
        required_state = CATEGORY_TO_STATE.get(category)

        for pos, finding in enumerate(findings):
            prefix = f"{category}[{pos}]"
            ticker = _norm_ticker(finding.get("ticker"))

            if not ticker:
                errors.append(f"{prefix}: missing ticker")
                continue

            if ticker not in idx.known_tickers:
                errors.append(f"{prefix}: ticker {ticker} was not supplied")

            for related in finding.get("related_tickers", []) or []:
                related_ticker = _norm_ticker(related)
                if related_ticker and related_ticker not in idx.known_tickers:
                    errors.append(
                        f"{prefix}: related ticker {related_ticker} was not supplied"
                    )

            if required_state:
                actual_state = idx.states.get(ticker)
                if not actual_state:
                    errors.append(
                        f"{prefix}: deterministic state for {ticker} is unavailable"
                    )
                elif actual_state != required_state:
                    errors.append(
                        f"{prefix}: {ticker} is {actual_state}, not {required_state}"
                    )

            title = str(finding.get("title") or "")
            explanation = str(finding.get("explanation") or "")
            text = f"{title} {explanation}"

            if not NUMERIC_RE.search(explanation):
                errors.append(f"{prefix}: explanation lacks numeric evidence")
            if not EVIDENCE_TERM_RE.search(explanation):
                errors.append(
                    f"{prefix}: explanation does not name a quantitative metric/horizon"
                )

            mode = idx.score_modes.get(ticker)
            if category == "pair_relationships" and mode != "PAIR":
                errors.append(f"{prefix}: {ticker} is not a PAIR signal")

            if mode == "PAIR":
                paired = idx.paired_tickers.get(ticker)

                if paired and not re.search(
                    rf"\b{re.escape(paired)}\b", text, re.I
                ):
                    errors.append(
                        f"{prefix}: PAIR finding for {ticker} must name paired ticker {paired}"
                    )

                if re.search(r"\b(?:rank|ranked|peer[- ]?group)\b", text, re.I):
                    errors.append(
                        f"{prefix}: PAIR finding for {ticker} must not use peer-group rank language"
                    )

                paired_is_benchmark = paired in idx.benchmark_tickers if paired else False
                pair_type = idx.pair_types.get(ticker, "")
                if (
                    "BENCHMARK" not in pair_type
                    and not paired_is_benchmark
                    and re.search(r"\bbenchmark\b", text, re.I)
                ):
                    errors.append(
                        f"{prefix}: PAIR finding for {ticker} uses generic benchmark language "
                        f"but paired ticker is {paired}"
                    )

    for pos, ticker_value in enumerate(
        analysis.get("dashboard_focus_tickers", []) or []
    ):
        ticker = _norm_ticker(ticker_value)
        if ticker and ticker not in idx.known_tickers:
            errors.append(
                f"dashboard_focus_tickers[{pos}]: {ticker} was not supplied"
            )


    # Horizon-language validation is applied to analytical model-authored fields.
    # methodology_note is canonicalized by Python before validation.
    horizon_fields = [
        ("headline", str(analysis.get("headline") or "")),
        ("market_regime", str(analysis.get("market_regime") or "")),
        ("executive_summary", str(analysis.get("executive_summary") or "")),
    ]
    for category in FINDING_CATEGORIES:
        for pos, finding in enumerate(analysis.get(category, []) or []):
            horizon_fields.append(
                (f"{category}[{pos}].title", str(finding.get("title") or ""))
            )
            horizon_fields.append(
                (
                    f"{category}[{pos}].explanation",
                    str(finding.get("explanation") or ""),
                )
            )
    for pos, value in enumerate(analysis.get("cross_market_confirmations", []) or []):
        horizon_fields.append((f"cross_market_confirmations[{pos}]", str(value)))
    for pos, value in enumerate(analysis.get("risks_or_conflicts", []) or []):
        horizon_fields.append((f"risks_or_conflicts[{pos}]", str(value)))

    bad_horizon = re.compile(
        r"\b5[- ]?(?:bar|day|trading[- ]?bar)?\s+relative[- ]?strength\b",
        re.I,
    )
    for field_path, field_text in horizon_fields:
        if bad_horizon.search(field_text):
            errors.append(
                f"{field_path}: 5-bar observations must be described as score changes; "
                "relative-strength evidence is limited to 20/63 bars"
            )


    if errors:
        raise AIOutputValidationError(errors)

    return {
        "status": "passed",
        "validator_version": VALIDATOR_VERSION,
        "checks": {
            "ticker_grounding": True,
            "state_category_consistency": True,
            "pair_semantics": True,
            "language_guardrails": True,
            "numeric_evidence": True,
            "horizon_precision": True,
        },
    }
