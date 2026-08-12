from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.common import (
    WEEKLY_RECOMMENDATION_AI_DIR,
    WEEKLY_RECOMMENDATION_HISTORY_PATH,
    WEEKLY_RECOMMENDATION_LATEST_PATH,
    WEEKLY_RECOMMENDATION_OUTCOMES_PATH,
    WEEKLY_RECOMMENDATION_SPEC_PATH,
    WEEKLY_RECOMMENDATION_STATUS_PATH,
)


NEW_YORK = ZoneInfo("America/New_York")
BULL_STATES = {"PERSISTENT_LEADER", "REACCELERATING", "ACCELERATING", "EMERGING"}
BEAR_STATES = {"WEAKENING", "ROTATION_OUT"}
SELECTABLE_EVIDENCE_POINTS = 8
MAX_CONFLICTS_FOR_SELECTABLE = 1
MAX_FINALISTS_PER_DIRECTION = 3
MAX_ACTIONS = 2


class WeeklyCommitteeAction(BaseModel):
    ticker: str = Field(description="Ticker from the supplied selectable finalist list.")
    direction: Literal["FAVOR", "AVOID"]
    evidence_ids: list[str] = Field(min_length=2, max_length=4)
    acknowledged_conflict_ids: list[str] = Field(default_factory=list, max_length=2)
    rationale: str = Field(
        description="Brief qualitative committee rationale. Do not include numbers or unsupported facts."
    )


class WeeklyCommitteeDecision(BaseModel):
    recommendation_date: str
    committee_summary: str = Field(
        description="Brief qualitative summary. Do not include numbers or unsupported facts."
    )
    actions: list[WeeklyCommitteeAction] = Field(default_factory=list, max_length=MAX_ACTIONS)
    no_action_reason: str | None = None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_spec() -> dict:
    if not WEEKLY_RECOMMENDATION_SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Missing weekly recommendation spec: {WEEKLY_RECOMMENDATION_SPEC_PATH}"
        )
    return json.loads(WEEKLY_RECOMMENDATION_SPEC_PATH.read_text(encoding="utf-8"))


def _requested_providers() -> list[str]:
    raw = os.getenv("AI_PROVIDERS", "").strip() or os.getenv("AI_PROVIDER", "").strip()
    providers: list[str] = []
    for item in raw.split(","):
        name = item.strip().lower()
        if name and name not in providers:
            providers.append(name)
    return providers


def _latest_market_date(latest: pd.DataFrame) -> pd.Timestamp | None:
    if latest.empty or "date" not in latest.columns:
        return None
    value = pd.to_datetime(latest["date"], errors="coerce").max()
    if pd.isna(value):
        return None
    if getattr(value, "tzinfo", None) is not None:
        value = value.tz_localize(None)
    return pd.Timestamp(value).normalize()


def _week_ending_friday(market_date: pd.Timestamp) -> pd.Timestamp:
    return (market_date - pd.Timedelta(days=market_date.weekday()) + pd.Timedelta(days=4)).normalize()


def _capture_gate(
    market_date: pd.Timestamp,
    now_et: datetime | None = None,
    force: bool | None = None,
) -> tuple[bool, str, pd.Timestamp]:
    week_ending = _week_ending_friday(market_date)
    if force is None:
        force = os.getenv("WEEKLY_RECOMMENDATION_FORCE", "").strip().lower() in {
            "1", "true", "yes", "y"
        }
    if force:
        return True, "forced_for_validation", week_ending

    now_et = now_et or datetime.now(NEW_YORK)
    local_date = pd.Timestamp(now_et.date())

    # Normal case: Friday after a short post-close buffer, with Friday market data.
    if now_et.weekday() == 4 and now_et.time() >= time(16, 15):
        if market_date == local_date:
            return True, "friday_after_close", week_ending
        # Friday market holiday: use Thursday's close if it is the immediately prior day.
        if market_date == local_date - pd.Timedelta(days=1) and market_date.weekday() == 3:
            return True, "friday_market_holiday_prior_close", week_ending
        return False, "friday_after_close_but_market_data_not_current", week_ending

    # Recovery case: allow a weekend rerun if Friday data exists and no recommendation was captured yet.
    if now_et.weekday() in {5, 6} and market_date.weekday() == 4:
        if 0 <= (local_date - market_date).days <= 2:
            return True, "weekend_recovery_for_friday_close", week_ending

    if now_et.weekday() == 4 and now_et.time() < time(16, 15):
        return False, "friday_before_post_close_buffer", week_ending
    return False, "not_weekly_capture_window", week_ending


def _direction_stability(history: pd.DataFrame, market_date: pd.Timestamp) -> pd.DataFrame:
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h[h["date"].notna() & (h["date"] <= market_date)].copy()
    rank_ok = h["rank_eligible"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    mode_ok = h["score_mode"].astype(str).str.upper().eq("CROSS_SECTIONAL")
    h = h[rank_ok & mode_ok].copy()
    h = h.sort_values(["ticker", "date"])
    h["favor_direction"] = h["rotation_state_confirmed"].isin(BULL_STATES)
    h["avoid_direction"] = h["rotation_state_confirmed"].isin(BEAR_STATES)
    h["favor_stability_5"] = h.groupby("ticker")["favor_direction"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    h["avoid_stability_5"] = h.groupby("ticker")["avoid_direction"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    return (
        h.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "favor_stability_5", "avoid_stability_5"]]
    )


def _attention_index(attention: dict | None) -> dict[str, list[tuple[str, str]]]:
    attention = attention or {}
    indexed: dict[str, list[tuple[str, str]]] = {}

    def add(ticker: str | None, conflict_id: str, text: str) -> None:
        t = str(ticker or "").strip().upper()
        if t:
            indexed.setdefault(t, []).append((conflict_id, text))

    for item in attention.get("sector_divergences", []) or []:
        sector = str(item.get("sector") or "peer group")
        improver = str(item.get("improver") or "").upper()
        deteriorator = str(item.get("deteriorator") or "").upper()
        if improver and deteriorator:
            text = f"Deterministic sector divergence is present in {sector}: {improver} versus {deteriorator}."
            add(improver, "C_SECTOR_DIVERGENCE", text)
            add(deteriorator, "C_SECTOR_DIVERGENCE", text)
            other = item.get("other_weak") or {}
            add(other.get("ticker"), "C_SECTOR_DIVERGENCE", text)

    for item in attention.get("extreme_cmf20", []) or []:
        t = str(item.get("ticker") or "").upper()
        if t:
            add(t, "C_EXTREME_CMF", f"{t} has an extreme CMF reading in the deterministic attention engine.")

    return indexed


def _safe_float(value, default: float = math.nan) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _build_candidate_record(
    row: pd.Series,
    direction: str,
    stability: float,
    attention_conflicts: list[tuple[str, str]],
) -> dict:
    ticker = str(row.get("ticker") or "").upper()
    state = str(row.get("rotation_state_confirmed") or row.get("rotation_state") or "").upper()
    age = _safe_int(row.get("confirmed_state_age"), 0)
    group_pct = _safe_float(row.get("group_percentile"))
    score = _safe_float(row.get("rotation_score"))
    rank = _safe_float(row.get("group_rank"))
    group_size = _safe_float(row.get("group_size"))
    rs20 = _safe_float(row.get("signal_rs20"))
    rs63 = _safe_float(row.get("signal_rs63"))
    score_change_5 = _safe_float(row.get("score_change_5"))
    cmf20 = _safe_float(row.get("cmf20"))
    rel_dvol = _safe_float(row.get("relative_dollar_volume"))
    pending_days = _safe_int(row.get("pending_state_days"), 0)
    pending_state = str(row.get("pending_rotation_state") or "").strip().upper()

    if direction == "FAVOR":
        points = 2 if state in {"PERSISTENT_LEADER", "REACCELERATING"} else 1
        points += int(stability >= 0.80)
        points += int(age >= 10)
        points += int(group_pct >= 80)
        points += int(score_change_5 > 0)
        points += int(cmf20 > 0)
        points += int(pending_days == 0)
        points += int(rs20 > 0)
        directional_conflicts = [
            ("C_PENDING_STATE", f"A pending confirmed-state change toward {pending_state or 'another state'} is in progress.")
            if pending_days > 0 else None,
            ("C_SCORE_REVERSAL", "The five-observation rotation-score change is negative despite a bullish weekly candidate.")
            if score_change_5 < 0 else None,
            ("C_CMF_DIRECTION", "CMF is negative despite a bullish weekly candidate.")
            if cmf20 < 0 else None,
            ("C_MIXED_HORIZON_RS", "Short- and medium-horizon relative strength point in opposite directions.")
            if rs20 < 0 < rs63 else None,
        ]
    else:
        points = 2 if state == "ROTATION_OUT" else 1
        points += int(stability >= 0.80)
        points += int(age >= 10)
        points += int(group_pct <= 20)
        points += int(score_change_5 < 0)
        points += int(cmf20 < 0)
        points += int(pending_days == 0)
        points += int(rs20 < 0)
        directional_conflicts = [
            ("C_PENDING_STATE", f"A pending confirmed-state change toward {pending_state or 'another state'} is in progress.")
            if pending_days > 0 else None,
            ("C_SCORE_REVERSAL", "The five-observation rotation-score change is positive despite a defensive weekly candidate.")
            if score_change_5 > 0 else None,
            ("C_CMF_DIRECTION", "CMF is positive despite a defensive weekly candidate.")
            if cmf20 > 0 else None,
            ("C_MIXED_HORIZON_RS", "Short- and medium-horizon relative strength point in opposite directions.")
            if rs20 > 0 > rs63 else None,
        ]

    conflicts = [item for item in directional_conflicts if item is not None]
    # De-duplicate attention conflict IDs so the validator has stable identifiers.
    seen_conflict_ids = {cid for cid, _ in conflicts}
    for cid, text in attention_conflicts:
        if cid not in seen_conflict_ids:
            conflicts.append((cid, text))
            seen_conflict_ids.add(cid)

    evidence = {
        "E_STATE": f"Confirmed state is {state} with age {age} observations and five-observation directional stability {stability * 100:.0f}%.",
        "E_PEER": f"Rotation score is {score:.2f}; peer-group percentile is {group_pct:.2f}; rank is {rank:.0f} of {group_size:.0f}.",
        "E_RS63": f"63-bar benchmark-relative strength is {rs63 * 100:+.2f} percentage points.",
        "E_SCORE_WEEK": f"Rotation score changed {score_change_5:+.2f} points over the latest five-observation comparison.",
        "E_FLOW": f"CMF20 is {cmf20:+.3f}; relative dollar volume is {rel_dvol:.2f}.",
        "E_RS20": f"20-bar benchmark-relative strength is {rs20 * 100:+.2f} percentage points.",
    }
    conflict_map = {cid: text for cid, text in conflicts}
    selectable = points >= SELECTABLE_EVIDENCE_POINTS and len(conflict_map) <= MAX_CONFLICTS_FOR_SELECTABLE

    return {
        "ticker": ticker,
        "exposure": str(row.get("exposure") or ticker),
        "rotation_group": str(row.get("rotation_group") or ""),
        "primary_benchmark": str(row.get("primary_benchmark") or "").upper(),
        "direction": direction,
        "confirmed_state": state,
        "confirmed_state_age": age,
        "stability_5": round(float(stability), 4),
        "rotation_score": score,
        "group_percentile": group_pct,
        "group_rank": rank,
        "group_size": group_size,
        "signal_rs20": rs20,
        "signal_rs63": rs63,
        "score_change_5": score_change_5,
        "cmf20": cmf20,
        "relative_dollar_volume": rel_dvol,
        "evidence_points": int(points),
        "evidence_confidence": "HIGH" if selectable else "MODERATE",
        "selectable": bool(selectable),
        "evidence": evidence,
        "conflicts": conflict_map,
    }


def build_weekly_finalists(
    latest: pd.DataFrame,
    history: pd.DataFrame,
    market_date: pd.Timestamp,
    deterministic_attention: dict | None = None,
) -> dict:
    snapshot = latest.copy()
    snapshot["date"] = pd.to_datetime(snapshot["date"], errors="coerce")
    snapshot = snapshot[snapshot["date"].eq(market_date)].copy()
    rank_ok = snapshot["rank_eligible"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    mode_ok = snapshot["score_mode"].astype(str).str.upper().eq("CROSS_SECTIONAL")
    score_ok = pd.to_numeric(snapshot["rotation_score"], errors="coerce").notna()
    snapshot = snapshot[rank_ok & mode_ok & score_ok].copy()

    stability = _direction_stability(history, market_date)
    snapshot = snapshot.merge(stability, on="ticker", how="left", validate="one_to_one")
    attention_idx = _attention_index(deterministic_attention)

    candidates: list[dict] = []
    for _, row in snapshot.iterrows():
        state = str(row.get("rotation_state_confirmed") or row.get("rotation_state") or "").upper()
        age = _safe_int(row.get("confirmed_state_age"), 0)
        rs63 = _safe_float(row.get("signal_rs63"))
        if state in BULL_STATES:
            stab = _safe_float(row.get("favor_stability_5"), 0.0)
            if age >= 5 and stab >= 0.60 and rs63 > 0:
                candidates.append(
                    _build_candidate_record(row, "FAVOR", stab, attention_idx.get(str(row["ticker"]).upper(), []))
                )
        elif state in BEAR_STATES:
            stab = _safe_float(row.get("avoid_stability_5"), 0.0)
            if age >= 5 and stab >= 0.60 and rs63 < 0:
                candidates.append(
                    _build_candidate_record(row, "AVOID", stab, attention_idx.get(str(row["ticker"]).upper(), []))
                )

    finalists: list[dict] = []
    for direction in ["FAVOR", "AVOID"]:
        group = [c for c in candidates if c["direction"] == direction]
        if direction == "FAVOR":
            group.sort(
                key=lambda c: (
                    c["selectable"], c["evidence_points"], c["rotation_score"], c["group_percentile"]
                ),
                reverse=True,
            )
        else:
            group.sort(
                key=lambda c: (
                    0 if c["selectable"] else 1,
                    -c["evidence_points"],
                    c["rotation_score"],
                    c["group_percentile"],
                )
            )
        finalists.extend(group[:MAX_FINALISTS_PER_DIRECTION])

    selectable = [c for c in finalists if c["selectable"]]
    return {
        "method": "WEEKLY_FINALIST_EVIDENCE_V1",
        "candidate_count": len(candidates),
        "finalist_count": len(finalists),
        "selectable_count": len(selectable),
        "finalists": finalists,
        "selectable_finalists": selectable,
    }


def _previous_week_actions() -> list[dict]:
    if not WEEKLY_RECOMMENDATION_HISTORY_PATH.exists():
        return []
    df = pd.read_csv(WEEKLY_RECOMMENDATION_HISTORY_PATH, dtype=str, keep_default_na=False)
    if df.empty or "recommendation_date" not in df.columns:
        return []
    df = df[df.get("action_type", "").ne("NO_ACTION")].copy()
    if df.empty:
        return []
    dates = pd.to_datetime(df["recommendation_date"], errors="coerce")
    last_date = dates.max()
    if pd.isna(last_date):
        return []
    rows = df[dates.eq(last_date)]
    return rows[[c for c in ["ticker", "direction", "evidence_confidence"] if c in rows.columns]].to_dict("records")


def _build_prompt(
    market_date: str,
    selectable: list[dict],
    previous_actions: list[dict],
    correction: str | None = None,
) -> str:
    payload = {
        "recommendation_date": market_date,
        "selectable_finalists": selectable,
        "previous_week_actions": previous_actions,
    }
    instructions = f"""
You are the skeptical weekly investment committee for a market-rotation dashboard.

This is a prospective SHADOW recommendation. The deterministic engine has already decided which finalists
are selectable. You may REJECT finalists, but you may not introduce another ticker or change its direction.

Return zero, one, or at most two actions. Never select more than one FAVOR and never select more than one AVOID.
A no-action week is valid and preferred when the evidence is materially conflicted.

For every selected action:
- choose only a ticker in selectable_finalists;
- copy its supplied direction;
- choose two to four supplied evidence IDs;
- if the finalist has any supplied conflicts, acknowledge at least one supplied conflict ID;
- write a short qualitative rationale that contains NO NUMBERS and introduces no new factual claims.

The committee_summary and no_action_reason must also contain no numeric claims.
Do not claim a probability of profit, expected return, guaranteed outcome, or calibrated accuracy.
Do not use the phrase 'confidence percentage'. The UI's Evidence Confidence label is deterministic and is not yours to change.

Recommendation date: {market_date}

DATA:
{json.dumps(payload, indent=2)}
""".strip()
    if correction:
        instructions += "\n\nCORRECTION REQUIRED:\n" + correction.strip()
    return instructions


def _call_gemini(
    market_date: str,
    selectable: list[dict],
    previous_actions: list[dict],
    correction: str | None = None,
) -> tuple[dict, str]:
    if "gemini" not in _requested_providers():
        raise RuntimeError("Gemini is not enabled in AI_PROVIDERS/AI_PROVIDER")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY secret is missing")

    from google import genai

    model = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.6-flash"
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=_build_prompt(market_date, selectable, previous_actions, correction),
        config={
            "response_mime_type": "application/json",
            "response_schema": WeeklyCommitteeDecision,
        },
    )
    if not getattr(response, "text", None):
        raise RuntimeError("Gemini returned no weekly recommendation text")
    decision = WeeklyCommitteeDecision.model_validate_json(response.text).model_dump()
    return decision, model


def _validate_ai_decision(decision: dict, market_date: str, selectable: list[dict]) -> list[str]:
    errors: list[str] = []
    if str(decision.get("recommendation_date") or "") != market_date:
        errors.append("recommendation_date must equal the supplied market date")

    actions = decision.get("actions", []) or []
    if len(actions) > MAX_ACTIONS:
        errors.append("at most two weekly actions are allowed")

    allowed = {(c["ticker"], c["direction"]): c for c in selectable}
    seen_tickers: set[str] = set()
    seen_directions: set[str] = set()

    for pos, action in enumerate(actions):
        ticker = str(action.get("ticker") or "").upper()
        direction = str(action.get("direction") or "").upper()
        key = (ticker, direction)
        if key not in allowed:
            errors.append(f"actions[{pos}] is not a selectable finalist")
            continue
        if ticker in seen_tickers:
            errors.append(f"actions[{pos}] duplicates ticker {ticker}")
        if direction in seen_directions:
            errors.append(f"actions[{pos}] duplicates direction {direction}")
        seen_tickers.add(ticker)
        seen_directions.add(direction)

        candidate = allowed[key]
        evidence_ids = list(action.get("evidence_ids") or [])
        if not (2 <= len(evidence_ids) <= 4):
            errors.append(f"actions[{pos}] must choose two to four evidence IDs")
        unknown_evidence = sorted(set(evidence_ids) - set(candidate["evidence"]))
        if unknown_evidence:
            errors.append(f"actions[{pos}] uses unknown evidence IDs: {unknown_evidence}")

        conflict_ids = list(action.get("acknowledged_conflict_ids") or [])
        unknown_conflicts = sorted(set(conflict_ids) - set(candidate["conflicts"]))
        if unknown_conflicts:
            errors.append(f"actions[{pos}] uses unknown conflict IDs: {unknown_conflicts}")
        if candidate["conflicts"] and not conflict_ids:
            errors.append(f"actions[{pos}] must acknowledge at least one supplied conflict")

        rationale = str(action.get("rationale") or "")
        if re.search(r"\d", rationale):
            errors.append(f"actions[{pos}] rationale must not contain numeric claims")

    summary = str(decision.get("committee_summary") or "")
    if re.search(r"\d", summary):
        errors.append("committee_summary must not contain numeric claims")
    no_action_reason = str(decision.get("no_action_reason") or "")
    if re.search(r"\d", no_action_reason):
        errors.append("no_action_reason must not contain numeric claims")
    if not actions and not no_action_reason.strip():
        errors.append("no_action_reason is required when no actions are selected")

    return errors


def _run_committee(market_date: str, selectable: list[dict]) -> dict:
    previous_actions = _previous_week_actions()
    decision, model = _call_gemini(market_date, selectable, previous_actions)
    errors = _validate_ai_decision(decision, market_date, selectable)
    retry_used = False
    first_errors: list[str] = []
    if errors:
        retry_used = True
        first_errors = errors
        correction = "Revise the response to fix only these validation issues:\n" + "\n".join(
            f"- {error}" for error in errors
        )
        decision, model = _call_gemini(market_date, selectable, previous_actions, correction)
        errors = _validate_ai_decision(decision, market_date, selectable)
    if errors:
        raise ValueError("Weekly Gemini decision failed validation: " + "; ".join(errors))
    return {
        "provider": "gemini",
        "model": model,
        "retry_used": retry_used,
        "first_attempt_errors": first_errors,
        "decision": decision,
    }


def _existing_recommendation_dates() -> set[str]:
    if not WEEKLY_RECOMMENDATION_HISTORY_PATH.exists():
        return set()
    df = pd.read_csv(WEEKLY_RECOMMENDATION_HISTORY_PATH, dtype=str, keep_default_na=False)
    if df.empty or "recommendation_date" not in df.columns:
        return set()
    return {str(x) for x in df["recommendation_date"] if str(x).strip()}


def _prior_action_keys_before(recommendation_date: str) -> set[tuple[str, str]]:
    if not WEEKLY_RECOMMENDATION_HISTORY_PATH.exists():
        return set()
    df = pd.read_csv(WEEKLY_RECOMMENDATION_HISTORY_PATH, dtype=str, keep_default_na=False)
    if df.empty:
        return set()
    d = pd.to_datetime(df.get("recommendation_date"), errors="coerce")
    current = pd.Timestamp(recommendation_date)
    prior_dates = d[d < current]
    if prior_dates.empty:
        return set()
    last = prior_dates.max()
    rows = df[d.eq(last) & df.get("action_type", "").ne("NO_ACTION")]
    return {
        (str(r.get("ticker") or "").upper(), str(r.get("direction") or "").upper())
        for _, r in rows.iterrows()
    }


def _append_history(rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    if WEEKLY_RECOMMENDATION_HISTORY_PATH.exists():
        existing = pd.read_csv(WEEKLY_RECOMMENDATION_HISTORY_PATH, dtype=str, keep_default_na=False)
        combined = pd.concat([existing, new], ignore_index=True, sort=False)
    else:
        combined = new
    combined.to_csv(WEEKLY_RECOMMENDATION_HISTORY_PATH, index=False)


def _record_weekly_recommendation(
    market_date: pd.Timestamp,
    week_ending: pd.Timestamp,
    finalist_bundle: dict,
    spec: dict,
) -> dict:
    recommendation_date = market_date.strftime("%Y-%m-%d")
    if recommendation_date in _existing_recommendation_dates():
        return {
            "status": "already_recorded",
            "recommendation_date": recommendation_date,
            "appended_rows": 0,
            "action_count": 0,
            "selectable_finalists": finalist_bundle.get("selectable_count", 0),
        }

    selectable = finalist_bundle.get("selectable_finalists", [])
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    model_version = str(spec.get("model_version") or "WEEKLY_RECOMMENDATION_SHADOW_V1")

    if not selectable:
        committee = {
            "provider": "deterministic_no_finalist",
            "model": None,
            "retry_used": False,
            "first_attempt_errors": [],
            "decision": {
                "recommendation_date": recommendation_date,
                "committee_summary": "No selectable high-evidence finalist passed the frozen weekly gates.",
                "actions": [],
                "no_action_reason": "No selectable high-evidence finalist passed the frozen weekly gates.",
            },
        }
    else:
        committee = _run_committee(recommendation_date, selectable)

    decision = committee["decision"]
    finalist_index = {
        (c["ticker"], c["direction"]): c for c in finalist_bundle.get("finalists", [])
    }
    prior_keys = _prior_action_keys_before(recommendation_date)
    rows: list[dict] = []

    actions = decision.get("actions", []) or []
    if actions:
        for slot, action in enumerate(actions, start=1):
            ticker = str(action["ticker"]).upper()
            direction = str(action["direction"]).upper()
            candidate = finalist_index[(ticker, direction)]
            evidence_ids = list(action.get("evidence_ids") or [])
            conflict_ids = list(action.get("acknowledged_conflict_ids") or [])
            action_key = f"{model_version}|{recommendation_date}|{ticker}|{direction}"
            action_id = hashlib.sha256(action_key.encode("utf-8")).hexdigest()[:24]
            rows.append({
                "action_id": action_id,
                "model_version": model_version,
                "recommendation_date": recommendation_date,
                "week_ending_date": week_ending.strftime("%Y-%m-%d"),
                "market_data_as_of": recommendation_date,
                "created_at_utc": created_at,
                "provider": committee["provider"],
                "provider_model": committee.get("model") or "",
                "action_type": "RECOMMENDATION",
                "action_slot": slot,
                "ticker": ticker,
                "exposure": candidate["exposure"],
                "rotation_group": candidate["rotation_group"],
                "primary_benchmark": candidate["primary_benchmark"],
                "direction": direction,
                "recommendation_status": "REAFFIRMED" if (ticker, direction) in prior_keys else "NEW",
                "confirmed_state": candidate["confirmed_state"],
                "confirmed_state_age": candidate["confirmed_state_age"],
                "evidence_confidence": candidate["evidence_confidence"],
                "evidence_points": candidate["evidence_points"],
                "rotation_score": candidate["rotation_score"],
                "group_percentile": candidate["group_percentile"],
                "signal_rs20": candidate["signal_rs20"],
                "signal_rs63": candidate["signal_rs63"],
                "score_change_5": candidate["score_change_5"],
                "cmf20": candidate["cmf20"],
                "relative_dollar_volume": candidate["relative_dollar_volume"],
                "evidence_ids_json": json.dumps(evidence_ids),
                "evidence_text_json": json.dumps([candidate["evidence"][eid] for eid in evidence_ids]),
                "conflict_ids_json": json.dumps(conflict_ids),
                "conflict_text_json": json.dumps([candidate["conflicts"][cid] for cid in conflict_ids]),
                "rationale": str(action.get("rationale") or ""),
                "committee_summary": str(decision.get("committee_summary") or ""),
                "no_action_reason": "",
            })
    else:
        action_key = f"{model_version}|{recommendation_date}|NO_ACTION"
        rows.append({
            "action_id": hashlib.sha256(action_key.encode("utf-8")).hexdigest()[:24],
            "model_version": model_version,
            "recommendation_date": recommendation_date,
            "week_ending_date": week_ending.strftime("%Y-%m-%d"),
            "market_data_as_of": recommendation_date,
            "created_at_utc": created_at,
            "provider": committee["provider"],
            "provider_model": committee.get("model") or "",
            "action_type": "NO_ACTION",
            "action_slot": 0,
            "ticker": "",
            "exposure": "",
            "rotation_group": "",
            "primary_benchmark": "",
            "direction": "",
            "recommendation_status": "NO_ACTION",
            "confirmed_state": "",
            "confirmed_state_age": "",
            "evidence_confidence": "",
            "evidence_points": "",
            "rotation_score": "",
            "group_percentile": "",
            "signal_rs20": "",
            "signal_rs63": "",
            "score_change_5": "",
            "cmf20": "",
            "relative_dollar_volume": "",
            "evidence_ids_json": "[]",
            "evidence_text_json": "[]",
            "conflict_ids_json": "[]",
            "conflict_text_json": "[]",
            "rationale": "",
            "committee_summary": str(decision.get("committee_summary") or ""),
            "no_action_reason": str(decision.get("no_action_reason") or ""),
        })

    _append_history(rows)

    latest_payload = {
        "status": "ok",
        "model_version": model_version,
        "recommendation_date": recommendation_date,
        "week_ending_date": week_ending.strftime("%Y-%m-%d"),
        "market_data_as_of": recommendation_date,
        "created_at_utc": created_at,
        "shadow_only": True,
        "provider": committee["provider"],
        "provider_model": committee.get("model"),
        "retry_used": committee.get("retry_used", False),
        "finalist_bundle": finalist_bundle,
        "decision": decision,
    }
    _write_json(WEEKLY_RECOMMENDATION_LATEST_PATH, latest_payload)
    _write_json(WEEKLY_RECOMMENDATION_AI_DIR / f"{recommendation_date}.json", latest_payload)

    return {
        "status": "recorded",
        "recommendation_date": recommendation_date,
        "appended_rows": len(rows),
        "action_count": len(actions),
        "selectable_finalists": finalist_bundle.get("selectable_count", 0),
        "provider": committee["provider"],
        "provider_model": committee.get("model"),
    }


def _relative_series(closes: pd.DataFrame, ticker: str, benchmark: str) -> pd.Series:
    if ticker not in closes.columns or benchmark not in closes.columns:
        return pd.Series(dtype=float)
    return (closes[ticker] / closes[benchmark]).dropna().sort_index()


def update_weekly_recommendation_outcomes(ohlcv: pd.DataFrame) -> dict:
    if not WEEKLY_RECOMMENDATION_HISTORY_PATH.exists():
        empty = pd.DataFrame()
        empty.to_csv(WEEKLY_RECOMMENDATION_OUTCOMES_PATH, index=False)
        return {"action_rows": 0, "matured_21": 0, "matured_63": 0, "matured_84": 0, "matured_126": 0}

    history = pd.read_csv(WEEKLY_RECOMMENDATION_HISTORY_PATH, dtype=str, keep_default_na=False)
    actions = history[history.get("action_type", "").eq("RECOMMENDATION")].copy()
    if actions.empty:
        pd.DataFrame().to_csv(WEEKLY_RECOMMENDATION_OUTCOMES_PATH, index=False)
        return {"action_rows": 0, "matured_21": 0, "matured_63": 0, "matured_84": 0, "matured_126": 0}

    data = ohlcv.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].dt.tz is not None:
        data["date"] = data["date"].dt.tz_localize(None)
    data["ticker"] = data["ticker"].astype(str).str.upper()
    closes = (
        data.dropna(subset=["date", "ticker", "close"])
        .pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
    )
    as_of = closes.index.max() if not closes.empty else pd.NaT

    rows: list[dict] = []
    for action in actions.itertuples(index=False):
        ticker = str(action.ticker).upper()
        direction = str(action.direction).upper()
        benchmark = ""
        # The primary benchmark is not stored in the weekly history; infer it from OHLCV-independent history
        # is impossible here, so persist it at action time if present. Older development rows may be blank.
        if hasattr(action, "primary_benchmark"):
            benchmark = str(action.primary_benchmark).upper()
        if not benchmark:
            # No silently guessed benchmark: leave outcomes unresolved.
            rows.append({
                "action_id": str(action.action_id),
                "model_version": str(action.model_version),
                "recommendation_date": str(action.recommendation_date),
                "ticker": ticker,
                "direction": direction,
                "primary_benchmark": "",
                "outcome_as_of_market_date": as_of.strftime("%Y-%m-%d") if pd.notna(as_of) else "",
            })
            continue

        relative = _relative_series(closes, ticker, benchmark)
        rec_date = pd.Timestamp(action.recommendation_date)
        if rec_date not in relative.index:
            rows.append({
                "action_id": str(action.action_id),
                "model_version": str(action.model_version),
                "recommendation_date": str(action.recommendation_date),
                "ticker": ticker,
                "direction": direction,
                "primary_benchmark": benchmark,
                "outcome_as_of_market_date": as_of.strftime("%Y-%m-%d") if pd.notna(as_of) else "",
            })
            continue

        pos = int(relative.index.get_loc(rec_date))
        entry = float(relative.iloc[pos])
        result = {
            "action_id": str(action.action_id),
            "model_version": str(action.model_version),
            "recommendation_date": str(action.recommendation_date),
            "ticker": ticker,
            "direction": direction,
            "primary_benchmark": benchmark,
            "outcome_as_of_market_date": as_of.strftime("%Y-%m-%d") if pd.notna(as_of) else "",
        }
        fwd: dict[int, float] = {}
        for horizon in [21, 63, 84, 126]:
            if pos + horizon < len(relative):
                value = float(relative.iloc[pos + horizon] / entry - 1.0)
                fwd[horizon] = value
                result[f"fwd_relative_return_{horizon}"] = value
                sign = 1.0 if direction == "FAVOR" else -1.0
                result[f"signed_return_{horizon}"] = sign * value
            else:
                result[f"fwd_relative_return_{horizon}"] = np.nan
                result[f"signed_return_{horizon}"] = np.nan

        if pos + 126 < len(relative):
            path = relative.iloc[pos : pos + 127].to_numpy(dtype=float) / entry - 1.0
            result["fwd_relative_mdd_126"] = float(np.nanmin(path))
            result["fwd_relative_mru_126"] = float(np.nanmax(path))
            band = float(np.nanmedian([fwd[63], fwd[84], fwd[126]]))
            result["band_median_relative_return"] = band
            if direction == "FAVOR":
                result["primary_success"] = bool(band >= 0.02 and result["fwd_relative_mdd_126"] >= -0.10)
            else:
                result["primary_success"] = bool(band <= -0.02 and result["fwd_relative_mru_126"] <= 0.10)
        else:
            result["fwd_relative_mdd_126"] = np.nan
            result["fwd_relative_mru_126"] = np.nan
            result["band_median_relative_return"] = np.nan
            result["primary_success"] = np.nan
        rows.append(result)

    outcomes = pd.DataFrame(rows)
    outcomes.to_csv(WEEKLY_RECOMMENDATION_OUTCOMES_PATH, index=False, float_format="%.8f")
    return {
        "action_rows": int(len(outcomes)),
        "matured_21": int(outcomes.get("fwd_relative_return_21", pd.Series(dtype=float)).notna().sum()),
        "matured_63": int(outcomes.get("fwd_relative_return_63", pd.Series(dtype=float)).notna().sum()),
        "matured_84": int(outcomes.get("fwd_relative_return_84", pd.Series(dtype=float)).notna().sum()),
        "matured_126": int(outcomes.get("fwd_relative_return_126", pd.Series(dtype=float)).notna().sum()),
    }


def run_weekly_recommendation_shadow(
    ohlcv: pd.DataFrame,
    latest: pd.DataFrame,
    history: pd.DataFrame,
    deterministic_attention: dict | None = None,
    now_et: datetime | None = None,
    force_capture: bool | None = None,
) -> dict:
    spec = _load_spec()
    market_date = _latest_market_date(latest)
    if market_date is None:
        status = {"status": "error", "error": "No latest market date available"}
        _write_json(WEEKLY_RECOMMENDATION_STATUS_PATH, status)
        return status

    capture, gate_reason, week_ending = _capture_gate(market_date, now_et=now_et, force=force_capture)
    capture_status: dict = {
        "status": "not_due",
        "recommendation_date": market_date.strftime("%Y-%m-%d"),
        "week_ending_date": week_ending.strftime("%Y-%m-%d"),
        "gate_reason": gate_reason,
        "action_count": 0,
        "appended_rows": 0,
    }

    if capture:
        finalists = build_weekly_finalists(
            latest=latest,
            history=history,
            market_date=market_date,
            deterministic_attention=deterministic_attention,
        )
        try:
            capture_status = _record_weekly_recommendation(
                market_date=market_date,
                week_ending=week_ending,
                finalist_bundle=finalists,
                spec=spec,
            )
            capture_status["gate_reason"] = gate_reason
        except Exception as exc:
            # Do not write an immutable weekly recommendation when the AI call or validation fails.
            # A later Friday/weekend rerun may retry safely.
            capture_status = {
                "status": "capture_error",
                "recommendation_date": market_date.strftime("%Y-%m-%d"),
                "week_ending_date": week_ending.strftime("%Y-%m-%d"),
                "gate_reason": gate_reason,
                "error": f"{type(exc).__name__}: {exc}",
                "action_count": 0,
                "appended_rows": 0,
                "selectable_finalists": finalists.get("selectable_count", 0),
            }

    outcomes = update_weekly_recommendation_outcomes(ohlcv)
    status = {
        "status": "ok" if capture_status.get("status") != "capture_error" else "partial",
        "model_version": spec.get("model_version"),
        "shadow_only": True,
        "dashboard_effect": "none",
        "capture": capture_status,
        "outcomes": outcomes,
        "history_file": WEEKLY_RECOMMENDATION_HISTORY_PATH.name,
        "outcomes_file": WEEKLY_RECOMMENDATION_OUTCOMES_PATH.name,
        "latest_file": WEEKLY_RECOMMENDATION_LATEST_PATH.name,
    }
    _write_json(WEEKLY_RECOMMENDATION_STATUS_PATH, status)
    return status
