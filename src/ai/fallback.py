from __future__ import annotations


def build_fallback_analysis(payload: dict, reason: str) -> dict:
    groups = payload.get("group_summaries", [])

    all_items = []
    for group in groups:
        all_items.extend(group.get("leaders", []))
        all_items.extend(group.get("biggest_20d_improvers", []))
        all_items.extend(group.get("biggest_20d_weakeners", []))

    # Deduplicate to the most recently seen snapshot per ticker.
    by_ticker = {}
    for item in all_items:
        ticker = item.get("ticker")
        if ticker:
            by_ticker[ticker] = item

    def items_for_state(state: str, limit: int = 5):
        rows = [x for x in by_ticker.values() if x.get("state") == state]
        if state in {"EMERGING", "ACCELERATING", "REACCELERATING"}:
            rows.sort(
                key=lambda x: (
                    x.get("score_change_20") is not None,
                    x.get("score_change_20") or -999,
                ),
                reverse=True,
            )
        elif state == "PERSISTENT_LEADER":
            rows.sort(key=lambda x: x.get("score") or -999, reverse=True)
        else:
            rows.sort(key=lambda x: x.get("score_change_20") or 999)
        return rows[:limit]

    def convert(item, label):
        related = []
        for key in ["primary_benchmark", "parent_benchmark", "paired_ticker"]:
            value = item.get(key)
            if value and value not in related:
                related.append(value)
        pending_text = ""
        raw_state = item.get("raw_state")
        confirmed_state = item.get("state")
        if raw_state and raw_state != confirmed_state:
            pending_text = (
                f" Current raw condition {raw_state} is pending "
                f"{item.get('pending_state_days', 0)}/{item.get('state_confirmation_bars', 3)} observations."
            )
        return {
            "ticker": item["ticker"],
            "title": f"{label}: {item.get('exposure') or item['ticker']}",
            "explanation": (
                f"Confirmed deterministic trend {confirmed_state}; "
                f"rotation score {item.get('score')}; "
                f"5-bar score change {item.get('score_change_5')}; "
                f"20-bar score change {item.get('score_change_20')}."
                f"{pending_text}"
            ),
            "confidence": "medium",
            "related_tickers": related[:4],
        }

    pair_items = []
    for p in payload.get("pair_signals", []):
        pair_items.append(
            {
                "ticker": p.get("ticker", ""),
                "title": f"Pair relationship: {p.get('exposure') or p.get('ticker','')}",
                "explanation": (
                    f"Confirmed pair signal {p.get('pair_signal') or 'PAIR_MIXED'} versus "
                    f"{p.get('paired_ticker') or 'paired market'}; "
                    f"20-bar spread {p.get('pair_spread_20_pct_points')} percentage points; "
                    f"63-bar spread {p.get('pair_spread_63_pct_points')} percentage points."
                    + (
                        f" Raw pair condition {p.get('pair_signal_raw')} is pending "
                        f"{p.get('pending_pair_signal_days', 0)}/{p.get('pair_confirmation_bars', 3)} observations."
                        if p.get('pair_signal_raw') and p.get('pair_signal_raw') != p.get('pair_signal')
                        else ""
                    )
                ),
                "confidence": "medium",
                "related_tickers": [p.get("paired_ticker")]
                if p.get("paired_ticker") else [],
            }
        )

    emerging = [convert(x, "Emerging") for x in items_for_state("EMERGING")]
    accelerating = [convert(x, "Accelerating") for x in items_for_state("ACCELERATING")]
    persistent = [convert(x, "Persistent leader") for x in items_for_state("PERSISTENT_LEADER")]
    reaccelerating = [convert(x, "Reaccelerating") for x in items_for_state("REACCELERATING")]
    weakening = [convert(x, "Weakening") for x in items_for_state("WEAKENING")]
    rotation_out = [convert(x, "Rotation out") for x in items_for_state("ROTATION_OUT")]

    focus = []
    for bucket in [emerging, accelerating, persistent, reaccelerating]:
        for item in bucket:
            if item["ticker"] not in focus:
                focus.append(item["ticker"])

    return {
        "headline": "Quantitative rotation dashboard",
        "market_regime": "AI interpretation unavailable",
        "executive_summary": (
            "The dashboard is using deterministic rotation calculations only. "
            f"AI was not used for this run: {reason}"
        ),
        "emerging_rotations": emerging,
        "accelerating_rotations": accelerating,
        "persistent_leaders": persistent,
        "reaccelerating_rotations": reaccelerating,
        "weakening_rotations": weakening,
        "rotation_out": rotation_out,
        "geographic_rotations": [],
        "pair_relationships": pair_items,
        "cross_market_confirmations": [],
        "risks_or_conflicts": [
            "Rotation scores are comparative signals, not forecasts.",
            "ETF trading volume is not the same as ETF net creations/redemptions.",
        ],
        "dashboard_focus_tickers": focus[:8],
        "methodology_note": "Deterministic fallback summary; no AI API call was used.",
    }
