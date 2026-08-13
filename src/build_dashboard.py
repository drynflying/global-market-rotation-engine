from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from src.common import (
    DOCS_DATA_DIR,
    DOCS_DIR,
    WEEKLY_RECOMMENDATION_LATEST_PATH,
)


def _fmt(value, digits=1, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def _state_class(state: str) -> str:
    return {
        "EMERGING": "good2",
        "ACCELERATING": "good",
        "PERSISTENT_LEADER": "good",
        "REACCELERATING": "good2",
        "NEUTRAL": "neutral",
        "WEAKENING": "warn",
        "ROTATION_OUT": "bad",
    }.get(state, "muted")


def _pair_class(signal: str) -> str:
    return {
        "PAIR_LEADING": "good",
        "PAIR_LAGGING": "bad",
        "PAIR_MIXED": "warn",
    }.get(signal, "muted")


def build_dashboard(
    latest: pd.DataFrame,
    history: pd.DataFrame,
    ai_analysis: dict,
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    latest = latest.copy()
    latest["date"] = pd.to_datetime(latest["date"])
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])

    as_of_date = pd.Timestamp(latest["date"].max()).normalize()
    as_of = as_of_date.strftime("%Y-%m-%d")

    def _format_date(value) -> str:
        try:
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                raise ValueError
            return ts.strftime("%b %d, %Y").replace(" 0", " ")
        except Exception:
            return "—"

    def _upcoming_weekly_evaluation(reference_date: pd.Timestamp, last_week_ending=None) -> pd.Timestamp:
        reference_date = pd.Timestamp(reference_date).normalize()
        if last_week_ending:
            try:
                candidate = pd.Timestamp(last_week_ending).normalize() + pd.Timedelta(days=7)
                while candidate < reference_date:
                    candidate += pd.Timedelta(days=7)
                return candidate
            except Exception:
                pass
        days_to_friday = (4 - reference_date.weekday()) % 7
        return reference_date + pd.Timedelta(days=days_to_friday)

    weekly_latest = None
    try:
        if WEEKLY_RECOMMENDATION_LATEST_PATH.exists():
            weekly_latest = json.loads(
                WEEKLY_RECOMMENDATION_LATEST_PATH.read_text(encoding="utf-8")
            )
    except Exception:
        weekly_latest = None

    weekly_recommendation_html = ""
    if weekly_latest and weekly_latest.get("status") == "ok":
        decision = weekly_latest.get("decision") or {}
        actions = (decision.get("actions") or [])[:2]
        finalist_bundle = weekly_latest.get("finalist_bundle") or {}
        finalists = finalist_bundle.get("finalists") or []
        finalist_lookup = {
            (str(item.get("ticker") or "").upper(), str(item.get("direction") or "").upper()): item
            for item in finalists
        }
        recommendation_date = weekly_latest.get("recommendation_date")
        week_ending_date = weekly_latest.get("week_ending_date")
        next_eval = _upcoming_weekly_evaluation(as_of_date, week_ending_date)

        action_cards = []
        for action in actions:
            ticker = str(action.get("ticker") or "").upper()
            direction = str(action.get("direction") or "").upper()
            candidate = finalist_lookup.get((ticker, direction), {})
            exposure = str(candidate.get("exposure") or ticker)
            confidence = str(candidate.get("evidence_confidence") or "HIGH").upper()
            rationale = str(action.get("rationale") or "")
            direction_class = "weekly-favor" if direction == "FAVOR" else "weekly-avoid"
            action_cards.append(
                f"""
                <article class='weekly-action'>
                  <div class='weekly-action-head'>
                    <div>
                      <span class='weekly-direction {direction_class}'>{html.escape(direction)}</span>
                      <strong class='weekly-ticker'>{html.escape(ticker)}</strong>
                    </div>
                    <span class='weekly-confidence'>Evidence Confidence: {html.escape(confidence)}</span>
                  </div>
                  <div class='weekly-exposure'>{html.escape(exposure)}</div>
                  <p>{html.escape(rationale)}</p>
                </article>
                """
            )

        if action_cards:
            body = f"<div class='weekly-actions'>{''.join(action_cards)}</div>"
        else:
            reason = str(
                decision.get("no_action_reason")
                or "No high-confidence action met the frozen weekly selection rules."
            )
            body = (
                "<div class='weekly-empty'><strong>No high-confidence action this week.</strong>"
                f"<div>{html.escape(reason)}</div></div>"
            )

        committee_summary = str(decision.get("committee_summary") or "")
        summary_html = (
            f"<div class='weekly-committee'>{html.escape(committee_summary)}</div>"
            if committee_summary else ""
        )
        weekly_recommendation_html = f"""
        <section class='weekly-panel'>
          <div class='weekly-panel-head'>
            <div>
              <div class='weekly-eyebrow'>Weekly High-Conviction Actions · Prospective Shadow</div>
              <h2>Recommendation date: {_format_date(recommendation_date)}</h2>
            </div>
            <div class='weekly-next'>Next evaluation: {_format_date(next_eval)} after close</div>
          </div>
          {body}
          {summary_html}
          <div class='weekly-disclaimer'>Evidence Confidence reflects agreement among the frozen quantitative evidence and AI committee review; it is not a probability of investment success.</div>
        </section>
        """
    else:
        next_eval = _upcoming_weekly_evaluation(as_of_date)
        weekly_recommendation_html = f"""
        <section class='weekly-panel weekly-panel-pending'>
          <div class='weekly-panel-head'>
            <div>
              <div class='weekly-eyebrow'>Weekly High-Conviction Actions · Prospective Shadow</div>
              <h2>No weekly recommendation has been generated yet.</h2>
            </div>
            <div class='weekly-next'>Next scheduled evaluation: {_format_date(next_eval)} after close</div>
          </div>
          <div class='weekly-empty'>The weekly layer evaluates the completed week's evidence after the Friday close and can return zero, one, or two actions. The latest dated recommendation will remain visible here throughout the following week.</div>
          <div class='weekly-disclaimer'>Evidence Confidence will describe evidence agreement, not a probability of investment success.</div>
        </section>
        """
    scored = latest[
        latest["rotation_score"].notna()
        & latest["rank_eligible"].astype(bool)
    ].copy()
    cross_scored = scored[scored["score_mode"].eq("CROSS_SECTIONAL")].copy()
    pair_scored = scored[scored["score_mode"].eq("PAIR")].copy()

    # Deterministic attention flags are calculated entirely in Python.
    # They are intentionally separate from AI-selected commentary so that
    # extreme or conflicting quantitative conditions cannot be omitted.
    lowest_scores = cross_scored.nsmallest(5, "rotation_score").copy()

    extreme_cmf = cross_scored[cross_scored["cmf20"].notna()].copy()
    extreme_cmf = extreme_cmf[extreme_cmf["cmf20"].abs() >= 0.40].copy()
    if not extreme_cmf.empty:
        extreme_cmf["_abs_cmf"] = extreme_cmf["cmf20"].abs()
        extreme_cmf = extreme_cmf.nlargest(5, "_abs_cmf")

    score_extremes_63 = []
    for ticker, g in history[history["rotation_score"].notna()].groupby("ticker"):
        g = g.sort_values("date").tail(63)
        if len(g) < 63:
            continue
        current_score = float(g.iloc[-1]["rotation_score"])
        low_score = float(g["rotation_score"].min())
        high_score = float(g["rotation_score"].max())
        if high_score - low_score < 1.0:
            continue
        if np.isclose(current_score, high_score, atol=0.01):
            score_extremes_63.append(
                {"ticker": ticker, "kind": "HIGH", "score": current_score,
                 "low": low_score, "high": high_score}
            )
        elif np.isclose(current_score, low_score, atol=0.01):
            score_extremes_63.append(
                {"ticker": ticker, "kind": "LOW", "score": current_score,
                 "low": low_score, "high": high_score}
            )

    highs_63 = sorted(
        (x for x in score_extremes_63 if x["kind"] == "HIGH"),
        key=lambda x: x["score"], reverse=True,
    )[:5]
    lows_63 = sorted(
        (x for x in score_extremes_63 if x["kind"] == "LOW"),
        key=lambda x: x["score"],
    )[:5]

    sector_divergences = []
    sector_rows = cross_scored[
        cross_scored["sector"].notna() & cross_scored["score_change_20"].notna()
    ].copy()
    for sector, g in sector_rows.groupby("sector"):
        sector_name = str(sector).strip()
        if not sector_name or len(g) < 2:
            continue
        improver = g.loc[g["score_change_20"].idxmax()]
        deteriorator = g.loc[g["score_change_20"].idxmin()]
        improvement = float(improver["score_change_20"])
        deterioration = float(deteriorator["score_change_20"])
        divergence = improvement - deterioration
        if improvement < 15 or deterioration > -15 or divergence < 40:
            continue

        other_weak = g[
            g["ticker"].ne(deteriorator["ticker"])
            & g["rotation_state"].isin(["ROTATION_OUT", "WEAKENING"])
        ].sort_values("rotation_score").head(1)
        other_weak_payload = None
        if not other_weak.empty:
            w = other_weak.iloc[0]
            other_weak_payload = {
                "ticker": str(w["ticker"]),
                "state": str(w["rotation_state"]),
                "score": float(w["rotation_score"]),
            }

        sector_divergences.append(
            {
                "sector": sector_name,
                "improver": str(improver["ticker"]),
                "improvement": improvement,
                "deteriorator": str(deteriorator["ticker"]),
                "deterioration": deterioration,
                "divergence": divergence,
                "other_weak": other_weak_payload,
            }
        )
    sector_divergences.sort(key=lambda x: x["divergence"], reverse=True)
    sector_divergences = sector_divergences[:6]

    pair_state_tensions = []
    strong_states = {
        "EMERGING", "ACCELERATING", "PERSISTENT_LEADER", "REACCELERATING"
    }
    weak_states = {"WEAKENING", "ROTATION_OUT"}
    for _, r in pair_scored.sort_values("ticker").iterrows():
        pair_signal = str(r.get("pair_signal") or "")
        rotation_state = str(r.get("rotation_state") or "")
        conflict = (
            pair_signal == "PAIR_LEADING" and rotation_state in weak_states
        ) or (
            pair_signal == "PAIR_LAGGING" and rotation_state in strong_states
        )
        if conflict:
            pair_state_tensions.append(r)

    pending_state_rows = scored[scored.get("pending_state_days", 0).fillna(0) > 0].copy()
    if not pending_state_rows.empty:
        pending_state_rows["_pending_move"] = pending_state_rows["score_change_5"].abs().fillna(0)
        pending_state_rows = pending_state_rows.sort_values(
            ["pending_state_days", "_pending_move", "rotation_score"],
            ascending=[False, False, False],
        ).head(12)

    pending_pair_rows = pair_scored[
        pair_scored.get("pending_pair_signal_days", 0).fillna(0) > 0
    ].copy()
    if not pending_pair_rows.empty:
        pending_pair_rows = pending_pair_rows.sort_values(
            ["pending_pair_signal_days", "ticker"], ascending=[False, True]
        )

    leaders = cross_scored.nlargest(15, "rotation_score")
    movers = cross_scored[
        cross_scored["score_change_20"].notna()
    ].nlargest(15, "score_change_20")
    weakening = cross_scored[
        cross_scored["score_change_20"].notna()
    ].nsmallest(10, "score_change_20")

    focus = ai_analysis.get("dashboard_focus_tickers", [])[:8]
    if not focus:
        focus = movers["ticker"].head(8).tolist()

    series = {}
    for ticker in focus:
        g = history[
            (history["ticker"] == ticker)
            & history["rotation_score"].notna()
        ].sort_values("date").tail(63)
        series[ticker] = [
            {"date": d.strftime("%Y-%m-%d"), "score": round(float(s), 2)}
            for d, s in zip(g["date"], g["rotation_score"])
        ]

    latest_json = latest.replace({np.nan: None}).to_dict(orient="records")
    (DOCS_DATA_DIR / "rotation_latest.json").write_text(
        json.dumps(latest_json, default=str, indent=2),
        encoding="utf-8",
    )
    (DOCS_DATA_DIR / "ai_analysis.json").write_text(
        json.dumps(ai_analysis, indent=2),
        encoding="utf-8",
    )

    provider_results = ai_analysis.get("provider_results", {})
    successful = ai_analysis.get("successful_providers", [])
    requested = ai_analysis.get("requested_providers", [])
    failed = ai_analysis.get("failed_providers", [])
    primary_provider = ai_analysis.get("primary_provider", "deterministic_fallback")
    consensus = ai_analysis.get("consensus", {})

    def state_cell(r) -> str:
        confirmed = str(r.get("rotation_state") or "—")
        raw = str(r.get("rotation_state_raw") or confirmed)
        pending_days = int(r.get("pending_state_days", 0) or 0)
        confirmation_bars = int(r.get("state_confirmation_bars", 3) or 3)
        pending_html = ""
        if raw != confirmed and pending_days > 0:
            pending_html = (
                f"<div class='pending-line'>Current: "
                f"<span class='{_state_class(raw)}'>{html.escape(raw)}</span> · "
                f"{pending_days}/{confirmation_bars}</div>"
            )
        return (
            f"<span class='badge {_state_class(confirmed)}'>{html.escape(confirmed)}</span>"
            f"{pending_html}"
        )

    def pair_signal_cell(r) -> str:
        confirmed = str(r.get("pair_signal") or "—")
        raw = str(r.get("pair_signal_raw") or confirmed)
        pending_days = int(r.get("pending_pair_signal_days", 0) or 0)
        confirmation_bars = int(r.get("pair_confirmation_bars", 3) or 3)
        pending_html = ""
        if raw != confirmed and pending_days > 0:
            pending_html = (
                f"<div class='pending-line'>Current: "
                f"<span class='{_pair_class(raw)}'>{html.escape(raw)}</span> · "
                f"{pending_days}/{confirmation_bars}</div>"
            )
        return (
            f"<span class='badge {_pair_class(confirmed)}'>{html.escape(confirmed)}</span>"
            f"{pending_html}"
        )

    def rows(frame):
        result = []
        for _, r in frame.iterrows():
            result.append(
                f"""
                <tr>
                  <td><strong>{html.escape(str(r['ticker']))}</strong></td>
                  <td>{html.escape(str(r.get('exposure') or ''))}</td>
                  <td>{_fmt(r.get('rotation_score'), 1)}</td>
                  <td>{_fmt(r.get('score_change_20'), 1)}</td>
                  <td>{_fmt(r.get('rs20') * 100 if pd.notna(r.get('rs20')) else None, 1, '%')}</td>
                  <td>{state_cell(r)}</td>
                </tr>
                """
            )
        return "\n".join(result)

    def ai_cards(key, title):
        items = ai_analysis.get(key, [])
        cards = []
        for item in items[:5]:
            related = ", ".join(item.get("related_tickers", [])[:4])
            related_html = (
                f"<div class='related'>Related: {html.escape(related)}</div>"
                if related else ""
            )
            cards.append(
                f"""
                <article class="mini-card">
                  <div class="mini-head">
                    <strong>{html.escape(item.get('ticker',''))}</strong>
                    <span>{html.escape(item.get('confidence',''))}</span>
                  </div>
                  <div class="mini-title">{html.escape(item.get('title',''))}</div>
                  <p>{html.escape(item.get('explanation',''))}</p>
                  {related_html}
                </article>
                """
            )
        return (
            f"<section><h2>{html.escape(title)}</h2>"
            f"<div class='mini-grid'>{''.join(cards) or '<p class=\"muted\">No items.</p>'}</div>"
            f"</section>"
        )

    lowest_scores_html = "".join(
        f"<li><strong>{html.escape(str(r['ticker']))}</strong> — "
        f"score {_fmt(r.get('rotation_score'),1)}, "
        f"{html.escape(str(r.get('rotation_state') or '—'))}; "
        f"20-bar Δ {_fmt(r.get('score_change_20'),1)}; "
        f"CMF20 {_fmt(r.get('cmf20'),3)}</li>"
        for _, r in lowest_scores.iterrows()
    ) or "<li class='muted'>No scored securities available.</li>"

    extreme_cmf_html = "".join(
        f"<li><strong>{html.escape(str(r['ticker']))}</strong> — "
        f"CMF20 {_fmt(r.get('cmf20'),3)}, score {_fmt(r.get('rotation_score'),1)}, "
        f"{html.escape(str(r.get('rotation_state') or '—'))}</li>"
        for _, r in extreme_cmf.iterrows()
    ) or "<li class='muted'>No |CMF20| readings at or above 0.40.</li>"

    highs_63_html = "".join(
        f"<li><strong>{html.escape(x['ticker'])}</strong> — "
        f"latest {_fmt(x['score'],1)} equals its 63-bar high "
        f"(range {_fmt(x['low'],1)}–{_fmt(x['high'],1)})</li>"
        for x in highs_63
    ) or "<li class='muted'>No current 63-bar score highs.</li>"

    lows_63_html = "".join(
        f"<li><strong>{html.escape(x['ticker'])}</strong> — "
        f"latest {_fmt(x['score'],1)} equals its 63-bar low "
        f"(range {_fmt(x['low'],1)}–{_fmt(x['high'],1)})</li>"
        for x in lows_63
    ) or "<li class='muted'>No current 63-bar score lows.</li>"

    sector_divergence_html_parts = []
    for item in sector_divergences:
        extra = ""
        if item["other_weak"]:
            w = item["other_weak"]
            extra = (
                f"; {html.escape(w['ticker'])} is "
                f"{html.escape(w['state'])} at score {_fmt(w['score'],1)}"
            )
        sector_divergence_html_parts.append(
            f"<li><strong>{html.escape(item['sector'])}</strong> — "
            f"{html.escape(item['improver'])} 20-bar Δ "
            f"{_fmt(item['improvement'],1)} vs "
            f"{html.escape(item['deteriorator'])} "
            f"{_fmt(item['deterioration'],1)} "
            f"({_fmt(item['divergence'],1)}-point divergence){extra}</li>"
        )
    sector_divergences_html = "".join(sector_divergence_html_parts) or (
        "<li class='muted'>No sector has both a ≥15-point improver and "
        "≤−15-point deteriorator with a ≥40-point divergence.</li>"
    )

    pair_state_tensions_html = "".join(
        f"<li><strong>{html.escape(str(r['ticker']))}</strong> — "
        f"{html.escape(str(r.get('pair_signal') or '—'))} versus "
        f"{html.escape(str(r.get('paired_ticker') or '—'))}, while its "
        f"pair-mode rotation state is "
        f"{html.escape(str(r.get('rotation_state') or '—'))}; "
        f"20/63-bar pair spreads "
        f"{_fmt(r.get('pair_spread_20') * 100 if pd.notna(r.get('pair_spread_20')) else None,1,'%')} / "
        f"{_fmt(r.get('pair_spread_63') * 100 if pd.notna(r.get('pair_spread_63')) else None,1,'%')}</li>"
        for r in pair_state_tensions
    ) or "<li class='muted'>No pair-signal / rotation-state tensions.</li>"

    pending_state_html = "".join(
        f"<li><strong>{html.escape(str(r['ticker']))}</strong> — confirmed "
        f"{html.escape(str(r.get('rotation_state') or '—'))}; current raw condition "
        f"{html.escape(str(r.get('rotation_state_raw') or '—'))} is pending "
        f"{int(r.get('pending_state_days',0) or 0)}/{int(r.get('state_confirmation_bars',3) or 3)}; "
        f"score {_fmt(r.get('rotation_score'),1)}, 5-bar Δ {_fmt(r.get('score_change_5'),1)}</li>"
        for _, r in pending_state_rows.iterrows()
    ) or "<li class='muted'>No raw rotation-state changes awaiting confirmation.</li>"

    pending_pair_html = "".join(
        f"<li><strong>{html.escape(str(r['ticker']))}</strong> vs "
        f"{html.escape(str(r.get('paired_ticker') or '—'))} — confirmed "
        f"{html.escape(str(r.get('pair_signal') or '—'))}; current raw pair condition "
        f"{html.escape(str(r.get('pair_signal_raw') or '—'))} is pending "
        f"{int(r.get('pending_pair_signal_days',0) or 0)}/{int(r.get('pair_confirmation_bars',3) or 3)}</li>"
        for _, r in pending_pair_rows.iterrows()
    ) or "<li class='muted'>No raw pair-signal changes awaiting confirmation.</li>"

    confirmations = ai_analysis.get("cross_market_confirmations", []) or []
    risks_or_conflicts = ai_analysis.get("risks_or_conflicts", []) or []

    confirmations_html = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in confirmations[:8]
    ) or "<li class='muted'>No cross-market confirmations supplied.</li>"

    risks_or_conflicts_html = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in risks_or_conflicts[:8]
    ) or "<li class='muted'>No material risks or conflicts supplied.</li>"

    provider_cards = []
    for name in requested:
        result = provider_results.get(name, {})
        status = result.get("status", "unknown")
        model = result.get("model") or "—"
        err = result.get("error")
        validation = result.get("validation") or {}
        if status == "validation_error":
            public_provider_message = (
                "AI response did not pass deterministic grounding checks and was withheld."
            )
        elif status == "error":
            public_provider_message = (
                "Provider did not return a usable response for this run."
            )
        else:
            public_provider_message = ""
        validation_status = validation.get("status")
        validation_version = validation.get("validator_version")
        validation_line = ""
        if validation_status:
            attempt_count = validation.get("attempt_count")
            retry_used = validation.get("retry_used", False)
            retry_text = ""
            if attempt_count:
                retry_text = f" · attempt {html.escape(str(attempt_count))}"
            if retry_used and validation_status == "passed":
                retry_text += " · corrected on retry"
            validation_line = (
                f"<div class='muted small'>validation: "
                f"{html.escape(str(validation_status))}"
                f"{' · ' + html.escape(str(validation_version)) if validation_version else ''}"
                f"{retry_text}"
                f"</div>"
            )
        provider_cards.append(
            f"""
            <div class="provider-card">
              <div><strong>{html.escape(name.title())}</strong></div>
              <div class="provider-status {'good' if status == 'success' else 'warn'}">
                {html.escape(status)}
              </div>
              <div class="muted small">{html.escape(str(model))}</div>
              {validation_line}
              {f'<div class="muted small">{html.escape(public_provider_message)}</div>' if public_provider_message else ''}
            </div>
            """
        )
    if not provider_cards:
        provider_cards.append(
            "<div class='provider-card'><strong>Deterministic fallback</strong>"
            "<div class='muted small'>No AI provider enabled.</div></div>"
        )

    consensus_html = ""
    if consensus.get("provider_count", 0) >= 2:
        strong = consensus.get("strong_consensus", [])[:8]
        disagreements = consensus.get("disagreements", [])[:8]

        strong_rows = "".join(
            f"<tr><td><strong>{html.escape(x['ticker'])}</strong></td>"
            f"<td>{html.escape(x['dominant_assessment'])}</td>"
            f"<td>{x['agreement_count']}/{x['provider_count']}</td></tr>"
            for x in strong
        ) or "<tr><td colspan='3' class='muted'>No unanimous findings.</td></tr>"

        disagree_rows = "".join(
            f"<tr><td><strong>{html.escape(x['ticker'])}</strong></td>"
            f"<td>{html.escape(', '.join(f'{k}: {v}' for k,v in x['assessment_counts'].items()))}</td>"
            f"<td>{html.escape(', '.join(x['providers_mentioning']))}</td></tr>"
            for x in disagreements
        ) or "<tr><td colspan='3' class='muted'>No model disagreements.</td></tr>"

        consensus_html = f"""
        <section>
          <h2>AI model consensus</h2>
          <div class="split">
            <div>
              <h3>Strong agreement</h3>
              <div class="table-wrap"><table>
                <thead><tr><th>Ticker</th><th>Assessment</th><th>Agreement</th></tr></thead>
                <tbody>{strong_rows}</tbody>
              </table></div>
            </div>
            <div>
              <h3>Disagreements</h3>
              <div class="table-wrap"><table>
                <thead><tr><th>Ticker</th><th>Assessments</th><th>Models</th></tr></thead>
                <tbody>{disagree_rows}</tbody>
              </table></div>
            </div>
          </div>
        </section>
        """
    else:
        consensus_html = """
        <section class="consensus-note">
          <strong>AI consensus:</strong>
          One AI analyst is active. Model agreement/disagreement will appear
          automatically when a second provider is enabled.
        </section>
        """

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Rotation Dashboard</title>
<style>
:root {{
  color-scheme: light dark;
  --bg:#0b1220; --panel:#111a2b; --panel2:#172235; --text:#eef4ff;
  --muted:#9aa9bf; --line:#27354a; --accent:#66b3ff; --good:#4ad295;
  --good2:#9ed66e; --warn:#ffc857; --bad:#ff7a7a;
}}
@media (prefers-color-scheme: light) {{
  :root {{
    --bg:#f5f7fb; --panel:#ffffff; --panel2:#f8fafc; --text:#182132;
    --muted:#5e6b7e; --line:#d9e0ea; --accent:#1769aa; --good:#18794e;
    --good2:#4f7d16; --warn:#9a6700; --bad:#b42318;
  }}
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  background:var(--bg); color:var(--text);
}}
main {{ max-width:1200px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 4px; font-size:28px; }}
h2 {{ margin-top:28px; font-size:18px; }}
h3 {{ font-size:14px; color:var(--muted); }}
.sub {{ color:var(--muted); margin-bottom:22px; }}
.grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
.card, .mini-card, .provider-card {{
  background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:16px;
}}
.metric {{ font-size:28px; font-weight:700; margin-top:6px; }}
.muted {{ color:var(--muted); }}
.small {{ font-size:12px; margin-top:5px; }}
.weekly-panel {{
  margin:18px 0 24px; padding:18px; border:1px solid var(--line);
  border-radius:14px; background:linear-gradient(180deg,var(--panel),var(--panel2));
}}
.weekly-panel-pending {{ border-style:dashed; }}
.weekly-panel-head {{
  display:flex; justify-content:space-between; align-items:flex-start; gap:16px;
}}
.weekly-panel h2 {{ margin:4px 0 0; font-size:20px; }}
.weekly-eyebrow {{
  color:var(--accent); font-size:12px; font-weight:700; text-transform:uppercase;
  letter-spacing:.06em;
}}
.weekly-next {{ color:var(--muted); font-size:12px; text-align:right; padding-top:4px; }}
.weekly-actions {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-top:14px; }}
.weekly-action {{
  background:var(--bg); border:1px solid var(--line); border-radius:12px; padding:14px;
}}
.weekly-action-head {{ display:flex; justify-content:space-between; gap:12px; align-items:center; }}
.weekly-direction {{
  display:inline-block; padding:3px 7px; margin-right:7px; border-radius:999px;
  font-size:11px; font-weight:800; letter-spacing:.04em;
}}
.weekly-favor {{ color:var(--good); border:1px solid var(--good); }}
.weekly-avoid {{ color:var(--bad); border:1px solid var(--bad); }}
.weekly-ticker {{ font-size:17px; }}
.weekly-confidence {{ color:var(--good2); font-size:12px; white-space:nowrap; }}
.weekly-exposure {{ color:var(--muted); font-size:12px; margin-top:8px; }}
.weekly-action p {{ margin:8px 0 0; line-height:1.45; }}
.weekly-empty {{
  margin-top:14px; padding:14px; border:1px solid var(--line); border-radius:12px;
  background:var(--bg); color:var(--muted); line-height:1.45;
}}
.weekly-empty strong {{ color:var(--text); display:block; margin-bottom:4px; }}
.weekly-committee {{ margin-top:12px; color:var(--muted); line-height:1.45; }}
.weekly-disclaimer {{ margin-top:12px; color:var(--muted); font-size:11px; line-height:1.4; }}
.ai-summary {{
  background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:18px; margin-top:18px;
}}
.providers {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:12px; }}
.provider-card {{ padding:12px; }}
.provider-status {{ text-transform:uppercase; font-size:12px; margin-top:4px; }}
.mini-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
.mini-head {{ display:flex; justify-content:space-between; color:var(--muted); }}
.mini-title {{ margin-top:8px; font-weight:700; }}
.mini-card p {{ margin-bottom:6px; color:var(--muted); }}
.related {{ color:var(--muted); font-size:12px; }}
table {{
  width:100%; border-collapse:collapse; background:var(--panel);
  border:1px solid var(--line); border-radius:14px; overflow:hidden;
}}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; }}
th {{ color:var(--muted); font-size:13px; }}
.badge {{
  padding:3px 7px; border-radius:999px; border:1px solid currentColor;
  font-size:12px; white-space:nowrap;
}}
.good {{ color:var(--good); }}
.good2 {{ color:var(--good2); }}
.warn {{ color:var(--warn); }}
.bad {{ color:var(--bad); }}
.neutral,.muted {{ color:var(--muted); }}
.chart-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
.chart-card {{
  background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:14px;
}}
.spark {{ width:100%; height:120px; }}
.legend {{ display:flex; justify-content:space-between; color:var(--muted); font-size:12px; }}
.split {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.consensus-note {{
  margin-top:24px; padding:14px; border:1px solid var(--line);
  border-radius:14px; background:var(--panel2); color:var(--muted);
}}
.insight-card {{
  background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:16px;
}}
.insight-card h3 {{ margin-top:0; }}
.insight-list {{ margin:8px 0 0; padding-left:20px; }}
.insight-list li {{ margin:8px 0; color:var(--muted); line-height:1.45; }}
.attention-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }}
.attention-note {{ color:var(--muted); font-size:12px; margin-top:-6px; margin-bottom:12px; }}
.attention-subhead {{ margin:12px 0 4px; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
.pending-line {{ margin-top:5px; color:var(--muted); font-size:11px; line-height:1.25; }}
.trend-rule {{ margin:10px 0 18px; padding:10px 12px; border:1px solid var(--line); border-radius:10px; background:var(--panel2); color:var(--muted); font-size:12px; }}
@media (max-width:800px) {{
  .grid,.mini-grid,.chart-grid,.providers,.split,.attention-grid,.weekly-actions {{ grid-template-columns:1fr; }}
  .weekly-panel-head,.weekly-action-head {{ flex-direction:column; align-items:flex-start; }}
  .weekly-next {{ text-align:left; }}
  main {{ padding:14px; }}
  .table-wrap {{ overflow-x:auto; }}
}}
</style>
</head>
<body>
<main>
  <h1>Market Rotation Dashboard</h1>
  <div class="sub">As of {as_of} · Longer-horizon rotation monitoring, not a daily trading signal</div>
  <div class="trend-rule"><strong>Trend confirmation:</strong> scores and raw conditions update daily, but a new rotation state or pair signal must persist for 3 consecutive observations before it becomes the confirmed dashboard trend. Pending raw changes remain visible as early warnings.</div>

  <div class="grid">
    <div class="card"><div class="muted">Scored signals</div><div class="metric">{len(scored)}</div></div>
    <div class="card"><div class="muted">Emerging / accelerating</div><div class="metric">{int(scored['rotation_state'].isin(['EMERGING','ACCELERATING','REACCELERATING']).sum())}</div></div>
    <div class="card"><div class="muted">Persistent leaders</div><div class="metric">{int((scored['rotation_state']=='PERSISTENT_LEADER').sum())}</div></div>
    <div class="card"><div class="muted">Weakening / out</div><div class="metric">{int(scored['rotation_state'].isin(['WEAKENING','ROTATION_OUT']).sum())}</div></div>
  </div>

  <div class="ai-summary">
    <div class="muted">AI interpretation · primary analyst: {html.escape(str(primary_provider))}</div>
    <h2 style="margin-top:6px">{html.escape(ai_analysis.get('headline','Rotation summary'))}</h2>
    <p><strong>{html.escape(ai_analysis.get('market_regime',''))}</strong></p>
    <p>{html.escape(ai_analysis.get('executive_summary',''))}</p>
    <div class="providers">{''.join(provider_cards)}</div>
  </div>

  {weekly_recommendation_html}

  {ai_cards('emerging_rotations','Emerging rotations')}
  {ai_cards('accelerating_rotations','Accelerating rotations')}
  {ai_cards('persistent_leaders','Persistent leaders')}
  {ai_cards('reaccelerating_rotations','Reaccelerating rotations')}
  {ai_cards('weakening_rotations','Weakening rotations')}
  {ai_cards('rotation_out','Rotation out')}

  <section>
    <h2>Deterministic attention flags</h2>
    <div class="attention-note">Generated directly from the quantitative dataset and 63-bar score history; these items are not AI-selected.</div>
    <div class="attention-grid">
      <div class="insight-card">
        <h3>Absolute weakness and CMF extremes</h3>
        <div class="attention-subhead">Lowest current cross-sectional scores</div>
        <ul class="insight-list">{lowest_scores_html}</ul>
        <div class="attention-subhead">Extreme CMF20 (|CMF20| ≥ 0.40)</div>
        <ul class="insight-list">{extreme_cmf_html}</ul>
      </div>
      <div class="insight-card">
        <h3>63-bar score extremes</h3>
        <div class="attention-subhead">At 63-bar highs</div>
        <ul class="insight-list">{highs_63_html}</ul>
        <div class="attention-subhead">At 63-bar lows</div>
        <ul class="insight-list">{lows_63_html}</ul>
      </div>
      <div class="insight-card">
        <h3>Sector divergences</h3>
        <ul class="insight-list">{sector_divergences_html}</ul>
      </div>
      <div class="insight-card">
        <h3>Pair signal / rotation-state tensions</h3>
        <ul class="insight-list">{pair_state_tensions_html}</ul>
      </div>
      <div class="insight-card">
        <h3>Pending trend confirmations</h3>
        <ul class="insight-list">{pending_state_html}</ul>
      </div>
      <div class="insight-card">
        <h3>Pending pair confirmations</h3>
        <ul class="insight-list">{pending_pair_html}</ul>
      </div>
    </div>
  </section>

  <section>
    <h2>Cross-market confirmations and risks</h2>
    <div class="split">
      <div class="insight-card">
        <h3>Cross-market confirmations</h3>
        <ul class="insight-list">{confirmations_html}</ul>
      </div>
      <div class="insight-card">
        <h3>Risks / conflicts</h3>
        <ul class="insight-list">{risks_or_conflicts_html}</ul>
      </div>
    </div>
  </section>

  {consensus_html}

  <section>
    <h2>63-bar score trends selected for attention</h2>
    <div id="charts" class="chart-grid"></div>
  </section>

  <section>
    <h2>Highest current rotation scores</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Ticker</th><th>Exposure</th><th>Score</th><th>20-bar Δ</th><th>RS20</th><th>Confirmed Trend</th></tr></thead>
        <tbody>{rows(leaders)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Biggest 20-bar improvements</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Ticker</th><th>Exposure</th><th>Score</th><th>20-bar Δ</th><th>RS20</th><th>Confirmed Trend</th></tr></thead>
        <tbody>{rows(movers)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Biggest 20-bar deterioration</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Ticker</th><th>Exposure</th><th>Score</th><th>20-bar Δ</th><th>RS20</th><th>Confirmed Trend</th></tr></thead>
        <tbody>{rows(weakening)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Pair signals</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Ticker</th><th>Exposure</th><th>Pair</th><th>Score</th><th>20-bar spread</th><th>63-bar spread</th><th>Confirmed Pair Signal</th><th>Confirmed Trend</th></tr></thead>
        <tbody>
          {''.join(
              f"<tr><td><strong>{html.escape(str(r['ticker']))}</strong></td>"
              f"<td>{html.escape(str(r.get('exposure') or ''))}</td>"
              f"<td>{html.escape(str(r.get('paired_ticker') or ''))}</td>"
              f"<td>{_fmt(r.get('rotation_score'),1)}</td>"
              f"<td>{_fmt(r.get('pair_spread_20')*100 if pd.notna(r.get('pair_spread_20')) else None,1,'%')}</td>"
              f"<td>{_fmt(r.get('pair_spread_63')*100 if pd.notna(r.get('pair_spread_63')) else None,1,'%')}</td>"
              f"<td>{pair_signal_cell(r)}</td>"
              f"<td>{state_cell(r)}</td></tr>"
              for _, r in pair_scored.sort_values('ticker').iterrows()
          ) or '<tr><td colspan="8" class="muted">No pair signals available.</td></tr>'}
        </tbody>
      </table>
    </div>
  </section>

  <p class="muted" style="margin-top:30px">
    Quantitative scores are deterministic comparative indicators. Confirmed trend states use a 3-observation persistence rule; raw conditions remain available as early warnings. AI commentary interprets supplied values and confirmed states; it does not calculate or alter them. Weekly high-conviction actions are a prospective shadow layer and Evidence Confidence is not a calibrated probability of success.
  </p>
</main>

<script>
const series = {json.dumps(series)};

function makeSparkline(points) {{
  const width = 520, height = 120, pad = 8;
  const vals = points.map(p => p.score).filter(v => Number.isFinite(v));
  if (vals.length < 2) return '<div class="muted">Not enough score history.</div>';
  const min = Math.min(...vals), max = Math.max(...vals);
  const span = Math.max(max-min, 1);
  const xy = vals.map((v,i) => {{
    const x = pad + i * (width-2*pad) / Math.max(vals.length-1,1);
    const y = height-pad - (v-min) * (height-2*pad) / span;
    return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
  }}).join(' ');
  return `
    <svg class="spark" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Rotation score history">
      <line x1="${{pad}}" y1="${{height/2}}" x2="${{width-pad}}" y2="${{height/2}}" stroke="var(--line)" />
      <polyline points="${{xy}}" fill="none" stroke="var(--accent)" stroke-width="3" />
    </svg>
    <div class="legend"><span>low ${{min.toFixed(1)}}</span><span>latest ${{vals[vals.length-1].toFixed(1)}}</span><span>high ${{max.toFixed(1)}}</span></div>
  `;
}}

const charts = document.getElementById('charts');
Object.entries(series).forEach(([ticker, points]) => {{
  const card = document.createElement('div');
  card.className = 'chart-card';
  card.innerHTML = `<strong>${{ticker}}</strong>${{makeSparkline(points)}}`;
  charts.appendChild(card);
}});
</script>
</body>
</html>
"""
    (DOCS_DIR / "index.html").write_text(page, encoding="utf-8")
