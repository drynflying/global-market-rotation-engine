from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = ROOT / "results" / "clean_sheet_r4"
DEFAULT_SPEC = Path(__file__).with_name("r4_spec.json")
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "r4_dashboard.html"

HORIZONS = [21, 63, 126, 189]


def _read_json(path: Path, default):
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _fmt_int(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{int(value):,}"
    except Exception:
        return "—"


def _fmt_num(value, digits=3) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "—"


def _fmt_pct(value, digits=1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "—"


def _fmt_pp(value, digits=2) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) * 100:+.{digits}f} pp"
    except Exception:
        return "—"


def _fmt_date(value) -> str:
    try:
        if value is None or pd.isna(value) or str(value).strip() in {"", "null", "None"}:
            return "—"
        return pd.Timestamp(value).strftime("%b %d, %Y").replace(" 0", " ")
    except Exception:
        return "—"


def _status_class(status: str) -> str:
    return {
        "INSUFFICIENT": "status-muted",
        "EARLY": "status-info",
        "PROMISING": "status-good",
        "MIXED": "status-warn",
        "CONFIRMED_PROSPECTIVE": "status-good",
        "FAILED_TO_CONFIRM": "status-bad",
    }.get(str(status or "").upper(), "status-muted")


def _status_label(status: str) -> str:
    return {
        "INSUFFICIENT": "Insufficient",
        "EARLY": "Early",
        "PROMISING": "Promising",
        "MIXED": "Mixed",
        "CONFIRMED_PROSPECTIVE": "Confirmed prospective",
        "FAILED_TO_CONFIRM": "Failed to confirm",
    }.get(str(status or "").upper(), str(status or "Unknown").replace("_", " ").title())


def _safe_for_script(value) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def _assessment_row(assessment: dict, horizon: int) -> dict:
    return (assessment.get("by_horizon") or {}).get(str(horizon), {}) or {}


def _ranking_metric(row: dict, method: str, name: str):
    return (((row.get("ranking") or {}).get(method) or {}).get(name))


def _avoid_metric(row: dict, name: str):
    return ((row.get("avoid") or {}).get(name))


def _horizon_card(h: int, spec: dict, assessment: dict, primary: int) -> str:
    row = _assessment_row(assessment, h)
    label = spec.get("horizon_labels", {}).get(str(h), f"{h}d")
    status = row.get("evidence_status", "INSUFFICIENT")
    months = row.get("evaluated_months", 0)
    ridge_ic = _ranking_metric(row, "ridge", "mean_ic")
    equal_ic = _ranking_metric(row, "equal_weight", "mean_ic")
    auc = _avoid_metric(row, "mean_roc_auc")
    lift = _avoid_metric(row, "mean_top_risk_quintile_lift")
    rel = _avoid_metric(row, "mean_top_risk_quintile_mean_rel_return")
    ridge_gate = bool(_ranking_metric(row, "ridge", "favor_gate_pass"))
    equal_gate = bool(_ranking_metric(row, "equal_weight", "favor_gate_pass"))
    avoid_gate = bool(_avoid_metric(row, "avoid_gate_pass"))
    primary_tag = "<span class='primary-pill'>Primary</span>" if h == primary else ""

    return f"""
    <button class="horizon-card {'primary-card' if h == primary else ''}" data-horizon="{h}" type="button">
      <div class="horizon-top">
        <div><span class="horizon-label">{html.escape(label)}</span>{primary_tag}</div>
        <span class="evidence-badge {_status_class(status)}">{html.escape(_status_label(status))}</span>
      </div>
      <div class="metric-grid compact">
        <div><span>Evaluated months</span><strong>{_fmt_int(months)}</strong></div>
        <div><span>Ridge IC</span><strong>{_fmt_num(ridge_ic)}</strong></div>
        <div><span>Equal-weight IC</span><strong>{_fmt_num(equal_ic)}</strong></div>
        <div><span>AVOID AUC</span><strong>{_fmt_num(auc)}</strong></div>
        <div><span>AVOID lift</span><strong>{_fmt_num(lift, 2)}×</strong></div>
        <div><span>High-risk vs SPY</span><strong>{_fmt_pct(rel)}</strong></div>
      </div>
      <div class="gate-row">
        <span class="gate {'pass' if ridge_gate else ''}">Ridge FAVOR {'PASS' if ridge_gate else 'not earned'}</span>
        <span class="gate {'pass' if equal_gate else ''}">Equal FAVOR {'PASS' if equal_gate else 'not earned'}</span>
        <span class="gate {'pass' if avoid_gate else ''}">AVOID {'PASS' if avoid_gate else 'not earned'}</span>
      </div>
    </button>
    """


def _features_table(spec: dict) -> str:
    rows = []
    primary = int(spec.get("primary_research_horizon_bars", 126))
    for h in HORIZONS:
        label = spec.get("horizon_labels", {}).get(str(h), str(h))
        features = (spec.get("candidate_features_by_horizon") or {}).get(str(h), [])
        chips = "".join(f"<span class='feature-chip'>{html.escape(str(f))}</span>" for f in features)
        rows.append(
            f"""
            <div class="feature-row {'feature-primary' if h == primary else ''}">
              <div class="feature-horizon">{html.escape(label)}</div>
              <div class="feature-list">{chips or '—'}</div>
            </div>
            """
        )
    return "".join(rows)


def _rank_table(rows: list[dict], value_key: str, value_label: str, risk: bool = False) -> str:
    if not rows:
        return "<div class='empty-mini'>No prospective ranking has been issued yet.</div>"
    body = []
    for i, item in enumerate(rows[:10], 1):
        ticker = html.escape(str(item.get("ticker", "—")))
        value = item.get(value_key)
        rank_pct = item.get("avoid_risk_rank_pct" if risk else (
            "ridge_rank_pct" if value_key == "ridge_prediction" else "equal_weight_rank_pct"
        ))
        value_text = _fmt_pct(value, 1) if risk else _fmt_num(value, 4)
        body.append(
            f"""
            <tr>
              <td class="rank-num">{i}</td>
              <td><strong>{ticker}</strong></td>
              <td class="mono">{value_text}</td>
              <td class="mono">{_fmt_pct(rank_pct, 0)}</td>
            </tr>
            """
        )
    return f"""
      <table class="rank-table">
        <thead><tr><th>#</th><th>Ticker</th><th>{html.escape(value_label)}</th><th>Percentile</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    """


def _latest_sections(latest: dict, spec: dict) -> str:
    by_h = latest.get("by_horizon") or {}
    sections = []
    for h in HORIZONS:
        label = spec.get("horizon_labels", {}).get(str(h), str(h))
        data = by_h.get(str(h), {}) or {}
        sections.append(
            f"""
            <div class="latest-horizon-panel" data-horizon-panel="{h}">
              <div class="rank-grid">
                <section class="panel rank-panel">
                  <div class="panel-head"><div><span class="eyebrow">Return ranking</span><h3>Ridge · {html.escape(label)}</h3></div></div>
                  {_rank_table(data.get("ridge_top_10") or [], "ridge_prediction", "Score")}
                </section>
                <section class="panel rank-panel">
                  <div class="panel-head"><div><span class="eyebrow">Simple baseline</span><h3>Equal weight · {html.escape(label)}</h3></div></div>
                  {_rank_table(data.get("equal_weight_top_10") or [], "equal_weight_score", "Score")}
                </section>
                <section class="panel rank-panel risk-panel">
                  <div class="panel-head"><div><span class="eyebrow">Negative screen</span><h3>Highest AVOID risk · {html.escape(label)}</h3></div></div>
                  {_rank_table(data.get("avoid_highest_risk_10") or [], "avoid_probability", "Probability", risk=True)}
                </section>
              </div>
            </div>
            """
        )
    return "".join(sections)


def build_dashboard(
    result_dir: Path,
    spec_path: Path,
    output_path: Path,
) -> Path:
    spec = _read_json(spec_path, {})
    status = _read_json(result_dir / "r4_status.json", {})
    assessment = _read_json(result_dir / "r4_assessment.json", {"by_horizon": {}})
    latest = _read_json(result_dir / "r4_latest.json", {"by_horizon": {}})
    ranking = _read_csv(result_dir / "r4_monthly_ranking_metrics.csv")
    avoid = _read_csv(result_dir / "r4_monthly_avoid_metrics.csv")

    primary = int(spec.get("primary_research_horizon_bars", 126))
    inception = _fmt_date(status.get("prospective_inception_date") or spec.get("inception_date"))
    first_anchor = _fmt_date(status.get("first_eligible_anchor_date") or spec.get("first_eligible_anchor_date"))
    latest_market = _fmt_date(status.get("latest_market_date"))
    latest_anchor = _fmt_date(status.get("latest_issued_anchor_date"))
    prediction_rows = int(status.get("prediction_rows") or 0)
    matured_rows = int(status.get("matured_outcome_rows") or 0)
    universe = int(status.get("frozen_universe_count") or 0)
    latest_anchor_raw = status.get("latest_issued_anchor_date")

    primary_row = _assessment_row(assessment, primary)
    primary_status = primary_row.get("evidence_status", "INSUFFICIENT")
    primary_months = int(primary_row.get("evaluated_months") or 0)
    primary_ridge_ic = _ranking_metric(primary_row, "ridge", "mean_ic")
    primary_avoid_auc = _avoid_metric(primary_row, "mean_roc_auc")
    primary_lift = _avoid_metric(primary_row, "mean_top_risk_quintile_lift")
    primary_rel = _avoid_metric(primary_row, "mean_top_risk_quintile_mean_rel_return")

    ranking_records = []
    if not ranking.empty:
        for col in ["horizon_bars", "ic", "top_decile_mean_rel_return", "q5_minus_q1_spread"]:
            if col in ranking.columns:
                ranking[col] = pd.to_numeric(ranking[col], errors="coerce")
        ranking_records = ranking.where(pd.notna(ranking), None).to_dict(orient="records")

    avoid_records = []
    if not avoid.empty:
        for col in ["horizon_bars", "roc_auc", "top_risk_quintile_lift", "top_risk_quintile_mean_rel_return"]:
            if col in avoid.columns:
                avoid[col] = pd.to_numeric(avoid[col], errors="coerce")
        avoid_records = avoid.where(pd.notna(avoid), None).to_dict(orient="records")

    initial_horizon = str(primary)
    waiting = prediction_rows == 0
    if waiting:
        latest_body = f"""
        <section class="waiting-card">
          <div class="waiting-icon">⏳</div>
          <div>
            <span class="eyebrow">Prospective clock is running</span>
            <h2>Waiting for the first completed month</h2>
            <p>The first eligible cohort is <strong>{first_anchor}</strong>. R4 will issue it only after market data from the next calendar month exists. Nothing needs to be entered manually.</p>
          </div>
        </section>
        """
    else:
        latest_body = f"""
        <section class="section-head-row">
          <div><span class="eyebrow">Latest immutable cohort</span><h2>{latest_anchor}</h2></div>
          <div class="subtle">These are shadow research rankings, not production recommendations.</div>
        </section>
        {_latest_sections(latest, spec)}
        """

    horizon_cards = "".join(
        _horizon_card(h, spec, assessment, primary) for h in HORIZONS
    )

    source_links = """
      <a href="r4-data/r4_report.md">Research report</a>
      <a href="r4-data/r4_assessment.csv">Assessment CSV</a>
      <a href="r4-data/r4_predictions.csv">Predictions CSV</a>
      <a href="r4-data/r4_outcomes.csv">Outcomes CSV</a>
      <a href="r4-data/r4_status.json">Status JSON</a>
    """

    payload = {
        "ranking": ranking_records,
        "avoid": avoid_records,
        "horizonLabels": spec.get("horizon_labels", {}),
        "initialHorizon": int(primary),
    }

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>R4 Clean-Sheet Forward Research</title>
<style>
:root {{
  --bg:#071018;
  --bg2:#0b1621;
  --panel:#101e2b;
  --panel2:#0d1924;
  --line:#213445;
  --text:#e9f1f7;
  --muted:#8fa5b6;
  --blue:#58a6ff;
  --cyan:#48d5c8;
  --green:#5dd39e;
  --amber:#f6c85f;
  --red:#ff7b72;
  --purple:#b99cff;
  --shadow:0 18px 45px rgba(0,0,0,.22);
}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{
  margin:0;
  background:
    radial-gradient(circle at 15% -10%, rgba(72,213,200,.09), transparent 35%),
    radial-gradient(circle at 90% 0%, rgba(88,166,255,.08), transparent 30%),
    var(--bg);
  color:var(--text);
  font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
a{{color:var(--blue);text-decoration:none}}
a:hover{{text-decoration:underline}}
.shell{{max-width:1480px;margin:0 auto;padding:28px 28px 64px}}
.topbar{{display:flex;justify-content:space-between;align-items:flex-start;gap:22px;margin-bottom:24px}}
.brandline{{display:flex;align-items:center;gap:10px;color:var(--cyan);font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}}
.brand-dot{{width:9px;height:9px;border-radius:50%;background:var(--cyan);box-shadow:0 0 16px rgba(72,213,200,.8)}}
h1{{font-size:clamp(28px,4vw,48px);line-height:1.02;margin:10px 0 10px;letter-spacing:-.04em}}
.hero-sub{{color:var(--muted);font-size:15px;max-width:780px;line-height:1.6}}
.top-actions{{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}}
.btn{{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border:1px solid var(--line);background:rgba(16,30,43,.78);border-radius:10px;color:var(--text);font-size:13px;font-weight:700}}
.btn:hover{{border-color:#3a5871;text-decoration:none}}
.hero-grid{{display:grid;grid-template-columns:1.55fr .95fr;gap:18px;margin:22px 0}}
.panel{{background:linear-gradient(180deg,rgba(16,30,43,.96),rgba(13,25,36,.96));border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}}
.hero-main{{padding:24px}}
.hero-status{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}}
.eyebrow{{display:block;color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px}}
.hero-main h2,.panel h2{{margin:0;font-size:25px;letter-spacing:-.025em}}
.evidence-badge{{display:inline-flex;align-items:center;padding:6px 9px;border-radius:999px;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;border:1px solid}}
.status-muted{{color:#bac7d1;border-color:#425463;background:rgba(143,165,182,.08)}}
.status-info{{color:#8cc4ff;border-color:#345b7e;background:rgba(88,166,255,.08)}}
.status-good{{color:#82e5b8;border-color:#356c56;background:rgba(93,211,158,.08)}}
.status-warn{{color:#f7d889;border-color:#6c5c34;background:rgba(246,200,95,.08)}}
.status-bad{{color:#ff9a92;border-color:#70403c;background:rgba(255,123,114,.08)}}
.primary-kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:22px}}
.kpi{{padding:13px;border:1px solid var(--line);background:rgba(7,16,24,.35);border-radius:12px}}
.kpi span,.metric-grid span{{display:block;color:var(--muted);font-size:11px;margin-bottom:5px}}
.kpi strong{{font-size:19px}}
.guardrail{{margin-top:16px;padding:12px 14px;border-left:3px solid var(--purple);background:rgba(185,156,255,.06);color:#c7b8ee;font-size:12px;line-height:1.55}}
.progress-panel{{padding:20px}}
.progress-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}}
.progress-box{{padding:13px;border-radius:12px;background:rgba(7,16,24,.35);border:1px solid var(--line)}}
.progress-box span{{display:block;font-size:11px;color:var(--muted);margin-bottom:4px}}
.progress-box strong{{font-size:17px}}
.section-title{{display:flex;justify-content:space-between;align-items:end;gap:12px;margin:32px 0 13px}}
.section-title h2{{margin:0;font-size:22px}}
.section-title p{{margin:0;color:var(--muted);font-size:12px}}
.horizon-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.horizon-card{{text-align:left;color:inherit;padding:16px;border-radius:14px;background:var(--panel2);border:1px solid var(--line);cursor:pointer;transition:.15s transform,.15s border-color,.15s background;font:inherit}}
.horizon-card:hover{{transform:translateY(-1px);border-color:#3a5871}}
.horizon-card.active{{border-color:var(--cyan);box-shadow:0 0 0 1px rgba(72,213,200,.12) inset}}
.primary-card{{background:linear-gradient(180deg,rgba(72,213,200,.055),var(--panel2))}}
.horizon-top{{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:12px}}
.horizon-label{{font-size:20px;font-weight:850}}
.primary-pill{{display:inline-flex;margin-left:7px;padding:3px 7px;border-radius:999px;background:rgba(72,213,200,.11);color:var(--cyan);font-size:9px;font-weight:900;text-transform:uppercase;vertical-align:3px}}
.metric-grid.compact{{display:grid;grid-template-columns:1fr 1fr;gap:9px}}
.metric-grid.compact div{{padding:8px 0;border-top:1px solid rgba(33,52,69,.72)}}
.metric-grid.compact strong{{font-size:14px}}
.gate-row{{display:flex;flex-wrap:wrap;gap:5px;margin-top:11px}}
.gate{{font-size:9px;font-weight:800;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:4px 6px;text-transform:uppercase}}
.gate.pass{{color:var(--green);border-color:#356c56;background:rgba(93,211,158,.07)}}
.waiting-card{{display:flex;gap:18px;align-items:center;padding:22px;border:1px dashed #38526a;border-radius:16px;background:linear-gradient(120deg,rgba(88,166,255,.055),rgba(72,213,200,.035));margin-top:14px}}
.waiting-icon{{font-size:32px}}
.waiting-card h2{{margin:3px 0 7px}}
.waiting-card p{{margin:0;color:var(--muted);line-height:1.55}}
.section-head-row{{display:flex;justify-content:space-between;align-items:end;margin:22px 0 12px}}
.subtle{{color:var(--muted);font-size:12px}}
.rank-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.rank-panel{{padding:16px;box-shadow:none}}
.risk-panel{{border-color:#463638}}
.panel-head h3{{margin:2px 0 10px;font-size:16px}}
.rank-table{{width:100%;border-collapse:collapse;font-size:12px}}
.rank-table th{{color:var(--muted);font-weight:700;text-align:right;padding:7px 6px;border-bottom:1px solid var(--line)}}
.rank-table th:nth-child(2){{text-align:left}}
.rank-table td{{padding:7px 6px;border-bottom:1px solid rgba(33,52,69,.55);text-align:right}}
.rank-table td:nth-child(2){{text-align:left}}
.rank-num{{color:var(--muted);width:28px}}
.mono{{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.empty-mini{{color:var(--muted);font-size:12px;padding:18px 4px}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.chart-panel{{padding:16px}}
.chart-wrap{{height:245px;position:relative}}
canvas{{width:100%;height:100%}}
.chart-empty{{display:flex;height:100%;align-items:center;justify-content:center;color:var(--muted);font-size:12px;border:1px dashed var(--line);border-radius:10px}}
.feature-panel{{padding:18px}}
.feature-row{{display:grid;grid-template-columns:70px 1fr;gap:12px;padding:11px 0;border-bottom:1px solid rgba(33,52,69,.65)}}
.feature-row:last-child{{border-bottom:0}}
.feature-horizon{{font-weight:850}}
.feature-primary .feature-horizon{{color:var(--cyan)}}
.feature-list{{display:flex;flex-wrap:wrap;gap:7px}}
.feature-chip{{padding:5px 8px;border:1px solid var(--line);border-radius:8px;background:rgba(7,16,24,.35);font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#c8d7e2}}
.footer-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:28px}}
.footer-panel{{padding:18px}}
.footer-panel p{{color:var(--muted);font-size:12px;line-height:1.6}}
.source-links{{display:flex;flex-wrap:wrap;gap:10px}}
.source-links a{{font-size:12px;padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:rgba(7,16,24,.35)}}
.footer-note{{margin-top:22px;color:#647b8d;font-size:11px;line-height:1.6}}
.latest-horizon-panel{{display:none}}
.latest-horizon-panel.active{{display:block}}
@media(max-width:1100px){{
  .hero-grid{{grid-template-columns:1fr}}
  .horizon-grid{{grid-template-columns:1fr 1fr}}
  .rank-grid{{grid-template-columns:1fr}}
}}
@media(max-width:720px){{
  .shell{{padding:20px 14px 44px}}
  .topbar{{display:block}}
  .top-actions{{justify-content:flex-start;margin-top:14px}}
  .primary-kpis{{grid-template-columns:1fr 1fr}}
  .horizon-grid{{grid-template-columns:1fr}}
  .chart-grid,.footer-grid{{grid-template-columns:1fr}}
  .hero-status{{display:block}}
  .hero-status .evidence-badge{{margin-top:10px}}
  .section-title{{display:block}}
  .section-title p{{margin-top:6px}}
}}
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div>
      <div class="brandline"><span class="brand-dot"></span>Clean-Sheet Forward Research</div>
      <h1>R4 Prospective Research Dashboard</h1>
      <div class="hero-sub">A read-only view of the frozen clean-sheet models. Predictions are issued prospectively, outcomes mature automatically, and evidence gates cannot be changed by this dashboard.</div>
    </div>
    <nav class="top-actions">
      <a class="btn" href="index.html">← Production dashboard</a>
      <a class="btn" href="r4-data/r4_report.md">Open research report</a>
    </nav>
  </header>

  <section class="hero-grid">
    <div class="panel hero-main">
      <div class="hero-status">
        <div>
          <span class="eyebrow">Primary research horizon · 6 months</span>
          <h2>Forward evidence: {_status_label(primary_status)}</h2>
        </div>
        <span class="evidence-badge {_status_class(primary_status)}">{html.escape(_status_label(primary_status))}</span>
      </div>
      <div class="primary-kpis">
        <div class="kpi"><span>Evaluated months</span><strong>{_fmt_int(primary_months)}</strong></div>
        <div class="kpi"><span>Ridge rank IC</span><strong>{_fmt_num(primary_ridge_ic)}</strong></div>
        <div class="kpi"><span>AVOID AUC</span><strong>{_fmt_num(primary_avoid_auc)}</strong></div>
        <div class="kpi"><span>AVOID precision lift</span><strong>{_fmt_num(primary_lift,2)}×</strong></div>
      </div>
      <div class="guardrail">R4 is a prospective shadow study. A high ranking is not a recommendation. FAVOR is only earned if the frozen prospective evidence gate passes; AVOID is only earned if its separate frozen gate passes.</div>
    </div>

    <aside class="panel progress-panel">
      <span class="eyebrow">Study progress</span>
      <h2>Prospective since {inception}</h2>
      <div class="progress-grid">
        <div class="progress-box"><span>Latest market date</span><strong>{latest_market}</strong></div>
        <div class="progress-box"><span>Latest cohort</span><strong>{latest_anchor}</strong></div>
        <div class="progress-box"><span>Prediction rows</span><strong>{_fmt_int(prediction_rows)}</strong></div>
        <div class="progress-box"><span>Matured outcomes</span><strong>{_fmt_int(matured_rows)}</strong></div>
        <div class="progress-box"><span>Frozen universe</span><strong>{_fmt_int(universe)}</strong></div>
        <div class="progress-box"><span>First eligible anchor</span><strong>{first_anchor}</strong></div>
      </div>
    </aside>
  </section>

  <div class="section-title">
    <div><span class="eyebrow">Evidence by horizon</span><h2>1M / 3M / 6M / 9M</h2></div>
    <p>Click a horizon to change rankings and charts below.</p>
  </div>
  <section class="horizon-grid">{horizon_cards}</section>

  {latest_body}

  <div class="section-title">
    <div><span class="eyebrow">Prospective performance</span><h2>Monthly evidence</h2></div>
    <p>Charts remain blank until outcomes mature.</p>
  </div>
  <section class="chart-grid">
    <div class="panel chart-panel">
      <div class="panel-head"><span class="eyebrow">Return ranking</span><h3>Monthly Spearman IC</h3></div>
      <div class="chart-wrap"><canvas id="rankingChart"></canvas><div id="rankingEmpty" class="chart-empty">No matured monthly ranking observations yet.</div></div>
    </div>
    <div class="panel chart-panel">
      <div class="panel-head"><span class="eyebrow">Negative screen</span><h3>AVOID precision lift</h3></div>
      <div class="chart-wrap"><canvas id="avoidChart"></canvas><div id="avoidEmpty" class="chart-empty">No matured monthly AVOID observations yet.</div></div>
    </div>
  </section>

  <div class="section-title">
    <div><span class="eyebrow">Frozen design</span><h2>Model inputs</h2></div>
    <p>These feature sets are not re-selected during R4.</p>
  </div>
  <section class="panel feature-panel">{_features_table(spec)}</section>

  <section class="footer-grid">
    <div class="panel footer-panel">
      <span class="eyebrow">Research controls</span>
      <h3>What R4 is allowed to do</h3>
      <p>R4 may refresh the same frozen Ridge and Logistic models annually using newly matured data. It may not rediscover features, tune thresholds, change hyperparameters, rewrite old predictions, or add new tickers to the V1 frozen universe.</p>
    </div>
    <div class="panel footer-panel">
      <span class="eyebrow">Audit files</span>
      <h3>Download the underlying records</h3>
      <div class="source-links">{source_links}</div>
    </div>
  </section>

  <div class="footer-note">
    As of {latest_market}. R4 inception {inception}. Latest cohort {latest_anchor}. Production logic changed: no. Historical reoptimization performed: no. Dashboard is informational research output only.
  </div>
</div>

<script>
const DATA={_safe_for_script(payload)};
let selectedHorizon=DATA.initialHorizon;

function activateHorizon(h){{
  selectedHorizon=Number(h);
  document.querySelectorAll('.horizon-card').forEach(el=>el.classList.toggle('active', Number(el.dataset.horizon)===selectedHorizon));
  document.querySelectorAll('.latest-horizon-panel').forEach(el=>el.classList.toggle('active', Number(el.dataset.horizonPanel)===selectedHorizon));
  drawAll();
}}
document.querySelectorAll('.horizon-card').forEach(el=>el.addEventListener('click',()=>activateHorizon(el.dataset.horizon)));

function css(name){{return getComputedStyle(document.documentElement).getPropertyValue(name).trim();}}
function prepCanvas(canvas){{
  const dpr=window.devicePixelRatio||1;
  const rect=canvas.getBoundingClientRect();
  canvas.width=Math.max(1,Math.floor(rect.width*dpr));
  canvas.height=Math.max(1,Math.floor(rect.height*dpr));
  const ctx=canvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {{ctx,w:rect.width,h:rect.height}};
}}
function lineChart(canvasId, emptyId, series, baseline){{
  const canvas=document.getElementById(canvasId), empty=document.getElementById(emptyId);
  const rows=series.flatMap(s=>s.rows);
  if(!rows.length){{canvas.style.display='none';empty.style.display='flex';return;}}
  canvas.style.display='block';empty.style.display='none';
  const {{ctx,w,h}}=prepCanvas(canvas);
  ctx.clearRect(0,0,w,h);
  const pad={{l:42,r:16,t:18,b:34}};
  const all=series.flatMap(s=>s.rows.map(r=>r.y)).filter(Number.isFinite);
  if(Number.isFinite(baseline)) all.push(baseline);
  let ymin=Math.min(...all), ymax=Math.max(...all);
  if(ymin===ymax){{ymin-=.1;ymax+=.1;}}
  const extra=(ymax-ymin)*.14||.1; ymin-=extra;ymax+=extra;
  const dates=[...new Set(rows.map(r=>r.x))].sort();
  const X=x=>pad.l+(dates.length<=1?0.5:(dates.indexOf(x)/(dates.length-1)))*(w-pad.l-pad.r);
  const Y=y=>pad.t+(1-(y-ymin)/(ymax-ymin))*(h-pad.t-pad.b);
  ctx.strokeStyle=css('--line');ctx.lineWidth=1;
  for(let i=0;i<4;i++){{const y=pad.t+i*(h-pad.t-pad.b)/3;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();}}
  ctx.fillStyle=css('--muted');ctx.font='10px system-ui';
  ctx.textAlign='right';ctx.fillText(ymax.toFixed(2),pad.l-7,pad.t+3);ctx.fillText(ymin.toFixed(2),pad.l-7,h-pad.b+3);
  if(Number.isFinite(baseline)){{ctx.setLineDash([4,4]);ctx.strokeStyle='#52697c';ctx.beginPath();ctx.moveTo(pad.l,Y(baseline));ctx.lineTo(w-pad.r,Y(baseline));ctx.stroke();ctx.setLineDash([]);}}
  series.forEach(s=>{{
    if(!s.rows.length)return;
    ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.lineWidth=2;ctx.beginPath();
    s.rows.forEach((r,i)=>{{const x=X(r.x),y=Y(r.y);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}});
    ctx.stroke();
    s.rows.forEach(r=>{{ctx.beginPath();ctx.arc(X(r.x),Y(r.y),2.5,0,Math.PI*2);ctx.fill();}});
  }});
  ctx.textAlign='center';ctx.fillStyle=css('--muted');
  dates.slice(-4).forEach(d=>ctx.fillText(String(d).slice(0,7),X(d),h-10));
  let lx=pad.l;series.forEach(s=>{{ctx.fillStyle=s.color;ctx.fillRect(lx,pad.t-12,10,3);ctx.fillStyle=css('--muted');ctx.textAlign='left';ctx.fillText(s.name,lx+14,pad.t-8);lx+=90;}});
}}
function drawAll(){{
  const rank=DATA.ranking.filter(r=>Number(r.horizon_bars)===selectedHorizon);
  const ridge=rank.filter(r=>r.method==='ridge' && Number.isFinite(Number(r.ic))).map(r=>({{x:r.anchor_date,y:Number(r.ic)}}));
  const equal=rank.filter(r=>r.method==='equal_weight' && Number.isFinite(Number(r.ic))).map(r=>({{x:r.anchor_date,y:Number(r.ic)}}));
  lineChart('rankingChart','rankingEmpty',[
    {{name:'Ridge',color:css('--cyan'),rows:ridge}},
    {{name:'Equal',color:css('--blue'),rows:equal}}
  ],0);
  const av=DATA.avoid.filter(r=>Number(r.horizon_bars)===selectedHorizon && Number.isFinite(Number(r.top_risk_quintile_lift))).map(r=>({{x:r.anchor_date,y:Number(r.top_risk_quintile_lift)}}));
  lineChart('avoidChart','avoidEmpty',[{{name:'AVOID lift',color:css('--amber'),rows:av}}],1);
}}
window.addEventListener('resize',drawAll);
activateHorizon(selectedHorizon);
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static R4 prospective research dashboard.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = build_dashboard(args.result_dir, args.spec, args.output)
    print(f"R4 dashboard written to {path}")


if __name__ == "__main__":
    main()
