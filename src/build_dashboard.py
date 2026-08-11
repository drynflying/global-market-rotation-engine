from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import DOCS_DATA_DIR, DOCS_DIR


def _fmt(value, digits=1, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def _state_class(state: str) -> str:
    return {
        "ROTATION_IN": "good",
        "ACCUMULATING": "good2",
        "NEUTRAL": "neutral",
        "WEAKENING": "warn",
        "ROTATION_OUT": "bad",
    }.get(state, "muted")


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

    as_of = latest["date"].max().strftime("%Y-%m-%d")
    scored = latest[
        latest["rotation_score"].notna()
        & latest["rank_eligible"].astype(bool)
    ].copy()

    leaders = scored.nlargest(15, "rotation_score")
    movers = scored[scored["score_change_20"].notna()].nlargest(15, "score_change_20")
    weakening = scored[scored["score_change_20"].notna()].nsmallest(10, "score_change_20")

    groups = []
    for group, g in scored.groupby("rotation_group"):
        g = g.sort_values("rotation_score", ascending=False)
        groups.append(
            {
                "group": group,
                "count": int(len(g)),
                "leader": g.iloc[0]["ticker"] if len(g) else None,
                "leader_score": float(g.iloc[0]["rotation_score"]) if len(g) else None,
                "median_score": float(g["rotation_score"].median()) if len(g) else None,
            }
        )
    groups = sorted(groups, key=lambda x: x["group"])

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
            {
                "date": d.strftime("%Y-%m-%d"),
                "score": round(float(s), 2),
            }
            for d, s in zip(g["date"], g["rotation_score"])
        ]

    # Machine-readable copies beside the dashboard.
    latest_json = latest.replace({np.nan: None}).to_dict(orient="records")
    (DOCS_DATA_DIR / "rotation_latest.json").write_text(
        json.dumps(latest_json, default=str, indent=2),
        encoding="utf-8",
    )
    (DOCS_DATA_DIR / "ai_analysis.json").write_text(
        json.dumps(ai_analysis, indent=2),
        encoding="utf-8",
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
                  <td><span class="badge {_state_class(str(r.get('rotation_state')))}">{html.escape(str(r.get('rotation_state')))}</span></td>
                </tr>
                """
            )
        return "\n".join(result)

    def ai_cards(key, title):
        items = ai_analysis.get(key, [])
        cards = []
        for item in items[:5]:
            cards.append(
                f"""
                <article class="mini-card">
                  <div class="mini-head">
                    <strong>{html.escape(item.get('ticker',''))}</strong>
                    <span>{html.escape(item.get('confidence',''))}</span>
                  </div>
                  <div class="mini-title">{html.escape(item.get('title',''))}</div>
                  <p>{html.escape(item.get('explanation',''))}</p>
                </article>
                """
            )
        return f"<section><h2>{html.escape(title)}</h2><div class='mini-grid'>{''.join(cards) or '<p class=\"muted\">No items.</p>'}</div></section>"

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
.sub {{ color:var(--muted); margin-bottom:22px; }}
.grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
.card, .mini-card {{
  background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:16px;
}}
.metric {{ font-size:28px; font-weight:700; margin-top:6px; }}
.muted {{ color:var(--muted); }}
.ai-summary {{
  background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:18px; margin-top:18px;
}}
.mini-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
.mini-head {{ display:flex; justify-content:space-between; color:var(--muted); }}
.mini-title {{ margin-top:8px; font-weight:700; }}
.mini-card p {{ margin-bottom:0; color:var(--muted); }}
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
@media (max-width:800px) {{
  .grid,.mini-grid,.chart-grid {{ grid-template-columns:1fr; }}
  main {{ padding:14px; }}
  .table-wrap {{ overflow-x:auto; }}
}}
</style>
</head>
<body>
<main>
  <h1>Market Rotation Dashboard</h1>
  <div class="sub">As of {as_of} · Longer-horizon rotation monitoring, not a daily trading signal</div>

  <div class="grid">
    <div class="card"><div class="muted">Scored signals</div><div class="metric">{len(scored)}</div></div>
    <div class="card"><div class="muted">Rotation in</div><div class="metric">{int((scored['rotation_state']=='ROTATION_IN').sum())}</div></div>
    <div class="card"><div class="muted">Accumulating</div><div class="metric">{int((scored['rotation_state']=='ACCUMULATING').sum())}</div></div>
    <div class="card"><div class="muted">Weakening / out</div><div class="metric">{int(scored['rotation_state'].isin(['WEAKENING','ROTATION_OUT']).sum())}</div></div>
  </div>

  <div class="ai-summary">
    <div class="muted">AI interpretation</div>
    <h2 style="margin-top:6px">{html.escape(ai_analysis.get('headline','Rotation summary'))}</h2>
    <p><strong>{html.escape(ai_analysis.get('market_regime',''))}</strong></p>
    <p>{html.escape(ai_analysis.get('executive_summary',''))}</p>
    <div class="muted">Provider status: {html.escape(ai_analysis.get('provider_status','unknown'))}</div>
  </div>

  {ai_cards('emerging_rotations','Emerging rotations')}
  {ai_cards('persistent_leaders','Persistent leaders')}
  {ai_cards('weakening_rotations','Weakening rotations')}

  <section>
    <h2>63-bar score trends selected for attention</h2>
    <div id="charts" class="chart-grid"></div>
  </section>

  <section>
    <h2>Highest current rotation scores</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Ticker</th><th>Exposure</th><th>Score</th><th>20-bar Δ</th><th>RS20</th><th>State</th></tr></thead>
        <tbody>{rows(leaders)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Biggest 20-bar improvements</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Ticker</th><th>Exposure</th><th>Score</th><th>20-bar Δ</th><th>RS20</th><th>State</th></tr></thead>
        <tbody>{rows(movers)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Biggest 20-bar deterioration</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Ticker</th><th>Exposure</th><th>Score</th><th>20-bar Δ</th><th>RS20</th><th>State</th></tr></thead>
        <tbody>{rows(weakening)}</tbody>
      </table>
    </div>
  </section>

  <p class="muted" style="margin-top:30px">
    The rotation score is a comparative quantitative indicator. It is not a forecast,
    recommendation, or direct measure of institutional creations/redemptions.
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
