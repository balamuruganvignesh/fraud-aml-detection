"""Persist cases to SQLite/JSON and render a self-contained HTML analyst report."""

from __future__ import annotations

import html
import json
import math
from collections import Counter
import sqlite3
from datetime import datetime
from pathlib import Path

import networkx as nx

from .models import CLOSE, ESCALATE, QUEUES, REVIEW, Case

ROLE_COLORS = {
    "funder": "#b42318", "origin": "#b42318", "source": "#d9480f", "collector": "#c2410c",
    "sink": "#9a3412", "exit": "#9a3412", "beneficiary": "#9a3412", "cycle_member": "#7c3aed",
    "intermediary": "#2563eb", "mule": "#0e7490", "subject": "#4b5563", "related": "#9ca3af",
}


# ------------------------------------------------------------------ persistence
def save_cases(conn: sqlite3.Connection, cases: list[Case]) -> None:
    with conn:
        for table in ("case_transactions", "case_accounts", "cases"):
            conn.execute(f"DELETE FROM {table}")
        for c in cases:
            r = c.llm_review or {}
            conn.execute(
                "INSERT INTO cases (case_id, risk_score, typologies, n_accounts, n_transactions, total_amount,"
                " is_alert, llm_source, llm_flag, llm_risk_level, llm_confidence, llm_summary, llm_json,"
                " disposition) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (c.case_id, c.risk_score, ",".join(c.typologies), len(c.accounts), len(c.txn_ids),
                 c.total_amount, int(c.is_alert), r.get("source"),
                 None if not r else int(r["flag"]), r.get("risk_level"), r.get("confidence"),
                 r.get("summary"), json.dumps(r) if r else None, c.disposition),
            )
            conn.executemany("INSERT INTO case_accounts VALUES (?,?,?)",
                             [(c.case_id, a, c.roles.get(a, "related")) for a in c.accounts])
            conn.executemany("INSERT INTO case_transactions VALUES (?,?)", [(c.case_id, t) for t in c.txn_ids])


def case_to_dict(c: Case) -> dict:
    return {
        "case_id": c.case_id, "risk_score": c.risk_score, "is_alert": c.is_alert,
        "typologies": c.typologies, "score_breakdown": c.score_breakdown,
        "accounts": [{"account_id": a, "role": c.roles.get(a, "related")} for a in c.accounts],
        "n_transactions": len(c.txn_ids), "total_amount": c.total_amount,
        "findings": [{"detector": f.detector, "source": f.source, "strength": round(f.strength, 3),
                      "details": f.details} for f in c.findings],
        "llm_review": c.llm_review,
        "disposition": c.disposition,
        "policy_override": c.policy_override,
    }


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


# ----------------------------------------------------------------------- graph
def case_svg(case: Case, width: int = 520) -> str:
    edges = case.evidence.get("graph", {}).get("edges", [])
    if not edges:
        return ""
    g = nx.DiGraph()
    for e in edges:
        g.add_edge(e["source"], e["target"], amount=e["amount"], count=e["count"])
    if nx.is_directed_acyclic_graph(g):
        # Money flows one way: lay out left-to-right by hop (funder -> source -> mules -> sink).
        for layer, nodes in enumerate(nx.topological_generations(g)):
            for n in nodes:
                g.nodes[n]["layer"] = layer
        pos = nx.multipartite_layout(g, subset_key="layer")
        tallest = max(Counter(nx.get_node_attributes(g, "layer").values()).values())
    else:
        pos = nx.circular_layout(g) if len(g) <= 8 else nx.spring_layout(g, seed=7)
        tallest = 8
    height = max(260, min(640, 34 * tallest + 60))
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    pad = 40

    def sx(x):
        return pad + (x - min(xs)) / ((max(xs) - min(xs)) or 1) * (width - 2 * pad - 30)  # room for labels

    def sy(y):
        return pad + (y - min(ys)) / ((max(ys) - min(ys)) or 1) * (height - 2 * pad)

    max_amt = max(d["amount"] for _, _, d in g.edges(data=True))
    parts = [f'<svg viewBox="0 0 {width} {height}" class="graph" role="img" '
             f'aria-label="Transaction graph for {case.case_id}">',
             '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
             'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--edge)"/></marker></defs>']
    r = 9
    for u, v, d in g.edges(data=True):
        x1, y1, x2, y2 = sx(pos[u][0]), sy(pos[u][1]), sx(pos[v][0]), sy(pos[v][1])
        dist = math.hypot(x2 - x1, y2 - y1) or 1
        x2s, y2s = x2 - (x2 - x1) / dist * (r + 3), y2 - (y2 - y1) / dist * (r + 3)
        w = 1 + 3 * d["amount"] / max_amt
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2s:.1f}" y2="{y2s:.1f}" stroke="var(--edge)" '
                     f'stroke-width="{w:.1f}" marker-end="url(#arr)"><title>{u} → {v}: '
                     f'${d["amount"]:,.0f} ({d["count"]} txn)</title></line>')
    for n in g.nodes:
        role = case.roles.get(n, "related")
        x, y = sx(pos[n][0]), sy(pos[n][1])
        parts.append(f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{ROLE_COLORS.get(role, "#9ca3af")}">'
                     f'<title>{n} ({role})</title></circle>'
                     f'<text x="{x + r + 3:.1f}" y="{y + 3:.1f}">{n[-5:]}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------- report
_CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1c1c1a;--muted:#6b6b66;--line:#e4e4df;--edge:#8a8a84;
--crit:#b42318;--high:#c2410c;--med:#a16207;--low:#4b5563;--ok:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecece8;--muted:#a3a39c;
--line:#33332f;--edge:#7a7a74}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 64px}h1{font-size:26px;margin:0 0 4px}
.muted{color:var(--muted)}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
gap:12px;margin:24px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.kpi b{display:block;font-size:24px;font-variant-numeric:tabular-nums}.kpi span{color:var(--muted);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}
.case{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}
.case h3{margin:0;font-size:18px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.pill{font-size:12px;font-weight:600;padding:2px 8px;border-radius:99px;color:#fff}
.CRITICAL{background:var(--crit)}.HIGH{background:var(--high)}.MEDIUM{background:var(--med)}.LOW{background:var(--low)}
.escalate{background:var(--crit)}.review{background:var(--med)}.close{background:var(--ok)}h2 small{font-size:13px;font-weight:400;color:var(--muted);margin-left:8px}.tag{font-size:12px;border:1px solid var(--line);
border-radius:6px;padding:1px 6px;color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:12px}
@media (max-width:820px){.grid{grid-template-columns:1fr}}.graph{width:100%;height:auto;background:var(--bg);
border-radius:8px}.graph text{font-size:10px;fill:var(--muted)}ul{margin:4px 0 8px;padding-left:20px}
h4{margin:12px 0 2px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
details summary{cursor:pointer;color:var(--muted);font-size:13px;margin-top:10px}.legend span{margin-right:12px;font-size:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
.scroll{overflow-x:auto}
"""


def _e(x) -> str:
    return html.escape(str(x))


def render_html(cases: list[Case], metrics: dict, run_info: dict, max_cases: int = 25) -> str:
    alerts = [c for c in cases if c.is_alert]
    queued = {q: [c for c in alerts if c.disposition == q] for q in QUEUES}
    sar = [c for c in alerts if (c.llm_review or {}).get("sar_recommended")]
    kpis = [
        (f"{run_info['n_transactions']:,}", "transactions analysed"),
        (f"{len(cases)}", "candidate cases"),
        (f"{len(alerts)}", f"alerts (score ≥ {run_info['alert_threshold']:.0f})"),
        (f"{len(queued[ESCALATE])}", "escalate now"),
        (f"{len(queued[REVIEW])}", "lower-priority review"),
        (f"{len(queued[CLOSE])}", "closed by LLM review"),
        (f"{len(sar)}", "SAR recommended"),
    ]
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>AML Alert Report</title><style>{_CSS}</style>"
           f"</head><body><main><h1>Fraud &amp; AML Alert Report</h1>"
           f"<div class='muted'>Generated {_e(run_info['generated_at'])} · reviewer: {_e(run_info['reviewer'])}"
           f"</div><div class='kpis'>"]
    out += [f"<div class='kpi'><b>{v}</b><span>{_e(l)}</span></div>" for v, l in kpis]
    out.append("</div>")

    if metrics:
        a, al, rg = metrics["accounts"], metrics["alerts"], metrics["rings"]
        out.append("<h2>Detection quality vs. ground truth</h2><div class='scroll'><table><tr><th>Metric</th>"
                   "<th>Value</th></tr>")
        rows = [("Account precision / recall / F1", f"{a['precision']:.2f} / {a['recall']:.2f} / {a['f1']:.2f}"),
                ("Alert case precision", f"{al['case_precision']:.2f} ({al['true_positive_cases']}/{al['total']})"),
                ("Planted rings detected", f"{rg['detected']}/{rg['total']}"),
                ("Rings detected by typology", ", ".join(f"{k}: {v}" for k, v in rg["by_typology"].items()))]
        if metrics.get("llm_review"):
            lr = metrics["llm_review"]
            rows.append(("LLM review: flagged / reviewed", f"{lr['flagged']}/{lr['reviewed']} "
                         f"(precision {lr['flag_precision']}, cleared {lr['false_alerts_cleared']}"
                         f"/{lr['false_alerts_total']} false alerts)"))
            for q, d in lr.get("queues", {}).items():
                rows.append((f"Queue '{q}': true / false alerts", f"{d['true_alerts']} / {d['false_alerts']}"))
        out += [f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in rows]
        out.append("</table></div>")

    titles = {ESCALATE: ("Escalate", "HIGH/CRITICAL, SAR recommended, or cycle/layering policy: investigate now"),
              REVIEW: ("Review", "flagged at MEDIUM/LOW without a SAR: lower-priority analyst check"),
              CLOSE: ("Closed", "not flagged by LLM review: close with rationale, keep for trends")}
    unreviewed = [c for c in alerts if c.disposition is None]
    for q, members in [*((q, queued[q]) for q in QUEUES), (None, unreviewed)]:
        if not members:
            continue
        title, note = titles.get(q, ("Unreviewed", "no LLM review"))
        out.append(f"<h2>{title} queue <small>{len(members)} · {_e(note)}</small></h2><div class='scroll'>"
                   "<table><tr><th>Case</th><th>Score</th><th>LLM risk</th><th>Typologies</th><th>Accounts</th>"
                   "<th>Txns</th><th>Amount</th></tr>")
        for c in members:
            r = c.llm_review or {}
            out.append(f"<tr><td><a href='#{c.case_id}'>{c.case_id}</a></td><td>{c.risk_score:.0f}</td>"
                       f"<td>{_e(r.get('risk_level', '–'))}{' (policy)' if c.policy_override else ''}</td>"
                       f"<td>{_e(', '.join(c.typologies))}</td>"
                       f"<td>{len(c.accounts)}</td><td>{len(c.txn_ids)}</td><td>${c.total_amount:,.0f}</td></tr>")
        out.append("</table></div>")
    priority = {q: i for i, q in enumerate(QUEUES)}
    alerts = sorted(alerts, key=lambda c: (priority.get(c.disposition, len(QUEUES)), -c.risk_score))

    out.append("<h2>Case files</h2><div class='legend'>" + "".join(
        f"<span><i style='background:{col}'></i>{_e(role)}</span>" for role, col in ROLE_COLORS.items()) + "</div>")
    for c in alerts[:max_cases]:
        out.append(_case_card(c))
    if len(alerts) > max_cases:
        out.append(f"<p class='muted'>{len(alerts) - max_cases} more alerts are in cases.json and the "
                   f"<code>cases</code> table.</p>")
    out.append("</main></body></html>")
    return "".join(out)


def _case_card(c: Case) -> str:
    r = c.llm_review or {}
    q = c.disposition
    flag = f"<span class='pill {q}'>{q.upper()}</span>" if q else ""
    if c.policy_override:
        flag += (f"<span class='tag'>policy escalation: {_e(', '.join(c.policy_override))} "
                 f"(LLM said {_e(r.get('risk_level', '–'))}{'' if r.get('flag') else ', not flagged'})</span>")
    sar = "<span class='tag'>SAR recommended</span>" if r.get("sar_recommended") else ""

    def ul(items):
        return "<ul>" + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>" if items else "<p class='muted'>None</p>"

    txn_rows = "".join(
        f"<tr><td>{_e(t['time'])}</td><td>{_e(t['from'])}</td><td>{_e(t['to'])}</td>"
        f"<td>${t['amount']:,.2f}</td></tr>" for t in c.evidence.get("transactions_sample", []))
    acct_rows = "".join(
        f"<tr><td>{_e(a['account_id'])}</td><td>{_e(a['role'])}</td><td>{_e(a['account_type'])}</td>"
        f"<td>{_e(a['country'])}</td><td>{_e(a['kyc_risk'])}</td><td>{_e(a['open_date'])}</td>"
        f"<td>{a['pagerank_percentile']}</td></tr>" for a in c.evidence.get("accounts", []))
    return (
        f"<section class='case' id='{c.case_id}'><h3>{c.case_id}"
        f"<span class='pill {_e(r.get('risk_level', 'LOW'))}'>{_e(r.get('risk_level', '–'))}</span>{flag}{sar}"
        f"<span class='tag'>score {c.risk_score:.0f}</span><span class='tag'>${c.total_amount:,.0f}</span></h3>"
        f"<p class='muted'>{_e(', '.join(c.typologies))} · reviewed by {_e(r.get('source', '–'))}"
        f" · confidence {r.get('confidence', '–')}</p>"
        f"<div class='grid'><div><h4>Summary</h4><p>{_e(r.get('summary', ''))}</p>"
        f"<h4>Key indicators</h4>{ul(r.get('key_indicators', []))}"
        f"<h4>Mitigating factors</h4>{ul(r.get('mitigating_factors', []))}"
        f"<h4>Recommended actions</h4>{ul(r.get('recommended_actions', []))}</div>"
        f"<div><h4>Transaction network</h4>{case_svg(c)}"
        f"<h4>Score breakdown</h4><p>{_e(', '.join(f'{k} {v}' for k, v in c.score_breakdown.items()))}</p></div></div>"
        f"<details><summary>Accounts ({len(c.accounts)})</summary><div class='scroll'><table><tr><th>Account</th>"
        f"<th>Role</th><th>Type</th><th>Country</th><th>KYC</th><th>Opened</th><th>PageRank pct</th></tr>"
        f"{acct_rows}</table></div></details>"
        f"<details><summary>Transactions ({len(c.txn_ids)})</summary><div class='scroll'><table><tr><th>Time</th>"
        f"<th>From</th><th>To</th><th>Amount</th></tr>{txn_rows}</table></div></details></section>"
    )


def run_info(n_accounts: int, n_txns: int, alert_threshold: float, reviewer: str) -> dict:
    return {"n_accounts": n_accounts, "n_transactions": n_txns, "alert_threshold": alert_threshold,
            "reviewer": reviewer, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
