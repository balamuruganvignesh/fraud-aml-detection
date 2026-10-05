"""Merge detector findings into investigation cases, score them, and assemble evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

from .config import DetectionConfig
from .graph_analysis import TransactionGraph
from .models import Case, Finding

# When an account plays several roles, keep the most informative one.
_ROLE_PRIORITY = ["funder", "origin", "source", "collector", "sink", "exit", "beneficiary",
                  "cycle_member", "intermediary", "mule", "subject"]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def build_cases(findings: list[Finding], g: TransactionGraph, cfg: DetectionConfig) -> list[Case]:
    """Findings that share an account are merged into one case (connected components).

    High-degree hubs (merchants, payroll employers, escrow agents) touch many unrelated
    flows, so they are excluded as bridges; otherwise one popular merchant would chain
    every nearby finding into a single giant case.
    """
    uf = _UnionFind(len(findings))
    first_seen: dict[str, int] = {}
    for i, f in enumerate(findings):
        for acct in f.accounts:
            if g.degree(acct) >= cfg.hub_min_degree:
                continue
            if acct in first_seen:
                uf.union(i, first_seen[acct])
            else:
                first_seen[acct] = i

    groups: dict[int, list[Finding]] = defaultdict(list)
    for i, f in enumerate(findings):
        groups[uf.find(i)].append(f)

    cases = []
    for group in groups.values():
        accounts = list(dict.fromkeys(a for f in group for a in f.accounts))
        txn_ids = sorted({t for f in group for t in f.txn_ids})
        roles: dict[str, str] = {}
        for f in group:
            for acct, role in f.roles.items():
                if acct not in roles or _ROLE_PRIORITY.index(role) < _ROLE_PRIORITY.index(roles[acct]):
                    roles[acct] = role
        score, breakdown = score_case(group, accounts, g, cfg)
        cases.append(Case(
            case_id="", accounts=accounts, txn_ids=txn_ids, findings=group, roles=roles,
            risk_score=score, score_breakdown=breakdown,
            total_amount=round(sum(g.txns[t].amount for t in txn_ids if t in g.txns), 2),
            is_alert=score >= cfg.alert_threshold,
        ))

    cases.sort(key=lambda c: (-c.risk_score, -c.total_amount))
    for i, case in enumerate(cases, start=1):
        case.case_id = f"CASE-{i:04d}"
    return cases


def score_case(group: list[Finding], accounts: list[str], g: TransactionGraph,
               cfg: DetectionConfig) -> tuple[float, dict[str, float]]:
    """Sum of detector weights x strongest hit per detector, plus a KYC bonus; capped at 100."""
    best: dict[str, float] = {}
    for f in group:
        best[f.detector] = max(best.get(f.detector, 0.0), f.strength)
    breakdown = {d: round(cfg.weights.get(d, 10.0) * s, 1) for d, s in best.items()}
    high_kyc = sum(1 for a in accounts if g.agg.nodes[a].get("kyc_risk") == "high")
    if high_kyc:
        breakdown["high_risk_kyc"] = min(2, high_kyc) * cfg.high_risk_kyc_bonus
    return round(min(100.0, sum(breakdown.values())), 1), breakdown


def attach_evidence(case: Case, g: TransactionGraph, max_txns: int = 40) -> dict:
    """Structured, LLM-ready description of the case (also used by the HTML report)."""
    txns = sorted((g.txns[t] for t in case.txn_ids if t in g.txns), key=lambda t: t.ts)
    shown = txns if len(txns) <= max_txns else txns[: max_txns // 2] + txns[-max_txns // 2:]
    def fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    community_ids = {g.communities().get(a) for a in case.accounts}
    profiles = [{**g.account_profile(a), "role": case.roles.get(a, "related")} for a in case.accounts]
    case.evidence = {
        "case_id": case.case_id,
        "risk_score": case.risk_score,
        "score_breakdown": case.score_breakdown,
        "typologies_detected": case.typologies,
        "detector_findings": [
            {"detector": f.detector, "source": f.source, "strength": round(f.strength, 2), **f.details}
            for f in case.findings
        ],
        "context": case_context(profiles, txns),
        "accounts": profiles,
        "n_transactions": len(txns),
        "total_amount": case.total_amount,
        "time_span": {"first": fmt(txns[0].ts), "last": fmt(txns[-1].ts)} if txns else {},
        "transactions_sample": [
            {"txn_id": t.txn_id, "time": fmt(t.ts), "from": t.sender, "to": t.receiver,
             "amount": round(t.amount, 2), "channel": t.channel} for t in shown
        ],
        "transactions_truncated": len(txns) > len(shown),
        "graph": {
            "edges": g.subgraph_edges(case.accounts, case.txn_ids),
            "n_communities_spanned": len(community_ids - {None}),
        },
    }
    return case.evidence


def case_context(profiles: list[dict], txns: list) -> dict:
    """Case-level facts that weigh for *or against* laundering, so the reviewer sees both sides."""
    from .config import HIGH_RISK_COUNTRIES

    ages = [p["account_age_days"] for p in profiles if p.get("account_age_days") is not None]
    foreign = sorted({p["country"] for p in profiles if p.get("country") not in (None, "US")})
    by_role: dict[str, list[dict]] = defaultdict(list)
    for p in profiles:
        by_role[p["role"]].append(p)
    endpoints = by_role["sink"] + by_role["exit"] + by_role["beneficiary"]
    return {
        "channel_mix": dict(Counter(t.channel for t in txns).most_common()),
        "countries_outside_us": foreign,
        "high_risk_countries": [c for c in foreign if c in HIGH_RISK_COUNTRIES],
        "accounts_opened_within_90d": sum(a < 90 for a in ages),
        "median_account_age_days": sorted(ages)[len(ages) // 2] if ages else None,
        "accounts_with_salary_income": sum(p["receives_salary"] for p in profiles),
        "accounts_with_everyday_card_spend": sum(p["everyday_card_purchases"] > 0 for p in profiles),
        "final_destinations": [
            {"account_id": p["account_id"], "type": p["account_type"], "country": p["country"],
             "age_days": p["account_age_days"], "runs_payroll_to_n_people": p["runs_payroll_to_n_people"],
             "unique_counterparties": p["unique_senders"] + p["unique_receivers"]}
            for p in endpoints
        ],
    }
