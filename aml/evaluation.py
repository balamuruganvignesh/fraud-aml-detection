"""Score detection output against the synthetic ground-truth labels."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .models import QUEUES, Case


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def evaluate(cases: list[Case], accounts: pd.DataFrame, rings: list[dict]) -> dict:
    if "is_laundering" not in accounts.columns:
        return {}
    truth = set(accounts.loc[accounts["is_laundering"] == 1, "account_id"])
    alerts = [c for c in cases if c.is_alert]
    flagged = {a for c in alerts for a in c.accounts}

    account_metrics = _prf(len(flagged & truth), len(flagged - truth), len(truth - flagged))

    def is_true_case(c: Case) -> bool:
        return sum(a in truth for a in c.accounts) / len(c.accounts) >= 0.5

    true_alerts = sum(is_true_case(c) for c in alerts)
    ring_hits, by_typology = [], defaultdict(lambda: [0, 0])
    for ring in rings:
        members = set(ring["accounts"])
        coverage = len(members & flagged) / len(members)
        hit = coverage >= 0.5
        ring_hits.append({"ring_id": ring["ring_id"], "typology": ring["typology"],
                          "coverage": round(coverage, 2), "detected": hit})
        by_typology[ring["typology"]][0] += hit
        by_typology[ring["typology"]][1] += 1

    reviewed = [c for c in alerts if c.llm_review]
    llm = {}
    if reviewed:
        flagged_true = sum(1 for c in reviewed if c.llm_review["flag"] and is_true_case(c))
        flagged_all = sum(1 for c in reviewed if c.llm_review["flag"])
        cleared_false = sum(1 for c in reviewed if not c.llm_review["flag"] and not is_true_case(c))
        llm = {
            "reviewed": len(reviewed),
            "flagged": flagged_all,
            "flag_precision": round(flagged_true / flagged_all, 3) if flagged_all else None,
            "false_alerts_cleared": cleared_false,
            "false_alerts_total": sum(1 for c in reviewed if not is_true_case(c)),
            "queues": {q: {"true_alerts": sum(is_true_case(c) for c in members),
                           "false_alerts": sum(not is_true_case(c) for c in members)}
                       for q in QUEUES
                       if (members := [c for c in reviewed if c.disposition == q])},
        }

    return {
        "accounts": account_metrics,
        "alerts": {"total": len(alerts), "true_positive_cases": true_alerts,
                   "case_precision": round(true_alerts / len(alerts), 3) if alerts else 0.0},
        "rings": {"total": len(rings), "detected": sum(r["detected"] for r in ring_hits),
                  "by_typology": {t: f"{h}/{n}" for t, (h, n) in sorted(by_typology.items())},
                  "detail": ring_hits},
        "llm_review": llm,
    }
