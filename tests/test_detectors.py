"""Detector unit tests on tiny hand-built graphs, plus an end-to-end run on synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from aml.config import DetectionConfig
from aml.database import connect, load_dataset, load_rules, run_sql_rules
from aml.graph_analysis import (TransactionGraph, detect_cycles, detect_fan_in, detect_fan_out,
                                detect_layering_chains)

CFG = DetectionConfig()


def make(txns: list[tuple[str, str, float, str]], countries: dict[str, str] | None = None):
    """txns: (sender, receiver, amount, 'YYYY-MM-DD HH:MM:SS')."""
    ids = sorted({a for s, r, *_ in txns for a in (s, r)})
    countries = countries or {}
    accounts = pd.DataFrame({
        "account_id": ids, "account_type": "individual",
        "country": [countries.get(a, "US") for a in ids],
        "open_date": "2020-01-01", "kyc_risk": "low", "is_laundering": 0,
    })
    tx = pd.DataFrame(txns, columns=["sender", "receiver", "amount", "timestamp"])
    tx.insert(0, "txn_id", [f"T{i}" for i in range(len(tx))])
    tx["channel"], tx["is_laundering"], tx["typology"], tx["ring_id"] = "wire", 0, None, None
    return accounts, tx


def test_cycle_requires_time_order():
    ordered = [("A", "B", 50_000, "2026-09-01 10:00:00"), ("B", "C", 49_000, "2026-09-02 10:00:00"),
               ("C", "A", 48_000, "2026-09-03 10:00:00")]
    found = detect_cycles(TransactionGraph(*make(ordered)), CFG)
    assert len(found) == 1 and set(found[0].accounts) == {"A", "B", "C"}

    # Same edges, but the money "returns" before it left: not a round trip.
    scrambled = [("A", "B", 50_000, "2026-09-03 10:00:00"), ("B", "C", 49_000, "2026-09-02 10:00:00"),
                 ("C", "A", 48_000, "2026-09-01 10:00:00")]
    assert detect_cycles(TransactionGraph(*make(scrambled)), CFG) == []


def test_cycle_rejects_large_value_loss():
    txns = [("A", "B", 50_000, "2026-09-01 10:00:00"), ("B", "C", 10_000, "2026-09-02 10:00:00"),
            ("C", "A", 9_000, "2026-09-03 10:00:00")]
    assert detect_cycles(TransactionGraph(*make(txns)), CFG) == []


def test_layering_chain_keeps_only_maximal_chain():
    txns = [("O", "H1", 100_000, "2026-09-01 08:00:00"), ("H1", "H2", 98_000, "2026-09-01 12:00:00"),
            ("H2", "H3", 96_000, "2026-09-01 20:00:00"), ("H3", "X", 95_000, "2026-09-02 06:00:00")]
    found = detect_layering_chains(TransactionGraph(*make(txns)), CFG)
    assert len(found) == 1
    assert found[0].details["path"] == "O -> H1 -> H2 -> H3 -> X"
    assert found[0].roles["O"] == "origin" and found[0].roles["X"] == "exit"


def test_layering_chain_ignores_slow_hops():
    txns = [("O", "H1", 100_000, "2026-09-01 08:00:00"), ("H1", "H2", 98_000, "2026-09-05 12:00:00"),
            ("H2", "H3", 96_000, "2026-09-09 20:00:00"), ("H3", "X", 95_000, "2026-09-12 06:00:00")]
    assert detect_layering_chains(TransactionGraph(*make(txns)), CFG) == []


def _smurfing(forward: bool):
    txns = [(f"M{i}", "COL", 9_500, f"2026-09-01 {8 + i:02d}:00:00") for i in range(8)]
    if forward:
        txns.append(("COL", "OFF", 70_000, "2026-09-03 09:00:00"))
    return txns


def test_fan_in_needs_onward_movement():
    assert len(detect_fan_in(TransactionGraph(*make(_smurfing(forward=True))), CFG)) == 1
    # A merchant collecting from many customers but keeping the money is not a collector.
    assert detect_fan_in(TransactionGraph(*make(_smurfing(forward=False))), CFG) == []


def test_scatter_gather_vs_payroll():
    scatter = [("SRC", f"M{i}", 9_200, f"2026-09-01 {8 + i:02d}:00:00") for i in range(8)]
    scatter += [(f"M{i}", "SINK", 9_000, f"2026-09-02 {8 + i:02d}:00:00") for i in range(8)]
    found = detect_fan_out(TransactionGraph(*make(scatter)), CFG)
    assert [f.detector for f in found] == ["scatter_gather"]
    assert found[0].roles["SINK"] == "sink"

    # Payroll: many recipients, normal salaries, money does not re-converge.
    payroll = [("EMP", f"P{i}", 4_000, f"2026-09-01 09:{i:02d}:00") for i in range(10)]
    assert detect_fan_out(TransactionGraph(*make(payroll)), CFG) == []


def test_sql_rules_parse_and_fire(tmp_path):
    assert set(load_rules()) >= {"structuring", "structuring_receiver", "rapid_pass_through",
                                 "high_velocity", "high_risk_jurisdiction"}
    txns = [("S", "R", 9_400, f"2026-09-0{d} 10:00:00") for d in range(1, 5)]
    txns.append(("R", "KY1", 30_000, "2026-09-05 10:00:00"))
    conn = connect(tmp_path / "t.db")
    load_dataset(conn, *make(txns, countries={"KY1": "KY"}))
    fired = {(f.detector, f.accounts[0]) for f in run_sql_rules(conn, CFG)}
    assert ("structuring", "S") in fired
    assert ("high_risk_jurisdiction", "R") in fired


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    from aml.pipeline import run_pipeline

    result = run_pipeline(tmp_path, n_accounts=600, rings_per_typology=2, seed=7, use_llm=False)
    m = result.metrics
    assert m["rings"]["detected"] == m["rings"]["total"] == 8
    assert m["accounts"]["recall"] >= 0.9
    assert all(c.llm_review for c in result.cases if c.is_alert)
    for name in ("report.html", "cases.json", "metrics.json", "aml.db"):
        assert (tmp_path / name).exists()
