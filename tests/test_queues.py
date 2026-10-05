"""Routing of LLM reviews into analyst work queues, and the schema migration that stores them."""

from __future__ import annotations

import sqlite3

import pytest

from aml.database import init_schema
from aml.models import CLOSE, ESCALATE, REVIEW, disposition, policy_override


@pytest.mark.parametrize("review, expected", [
    ({"flag": True, "risk_level": "CRITICAL", "sar_recommended": True}, ESCALATE),
    ({"flag": True, "risk_level": "HIGH", "sar_recommended": False}, ESCALATE),
    ({"flag": True, "risk_level": "MEDIUM", "sar_recommended": True}, ESCALATE),   # a SAR always escalates
    ({"flag": True, "risk_level": "MEDIUM", "sar_recommended": False}, REVIEW),
    ({"flag": True, "risk_level": "LOW", "sar_recommended": False}, REVIEW),
    ({"flag": False, "risk_level": "LOW", "sar_recommended": False}, CLOSE),
    ({"flag": False, "risk_level": "HIGH", "sar_recommended": False}, CLOSE),     # the flag decides closing
    (None, None),
])
def test_disposition(review, expected):
    assert disposition(review) == expected


MEDIUM_NO_SAR = {"flag": True, "risk_level": "MEDIUM", "sar_recommended": False}
NOT_FLAGGED = {"flag": False, "risk_level": "LOW", "sar_recommended": False}


@pytest.mark.parametrize("review, typologies, expected, override", [
    (MEDIUM_NO_SAR, ["cycle"], ESCALATE, ["cycle"]),
    (NOT_FLAGGED, ["layering_chain", "high_risk_jurisdiction"], ESCALATE, ["layering_chain"]),
    (MEDIUM_NO_SAR, ["fan_in", "structuring_receiver"], REVIEW, []),       # pooling-shaped: LLM decides
    ({"flag": True, "risk_level": "HIGH", "sar_recommended": True}, ["cycle"], ESCALATE, []),  # no override needed
    (None, ["cycle"], None, []),                                            # unreviewed stays unrouted
])
def test_cycle_and_layering_always_escalate(review, typologies, expected, override):
    assert disposition(review, typologies) == expected
    assert policy_override(review, typologies) == override


def test_old_database_gets_disposition_column(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE cases (case_id TEXT PRIMARY KEY, risk_score REAL)")  # pre-queue schema
    init_schema(conn)
    init_schema(conn)  # idempotent
    assert "disposition" in {r["name"] for r in conn.execute("PRAGMA table_info(cases)")}
