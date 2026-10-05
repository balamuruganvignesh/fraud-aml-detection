"""SQLite persistence and execution of the SQL detection rules."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import HIGH_RISK_COUNTRIES, SQL_DIR, DetectionConfig
from .models import Finding

_RULE_HEADER = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript((SQL_DIR / "schema.sql").read_text())
    # Databases created before the review queue existed lack the disposition column.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(cases)")}
    if "disposition" not in columns:
        conn.execute("ALTER TABLE cases ADD COLUMN disposition TEXT")


def load_dataset(conn: sqlite3.Connection, accounts: pd.DataFrame, transactions: pd.DataFrame) -> None:
    """Replace the accounts/transactions tables' contents with the given frames."""
    init_schema(conn)
    with conn:
        for table in ("case_transactions", "case_accounts", "cases", "rule_hits", "transactions", "accounts"):
            conn.execute(f"DELETE FROM {table}")
        accounts.to_sql("accounts", conn, if_exists="append", index=False)
        transactions.to_sql("transactions", conn, if_exists="append", index=False)


def load_rules(path: Path = SQL_DIR / "detection_rules.sql") -> dict[str, str]:
    """Split the rules file on '-- name: x' headers into {rule_name: sql}."""
    text = path.read_text()
    headers = list(_RULE_HEADER.finditer(text))
    rules = {}
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        rules[m.group(1)] = text[m.end():end].strip().rstrip(";")
    return rules


def _rule_params(cfg: DetectionConfig) -> dict:
    params = {k: v for k, v in asdict(cfg).items() if not isinstance(v, dict)}
    params["velocity_window_days"] = cfg.velocity_window_hours / 24.0
    params["high_risk_countries"] = json.dumps(list(HIGH_RISK_COUNTRIES))
    return params


def run_sql_rules(conn: sqlite3.Connection, cfg: DetectionConfig) -> list[Finding]:
    """Execute every SQL rule and return one Finding per (rule, account) hit."""
    params = _rule_params(cfg)
    findings: list[Finding] = []
    rows_to_store = []
    for name, sql in load_rules().items():
        for row in conn.execute(sql, params):
            txn_ids = sorted(set(filter(None, (row["txn_ids"] or "").split(","))))
            details = json.loads(row["details"]) if row["details"] else {}
            findings.append(Finding(
                detector=name,
                source="sql",
                accounts=[row["account_id"]],
                txn_ids=txn_ids,
                strength=_strength(name, row["score_hint"], cfg),
                details=details,
                roles={row["account_id"]: "subject"},
            ))
            rows_to_store.append((name, row["account_id"], row["score_hint"],
                                  row["details"], ",".join(txn_ids)))
    with conn:
        conn.execute("DELETE FROM rule_hits")
        conn.executemany("INSERT INTO rule_hits VALUES (?, ?, ?, ?, ?)", rows_to_store)
    return findings


def _strength(rule: str, hint: float | None, cfg: DetectionConfig) -> float:
    """Map a rule's raw score hint onto [0.5, 1.0]; 0.6 at the trigger threshold."""
    if hint is None:
        return 0.5
    baseline = {
        "structuring": cfg.structuring_min_count,
        "structuring_receiver": cfg.structuring_min_count,
        "rapid_pass_through": cfg.pass_through_min_pairs,
        "high_velocity": cfg.velocity_min_txns,
        "high_risk_jurisdiction": 1,
    }.get(rule, 1)
    return max(0.5, min(1.0, 0.6 + 0.4 * (hint - baseline) / max(baseline, 1)))


def fetch_frame(conn: sqlite3.Connection, sql: str, params: dict | tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)
