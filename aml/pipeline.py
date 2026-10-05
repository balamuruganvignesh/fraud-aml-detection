"""End-to-end pipeline: data -> SQLite -> SQL rules + graph detectors -> cases -> Gemini -> report."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cases import attach_evidence, build_cases
from .config import DEFAULT_OUTPUT_DIR, DetectionConfig, LLMConfig
from .data_generator import generate_dataset
from .database import connect, load_dataset, run_sql_rules
from .evaluation import evaluate
from .gemini_analyst import FALLBACK_SOURCE, GeminiAnalyst, rules_based_review
from .graph_analysis import TransactionGraph, run_graph_detectors
from .models import Case
from .report import case_to_dict, render_html, run_info, save_cases, write_json

log = logging.getLogger(__name__)

REQUIRED_TXN_COLUMNS = {"txn_id", "timestamp", "sender", "receiver", "amount"}
REQUIRED_ACCOUNT_COLUMNS = {"account_id"}


@dataclass
class PipelineResult:
    cases: list[Case]
    metrics: dict
    output_dir: Path
    timings: dict


def load_csv_dataset(accounts_csv: Path, transactions_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load your own data. Missing optional columns are filled with neutral defaults."""
    accounts = pd.read_csv(accounts_csv)
    txns = pd.read_csv(transactions_csv)
    for df, required, name in ((accounts, REQUIRED_ACCOUNT_COLUMNS, "accounts"),
                               (txns, REQUIRED_TXN_COLUMNS, "transactions")):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {sorted(missing)}")
    defaults = {"account_type": "individual", "country": "US", "open_date": "2000-01-01", "kyc_risk": "low"}
    for col, val in defaults.items():
        if col not in accounts.columns:
            accounts[col] = val
    if "is_laundering" not in accounts.columns:
        accounts["is_laundering"] = 0
    txns["timestamp"] = pd.to_datetime(txns["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for col, val in {"channel": "unknown", "is_laundering": 0, "typology": None, "ring_id": None}.items():
        if col not in txns.columns:
            txns[col] = val
    unknown = (set(txns["sender"]) | set(txns["receiver"])) - set(accounts["account_id"])
    if unknown:  # accounts seen only in transactions get default profiles
        accounts = pd.concat([accounts, pd.DataFrame({"account_id": sorted(unknown), **defaults,
                                                      "is_laundering": 0})], ignore_index=True)
    return accounts, txns


def run_pipeline(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    n_accounts: int = 1500,
    days: int = 30,
    rings_per_typology: int = 3,
    seed: int = 42,
    accounts_csv: Path | None = None,
    transactions_csv: Path | None = None,
    use_llm: bool = True,
    llm_cfg: LLMConfig | None = None,
    det_cfg: DetectionConfig | None = None,
) -> PipelineResult:
    det_cfg = det_cfg or DetectionConfig()
    llm_cfg = llm_cfg or LLMConfig.from_env()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def step(name: str, t0: float) -> float:
        timings[name] = round(time.perf_counter() - t0, 2)
        log.info("%-18s %6.2fs", name, timings[name])
        return time.perf_counter()

    t = time.perf_counter()
    if accounts_csv and transactions_csv:
        accounts, txns = load_csv_dataset(accounts_csv, transactions_csv)
        rings: list[dict] = []
    else:
        ds = generate_dataset(n_accounts, days, rings_per_typology, seed)
        accounts, txns, rings = ds.accounts, ds.transactions, ds.rings
        write_json(output_dir / "ground_truth.json", {"rings": rings, "legitimate_lookalikes": ds.decoys})
    log.info("Loaded %d accounts, %d transactions", len(accounts), len(txns))
    t = step("load_data", t)

    conn = connect(output_dir / "aml.db")
    load_dataset(conn, accounts, txns)
    sql_findings = run_sql_rules(conn, det_cfg)
    t = step("sql_rules", t)

    graph = TransactionGraph(accounts, txns)
    graph_findings = run_graph_detectors(graph, det_cfg)
    log.info("Graph: %d nodes, %d edges | findings: %d sql, %d graph", graph.agg.number_of_nodes(),
             graph.agg.number_of_edges(), len(sql_findings), len(graph_findings))
    t = step("graph_detectors", t)

    cases = build_cases(sql_findings + graph_findings, graph, det_cfg)
    for c in cases:
        attach_evidence(c, graph, llm_cfg.max_txns_in_prompt)
    alerts = [c for c in cases if c.is_alert]
    log.info("%d cases, %d alerts", len(cases), len(alerts))
    t = step("case_building", t)

    analyst = GeminiAnalyst(llm_cfg if use_llm else LLMConfig(api_key=None))
    if use_llm and not analyst.enabled:
        log.warning("No GEMINI_API_KEY found — using the offline rules-based reviewer")
    analyst.review_all(alerts[: llm_cfg.max_cases])
    for c in alerts[llm_cfg.max_cases:]:  # beyond the LLM budget: deterministic review
        c.llm_review = {**rules_based_review(c), "source": FALLBACK_SOURCE}
    t = step("llm_review", t)

    metrics = evaluate(cases, accounts, rings) if rings else {}
    save_cases(conn, cases)
    conn.close()
    reviewer = llm_cfg.model if analyst.enabled else FALLBACK_SOURCE
    write_json(output_dir / "cases.json", [case_to_dict(c) for c in cases])
    write_json(output_dir / "metrics.json", metrics)
    info = run_info(len(accounts), len(txns), det_cfg.alert_threshold, reviewer)
    (output_dir / "report.html").write_text(render_html(cases, metrics, info))
    step("reporting", t)
    return PipelineResult(cases=cases, metrics=metrics, output_dir=output_dir, timings=timings)
