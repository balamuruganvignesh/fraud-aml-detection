"""CLI: ``python -m aml run [options]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, LLMConfig
from .models import QUEUES
from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aml", description="Fraud & AML graph detection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the full detection pipeline")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    run.add_argument("--accounts", type=int, default=1500, help="Synthetic accounts to generate")
    run.add_argument("--days", type=int, default=30, help="Days of synthetic activity")
    run.add_argument("--rings", type=int, default=3, help="Planted rings per typology")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--accounts-csv", type=Path, help="Use your own accounts CSV instead of synthetic data")
    run.add_argument("--transactions-csv", type=Path, help="Use your own transactions CSV")
    run.add_argument("--no-llm", action="store_true", help="Skip Gemini; use the offline reviewer")
    run.add_argument("--model", help="Gemini model (default: $GEMINI_MODEL or gemini-3.5-flash-lite)")
    run.add_argument("--max-llm-cases", type=int, help="Max alerts sent to Gemini (default 15)")
    run.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if bool(args.accounts_csv) != bool(args.transactions_csv):
        parser.error("--accounts-csv and --transactions-csv must be given together")

    result = run_pipeline(
        args.output, n_accounts=args.accounts, days=args.days, rings_per_typology=args.rings, seed=args.seed,
        accounts_csv=args.accounts_csv, transactions_csv=args.transactions_csv, use_llm=not args.no_llm,
        llm_cfg=LLMConfig.from_env(model=args.model, max_cases=args.max_llm_cases),
    )

    alerts = [c for c in result.cases if c.is_alert]
    priority = {q: i for i, q in enumerate(QUEUES)}
    alerts.sort(key=lambda c: (priority.get(c.disposition, len(QUEUES)), -c.risk_score))
    print(f"\n{'Case':<10} {'Queue':<9} {'Score':>5}  {'LLM':<9} {'Typologies':<50} {'Amount':>12}")
    for c in alerts[:20]:
        r = c.llm_review or {}
        queue = f"{c.disposition}{'+' if c.policy_override else ''}" if c.disposition else "-"
        print(f"{c.case_id:<10} {queue:<9} {c.risk_score:>5.0f}  {r.get('risk_level', '-'):<9} "
              f"{', '.join(c.typologies)[:50]:<50} ${c.total_amount:>11,.0f}")
    counts = {q: sum(c.disposition == q for c in alerts) for q in QUEUES}
    print("Queues: " + " · ".join(f"{q} {n}" for q, n in counts.items())
          + "   (+ = escalated by cycle/layering policy over the LLM's rating)")
    if len(alerts) > 20:
        print(f"... {len(alerts) - 20} more")
    m = result.metrics
    if m:
        a = m["accounts"]
        print(f"\nRings detected: {m['rings']['detected']}/{m['rings']['total']} {m['rings']['by_typology']}")
        print(f"Accounts  P={a['precision']:.2f} R={a['recall']:.2f} F1={a['f1']:.2f} | "
              f"alert precision {m['alerts']['case_precision']:.2f}")
        if m.get("llm_review"):
            lr = m["llm_review"]
            print("Queue accuracy (true / false alerts): " + " · ".join(
                f"{q} {d['true_alerts']}/{d['false_alerts']}" for q, d in lr["queues"].items()))
    print(f"\nOutputs in {result.output_dir}/: report.html, cases.json, metrics.json, aml.db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
