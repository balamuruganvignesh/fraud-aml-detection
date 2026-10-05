"""Central configuration: detection thresholds, scoring weights and runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = PROJECT_ROOT / "sql"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

# FATF grey/black-list style jurisdictions used by the synthetic generator and rules.
HIGH_RISK_COUNTRIES = ("KY", "PA", "VG", "IR", "KP", "MM", "SY", "YE")

# US Currency Transaction Report threshold; structuring = deliberately staying under it.
REPORTING_THRESHOLD = 10_000.0


@dataclass(frozen=True)
class DetectionConfig:
    # --- SQL rule thresholds -------------------------------------------------
    structuring_low: float = 8_000.0          # "just below the threshold" band
    structuring_high: float = REPORTING_THRESHOLD
    structuring_min_count: int = 3            # sub-threshold txns in a 7-day window
    velocity_window_hours: float = 24.0
    velocity_min_txns: int = 8                # outgoing txns inside the window
    pass_through_hours: float = 48.0          # money out within N hours of money in
    pass_through_min_ratio: float = 0.80      # out amount / in amount
    pass_through_min_amount: float = 3_000.0
    pass_through_min_pairs: int = 2
    high_risk_min_amount: float = 5_000.0

    # --- Graph detector thresholds ------------------------------------------
    cycle_max_length: int = 6
    cycle_min_amount: float = 2_000.0         # ignore small edges when enumerating cycles
    cycle_max_days: float = 21.0              # whole loop must complete within this
    cycle_min_retention: float = 0.70         # each hop keeps >= 70% of the previous amount
    chain_min_hops: int = 3
    chain_max_hours_per_hop: float = 48.0
    chain_min_amount: float = 5_000.0
    chain_min_retention: float = 0.85
    fan_window_hours: float = 72.0
    fan_min_counterparties: int = 6
    fan_min_total: float = 30_000.0
    fan_forward_days: float = 7.0
    fan_in_min_forward_ratio: float = 0.60    # collector forwards >= 60% of what it gathered
    gather_min_ratio: float = 0.50            # >= 50% of fan-out recipients pay the same sink

    # --- Case building -------------------------------------------------------
    hub_min_degree: int = 40                  # accounts this connected never bridge two cases

    # --- Scoring -------------------------------------------------------------
    weights: dict[str, float] = field(default_factory=lambda: {
        "cycle": 45.0,
        "layering_chain": 35.0,
        "fan_in": 35.0,
        "fan_out": 30.0,
        "scatter_gather": 45.0,
        "structuring": 20.0,
        "structuring_receiver": 25.0,
        "rapid_pass_through": 15.0,
        "high_velocity": 10.0,
        "high_risk_jurisdiction": 15.0,
    })
    high_risk_kyc_bonus: float = 5.0
    alert_threshold: float = 40.0             # cases at/above this become alerts


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None = None
    model: str = "gemini-3.5-flash-lite"
    max_cases: int = 15                        # cap on cases sent to Gemini per run
    max_txns_in_prompt: int = 40
    temperature: float = 0.2
    retries: int = 4
    min_interval_s: float = 2.0                # pacing between calls (free-tier friendly)

    @classmethod
    def from_env(cls, **overrides) -> "LLMConfig":
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:  # python-dotenv is optional
            pass
        base = cls(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            model=os.getenv("GEMINI_MODEL", cls.model),
        )
        return cls(**{**base.__dict__, **{k: v for k, v in overrides.items() if v is not None}})
