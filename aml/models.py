"""Plain data objects shared across detectors, case building and the LLM layer."""

from __future__ import annotations

from dataclasses import dataclass, field

# Analyst work queues, in priority order.
ESCALATE, REVIEW, CLOSE = "escalate", "review", "close"
QUEUES = (ESCALATE, REVIEW, CLOSE)


# Policy floor: typologies with no everyday explanation are never down-ranked by the LLM.
ALWAYS_ESCALATE = frozenset({"cycle", "layering_chain"})


def llm_disposition(review: dict | None) -> str | None:
    """Route an LLM review to a work queue on the review alone.

    escalate: flagged and HIGH/CRITICAL, or a SAR is recommended -> investigate now
    review:   flagged but only MEDIUM/LOW with no SAR -> lower-priority analyst check
    close:    not flagged -> close with rationale, keep for trend monitoring
    """
    if not review:
        return None
    if not review.get("flag"):
        return CLOSE
    if review.get("sar_recommended") or review.get("risk_level") in ("HIGH", "CRITICAL"):
        return ESCALATE
    return REVIEW


def policy_override(review: dict | None, typologies) -> list[str]:
    """Typologies that force escalation the LLM review would not have given on its own."""
    queue = llm_disposition(review)
    if queue is None or queue == ESCALATE:
        return []
    return sorted(ALWAYS_ESCALATE.intersection(typologies))


def disposition(review: dict | None, typologies=()) -> str | None:
    """Final queue: the LLM's routing, raised to escalate by the policy floor when it applies."""
    return ESCALATE if policy_override(review, typologies) else llm_disposition(review)


@dataclass
class Finding:
    """A single detector hit: which accounts/transactions matched and how strongly (0-1)."""

    detector: str                       # e.g. "cycle", "structuring"
    source: str                         # "sql" | "graph"
    accounts: list[str]
    txn_ids: list[str]
    strength: float = 1.0
    details: dict = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)


@dataclass
class Case:
    """A cluster of related findings that an analyst reviews as one investigation."""

    case_id: str
    accounts: list[str]
    txn_ids: list[str]
    findings: list[Finding]
    roles: dict[str, str]
    risk_score: float
    score_breakdown: dict[str, float]
    total_amount: float = 0.0
    is_alert: bool = False
    evidence: dict = field(default_factory=dict)
    llm_review: dict | None = None

    @property
    def disposition(self) -> str | None:
        return disposition(self.llm_review, self.typologies)

    @property
    def policy_override(self) -> list[str]:
        return policy_override(self.llm_review, self.typologies)

    @property
    def typologies(self) -> list[str]:
        return sorted({f.detector for f in self.findings})
