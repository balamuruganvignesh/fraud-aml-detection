"""Gemini-powered case triage: flag each case and write an analyst-ready summary.

The model receives the structured evidence for a case (detector findings, account
profiles, graph edges, transaction sample) and returns a schema-validated JSON review.
Without an API key, or if a call keeps failing, a deterministic rules-based reviewer
produces the same schema so the pipeline always completes.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from .config import LLMConfig
from .models import Case

log = logging.getLogger(__name__)

FALLBACK_SOURCE = "rules-fallback"
PERMANENT_ERRORS = {400, 401, 404}  # bad request, bad key, unknown/retired model
MAX_CONSECUTIVE_FAILURES = 2         # circuit breaker: stop calling once quota is clearly exhausted

SYSTEM_INSTRUCTION = """\
You are a senior Anti-Money Laundering (AML) investigator at a bank's Financial Intelligence Unit.
You review cases produced by an automated graph + rules detection system and decide whether each
case should be escalated to a human analyst, and you write the summary that analyst will read first.

Guidelines:
- Base every statement strictly on the evidence provided. Cite account IDs, amounts and timings.
- Name the typology using standard terms (structuring/smurfing, round-tripping, layering,
  scatter-gather / funnel accounts, rapid movement of funds, high-risk jurisdiction exposure).
- Weigh innocent explanations (payroll, merchants, rent, family remittances) and list them as
  mitigating factors when plausible. Do not flag on a single weak signal.
- "flag" = true means escalate for human investigation. Recommend a SAR only when the pattern
  is clear and multiple independent indicators agree.
- The summary is 3-5 sentences, plain English, written for a busy analyst.
"""


class AnalystReview(BaseModel):
    flag: bool = Field(description="Escalate this case to a human analyst")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the assessment, 0-1")
    typology: str = Field(description="Primary money-laundering typology, or 'none'")
    summary: str = Field(description="3-5 sentence analyst-facing narrative")
    key_indicators: list[str] = Field(description="Specific red flags with account IDs / amounts")
    mitigating_factors: list[str] = Field(description="Plausible innocent explanations, if any")
    recommended_actions: list[str] = Field(description="Concrete next investigative steps")
    sar_recommended: bool = Field(description="Whether a Suspicious Activity Report is warranted")


def build_prompt(evidence: dict) -> str:
    return (
        "Review the following AML case and return your assessment as JSON.\n\n"
        "Detector glossary: cycle = funds return to origin via a loop; layering_chain = rapid "
        "multi-hop transfers keeping most of the amount; fan_in = many senders into one collector "
        "that forwards the funds; scatter_gather = one source splits funds across intermediaries "
        "who re-converge on a sink; structuring = repeated amounts just under the $10,000 "
        "reporting threshold; rapid_pass_through = funds leave soon after arriving; "
        "high_risk_jurisdiction = large transfers involving high-risk countries.\n"
        "pagerank_percentile and community fields come from a NetworkX analysis of the full "
        "transaction graph.\n"
        "The 'context' block and per-account behaviour fields (account_age_days, receives_salary, "
        "runs_payroll_to_n_people, everyday_card_purchases, cash_deposits_received, channel) describe "
        "how ordinary each participant looks. Weigh them by typology:\n"
        "- Ordinary-looking participants do NOT excuse round-tripping (cycle) or layering chains: money "
        "that loops back to its origin or hops through several unrelated people within days has no "
        "everyday purpose, and criminals routinely recruit ordinary customers as money mules.\n"
        "- For convergence patterns (fan_in, structuring_receiver) the destination matters most: funds "
        "pooled by long-standing salaried individuals over electronic channels into an established "
        "domestic business that operates normally can be legitimate (e.g. a family down payment to "
        "escrow); cash deposits, new accounts, or offshore / high-risk / newly opened destinations "
        "point to smurfing.\n"
        "Weigh both sides explicitly before deciding 'flag'.\n\n"
        f"CASE EVIDENCE:\n{json.dumps(evidence, indent=1, default=str)}"
    )


class GeminiAnalyst:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = None
        self._last_call = float("-inf")
        self._consecutive_failures = 0
        if cfg.api_key:
            from google import genai
            self.client = genai.Client(api_key=cfg.api_key)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def review(self, case: Case) -> dict:
        """Return the review dict (AnalystReview fields + 'source') and store it on the case."""
        review = None
        if self.enabled:
            review = self._call_gemini(case)
        if review is None:
            review = {**rules_based_review(case), "source": FALLBACK_SOURCE}
        case.llm_review = review
        return review

    def review_all(self, cases: list[Case]) -> None:
        for i, case in enumerate(cases, start=1):
            log.info("Reviewing %s (%d/%d)", case.case_id, i, len(cases))
            self.review(case)

    def _call_gemini(self, case: Case) -> dict | None:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.cfg.temperature,
            response_mime_type="application/json",
            response_schema=AnalystReview,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        prompt = build_prompt(case.evidence)
        for attempt in range(1, self.cfg.retries + 1):
            wait = self._last_call + self.cfg.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self.client.models.generate_content(model=self.cfg.model, contents=prompt, config=config)
                parsed = resp.parsed
                if parsed is None:  # schema-violating output: validate the raw text ourselves
                    parsed = AnalystReview.model_validate_json(resp.text)
                self._consecutive_failures = 0
                return {**parsed.model_dump(), "source": self.cfg.model}
            except errors.ClientError as exc:
                # Bad request / bad key / unknown model will not fix itself; 403 and 429 can be
                # transient quota limits, so those are retried with backoff.
                if getattr(exc, "code", None) in PERMANENT_ERRORS:
                    log.error("Gemini rejected the request (%s); using the offline reviewer for the rest "
                              "of this run. Check GEMINI_API_KEY / GEMINI_MODEL.", exc)
                    self.client = None
                    return None
                log.warning("Gemini quota/rate limit on %s (attempt %d): %s", case.case_id, attempt, exc.code)
            except Exception as exc:  # server errors (503 overload), network, malformed JSON
                log.warning("Gemini call failed for %s (attempt %d): %s", case.case_id, attempt, str(exc)[:100])
            if attempt < self.cfg.retries:
                time.sleep(5 * 2 ** (attempt - 1))
        log.warning("Giving up on Gemini for %s after %d attempts; using fallback", case.case_id, self.cfg.retries)
        self._consecutive_failures += 1
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log.error("Gemini failed on %d cases in a row (quota exhausted or service down); using the "
                      "offline reviewer for the rest of this run", self._consecutive_failures)
            self.client = None
        return None


def rules_based_review(case: Case) -> dict:
    """Deterministic stand-in for Gemini, built from the same evidence."""
    ev = case.evidence
    typ = set(case.typologies)
    names = {
        "cycle": "round-tripping",
        "layering_chain": "layering",
        "fan_in": "structuring / smurfing into a collector account",
        "scatter_gather": "scatter-gather via mule accounts",
        "fan_out": "structured fan-out",
        "structuring": "structuring",
        "structuring_receiver": "structuring",
        "rapid_pass_through": "rapid movement of funds",
        "high_velocity": "high-velocity activity",
        "high_risk_jurisdiction": "high-risk jurisdiction exposure",
    }
    graph_types = [t for t in ("scatter_gather", "fan_in", "cycle", "layering_chain", "fan_out") if t in typ]
    primary = names[graph_types[0]] if graph_types else (names[sorted(typ)[0]] if typ else "none")

    score = case.risk_score
    level = "CRITICAL" if score >= 85 else "HIGH" if score >= 65 else "MEDIUM" if score >= 40 else "LOW"
    indicators = []
    ordered = sorted(case.findings, key=lambda f: (f.source != "graph", -f.strength))  # structure first
    for f in ordered[:8]:
        d = f.details
        if f.detector == "cycle":
            indicators.append(f"Funds looped {d['path']} ({d['loops_completed']}x, {d['retention']:.0%} retained "
                              f"over {d['loop_hours']}h)")
        elif f.detector == "layering_chain":
            indicators.append(f"{d['hops']}-hop chain {d['path']} in {d['total_hours']}h, "
                              f"${d['start_amount']:,.0f} -> ${d['end_amount']:,.0f}")
        elif f.detector == "fan_in":
            indicators.append(f"{d['distinct_senders']} senders paid ${d['window_total']:,.0f} into {d['collector']} "
                              f"within {d['window_hours']}h; {d['forwarded_ratio']:.0%} forwarded on")
        elif f.detector == "scatter_gather":
            indicators.append(f"{d['source']} split ${d['split_total']:,.0f} across {d['n_intermediaries']} accounts; "
                              f"{d['gather_ratio']:.0%} re-converged on {d['sink']}")
        elif f.detector == "fan_out":
            indicators.append(f"{d['source']} paid {d['distinct_receivers']} receivers ${d['window_total']:,.0f} "
                              f"mostly just under $10k")
        else:
            indicators.append(f"{f.detector.replace('_', ' ')} rule hit on {f.accounts[0]}: "
                              + ", ".join(f"{k}={v}" for k, v in d.items()))
    risky = [a for a in ev.get("accounts", []) if a.get("kyc_risk") == "high"]
    if risky:
        indicators.append("High-KYC-risk accounts involved: " + ", ".join(a["account_id"] for a in risky[:5]))

    first_seen = ev.get("time_span", {}).get("first", "")[:10]
    cutoff = (date.fromisoformat(first_seen) - timedelta(days=90)).isoformat() if first_seen else None
    young = [a for a in ev.get("accounts", []) if cutoff and a.get("open_date", "") >= cutoff]
    mitigating = []
    if len(graph_types) == 0:
        mitigating.append("Only rule-based signals fired; no network laundering structure was found.")
    if any(a.get("account_type") in ("merchant", "business") for a in ev.get("accounts", [])) and not young:
        mitigating.append("Established business accounts involved; confirm activity matches expected profile.")

    flag = score >= 40 and bool(graph_types or len(typ) >= 2)
    summary = (
        f"{case.case_id} links {len(case.accounts)} accounts and {ev.get('n_transactions', 0)} transactions "
        f"totalling ${case.total_amount:,.0f} between {ev.get('time_span', {}).get('first', '?')} and "
        f"{ev.get('time_span', {}).get('last', '?')}. The dominant pattern is {primary}. "
        f"Detectors triggered: {', '.join(sorted(typ))}. "
        + (f"{len(young)} account(s) were opened recently, consistent with purpose-built mule or shell accounts. "
           if young else "")
        + f"Composite risk score {score:.0f}/100."
    )
    return AnalystReview(
        flag=flag,
        risk_level=level,
        confidence=round(min(0.95, 0.4 + score / 200), 2),
        typology=primary,
        summary=summary,
        key_indicators=indicators,
        mitigating_factors=mitigating,
        recommended_actions=(
            ["Request source-of-funds documentation from the originating account(s)",
             "Review KYC files and beneficial ownership for all collector/sink accounts",
             "Place enhanced monitoring on intermediary accounts"]
            + (["Prepare SAR narrative for compliance sign-off"] if score >= 65 else [])
        ) if flag else ["Close as no further action; retain for trend monitoring"],
        sar_recommended=score >= 65 and bool(graph_types),
    ).model_dump()
