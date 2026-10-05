"""Gemini integration tests with a fake client (no network, no API key needed)."""

from __future__ import annotations

from types import SimpleNamespace

from google.genai import errors

from aml.config import LLMConfig
from aml.gemini_analyst import FALLBACK_SOURCE, AnalystReview, GeminiAnalyst, build_prompt
from aml.models import Case, Finding

REVIEW = AnalystReview(
    flag=True, risk_level="HIGH", confidence=0.86, typology="smurfing",
    summary="Eight accounts deposited just under $10k into COL, which wired the pot offshore.",
    key_indicators=["8 deposits of ~$9.5k within 7h"], mitigating_factors=[],
    recommended_actions=["Request source of funds"], sar_recommended=True,
)


def make_case() -> Case:
    f = Finding(detector="fan_in", source="graph", accounts=["COL", "M1"], txn_ids=["T1"], strength=0.9,
                details={"collector": "COL", "distinct_senders": 8, "window_total": 76000.0,
                         "window_hours": 7.0, "forwarded_ratio": 0.92, "sub_threshold_share": 1.0})
    case = Case(case_id="CASE-0001", accounts=["COL", "M1"], txn_ids=["T1"], findings=[f],
                roles={"COL": "collector", "M1": "mule"}, risk_score=72.0,
                score_breakdown={"fan_in": 31.5}, total_amount=76000.0, is_alert=True)
    case.evidence = {"case_id": case.case_id, "typologies_detected": ["fan_in"], "accounts": [],
                     "n_transactions": 1, "time_span": {"first": "2026-09-01 08:00", "last": "2026-09-03 09:00"}}
    return case


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def analyst_with(outcomes, monkeypatch) -> tuple[GeminiAnalyst, FakeModels]:
    monkeypatch.setattr("aml.gemini_analyst.time.sleep", lambda s: None)
    analyst = GeminiAnalyst(LLMConfig(api_key=None, model="gemini-test", retries=3, min_interval_s=0))
    fake = FakeModels(outcomes)
    analyst.client = SimpleNamespace(models=fake)
    return analyst, fake


def test_structured_output_is_used(monkeypatch):
    analyst, fake = analyst_with([SimpleNamespace(parsed=REVIEW, text=None)], monkeypatch)
    review = analyst.review(make_case())
    assert review["source"] == "gemini-test"
    assert review["flag"] is True and review["risk_level"] == "HIGH"
    cfg = fake.calls[0]["config"]
    assert cfg.response_mime_type == "application/json" and cfg.response_schema is AnalystReview
    assert "CASE-0001" in fake.calls[0]["contents"]


def test_raw_json_is_validated_when_parsed_missing(monkeypatch):
    analyst, _ = analyst_with([SimpleNamespace(parsed=None, text=REVIEW.model_dump_json())], monkeypatch)
    assert analyst.review(make_case())["typology"] == "smurfing"


def test_retries_then_succeeds(monkeypatch):
    analyst, fake = analyst_with([RuntimeError("503"), SimpleNamespace(parsed=REVIEW, text=None)], monkeypatch)
    assert analyst.review(make_case())["source"] == "gemini-test"
    assert len(fake.calls) == 2


def test_bad_request_falls_back_without_retry(monkeypatch):
    err = errors.ClientError(400, {"error": {"code": 400, "message": "bad model", "status": "INVALID_ARGUMENT"}})
    analyst, fake = analyst_with([err], monkeypatch)
    review = analyst.review(make_case())
    assert review["source"] == FALLBACK_SOURCE and len(fake.calls) == 1
    # Non-retryable errors disable Gemini for the rest of the run instead of failing per case.
    assert not analyst.enabled


def test_quota_403_is_retried(monkeypatch):
    err = errors.ClientError(403, {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}})
    analyst, fake = analyst_with([err, SimpleNamespace(parsed=REVIEW, text=None)], monkeypatch)
    assert analyst.review(make_case())["source"] == "gemini-test"
    assert analyst.enabled and len(fake.calls) == 2


def test_circuit_breaker_after_consecutive_failures(monkeypatch):
    err = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    analyst, fake = analyst_with([err] * 6, monkeypatch)
    analyst.review(make_case())
    assert analyst.enabled  # one bad case is not enough
    analyst.review(make_case())
    assert not analyst.enabled and len(fake.calls) == 6
    assert analyst.review(make_case())["source"] == FALLBACK_SOURCE  # no further calls
    assert len(fake.calls) == 6


def test_no_key_uses_fallback():
    analyst = GeminiAnalyst(LLMConfig(api_key=None))
    review = analyst.review(make_case())
    assert not analyst.enabled
    AnalystReview.model_validate({k: v for k, v in review.items() if k != "source"})
    assert review["source"] == FALLBACK_SOURCE and review["flag"] is True


def test_prompt_contains_evidence():
    prompt = build_prompt(make_case().evidence)
    assert "fan_in" in prompt and "structuring" in prompt
