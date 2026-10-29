"""Quesine Experiment 014 — Meta-Ethical Router (educational reference prototype).

This module chooses ONLY allowlisted, pre-reviewed interventions; it does not
make emergency diagnoses or authorize open-ended language-model output.

Threat model: attacker controls the prompt and can submit invalid/flooded signals,
but cannot access trusted-source HMAC keys or mutate the router's risk policy.
The prototype uses in-process replay protection; use shared durable storage and
separate protected queues in a production deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from hmac import compare_digest, new as hmac_new
from math import isfinite
from time import time
from typing import Callable, Mapping, Protocol
import json


class Mode(str, Enum):
    ABSTAIN = "ABSTAIN"
    BOUNDED_OVERRIDE = "BOUNDED_OVERRIDE"
    ESCALATE_ONLY = "ESCALATE_ONLY"


class EscalationStatus(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    QUEUED = "QUEUED"             # Accepted by transport; NOT human acknowledgement.
    FAILED = "FAILED"


@dataclass(frozen=True)
class SignedSignal:
    source: str
    event: str
    asset: str
    issued_at: int
    nonce: str
    mac: str

    def payload(self) -> bytes:
        data = dict(source=self.source, event=self.event, asset=self.asset,
                    issued_at=self.issued_at, nonce=self.nonce)
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class Source:
    secret: bytes
    events: frozenset[str]
    assets: frozenset[str]


@dataclass(frozen=True)
class Action:
    """Candidate action supported by measured, conservative harm bounds.

    q_wrong_upper must be evaluated for this *specific fixed message* under
    the relevant OOD/attack conditions. If unavailable, leave validated_ood=False.
    """
    message: str
    q_wrong_upper: float
    harm_wrong_upper: float
    residual_harm_upper: float
    validated_ood: bool


@dataclass(frozen=True)
class Policy:
    """Only operators/trusted configuration may provide these coefficients.

    All harm coefficients must use a consistent domain-specific harm scale.
    p_incident_lower and harm_silent_lower are pessimistic-for-inaction lower
    bounds, not a probability the LLM assigns to a user's story.
    """
    p_incident_lower: float
    harm_silent_lower: float
    p_incident_upper: float
    harm_silent_upper: float
    action: Action
    escalate_if_abstain_upper_at_least: float = 10.0


@dataclass(frozen=True)
class PilotAlert:
    event: str
    asset: str
    reason: str
    abstain_cost_lower: float | None
    abstain_cost_upper: float | None
    answer_cost_upper: float | None
    model_uncertain: bool
    proposed_action: str | None


class PilotSink(Protocol):
    def enqueue(self, alert: PilotAlert) -> bool:
        """Return True ONLY if accepted into a real, persistent human-review queue."""
        ...


@dataclass(frozen=True)
class Decision:
    mode: Mode
    message: str
    reason: str
    abstain_cost_lower: float | None = None
    abstain_cost_upper: float | None = None
    answer_cost_upper: float | None = None
    escalation: EscalationStatus = EscalationStatus.NOT_REQUESTED


class SignalVerifier:
    """Authentication, time window and replay checks; not a remote trust oracle."""

    def __init__(self, sources: Mapping[str, Source], *, ttl_s: int = 120,
                 clock: Callable[[], float] = time):
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self._sources = dict(sources)
        self._ttl_s = ttl_s
        self._clock = clock
        self._seen: dict[tuple[str, str], float] = {}

    def verify(self, signal: SignedSignal) -> bool:
        if not isinstance(signal, SignedSignal):
            return False
        src = self._sources.get(signal.source)
        if src is None or signal.event not in src.events or signal.asset not in src.assets:
            return False
        now = self._clock()
        if (not isinstance(signal.issued_at, int)
            or not signal.nonce
            or not signal.mac
            or signal.issued_at > now + 5
            or now - signal.issued_at > self._ttl_s):
            return False

        expected = hmac_new(src.secret, signal.payload(), sha256).hexdigest()
        if not compare_digest(expected, signal.mac):
            return False

        # Expire old nonces. Production: atomic check-and-set with TTL in Redis/DB.
        self._seen = {k: expiry for k, expiry in self._seen.items() if expiry > now}
        key = (signal.source, signal.nonce)
        if key in self._seen:
            return False
        self._seen[key] = now + self._ttl_s
        return True


def _unit(v: float) -> bool:
    return isinstance(v, (int, float)) and isfinite(v) and 0 <= v <= 1


def _nonnegative(v: float) -> bool:
    return isinstance(v, (int, float)) and isfinite(v) and v >= 0


def _validate_policy(p: Policy) -> None:
    a = p.action
    if not all(_unit(v) for v in (
        p.p_incident_lower, p.p_incident_upper, a.q_wrong_upper
    )):
        raise ValueError("Probability bounds must be finite and in [0, 1]")
    if p.p_incident_lower > p.p_incident_upper:
        raise ValueError("Inconsistent incident probability bounds")
    if not all(_nonnegative(v) for v in (
        p.harm_silent_lower, p.harm_silent_upper,
        a.harm_wrong_upper, a.residual_harm_upper,
        p.escalate_if_abstain_upper_at_least
    )) or p.harm_silent_lower > p.harm_silent_upper:
        raise ValueError("Invalid or inverted harm bounds")
    if a.harm_wrong_upper < a.residual_harm_upper:
        raise ValueError("This simplified loss formula requires harm_wrong >= residual")
    if not a.message.strip():
        raise ValueError("A bounded intervention must have approved content")


class MetaEthicalRouter:
    """Cost-aware abstention override with an authenticated and bounded action lane.

    No user-controlled words feed incident severity, probability or policy choice.
    No arbitrary LLM answer can be emitted through the override lane.
    """

    def __init__(self, *, verifier: SignalVerifier,
                 policies: Mapping[str, Policy], pilot: PilotSink,
                 spread_threshold: float = 0.7):
        if not _unit(spread_threshold):
            raise ValueError("spread_threshold must be finite and in [0, 1]")
        for p in policies.values():
            _validate_policy(p)
        self._verifier = verifier
        self._policies = dict(policies)
        self._pilot = pilot
        self._spread_threshold = spread_threshold

    def _queue(self, alert: PilotAlert) -> EscalationStatus:
        try:
            accepted = self._pilot.enqueue(alert)
            return (EscalationStatus.QUEUED if accepted is True
                    else EscalationStatus.FAILED)
        except Exception:
            # Do not assert a human was alerted when transport failed.
            return EscalationStatus.FAILED

    def route(self, *, user_text: str, hypothesis_spread: float,
              ood: bool, signal: SignedSignal | None) -> Decision:
        # `user_text` is intentionally excluded from authentication and risk.
        # At most it may be separately processed by the normal (non-override) path.
        del user_text

        if signal is None or not self._verifier.verify(signal):
            return Decision(
                Mode.ABSTAIN,
                "Urgency is unverified; no high-risk override is authorized. "
                "If someone may be in immediate danger, contact appropriate emergency "
                "responders and follow their instructions.",
                "UNTRUSTED_OR_REPLAYED_SIGNAL")

        policy = self._policies.get(signal.event)
        if policy is None:
            status = self._queue(PilotAlert(
                event=signal.event, asset=signal.asset,
                reason="AUTHENTICATED_BUT_UNMODELED_EVENT",
                abstain_cost_lower=None, abstain_cost_upper=None,
                answer_cost_upper=None, model_uncertain=True,
                proposed_action=None))
            return Decision(
                Mode.ESCALATE_ONLY,
                "Verified event has no risk model or authorized response; "
                "no unsupported action issued.",
                "NO_APPROVED_POLICY", escalation=status)

        # Unknown/invalid spread is treated as high uncertainty.
        uncertain = (not _unit(hypothesis_spread)
                     or hypothesis_spread >= self._spread_threshold
                     or ood)

        # Conservative intervals, using independent policy calibrations.
        ca_lo = policy.p_incident_lower * policy.harm_silent_lower
        ca_hi = policy.p_incident_upper * policy.harm_silent_upper
        a = policy.action
        # If OOD validation is absent, risk may be as high as 100% wrong.
        q_up = a.q_wrong_upper if (not ood or a.validated_ood) else 1.0
        cr_hi = q_up * a.harm_wrong_upper + (1 - q_up) * a.residual_harm_upper

        override = ca_lo > cr_hi  # Strict dominance. No numerical tie override.
        if override:
            status = EscalationStatus.NOT_REQUESTED
            if uncertain:
                status = self._queue(PilotAlert(
                    event=signal.event, asset=signal.asset,
                    reason="UNCERTAIN_BOUNDED_OVERRIDE",
                    abstain_cost_lower=ca_lo, abstain_cost_upper=ca_hi,
                    answer_cost_upper=cr_hi, model_uncertain=True,
                    proposed_action=a.message))
            return Decision(
                mode=Mode.BOUNDED_OVERRIDE,
                message=a.message + (" [Human review requested.]"
                                     if status == EscalationStatus.QUEUED else ""),
                reason="ROBUST_COST_DOMINANCE",
                abstain_cost_lower=ca_lo,
                abstain_cost_upper=ca_hi,
                answer_cost_upper=cr_hi,
                escalation=status)

        # Even if the robust comparison fails, high-stakes verified events
        # may warrant escalation without inventing a factual answer.
        if ca_hi >= policy.escalate_if_abstain_upper_at_least:
            status = self._queue(PilotAlert(
                event=signal.event, asset=signal.asset,
                reason="RISK_INTERVALS_OVERLAP_OR_ACTION_UNSAFE",
                abstain_cost_lower=ca_lo, abstain_cost_upper=ca_hi,
                answer_cost_upper=cr_hi, model_uncertain=uncertain,
                proposed_action=None))
            return Decision(
                mode=Mode.ESCALATE_ONLY,
                message="Risk comparison is inconclusive. No unverified answer issued.",
                reason="NO_ROBUSTLY_SAFE_OVERRIDE",
                abstain_cost_lower=ca_lo, abstain_cost_upper=ca_hi,
                answer_cost_upper=cr_hi,
                escalation=status)

        return Decision(Mode.ABSTAIN,
                        "No justified override; uncertainty remains unresolved.",
                        "NO_ROBUSTLY_SAFE_OVERRIDE", ca_lo, ca_hi, cr_hi)


def sign_in_trusted_adapter(source: str, event: str, asset: str,
                            issued_at: int, nonce: str, secret: bytes) -> SignedSignal:
    """For an isolated trusted adapter/test fixture, NEVER an attacker-facing API."""
    draft = SignedSignal(source, event, asset, issued_at, nonce, "")
    mac = hmac_new(secret, draft.payload(), sha256).hexdigest()
    return SignedSignal(source, event, asset, issued_at, nonce, mac)