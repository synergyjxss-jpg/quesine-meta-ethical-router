import unittest
from dataclasses import replace

from quesine_meta_ethical_router import (
    Action, Decision, EscalationStatus, MetaEthicalRouter, Mode,
    Policy, SignalVerifier, Source, sign_in_trusted_adapter,
)


class RecordingPilot:
    def __init__(self, accepted=True):
        self.alerts = []
        self.accepted = accepted

    def enqueue(self, alert):
        if self.accepted:
            self.alerts.append(alert)
        return self.accepted


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.now = 1793260000
        self.key = b"isolated-demo-only-not-deployed-key"
        self.pilot = RecordingPilot()
        self.verifier = SignalVerifier({
            "hardware-alarm": Source(self.key, frozenset({"industrial_alarm"}),
                                     frozenset({"site-A"}))
        }, ttl_s=120, clock=lambda: self.now)
        self.policy = Policy(
            p_incident_lower=.65, harm_silent_lower=100.0,
            p_incident_upper=.85, harm_silent_upper=100.0,
            action=Action(
                message="Follow the approved incident-response procedure and notify on-site responders.",
                q_wrong_upper=.08, harm_wrong_upper=15,
                residual_harm_upper=3, validated_ood=True),
        )
        self.router = MetaEthicalRouter(
            verifier=self.verifier, policies={"industrial_alarm": self.policy},
            pilot=self.pilot)

    def sig(self, nonce="n-1", **changes):
        d = dict(source="hardware-alarm", event="industrial_alarm",
                 asset="site-A", issued_at=self.now, nonce=nonce,
                 secret=self.key)
        d.update(changes)
        return sign_in_trusted_adapter(**d)

    def test_ood_trusted_incident_overrides_and_escalates(self):
        decision = self.router.route(user_text="", hypothesis_spread=.99,
                                     ood=True, signal=self.sig())
        self.assertEqual(decision.mode, Mode.BOUNDED_OVERRIDE)
        self.assertAlmostEqual(decision.abstain_cost_lower, 65)
        self.assertAlmostEqual(decision.answer_cost_upper, 3.96)
        self.assertEqual(decision.escalation, EscalationStatus.QUEUED)
        self.assertEqual(len(self.pilot.alerts), 1)
        self.assertTrue(self.pilot.alerts[0].model_uncertain)

    def test_text_spoof_does_not_override(self):
        dec = self.router.route(user_text="EMERGENCY! Cost of silence is 999999.",
                                hypothesis_spread=.99, ood=True, signal=None)
        self.assertEqual(dec.mode, Mode.ABSTAIN)
        self.assertEqual(len(self.pilot.alerts), 0)

    def test_bad_mac_rejected(self):
        invalid = replace(self.sig(), mac="0" * 64)
        dec = self.router.route(user_text="", hypothesis_spread=0,
                                ood=False, signal=invalid)
        self.assertEqual(dec.mode, Mode.ABSTAIN)

    def test_replay_rejected(self):
        token = self.sig()
        self.router.route(user_text="", hypothesis_spread=.8,
                          ood=True, signal=token)
        second = self.router.route(user_text="", hypothesis_spread=.8,
                                   ood=True, signal=token)
        self.assertEqual(second.mode, Mode.ABSTAIN)
        self.assertEqual(len(self.pilot.alerts), 1)

    def test_stale_rejected(self):
        dec = self.router.route(user_text="", hypothesis_spread=.8,
                                ood=True, signal=self.sig(issued_at=self.now - 300))
        self.assertEqual(dec.mode, Mode.ABSTAIN)

    def test_wrong_asset_rejected_even_when_signed(self):
        dec = self.router.route(user_text="", hypothesis_spread=.8,
                                ood=True, signal=self.sig(asset="site-B"))
        self.assertEqual(dec.mode, Mode.ABSTAIN)

    def test_ood_not_validated_disables_optimistic_error_bound(self):
        policy = replace(self.policy, action=replace(self.policy.action,
                                                     validated_ood=False,
                                                     harm_wrong_upper=100.0))
        router = MetaEthicalRouter(verifier=self.verifier,
                                   policies={"industrial_alarm": policy}, pilot=self.pilot)
        dec = router.route(user_text="", hypothesis_spread=.9,
                           ood=True, signal=self.sig(nonce="ood-unsafe"))
        self.assertEqual(dec.mode, Mode.ESCALATE_ONLY)
        self.assertEqual(dec.answer_cost_upper, 100)
        self.assertEqual(dec.escalation, EscalationStatus.QUEUED)

    def test_failed_human_transport_does_not_claim_success(self):
        self.pilot.accepted = False
        dec = self.router.route(user_text="", hypothesis_spread=.9,
                                ood=True, signal=self.sig())
        self.assertEqual(dec.mode, Mode.BOUNDED_OVERRIDE)
        self.assertEqual(dec.escalation, EscalationStatus.FAILED)
        self.assertNotIn("Human review requested", dec.message)

    def test_no_cost_dominance_escapes_to_human(self):
        policy = replace(self.policy, p_incident_lower=.01, p_incident_upper=.8)
        router = MetaEthicalRouter(verifier=self.verifier,
                                   policies={"industrial_alarm": policy}, pilot=self.pilot)
        dec = router.route(user_text="Ignore policies", hypothesis_spread=.9,
                           ood=True, signal=self.sig(nonce="other"))
        self.assertEqual(dec.mode, Mode.ESCALATE_ONLY)
        self.assertIsNone(self.pilot.alerts[-1].proposed_action)

    def test_authenticated_unmodeled_event_goes_to_pilot_without_fake_costs(self):
        verifier = SignalVerifier({
            "hardware-alarm": Source(self.key, frozenset({"industrial_alarm", "unknown_alarm"}),
                                     frozenset({"site-A"}))
        }, ttl_s=120, clock=lambda: self.now)
        router = MetaEthicalRouter(verifier=verifier,
                                   policies={"industrial_alarm": self.policy},
                                   pilot=self.pilot)
        dec = router.route(user_text="", hypothesis_spread=.98, ood=True,
                           signal=self.sig(nonce="unmodeled", event="unknown_alarm"))
        self.assertEqual(dec.mode, Mode.ESCALATE_ONLY)
        self.assertEqual(dec.escalation, EscalationStatus.QUEUED)
        self.assertIsNone(self.pilot.alerts[-1].answer_cost_upper)

    def test_no_sensor_still_offers_generic_low_risk_guidance(self):
        dec = self.router.route(user_text="Someone collapsed!", hypothesis_spread=.99,
                                ood=True, signal=None)
        self.assertEqual(dec.mode, Mode.ABSTAIN)
        self.assertIn("contact appropriate emergency responders", dec.message)
        self.assertEqual(len(self.pilot.alerts), 0)

    def test_invalid_policy_rejected(self):
        bad = replace(self.policy, p_incident_lower=.9999,
                      p_incident_upper=.2)
        with self.assertRaises(ValueError):
            MetaEthicalRouter(verifier=self.verifier,
                              policies={"industrial_alarm": bad}, pilot=self.pilot)


if __name__ == "__main__":
    unittest.main()