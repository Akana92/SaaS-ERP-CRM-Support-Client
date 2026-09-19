from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support.business_metrics import SCOPE, summarize_smoke


def record(identity, *, handoff=False, mode="fine_tuned", model_handoff=False):
    client = {"status": "human_escalation" if handoff else "answered", "escalation": handoff}
    return {"request_id": identity, "client": client, "admin": {
        "raw_model_result": {"human_escalation": model_handoff},
        "usage_calls": [{"call_id": "call-"+identity, "request_id": identity, "mode": mode,
                         "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                         "api_cost": 0.0, "latency_ms": 2000, "currency": "USD", "complete": True}]}}


def evidence():
    cases = []
    for identity, convo, handoff in (("r1", "c1", True), ("r2", "c1", True), ("r3", "c2", False)):
        detail = record(identity, handoff=handoff)
        cases.append({"conversation_id": convo, "public": {"request_id": identity, "client": detail["client"]},
                      "detail": detail})
    return {"scope": SCOPE, "training_started": False, "protected_data_read": False, "cases": cases,
            "comparison": {"status": "completed", "base": record("b", mode="base"), "fine_tuned": record("f")}}


class BusinessMetricTests(unittest.TestCase):
    def test_messages_conversations_and_comparison_have_separate_denominators(self):
        result = summarize_smoke(evidence())
        client = result["client"]
        self.assertEqual((client["requests"], client["conversations"]), (3, 2))
        self.assertEqual(client["escalated_request_rate"], 2/3)
        self.assertEqual(client["conversation_handoff_rate"], 1/2)
        self.assertEqual(client["usage"]["total_tokens"], 360)
        self.assertEqual(result["comparison"]["base"]["total_tokens"], 120)

    def test_idempotent_replay_is_counted_once(self):
        data = evidence()
        data["cases"].append(copy.deepcopy(data["cases"][0]))
        self.assertEqual(summarize_smoke(data), summarize_smoke(evidence()))

    def test_conflicting_replay_is_rejected(self):
        data = evidence()
        duplicate = copy.deepcopy(data["cases"][0])
        duplicate["conversation_id"] = "different"
        data["cases"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            summarize_smoke(data)

    def test_model_decision_is_distinct_from_server_handoff(self):
        client = summarize_smoke(evidence())["client"]
        self.assertEqual(client["raw_model_handoff"], {"yes": 0, "no": 3, "unavailable": 0})
        self.assertEqual(client["escalated_requests"], 2)

    def test_invalid_model_still_counts_real_cost_and_server_handoff(self):
        data = evidence()
        data["cases"][0]["detail"]["admin"]["raw_model_result"] = None
        data["cases"][0]["public"]["client"]["status"] = "fallback"
        result = summarize_smoke(data)["client"]
        self.assertEqual(result["raw_model_handoff"]["unavailable"], 1)
        self.assertEqual(result["usage"]["total_tokens"], 360)

    def test_missing_usage_is_unknown_not_zero_cost(self):
        data = evidence()
        call = data["cases"][0]["detail"]["admin"]["usage_calls"][0]
        call.update(complete=False, api_cost=None, total_tokens=None, output_tokens=None, latency_ms=None)
        usage = summarize_smoke(data)["client"]["usage"]
        self.assertIsNone(usage["api_cost"])
        self.assertIsNone(usage["total_tokens"])
        self.assertEqual(usage["latency"]["observed_calls"], 2)
        self.assertFalse(usage["complete"])

    def test_unknown_human_savings_and_total_cost_remain_null(self):
        result = summarize_smoke(evidence())
        self.assertTrue(all(v is None for v in result["unmeasured"].values()))
        self.assertEqual(result["client"]["usage"]["api_cost"], 0)

    def test_request_without_usage_cannot_make_partial_totals_complete(self):
        data = evidence()
        data["cases"][0]["detail"]["admin"]["usage_calls"] = []
        usage = summarize_smoke(data)["client"]["usage"]
        self.assertEqual(usage["requests_without_usage"], 1)
        self.assertEqual(usage["calls"], 2)
        self.assertFalse(usage["complete"])
        self.assertIsNone(usage["total_tokens"])
        self.assertIsNone(usage["api_cost"])

    def test_completed_pair_cannot_reuse_client_request(self):
        data = evidence()
        data["comparison"]["fine_tuned"] = data["cases"][0]["detail"]
        with self.assertRaisesRegex(ValueError, "populations overlap"):
            summarize_smoke(data)

    def test_bad_input_scope_is_rejected(self):
        for key, value in (("scope", "test"), ("training_started", True), ("protected_data_read", True)):
            data = evidence()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_smoke(data)

    def test_invalid_numbers_are_rejected(self):
        for field, value in (("total_tokens", -1), ("input_tokens", True), ("api_cost", float("nan")),
                             ("output_tokens", 1.2), ("total_tokens", 121)):
            data = evidence()
            data["cases"][0]["detail"]["admin"]["usage_calls"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                summarize_smoke(data)

    def test_duplicate_call_is_not_charged_twice(self):
        data = evidence()
        calls = data["cases"][0]["detail"]["admin"]["usage_calls"]
        calls.append(copy.deepcopy(calls[0]))
        self.assertEqual(summarize_smoke(data)["client"]["usage"]["calls"], 3)

    def test_failed_comparison_does_not_contaminate_client_metrics(self):
        data = evidence()
        data["comparison"]["status"] = "failed"
        result = summarize_smoke(data)
        self.assertFalse(result["comparison"]["available"])
        self.assertEqual(result["client"]["usage"]["calls"], 3)


if __name__ == "__main__":
    unittest.main()
