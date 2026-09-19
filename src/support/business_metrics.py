"""Descriptive metrics from authored live smoke evidence, not model evaluation.

This module has no GPU, server, training, dataset or network dependencies.
"""
from __future__ import annotations

from collections import Counter
import math
from statistics import mean, median

SCOPE = "new_manual_fictional_erp_smoke_not_quality_evaluation"


def _unique(rows, key):
    by_id = {}
    for row in rows:
        identity = row[key]
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"Missing {key}")
        if identity in by_id and by_id[identity] != row:
            raise ValueError(f"Conflicting duplicate {key}")
        by_id[identity] = row
    return list(by_id.values())


def _number(value, field):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {field}")
    if field.endswith("tokens") and not isinstance(value, int):
        raise ValueError(f"Non-integer {field}")
    return value


def _usage(records, mode):
    calls = []
    for record in records:
        for call in record["admin"]["usage_calls"]:
            if call["request_id"] != record["request_id"] or call["mode"] != mode:
                raise ValueError("Usage identity/mode mismatch")
            if type(call["complete"]) is not bool or call["currency"] != "USD":
                raise ValueError("Unsupported usage completeness/currency")
            for field in ("input_tokens", "output_tokens", "total_tokens", "api_cost", "latency_ms"):
                _number(call.get(field), field)
            if all(call.get(k) is not None for k in ("input_tokens", "output_tokens", "total_tokens")):
                if call["input_tokens"] + call["output_tokens"] != call["total_tokens"]:
                    raise ValueError("Inconsistent token total")
            calls.append(call)
    calls = _unique(calls, "call_id")
    missing_requests = sum(not record["admin"]["usage_calls"] for record in records)
    complete = bool(calls) and not missing_requests and all(c["complete"] for c in calls)
    totals = {}
    for field in ("input_tokens", "output_tokens", "total_tokens", "api_cost"):
        values = [c.get(field) for c in calls]
        totals[field] = sum(values) if complete and all(v is not None for v in values) else None
    observed = [c["latency_ms"] for c in calls if c["complete"] and c.get("latency_ms") is not None]
    latency = dict(observed_calls=len(observed), mean_ms=mean(observed) if observed else None,
                   median_ms=median(observed) if observed else None,
                   min_ms=min(observed) if observed else None, max_ms=max(observed) if observed else None,
                   p95_ms=None, p95_note="Not reported for this small purposive smoke sample",
                   scope="generation only; excludes queue, prompt preparation and delivery; not TTFT")
    return {"calls": len(calls), "requests_without_usage": missing_requests,
            "complete_calls": sum(c["complete"] for c in calls),
            "complete": complete, **totals, "currency": "USD", "latency": latency,
            "tokens_per_call": totals["total_tokens"] / len(calls) if totals["total_tokens"] is not None else None,
            "api_cost_per_call": totals["api_cost"] / len(calls) if totals["api_cost"] is not None else None}


def summarize_smoke(evidence):
    if (evidence.get("scope") != SCOPE or evidence.get("training_started") is not False
            or evidence.get("protected_data_read") is not False):
        raise ValueError("Expected authored live-smoke evidence, not training or evaluation data")
    rows = []
    for case in evidence["cases"]:
        detail, public = case["detail"], case["public"]
        if detail["request_id"] != public["request_id"] or detail["client"] != public["client"]:
            raise ValueError("Public/admin record mismatch")
        rows.append({"request_id": detail["request_id"], "conversation_id": case["conversation_id"],
                     "client": public["client"], "admin": detail["admin"]})
    records = _unique(rows, "request_id")
    if not records:
        raise ValueError("No client records")
    conversations = {}
    model_handoffs = Counter()
    for record in records:
        client = record["client"]
        if type(client["escalation"]) is not bool:
            raise ValueError("Missing escalation decision")
        if client["status"] not in {"answered", "needs_information", "human_escalation", "fallback", "server_clarification"}:
            raise ValueError("Unknown client status")
        conversation = record["conversation_id"]
        if not isinstance(conversation, str) or not conversation:
            raise ValueError("Missing conversation_id")
        conversations[conversation] = conversations.get(conversation, False) or client["escalation"]
        result = record["admin"]["raw_model_result"]
        flag = result["human_escalation"] if result is not None else None
        if flag is not None and type(flag) is not bool:
            raise ValueError("Invalid raw model escalation")
        model_handoffs["unavailable" if flag is None else "yes" if flag else "no"] += 1
    statuses = Counter(r["client"]["status"] for r in records)
    handoffs = sum(r["client"]["escalation"] for r in records)
    answered = sum(r["client"]["status"] == "answered" and not r["client"]["escalation"] for r in records)
    comparison = evidence.get("comparison")
    compare = {"available": False, "note": "No completed pair; excluded from client metrics"}
    if comparison and comparison.get("status") == "completed":
        pair = [comparison[mode] for mode in ("base", "fine_tuned")]
        client_ids = {r["request_id"] for r in records}
        if pair[0]["request_id"] == pair[1]["request_id"] or any(r["request_id"] in client_ids for r in pair):
            raise ValueError("Client/comparison populations overlap")
        compare = {"available": True, "base": _usage([pair[0]], "base"),
                   "fine_tuned": _usage([pair[1]], "fine_tuned"),
                   "note": "One authored pair; excluded from client traffic and routing denominators"}
    return {"schema_version": "business-smoke-metrics-v1", "scope": SCOPE,
            "population_note": "Small authored functional smoke sample, not representative production traffic or a quality test",
            "client": {"requests": len(records), "conversations": len(conversations), "statuses": dict(statuses),
                       "answered_without_handoff": answered, "answered_without_handoff_rate": answered / len(records),
                       "escalated_requests": handoffs, "escalated_request_rate": handoffs / len(records),
                       "conversations_with_handoff": sum(conversations.values()),
                       "conversation_handoff_rate": sum(conversations.values()) / len(conversations),
                       "raw_model_handoff": {k: model_handoffs[k] for k in ("yes", "no", "unavailable")},
                       "usage": _usage(records, "fine_tuned")},
            "comparison": compare,
            "unmeasured": {"confirmed_resolution_rate": None, "operator_minutes_saved": None,
                           "human_handling_baseline_minutes": None, "end_to_end_response_ms": None,
                           "electricity_cost": None, "hardware_cost": None, "total_cost_per_request": None},
            "limitations": ["Routing without handoff is not confirmed resolution or autonomous success",
                            "No human time study, production population, final test or new quality judging",
                            "API fee 0 does not measure total operating cost",
                            "One conversation may contain several requests and handoff cards",
                            "Reported model latency excludes queue and user-visible delivery"]}
