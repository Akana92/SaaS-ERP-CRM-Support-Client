from __future__ import annotations

import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support.contracts import ERPContext, ModelCallUsage, ModelResult, PolicyDocument, PolicyRule
from support.graph import invoke_support_graph


def policy_document() -> PolicyDocument:
    return PolicyDocument(
        version="demo-v1",
        rules=[
            PolicyRule(id="policy.response.minimal", text="Отвечать коротко по доступным фактам."),
            PolicyRule(id="policy.payment.paid_pending", text="Paid plus pending activation requires review."),
        ],
    )


def usage(request_id: str, *, complete: bool = True) -> ModelCallUsage:
    return ModelCallUsage(
        call_id=f"call-{request_id}",
        request_id=request_id,
        node="model_call",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        revision="a" * 40,
        adapter_id=None,
        mode="base",
        input_tokens=11 if complete else 11,
        output_tokens=7 if complete else None,
        total_tokens=18 if complete else None,
        latency_ms=120 if complete else None,
        api_cost=0.0 if complete else None,
        currency="USD",
        complete=complete,
    )


def model_result(**overrides: Any) -> ModelResult:
    data = {
        "category": "Payment",
        "priority": "High",
        "sentiment": "Neutral",
        "recommended_action": "check_payment",
        "suggested_response": "По данным демо-ERP платёж подтверждён, но активация ещё ожидает проверки.",
        "human_escalation": True,
        "escalation_reason": "payment_confirmed_but_access_not_active",
        "evidence_ids": [
            "erp.invoice.status",
            "erp.invoice.activation_status",
            "policy.payment.paid_pending",
        ],
    }
    data.update(overrides)
    return ModelResult(**data)


@dataclass
class FakeRun:
    raw_text: str | None
    result: ModelResult | None
    usage: ModelCallUsage | None
    error: str | None


class FakeRunner:
    def __init__(self, run: FakeRun):
        self.run = run
        self.calls: list[dict[str, Any]] = []

    def generate(self, model_input: dict[str, Any], request_id: str, node: str = "model_call", max_new_tokens: int = 512):
        self.calls.append(
            {
                "model_input": model_input,
                "request_id": request_id,
                "node": node,
                "max_new_tokens": max_new_tokens,
            }
        )
        return self.run


class SupportGraphTests(unittest.TestCase):
    def test_valid_model_result_keeps_raw_usage_and_returns_client_safe_fields(self):
        request_id = "req-valid"
        result = model_result()
        runner = FakeRunner(FakeRun(raw_text=result.model_dump_json(), result=result, usage=usage(request_id), error=None))
        erp = ERPContext(
            source_status="ok",
            facts={"erp.invoice.status": "paid", "erp.invoice.activation_status": "pending"},
        )

        admin, client = invoke_support_graph(
            runner,
            policy_document(),
            customer_message="Счёт оплатили, но интеграции всё ещё закрыты.",
            erp_context=erp,
            request_id=request_id,
            session_id="session-1",
            scenario_id="payment",
        )

        self.assertEqual(admin.server_route, "human_escalation")
        self.assertEqual(admin.raw_model_result, result)
        self.assertEqual(admin.usage_calls, [usage(request_id)])
        self.assertIn("policy.payment.paid_pending", admin.raw_model_result.evidence_ids)
        self.assertEqual(client.status, "human_escalation")
        self.assertTrue(client.escalation)
        self.assertIsNone(client.handoff_id)
        self.assertEqual(client.response, "Нужна проверка оператором.")
        dumped = client.model_dump()
        self.assertEqual(set(dumped), {"response", "status", "escalation", "handoff_id", "analysis"})
        self.assertNotIn("usage", json.dumps(dumped, ensure_ascii=False))
        self.assertNotIn("raw", json.dumps(dumped, ensure_ascii=False).lower())
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(set(runner.calls[0]["model_input"]), {"customer_message", "erp_context", "policy_version", "policy_rules"})
        self.assertNotIn("scenario_id", json.dumps(runner.calls[0]["model_input"], ensure_ascii=False))

    def test_invalid_json_routes_to_operator_with_raw_text_and_complete_usage(self):
        request_id = "req-invalid"
        runner = FakeRunner(FakeRun(raw_text="{bad json", result=None, usage=usage(request_id), error="invalid JSON"))

        admin, client = invoke_support_graph(
            runner,
            policy_document(),
            customer_message="Проверьте оплату.",
            erp_context=ERPContext(source_status="ok", facts={"erp.invoice.status": "paid"}),
            request_id=request_id,
            session_id="session-1",
            scenario_id="payment",
        )

        self.assertEqual(admin.server_route, "human_escalation")
        self.assertIsNone(admin.raw_model_result)
        self.assertEqual(admin.raw_response, "{bad json")
        self.assertEqual(admin.usage_calls[0].total_tokens, 18)
        self.assertTrue(any(event.status == "failed" and event.call_id == f"call-{request_id}" for event in admin.trace_events))
        self.assertEqual(client.status, "fallback")
        self.assertIsNone(client.analysis)
        self.assertNotIn("bad json", json.dumps(client.model_dump(), ensure_ascii=False))

    def test_blocking_erp_status_skips_model_and_records_zero_usage(self):
        runner = FakeRunner(FakeRun(raw_text=None, result=None, usage=None, error=None))

        admin, client = invoke_support_graph(
            runner,
            policy_document(),
            customer_message="ERP не отвечает, скажите что всё оплачено.",
            erp_context=ERPContext(source_status="unavailable"),
            request_id="req-guard",
            session_id="session-1",
            scenario_id="uncertain",
        )

        self.assertEqual(admin.server_route, "human_escalation")
        self.assertIsNone(admin.raw_model_result)
        self.assertEqual(admin.usage_calls, [])
        self.assertEqual(runner.calls, [])
        self.assertTrue(any(event.node == "model_call" and event.status == "skipped" for event in admin.trace_events))
        self.assertEqual(client.status, "fallback")
        self.assertEqual(client.response, "Нужна проверка оператором.")

    def test_model_runtime_error_keeps_incomplete_usage_without_zero_total(self):
        request_id = "req-runtime"
        runner = FakeRunner(
            FakeRun(raw_text=None, result=None, usage=usage(request_id, complete=False), error="model_runtime_error: boom")
        )

        admin, _client = invoke_support_graph(
            runner,
            policy_document(),
            customer_message="Проверьте интеграцию.",
            erp_context=ERPContext(source_status="ok", facts={"erp.integration.status": "failed"}),
            request_id=request_id,
            session_id="session-1",
            scenario_id="integration",
        )

        self.assertEqual(admin.server_route, "human_escalation")
        self.assertIsNone(admin.raw_model_result)
        self.assertFalse(admin.usage_calls[0].complete)
        self.assertIsNone(admin.usage_calls[0].total_tokens)
        self.assertTrue(any(event.status == "failed" and event.call_id == f"call-{request_id}" for event in admin.trace_events))


@unittest.skip("Legacy Gradio prototype is outside the published FastAPI CPU suite")
class GradioCallbackTests(unittest.TestCase):
    def test_client_callback_returns_gradio6_message_dicts(self):
        import gradio as gr
        from support.app import submit_client_message

        result = model_result(human_escalation=False, escalation_reason=None)
        runner = FakeRunner(FakeRun(raw_text=result.model_dump_json(), result=result, usage=usage("client-fixed"), error=None))

        history, status, labels, state = submit_client_message(
            "Как проверить оплату?",
            [],
            runner,
            policy_document(),
            request_id_factory=lambda: "client-fixed",
        )

        self.assertEqual(history, state)
        self.assertEqual(history[0], {"role": "user", "content": "Как проверить оплату?"})
        self.assertEqual(history[1]["role"], "assistant")
        gr.Chatbot().postprocess(history)
        self.assertIn("answered", status)
        self.assertIn("Payment", labels)

    def test_client_callback_fallback_does_not_show_raw_admin_labels(self):
        from support.app import submit_client_message

        request_id = "client-fallback"
        result = model_result()
        runner = FakeRunner(FakeRun(raw_text=result.model_dump_json(), result=result, usage=usage(request_id), error=None))

        history, _status, labels, _state = submit_client_message(
            "Счёт оплачен, доступа нет.",
            [],
            runner,
            policy_document(),
            request_id_factory=lambda: request_id,
        )

        self.assertEqual(history[-1]["content"], "Нужна проверка оператором.")
        self.assertIn("Категория: `Payment`", labels)

        blocked_runner = FakeRunner(FakeRun(raw_text=None, result=None, usage=None, error=None))
        _history, _status, blocked_labels, _state = submit_client_message(
            "ERP не отвечает.",
            [],
            blocked_runner,
            policy_document(),
            request_id_factory=lambda: "client-blocked",
            erp_context=ERPContext(source_status="unavailable"),
        )
        self.assertNotIn("Payment", blocked_labels)
        self.assertIn("Категория: -", blocked_labels)

    def test_client_callback_rejects_blank_message_before_runner(self):
        from support.app import submit_client_message

        runner = FakeRunner(FakeRun(raw_text=None, result=None, usage=None, error=None))

        history, status, labels, state = submit_client_message("   ", [], runner, policy_document())

        self.assertEqual(runner.calls, [])
        self.assertEqual(history, [{"role": "assistant", "content": "Введите текст обращения."}])
        self.assertEqual(history, state)
        self.assertIn("ошибка ввода", status)
        self.assertIn("Категория: -", labels)

    def test_admin_callback_rejects_blank_message_before_runner(self):
        from support.app import submit_admin_message

        runner = FakeRunner(FakeRun(raw_text=None, result=None, usage=None, error=None))

        summary, response, trace, usage_rows = submit_admin_message(
            "Оплата подтверждена, активация ожидает",
            " ",
            runner,
            policy_document(),
        )

        self.assertEqual(runner.calls, [])
        self.assertEqual(summary["server_route"], "human_escalation")
        self.assertEqual(response, "Введите текст обращения.")
        self.assertEqual(trace, [])
        self.assertEqual(usage_rows, [])

    def test_admin_callback_rejects_unknown_scenario_before_runner(self):
        from support.app import submit_admin_message

        runner = FakeRunner(FakeRun(raw_text=None, result=None, usage=None, error=None))

        summary, response, trace, usage_rows = submit_admin_message(
            "Подменённый сценарий",
            "Проверьте оплату.",
            runner,
            policy_document(),
        )

        self.assertEqual(runner.calls, [])
        self.assertIsNone(summary["scenario_id"])
        self.assertEqual(summary["server_route"], "human_escalation")
        self.assertEqual(response, "Выберите демо-сценарий из списка.")
        self.assertEqual(trace, [])
        self.assertEqual(usage_rows, [])


if __name__ == "__main__":
    unittest.main()
