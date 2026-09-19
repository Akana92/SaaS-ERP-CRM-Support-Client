from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support.contracts import (
    AdminResponse,
    AdminDemoRequest,
    ClientResponse,
    DevCase,
    ERPContext,
    ModelCallUsage,
    ModelResult,
    PolicyDocument,
    PolicyRule,
    SupportRequest,
    SupportState,
    TraceEvent,
    export_json_schemas,
    model_input_from_case,
    validate_evidence,
)


def valid_model_result(**overrides):
    data = {
        "category": "Payment",
        "priority": "High",
        "sentiment": "Negative",
        "recommended_action": "check_payment",
        "suggested_response": "Платёж подтверждён, но активация требует проверки оператором.",
        "human_escalation": True,
        "escalation_reason": "payment_confirmed_but_access_not_active",
        "evidence_ids": [
            "erp.invoice.INV-2048.status",
            "erp.module.warehouse.activation",
            "policy.manual_activation_review",
        ],
    }
    data.update(overrides)
    return ModelResult(**data)


def non_escalating_result(**overrides):
    data = {
        "category": "Settings",
        "priority": "Low",
        "sentiment": "Neutral",
        "recommended_action": "provide_instructions",
        "suggested_response": "Откройте настройки профиля и проверьте выбранный язык интерфейса.",
        "human_escalation": False,
        "escalation_reason": None,
        "evidence_ids": ["policy.settings_help"],
    }
    data.update(overrides)
    return ModelResult(**data)


def policy_document(**overrides):
    data = {
        "version": "policy-v1",
        "rules": [
            {"id": "policy.settings_help", "text": "Для настроек можно дать инструкцию клиенту."},
            {
                "id": "policy.manual_activation_review",
                "text": "Если оплата подтверждена, но модуль не активен, нужна проверка оператором.",
            },
        ],
    }
    data.update(overrides)
    return PolicyDocument(**data)


def valid_usage(**overrides):
    data = {
        "call_id": "call-1",
        "request_id": "req-1",
        "node": "model_call",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "rev-a",
        "adapter_id": None,
        "mode": "base",
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
        "latency_ms": 900,
        "api_cost": 0.0,
        "currency": "USD",
        "complete": True,
    }
    data.update(overrides)
    return ModelCallUsage(**data)


class ContractTests(unittest.TestCase):
    def test_blank_request_message_rejected(self):
        with self.assertRaises(ValidationError):
            SupportRequest(message="   ")

    def test_public_request_has_message_only_and_admin_demo_selects_scenario(self):
        request = SupportRequest(message="Проверьте оплату")
        self.assertEqual(request.message, "Проверьте оплату")
        with self.assertRaises(ValidationError):
            SupportRequest(
                message="Проверьте оплату",
                scenario_id="payment",
            )
        admin_request = AdminDemoRequest(
            message="Проверьте оплату",
            scenario_id="payment",
            model_mode="base",
        )
        self.assertEqual(admin_request.scenario_id, "payment")
        with self.assertRaises(ValidationError):
            AdminDemoRequest(
                message="Проверьте оплату",
                scenario_id="real-client-case",
                model_mode="base",
                org_id="must-not-be-client-controlled",
            )

    def test_bool_string_is_not_coerced(self):
        with self.assertRaises(ValidationError):
            valid_model_result(human_escalation="true")

    def test_escalation_reason_required_only_when_escalating(self):
        with self.assertRaises(ValidationError):
            valid_model_result(human_escalation=True, escalation_reason=None)
        with self.assertRaises(ValidationError):
            valid_model_result(
                human_escalation=False,
                escalation_reason="payment_dispute",
                recommended_action="provide_instructions",
            )
        result = valid_model_result(
            human_escalation=False,
            escalation_reason=None,
            recommended_action="provide_instructions",
        )
        self.assertFalse(result.human_escalation)
        with self.assertRaises(ValidationError):
            valid_model_result(
                recommended_action="escalate_human",
                human_escalation=False,
                escalation_reason=None,
            )
        result = valid_model_result(recommended_action="check_payment")
        self.assertTrue(result.human_escalation)
        with self.assertRaises(ValidationError):
            ModelResult(
                category="Other",
                priority="Low",
                sentiment="Neutral",
                recommended_action="provide_instructions",
                suggested_response="Ответ без источников допустим только с явным пустым списком.",
                human_escalation=False,
                escalation_reason=None,
            )
        with self.assertRaises(ValidationError):
            non_escalating_result(evidence_ids=["policy.settings_help", "policy.settings_help"])

    def test_erp_context_validates_source_and_evidence_ids(self):
        erp = ERPContext(
            source_status="ok",
            facts={"erp.invoice.INV-2048.status": "paid", "erp.invoice.amount": 1200.5},
        )
        self.assertEqual(erp.facts["erp.invoice.INV-2048.status"], "paid")
        with self.assertRaises(ValidationError):
            ERPContext(source_status="ok", facts={"crm.invoice.status": "paid"})
        with self.assertRaises(ValidationError):
            ERPContext(source_status="ok", facts={"erp.invoice.amount": math.inf})
        with self.assertRaises(ValidationError):
            ERPContext(source_status="unavailable", facts={"erp.account.id": "org-1"})
        with self.assertRaises(ValidationError):
            ERPContext(source_status="not_found", facts={"erp.account.id": "missing"})

    def test_validate_evidence_rejects_fake_ids(self):
        erp = ERPContext(
            source_status="ok",
            facts={
                "erp.invoice.INV-2048.status": "paid",
                "erp.module.warehouse.activation": "pending",
            },
        )
        result = valid_model_result()
        validate_evidence(result, erp, {"policy.manual_activation_review"})
        with self.assertRaises(ValueError):
            validate_evidence(result, erp, set())

    def test_usage_arithmetic_unknown_vs_zero_and_adapter_mode(self):
        usage = valid_usage()
        self.assertEqual(usage.total_tokens, 150)
        with self.assertRaises(ValidationError):
            valid_usage(total_tokens=151)
        with self.assertRaises(ValidationError):
            valid_usage(complete=False, input_tokens=None, output_tokens=None, total_tokens=0)
        incomplete = valid_usage(
            complete=False,
            input_tokens=120,
            output_tokens=None,
            total_tokens=None,
            latency_ms=None,
            api_cost=None,
        )
        self.assertIsNone(incomplete.total_tokens)
        with self.assertRaises(ValidationError):
            valid_usage(mode="base", adapter_id="adapter-a")
        valid_usage(mode="fine_tuned", adapter_id="adapter-a")
        with self.assertRaises(ValidationError):
            valid_usage(mode="fine_tuned", adapter_id=None)

    def test_client_response_does_not_leak_internal_fields(self):
        with self.assertRaises(ValueError):
            ClientResponse.from_model_result(valid_model_result(), status="human_escalation")
        response = ClientResponse.from_model_result(
            valid_model_result(),
            status="human_escalation",
            server_response="Передам обращение оператору для проверки.",
            handoff_id="handoff-1",
        )
        self.assertEqual(
            set(response.model_dump().keys()),
            {"response", "status", "escalation", "handoff_id", "analysis"},
        )
        self.assertEqual(response.handoff_id, "handoff-1")
        self.assertEqual(
            set(response.analysis.model_dump().keys()),
            {"category", "priority", "sentiment", "recommended_action"},
        )
        answered = ClientResponse.from_model_result(non_escalating_result(), status="answered")
        self.assertFalse(answered.escalation)
        with self.assertRaises(ValidationError):
            ClientResponse(
                response="Ответ готов.",
                status="answered",
                escalation=False,
                analysis={
                    "category": "Other",
                    "priority": "Low",
                    "sentiment": "Neutral",
                    "recommended_action": "request_information",
                },
            )
        with self.assertRaises(ValidationError):
            ClientResponse(
                response="Ответ готов.",
                status="needs_information",
                escalation=False,
                analysis={
                    "category": "Settings",
                    "priority": "Low",
                    "sentiment": "Neutral",
                    "recommended_action": "provide_instructions",
                },
            )
        fallback = ClientResponse(
            response="Сейчас не могу проверить данные, передам обращение оператору.",
            status="fallback",
            escalation=True,
            handoff_id="handoff-2",
            analysis=None,
        )
        self.assertEqual(fallback.status, "fallback")
        helper_fallback = ClientResponse.from_model_result(
            valid_model_result(),
            status="fallback",
            server_response="Сейчас не могу проверить данные, передам обращение оператору.",
        )
        self.assertIsNone(helper_fallback.analysis)
        with self.assertRaises(ValidationError):
            ClientResponse(
                response="Передам оператору.",
                status="human_escalation",
                escalation=True,
                raw_model_result={},
            )

    def test_admin_response_validates_usage_uniqueness_and_request_consistency(self):
        usage = valid_usage(call_id="call-1")
        trace = TraceEvent(
            node="model_call",
            description="Вызов модели",
            request_id="req-1",
            status="ok",
            duration_ms=910,
            call_id="call-1",
            token_count=150,
        )
        admin = AdminResponse(
            request_id="req-1",
            raw_model_result=non_escalating_result(),
            server_route="auto_answer",
            response="Откройте настройки профиля.",
            usage_calls=[usage],
            trace_events=[trace],
        )
        self.assertEqual(admin.usage_calls[0].call_id, "call-1")
        with self.assertRaises(ValidationError):
            AdminResponse(
                request_id="req-1",
                raw_model_result=None,
                server_route="human_escalation",
                response="Fallback.",
                usage_calls=[usage, valid_usage(call_id="call-1")],
                trace_events=[],
            )
        with self.assertRaises(ValidationError):
            AdminResponse(
                request_id="req-1",
                raw_model_result=None,
                server_route="human_escalation",
                response="Fallback.",
                usage_calls=[valid_usage(request_id="other")],
                trace_events=[],
            )
        with self.assertRaises(ValidationError):
            AdminResponse(
                request_id="req-1",
                raw_model_result=non_escalating_result(),
                server_route="auto_answer",
                response="Откройте настройки профиля.",
                usage_calls=[],
                trace_events=[],
            )
        with self.assertRaises(ValidationError):
            AdminResponse(
                request_id="req-1",
                raw_model_result=non_escalating_result(recommended_action="request_information"),
                server_route="auto_answer",
                response="Откройте настройки профиля.",
                usage_calls=[usage],
                trace_events=[trace],
            )
        with self.assertRaises(ValidationError):
            AdminResponse(
                request_id="req-1",
                raw_model_result=non_escalating_result(),
                server_route="auto_answer",
                response="Откройте настройки профиля.",
                usage_calls=[usage],
                trace_events=[
                    TraceEvent(
                        node="model_call",
                        description="Вызов модели",
                        request_id="req-1",
                        status="ok",
                        duration_ms=910,
                        call_id="missing-call",
                    )
                ],
            )

    def test_admin_response_allows_human_override_and_failure_usage(self):
        usage = valid_usage(call_id="call-1")
        non_escalating_raw = non_escalating_result()
        override = AdminResponse(
            request_id="req-1",
            raw_model_result=non_escalating_raw,
            server_route="human_escalation",
            response="Передам обращение оператору из-за серверной проверки.",
            usage_calls=[usage],
            trace_events=[
                TraceEvent(
                    node="server_checks",
                    description="Серверная проверка потребовала оператора",
                    request_id="req-1",
                    status="ok",
                    duration_ms=12,
                )
            ],
        )
        self.assertIs(override.raw_model_result, non_escalating_raw)
        with self.assertRaises(ValidationError):
            AdminResponse(
                request_id="req-1",
                raw_model_result=non_escalating_raw,
                server_route="human_escalation",
                response="Передам обращение оператору из-за серверной проверки.",
                usage_calls=[],
                trace_events=[
                    TraceEvent(
                        node="server_checks",
                        description="Серверная проверка потребовала оператора",
                        request_id="req-1",
                        status="ok",
                        duration_ms=12,
                    )
                ],
            )
        with self.assertRaises(ValidationError):
            AdminResponse(
                request_id="req-1",
                raw_model_result=non_escalating_raw,
                server_route="human_escalation",
                response="Передам обращение оператору из-за серверной проверки.",
                usage_calls=[
                    valid_usage(
                        complete=False,
                        input_tokens=120,
                        output_tokens=None,
                        total_tokens=None,
                        latency_ms=None,
                        api_cost=None,
                    )
                ],
                trace_events=[
                    TraceEvent(
                        node="model_call",
                        description="Модель вызвана, но учёт не завершён",
                        request_id="req-1",
                        status="failed",
                        duration_ms=1000,
                        call_id="call-1",
                    )
                ],
            )
        failed_usage = valid_usage(
            call_id="call-failed",
            complete=False,
            input_tokens=120,
            output_tokens=None,
            total_tokens=None,
            latency_ms=None,
            api_cost=None,
        )
        runtime_error = AdminResponse(
            request_id="req-1",
            raw_model_result=None,
            server_route="human_escalation",
            response="Модель не вернула корректный результат, передам обращение оператору.",
            usage_calls=[failed_usage],
            trace_events=[
                TraceEvent(
                    node="model_call",
                    description="Ошибка выполнения модели",
                    request_id="req-1",
                    status="failed",
                    duration_ms=1000,
                    call_id="call-failed",
                    error="model_runtime_error",
                )
            ],
        )
        self.assertIsNone(runtime_error.raw_model_result)
        no_call = AdminResponse(
            request_id="req-1",
            raw_model_result=None,
            server_route="human_escalation",
            response="ERP недоступна, передам обращение оператору.",
            usage_calls=[],
            trace_events=[
                TraceEvent(
                    node="erp_context",
                    description="ERP недоступна, модель не вызывалась",
                    request_id="req-1",
                    status="skipped",
                    duration_ms=50,
                )
            ],
        )
        self.assertEqual(no_call.usage_calls, [])

    def test_support_state_is_json_serializable(self):
        state = SupportState(
            request_id="req-1",
            session_id="session-1",
            scenario_id="payment",
            customer_message="Оплатили модуль, доступа нет.",
            model_mode="base",
            erp_context=ERPContext(source_status="ok", facts={"erp.account.id": "org-1"}),
            usage_calls=[valid_usage()],
        )
        encoded = state.to_jsonable()
        self.assertEqual(encoded["request_id"], "req-1")
        json.dumps(encoded, ensure_ascii=False)

    def test_dev_case_expected_and_model_input_rules(self):
        erp = ERPContext(
            source_status="ok",
            facts={
                "erp.invoice.INV-2048.status": "paid",
                "erp.module.warehouse.activation": "pending",
            },
        )
        case = DevCase(
            id="dev-001",
            family_id="payment-activation",
            split="development",
            source="synthetic_authored",
            review_status="pending_human_review",
            scenario_id="payment",
            customer_message="Мы оплатили склад, почему доступа нет?",
            org_id="org-demo-a",
            erp_context=erp,
            policy_version="policy-v1",
            policy_refs=["policy.manual_activation_review"],
            expected=valid_model_result(),
            expected_route="human_escalation",
            expected_model_call=True,
            fallback_response=None,
            notes_for_reviewer="Проверить, что модель не обещает активацию.",
            covered_risks=["payment_confirmed_but_access_not_active"],
        )
        model_input = model_input_from_case(case, policy_document())
        self.assertEqual(
            set(model_input.keys()),
            {"customer_message", "erp_context", "policy_version", "policy_rules"},
        )
        forbidden = {
            "scenario_id",
            "expected",
            "expected_route",
            "expected_model_call",
            "fallback_response",
            "notes_for_reviewer",
            "covered_risks",
            "org_id",
        }
        self.assertTrue(forbidden.isdisjoint(model_input))
        self.assertEqual(
            model_input["policy_rules"],
            model_input_from_case(
                DevCase(
                    id="dev-003",
                    family_id="settings-help",
                    split="development",
                    source="synthetic_authored",
                    review_status="pending_human_review",
                    scenario_id="information",
                    customer_message="Как поменять язык?",
                    org_id="org-demo-b",
                    erp_context=ERPContext(source_status="ok", facts={"erp.account.id": "org-demo-b"}),
                    policy_version="policy-v1",
                    policy_refs=["policy.settings_help"],
                    expected=non_escalating_result(),
                    expected_route="auto_answer",
                    expected_model_call=True,
                    fallback_response=None,
                    notes_for_reviewer="Проверить обычную инструкцию.",
                    covered_risks=["settings_question"],
                ),
                policy_document(),
            )["policy_rules"],
        )
        with self.assertRaises(ValueError):
            model_input_from_case(case, policy_document(version="policy-v2"))
        with self.assertRaises(ValidationError):
            DevCase(
                id="dev-002",
                family_id="erp-timeout",
                split="development",
                source="synthetic_authored",
                review_status="pending_human_review",
                scenario_id="uncertain",
                customer_message="Что с доступом?",
                org_id="org-demo-a",
                erp_context=ERPContext(source_status="unavailable"),
                policy_version="policy-v1",
                policy_refs=[],
                expected=valid_model_result(),
                expected_route="human_escalation",
                expected_model_call=False,
                fallback_response=None,
                notes_for_reviewer="No LLM path.",
                covered_risks=["erp_unavailable"],
            )
        blocked = DevCase(
            id="dev-004",
            family_id="erp-timeout",
            split="development",
            source="synthetic_authored",
            review_status="pending_human_review",
            scenario_id="uncertain",
            customer_message="Что с доступом?",
            org_id="org-demo-a",
            erp_context=ERPContext(source_status="unavailable"),
            policy_version="policy-v1",
            policy_refs=[],
            expected=None,
            expected_route="human_escalation",
            expected_model_call=False,
            fallback_response="ERP недоступна, обращение должен проверить оператор.",
            notes_for_reviewer="No LLM path.",
            covered_risks=["erp_unavailable"],
        )
        with self.assertRaises(ValueError):
            model_input_from_case(blocked, policy_document())
        with self.assertRaises(ValidationError):
            DevCase(
                id="dev-005",
                family_id="settings-help",
                split="development",
                source="synthetic_authored",
                review_status="pending_human_review",
                scenario_id="information",
                customer_message="Как поменять язык?",
                org_id="org-demo-a",
                erp_context=ERPContext(source_status="ok", facts={"erp.account.id": "org-demo-a"}),
                policy_version="policy-v1",
                policy_refs=["policy.settings_help"],
                expected=non_escalating_result(),
                expected_route="auto_answer",
                expected_model_call=True,
                fallback_response="Should not be here.",
                notes_for_reviewer="Model path.",
                covered_risks=["settings_question"],
            )

    def test_schema_export_is_deterministic_and_contains_required_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = export_json_schemas(tmp)
            first_snapshot = {path.name: path.read_text(encoding="utf-8") for path in first}
            second = export_json_schemas(tmp)
            second_snapshot = {path.name: path.read_text(encoding="utf-8") for path in second}
        self.assertEqual(first_snapshot, second_snapshot)
        self.assertIn("ModelResult.schema.json", first_snapshot)
        self.assertIn('"additionalProperties": false', first_snapshot["ModelResult.schema.json"])


if __name__ == "__main__":
    unittest.main()
