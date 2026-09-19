"""CPU regressions for public handoff consistency and requested repetition."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support.contracts import AdminResponse, ClientResponse, ModelCallUsage, ModelResult
from support.graph import _route_from_model, _client_status
from support import live_reply_guard as guards
from support.live_contracts import LiveAdminResponse, LiveClientResponse
from pydantic import ValidationError


class HandoffGuardTests(unittest.TestCase):
    def responses(self, text, action="provide_instructions"):
        result = ModelResult(category="Settings", priority="Low", sentiment="Neutral",
                             recommended_action=action, suggested_response=text,
                             human_escalation=False, escalation_reason=None, evidence_ids=[])
        route = _route_from_model(result)
        usage = ModelCallUsage(call_id="call", request_id="request", node="model_call", model_id="fake",
                               revision="a" * 40, mode="fine_tuned", adapter_id="fake-adapter",
                               input_tokens=10, output_tokens=20, total_tokens=30, latency_ms=1,
                               api_cost=0.0, currency="USD", complete=True)
        admin = AdminResponse(request_id="request", raw_model_result=result, raw_response=result.model_dump_json(),
                              server_route=route, response=text, usage_calls=[usage])
        return admin, ClientResponse.from_model_result(result, _client_status(route))

    def test_observed_required_specialist_preserves_raw_and_escalates_public(self):
        text = "Справка не подтверждает права этого сотрудника. Нужен специалист через этот чат; я не могу проверить его доступ."
        admin, client = self.responses(text)
        original = admin.model_dump(mode="json")
        self.assertEqual(client.status, "answered")
        safe_admin, safe_client = guards.apply_handoff_guard(admin, client)
        self.assertEqual(safe_admin.server_route, "human_escalation")
        self.assertTrue(safe_client.escalation)
        self.assertEqual(safe_client.status, "human_escalation")
        self.assertNotEqual(safe_client.response, text)
        self.assertEqual(safe_admin.response, safe_client.response)
        self.assertEqual(safe_admin.raw_model_result, admin.raw_model_result)
        self.assertEqual(safe_admin.raw_response, admin.raw_response)
        self.assertEqual(safe_client.analysis, client.analysis)
        self.assertEqual(safe_admin.usage_calls, admin.usage_calls)
        self.assertEqual(admin.model_dump(mode="json"), original)
        self.assertEqual(safe_admin.trace_events[-1].node, "handoff_guard")
        self.assertEqual(safe_admin.trace_events[-1].error, "explicit_handoff_not_routed")
        self.assertEqual(safe_admin.trace_events[-1].request_id, admin.request_id)

    def test_required_and_performed_handoffs(self):
        for text in ("Требуется оператор.", "Специалист необходим.", "Необходима проверка специалистом.",
                     "Нужна помощь специалиста.", "Передаю обращение специалисту.",
                     "Обращение передано оператору.", "Я не могу проверить доступ, нужен специалист."):
            with self.subTest(text=text):
                admin, client = self.responses(text, action="request_information")
                self.assertTrue(guards.apply_handoff_guard(admin, client)[1].escalation)

    def test_optional_negated_conditional_and_quoted_text_is_not_handoff(self):
        texts = ("Если нужна помощь — можно передать обращение специалисту.",
                 "Специалист не нужен.", "Не нужен специалист.", "Обращение не передано специалисту.",
                 "Не думаю, что нужен специалист.", "Не могу сказать, что требуется оператор.",
                 "Нужен ли специалист?", "Если ошибка повторится, нужен специалист.",
                 "Когда потребуется помощь, передаю обращение оператору.",
                 "При необходимости требуется оператор.", "Возможно, нужен специалист.",
                 "Например, нужен специалист.", "Предположим, требуется оператор.",
                 "Можно передать обращение специалисту.",
                 "В инструкции написано «Нужен специалист». Пока проверьте статус.",
                 'Сообщение "Требуется оператор" — пример текста.',
                 "Пример: `Передаю обращение специалисту`.",
                 "Специалист ранее помог с настройкой.", "Проверьте настройки самостоятельно.")
        for text in texts:
            with self.subTest(text=text):
                admin, client = self.responses(text)
                after = guards.apply_handoff_guard(admin, client)
                self.assertIs(after[0], admin)
                self.assertIs(after[1], client)

    def test_existing_escalation_or_missing_result_not_rewritten(self):
        admin, client = self.responses("Нужен специалист.")
        escalated_admin, escalated_client = guards.apply_handoff_guard(admin, client)
        after = guards.apply_handoff_guard(escalated_admin, escalated_client)
        self.assertIs(after[0], escalated_admin)
        self.assertIs(after[1], escalated_client)
        failed_admin = AdminResponse(request_id="request", raw_model_result=None,
                                     server_route="human_escalation", response="Нужен специалист.",
                                     trace_events=[dict(node="model_call", request_id="request", status="failed",
                                                        description="Model unavailable", duration_ms=0)])
        failed_client = ClientResponse(response="Нужен специалист.", status="fallback", escalation=True)
        self.assertEqual(guards.apply_handoff_guard(failed_admin, failed_client), (failed_admin, failed_client))

    def test_requested_one_line_repeat_is_allowed_but_unrequested_loop_is_not(self):
        text = "Печатная форма доступна в разделе Продажи → Счета → нужный счёт → Печатная форма. Просмотр не создаёт новый документ."
        admin, client = self.responses(text)
        history = [{"client": client.model_dump(mode="json")}]
        for message in ("Можно ещё раз, одной строкой, я записываю.", "Можно еще раз?", "Повторите инструкцию."):
            with self.subTest(message=message):
                self.assertIs(guards.apply_reply_guard(admin, client, history, message)[1], client)
        for message in ("Не надо ещё раз, я уже проверил.", "Не повторите ту же ошибку.", "Я уже проверил, ошибка осталась."):
            with self.subTest(message=message):
                safe_admin, safe_client = guards.apply_reply_guard(admin, client, history, message)
                self.assertFalse(safe_client.escalation)
                self.assertEqual(safe_client.status, "server_clarification")
                self.assertEqual(safe_admin.server_route, "server_clarification")
                self.assertEqual(safe_admin.raw_model_result, admin.raw_model_result)
                self.assertEqual(safe_client.analysis, client.analysis)

    def test_required_handoff_still_wins_over_repeat_guard(self):
        text = "Нужен специалист для проверки доступа к разделу. Самостоятельно проверить доступ в этом чате невозможно."
        admin, client = self.responses(text)
        history = [{"client": client.model_dump(mode="json")}]
        required_admin, required_client = guards.apply_handoff_guard(admin, client)
        self.assertEqual(guards.apply_reply_guard(required_admin, required_client, history, "Доступ всё ещё закрыт"),
                         (required_admin, required_client))
        self.assertTrue(required_client.escalation)

    def test_live_clarification_contracts_preserve_legacy_and_reject_invalid_states(self):
        text = "Подробная инструкция по настройке доступна в разделе справки; проверьте выбранный раздел и параметры."
        admin, client = self.responses(text)
        self.assertEqual(LiveAdminResponse.model_validate(admin.model_dump()).model_dump(), admin.model_dump())
        self.assertEqual(LiveClientResponse.model_validate(client.model_dump()).model_dump(), client.model_dump())
        safe_admin, safe_client = guards.apply_reply_guard(
            admin, client, [{"client": client.model_dump()}], "Уточняю вопрос")
        self.assertEqual(safe_admin.raw_response, admin.raw_response)
        self.assertEqual(safe_admin.usage_calls, admin.usage_calls)
        self.assertEqual(safe_admin.raw_model_result, admin.raw_model_result)
        self.assertEqual(LiveAdminResponse.model_validate_json(safe_admin.model_dump_json()), safe_admin)
        self.assertEqual(LiveClientResponse.model_validate_json(safe_client.model_dump_json()), safe_client)
        for override in ({"raw_model_result": None}, {"trace_events": []}, {"usage_calls": []},
                         {"raw_model_result": {**admin.raw_model_result.model_dump(),
                                               "human_escalation": True, "escalation_reason": "security_risk"}}):
            with self.subTest(override=override), self.assertRaises(ValidationError):
                LiveAdminResponse.model_validate({**safe_admin.model_dump(), **override})
        for override in ({"escalation": True}, {"handoff_id": "ticket"}, {"analysis": None},
                         {"analysis": {**client.analysis.model_dump(), "recommended_action": "escalate_human"}}):
            with self.subTest(override=override), self.assertRaises(ValidationError):
                LiveClientResponse.model_validate({**safe_client.model_dump(), **override})


if __name__ == "__main__":
    unittest.main()
