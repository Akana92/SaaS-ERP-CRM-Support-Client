from __future__ import annotations

import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient
from support.contracts import ModelCallUsage, ModelResult, PolicyDocument, PolicyRule
from support.live_app import create_live_app
from support import live_evaluation


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.chat_calls = []
        self.fail = False
        self.result_overrides = {}
        self.entered, self.release = threading.Event(), threading.Event()
        self.block = False

    def status(self):
        return dict(loaded=True, busy=self.block and self.entered.is_set() and not self.release.is_set(), active_mode="base",
                    inference_profile="bf16", precision_scope="serving_only_experiment")

    def count_prompt_tokens(self, model_input):
        # Deliberately small deterministic test counter; production uses the tokenizer.
        return 128 + len(json.dumps(model_input, ensure_ascii=False).encode("utf-8")) // 4

    def count_chat_prompt_tokens(self, messages):
        return 128 + len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) // 4

    def for_mode(self, mode):
        runtime = self
        class Runner:
            def generate(self, model_input, request_id, **kwargs):
                runtime.calls.append((mode, copy.deepcopy(model_input)))
                runtime.chat_calls.append(copy.deepcopy(kwargs.get("chat_messages")))
                if runtime.block:
                    runtime.entered.set()
                    if not runtime.release.wait(5):
                        raise RuntimeError("Test timeout")
                if runtime.fail:
                    raise RuntimeError("secret internal path")
                result = ModelResult(category="Settings", priority="Low", sentiment="Neutral",
                    recommended_action="provide_instructions", suggested_response="Проверьте настройки.",
                    human_escalation=False, escalation_reason=None, evidence_ids=["erp.help"])
                result = ModelResult.model_validate({**result.model_dump(), **runtime.result_overrides})
                usage = ModelCallUsage(call_id="call-" + request_id, request_id=request_id, node="model_call",
                    model_id="fake", revision="a" * 40, mode=mode, adapter_id="fake-adapter" if mode == "fine_tuned" else None,
                    input_tokens=11, output_tokens=7, total_tokens=18, latency_ms=1, api_cost=0.0, currency="USD", complete=True)
                return SimpleNamespace(raw_text=result.model_dump_json(), result=result, usage=usage, error=None)
        return Runner()


class LiveAppTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.audiences = [dict(id="customer", label="Справка", prompt="Как настроить?", org_id="org-one",
                               erp_context=dict(source_status="ok", facts={"erp.help": "Настройки"})),
                          dict(id="employee", label="Ошибка ERP", prompt="ERP не работает", org_id="org-one",
                               erp_context=dict(source_status="unavailable", facts={}))]
        policy = PolicyDocument(version="demo", rules=[PolicyRule(id="policy.one", text="Только факты.")])
        self.app = create_live_app(self.runtime, policy, self.audiences)
        self.client = TestClient(self.app, base_url="http://127.0.0.1")
        self.client.get("/api/client/bootstrap")
        self.conv = self.client.post("/api/client/conversations", json={"audience": "customer"}).json()["id"]

    def tearDown(self):
        self.runtime.release.set()
        self.app.state.service.close()

    def wait_job(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/admin/comparisons/{job_id}").json()
            if job["status"] in {"completed", "failed"}:
                return job
            time.sleep(0.01)
        self.fail("Comparison did not terminate")

    def send(self, key="message-0001", message="Как настроить?", conv=None):
        return self.client.post(f"/api/client/conversations/{conv or self.conv}/messages", json={"message": message, "idempotency_key": key})

    def test_client_uses_fine_tuned_and_next_turn_receives_owned_history(self):
        first = self.send(message="Бюджетирование: ошибка без текста.").json()
        second = self.send(key="dialogue-0002", message="Это запуск счетов на оплату, все поля заполнены.").json()
        self.assertEqual([mode for mode, _ in self.runtime.calls], ["fine_tuned", "fine_tuned"])
        effective = self.runtime.calls[-1][1]
        chat = self.runtime.chat_calls[-1]
        self.assertEqual([item["role"] for item in chat], ["system", "user", "assistant", "user"])
        self.assertIn("Бюджетирование: ошибка без текста.", chat[1]["content"])
        self.assertIn(first["client"]["response"], chat[2]["content"])
        self.assertIn("Это запуск счетов на оплату", chat[-1]["content"])
        self.assertEqual(effective["customer_message"], "Это запуск счетов на оплату, все поля заполнены.")
        self.assertEqual(effective["erp_context"]["facts"], {"erp.help": "Настройки"})
        record = self.app.state.service.requests[second["request_id"]]
        self.assertEqual(record["dialogue_context"]["included_request_ids"], [first["request_id"]])
        self.assertEqual(record["admin"]["usage_calls"][0]["mode"], "fine_tuned")
        self.assertEqual(self.app.state.service.snapshot()["runtime"]["client_mode"], "fine_tuned")
        self.assertNotIn("dialogue_context", second)
        detail = self.client.get("/api/admin/requests/" + second["request_id"]).json()
        self.assertEqual(detail["model_history"], [first])
        self.assertNotIn("model_history", second)

    def test_serving_identity_in_health_and_admin_state_is_copied(self):
        identity = dict(adapter_profile="dialogue-250", adapter_path="local/pilot/final_adapter",
                        adapter_model_sha256="a" * 64, client_mode="fine_tuned")
        app = create_live_app(self.runtime, self.app.state.service.policy, self.audiences, serving_info=identity)
        try:
            client = TestClient(app, base_url="http://127.0.0.1")
            identity["adapter_path"] = "changed-after-startup"
            health = client.get("/health").json()
            state = client.get("/api/admin/state").json()["runtime"]
            for result in (health, state):
                self.assertEqual(result["adapter_profile"], "dialogue-250")
                self.assertEqual(result["adapter_path"], "local/pilot/final_adapter")
                self.assertEqual(result["adapter_model_sha256"], "a" * 64)
                self.assertEqual(result["client_mode"], "fine_tuned")
            self.assertTrue(health["loaded"])
        finally:
            app.state.service.close()

    def test_new_adapter_does_not_present_old_validation_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            saved = dict(base=dict(accuracy={"category": dict(correct=120, total=150)}, counts=dict(cases_loaded=150)),
                         fine_tuned=dict(accuracy={"category": dict(correct=140, total=150)}))
            (report / "comparison.json").write_text(json.dumps(saved), encoding="utf-8")
            (report / "comparison.html").write_text("Archived full-v8 validation", encoding="utf-8")
            with patch("support.live_app.REPORT", report):
                for profile in (None, "full-v8", "dialogue-250"):
                    app = create_live_app(self.runtime, self.app.state.service.policy, self.audiences,
                                          serving_info=None if profile is None else {"adapter_profile": profile})
                    try:
                        client = TestClient(app, base_url="http://127.0.0.1")
                        result = client.get("/api/admin/evaluation").json()
                        if profile == "dialogue-250":
                            self.assertFalse(result["available"])
                            self.assertEqual(result["split"], "development")
                            self.assertEqual(result["metrics"], [])
                            self.assertIsNone(result["n"])
                            self.assertIsNone(result["report_url"])
                            self.assertIn("150", result["quality_note"])
                            self.assertIn("full-v8", result["quality_note"])
                            self.assertIn("development", result["quality_note"])
                        else:
                            self.assertTrue(result["available"])
                            self.assertEqual(result["split"], "validation")
                            self.assertEqual(result["n"], 150)
                            self.assertEqual(result["metrics"][0]["fine_tuned"], 140)
                            self.assertEqual(result["report_url"], "/admin/validation-report")
                        self.assertIn("Archived full-v8", client.get("/admin/validation-report").text)
                    finally:
                        app.state.service.close()

    def test_candidate_f_saved_evaluation_and_public_report(self):
        identity = dict(adapter_profile="quality-f", adapter_model_sha256=live_evaluation.ADAPTER_SHA256,
                        candidate_status="selected_for_local_demo")
        app = create_live_app(self.runtime, self.app.state.service.policy, self.audiences, serving_info=identity)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            result = client.get("/api/admin/evaluation").json()
            self.assertTrue(result["available"])
            self.assertEqual((result["split"], result["n"], result["turns"]), ("development", 100, 225))
            self.assertEqual(result["candidate_status"], "selected_for_local_demo")
            self.assertEqual(result["assessment"], "ai_assessed_descriptive")
            self.assertFalse(result["human_verified"])
            self.assertFalse(result["accepted_merge"])
            self.assertIsNone(result["full_success_rate"])
            self.assertEqual(result["method_status"], "QUALIFIED")
            self.assertEqual(result["dialogue_outcomes"], {"denominator": 100,
                "base": {"successful": 10, "failed": 90, "unknown": 0},
                "fine_tuned": {"successful": 49, "failed": 51, "unknown": 0}})
            self.assertEqual(result["unresolved"], {"criteria": 4, "rows": 4, "paired_cases": 3})
            self.assertEqual(len(result["metrics"]), 7)
            self.assertEqual(next(row for row in result["metrics"] if row["label"] == "all_five"),
                             dict(label="all_five", base=64, fine_tuned=175, denominator=225))
            report = client.get(result["report_url"])
            self.assertEqual(report.status_code, 200)
            self.assertEqual(report.headers["content-type"], "text/plain; charset=utf-8")
            self.assertEqual(report.headers["x-content-type-options"], "nosniff")
            bootstrap = client.get("/api/client/bootstrap").json()
            self.assertNotIn("adapter_profile", json.dumps(bootstrap))
            self.assertNotIn("assessment", json.dumps(bootstrap))
            self.assertEqual(self.runtime.calls, [])

    def test_candidate_f_missing_or_tampered_artifacts_never_fall_back(self):
        identity = dict(adapter_profile="quality-f", adapter_model_sha256=live_evaluation.ADAPTER_SHA256)
        app = create_live_app(self.runtime, self.app.state.service.policy, self.audiences, serving_info=identity)
        with tempfile.TemporaryDirectory() as directory, TestClient(app, base_url="http://127.0.0.1") as client:
            missing = Path(directory) / "missing.json"
            tampered = Path(directory) / "tampered.json"
            tampered.write_text('{"base": {"successful_cases": 100}}', encoding="utf-8")
            for name in ("AGGREGATE", "VERIFICATION"):
                for path in (missing, tampered):
                    with self.subTest(artifact=name, path=path.name), patch.object(live_evaluation, name, path):
                        result = client.get("/api/admin/evaluation").json()
                        self.assertFalse(result["available"])
                        self.assertEqual(result["metrics"], [])
                        self.assertIsNone(result["n"])
                        self.assertIsNone(result["report_url"])
                        self.assertNotIn("dialogue_outcomes", result)
                        self.assertIn("недоступен", result["quality_note"])
                        self.assertNotIn(directory, result["quality_note"])
            with patch.object(live_evaluation, "CANDIDATE_REPORT", missing):
                self.assertIsNone(client.get("/api/admin/evaluation").json()["report_url"])
                self.assertEqual(client.get("/admin/candidate-f-report").status_code, 404)

    def test_candidate_f_adapter_identity_must_match(self):
        for adapter_hash in (None, "a" * 64):
            app = create_live_app(self.runtime, self.app.state.service.policy, self.audiences,
                                 serving_info=dict(adapter_profile="quality-f", adapter_model_sha256=adapter_hash))
            with TestClient(app, base_url="http://127.0.0.1") as client:
                with patch.object(live_evaluation, "_pinned_json") as read:
                    result = client.get("/api/admin/evaluation").json()
                    read.assert_not_called()
                self.assertFalse(result["available"])
                self.assertEqual(result["split"], "development")
                self.assertEqual(result["metrics"], [])
                self.assertIsNone(result["report_url"])
                self.assertIn("адаптера", result["quality_note"])

    def test_default_health_response_remains_unchanged(self):
        self.assertEqual(self.client.get("/health").json(), dict(status="ok", loaded=True, busy=False))
        self.assertNotIn("adapter_profile", self.client.get("/api/admin/state").json()["runtime"])

    def test_general_live_audience_has_no_diagnostic_checklist_or_specific_status(self):
        app = create_live_app(self.runtime, self.app.state.service.policy)
        try:
            for audience in app.state.service.audiences:
                self.assertEqual(set(audience["erp_context"]["facts"]),
                    {"erp.requester.audience", "erp.help.context_scope", "erp.help.support_contact"})
        finally:
            app.state.service.close()

    def test_new_conversation_and_comparison_do_not_inherit_other_history(self):
        self.send(message="Private previous issue 81123456")
        new_conv = self.client.post("/api/client/conversations", json={"audience": "customer"}).json()["id"]
        self.send(conv=new_conv, key="other-0001", message="Новый вопрос")
        self.assertNotIn("81123456", self.runtime.calls[-1][1]["customer_message"])
        job = self.client.post("/api/admin/comparisons", json={"message": "Один новый вопрос", "audience": "customer", "idempotency_key": "compare-new-0001"}).json()
        self.wait_job(job["id"])
        for _, effective in self.runtime.calls[-2:]:
            self.assertNotIn("81123456", effective["customer_message"])
        self.assertEqual(self.runtime.calls[-2][1], self.runtime.calls[-1][1])
        self.assertEqual(self.runtime.chat_calls[-2], self.runtime.chat_calls[-1])
        self.assertNotIn("81123456", repr(self.runtime.chat_calls[-3:]))

    def test_context_budget_rejection_never_calls_gpu_or_invents_usage(self):
        self.runtime.count_chat_prompt_tokens = lambda _: 6001
        response = self.send(message="Слишком объёмный для тестового лимита вопрос").json()
        record = self.app.state.service.requests[response["request_id"]]
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual(record["admin"]["usage_calls"], [])
        self.assertEqual(record["dialogue_context"]["status"], "budget_exceeded")
        self.assertIn("слишком длинное", response["client"]["response"])
        self.assertTrue(response["client"]["escalation"])
        call_event = next(e for e in record["admin"]["trace_events"] if e["node"] == "model_call")
        self.assertEqual(call_event["status"], "skipped")

    def test_persisted_dialogue_is_used_by_model_after_service_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            policy = self.app.state.service.policy
            app = create_live_app(self.runtime, policy, self.audiences, store_path=path)
            client = TestClient(app, base_url="http://127.0.0.1")
            client.get("/api/client/bootstrap")
            conv = client.post("/api/client/conversations", json={"audience": "customer"}).json()["id"]
            first = client.post(f"/api/client/conversations/{conv}/messages", json={
                "message": "Счета на оплату: ошибка без описания", "idempotency_key": "history-before-1"}).json()
            app.state.service.close()
            fresh = create_live_app(self.runtime, policy, self.audiences, store_path=path)
            fresh_client = TestClient(fresh, base_url="http://127.0.0.1")
            fresh_client.cookies.update(client.cookies)
            try:
                second = fresh_client.post(f"/api/client/conversations/{conv}/messages", json={
                    "message": "Все поля уже заполнены", "idempotency_key": "history-after-2"}).json()
                effective = repr(self.runtime.chat_calls[-1])
                self.assertIn(first["message"], effective)
                self.assertIn(first["client"]["response"], effective)
                record = fresh.state.service.requests[second["request_id"]]
                self.assertEqual(record["dialogue_context"]["included_request_ids"], [first["request_id"]])
            finally:
                fresh.state.service.close()

    def test_repeated_reply_clarifies_without_rewriting_model_or_extra_call(self):
        repeated = "Пожалуйста, уточните название раздела, действие перед ошибкой, время и текст ошибки без секретных данных."
        self.runtime.result_overrides = {"suggested_response": repeated}
        self.send(message="Бюджетирование, ошибка без текста.")
        second = self.send(key="repeat-0002", message="Счета на оплату, все поля заполнены.").json()
        record = self.app.state.service.requests[second["request_id"]]
        self.assertFalse(second["client"]["escalation"])
        self.assertEqual(second["client"]["status"], "server_clarification")
        self.assertEqual(record["admin"]["server_route"], "server_clarification")
        self.assertNotEqual(second["client"]["response"], repeated)
        self.assertEqual(record["admin"]["raw_model_result"]["suggested_response"], repeated)
        self.assertEqual(record["admin"]["usage_calls"][0]["total_tokens"], 18)
        self.assertEqual(len(self.runtime.calls), 2)
        event = next(e for e in record["admin"]["trace_events"] if e["node"] == "response_guard")
        self.assertEqual(event["error"], "repeated_previous_response")
        self.assertIsNone(second["client"]["handoff_id"])
        self.assertEqual(self.app.state.service.queue, {})
        self.assertEqual(second["client"]["analysis"]["recommended_action"], "provide_instructions")
        filtered = self.client.get("/api/admin/requests", params={"route": "server_clarification"})
        self.assertEqual(filtered.status_code, 200, filtered.text)
        self.assertEqual([item["request_id"] for item in filtered.json()["items"]], [second["request_id"]])
        self.assertEqual(filtered.json()["items"][0]["server_route"], "server_clarification")

    def test_uncaught_noise_with_repeated_reply_never_creates_ticket(self):
        self.runtime.result_overrides = {"suggested_response": "Для общего вопроса поддержка доступна через этот чат. Конкретный счёт, оплата или SIM пока не проверены."}
        self.send(message="Как настроить интернет?")
        # Deliberately simulate any future input-detector miss: routing must remain safe.
        with patch("support.live_input_guard.inspect_message", return_value=None):
            second = self.send(key="missed-noise-2", message="asdg asdg").json()
        self.assertEqual(second["client"]["status"], "server_clarification")
        self.assertFalse(second["client"]["escalation"])
        self.assertEqual(self.app.state.service.queue, {})
        raw = self.app.state.service.requests[second["request_id"]]["admin"]["raw_model_result"]
        self.assertEqual(raw["recommended_action"], "provide_instructions")
        self.assertFalse(raw["human_escalation"])

    def test_repeat_clarification_survives_restart_and_next_turn_history(self):
        self.runtime.result_overrides = {"suggested_response": "Проверьте название раздела и доступные настройки. Уточните, какой именно параметр вы хотите изменить."}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repeat-history.json"
            app = create_live_app(self.runtime, self.app.state.service.policy, self.audiences, store_path=path)
            client = TestClient(app, base_url="http://127.0.0.1")
            client.get("/api/client/bootstrap")
            conv = client.post("/api/client/conversations", json={"audience": "customer"}).json()["id"]
            route = f"/api/client/conversations/{conv}/messages"
            try:
                client.post(route, json={"message": "Как настроить интернет?", "idempotency_key": "repeat-before-1"})
                second = client.post(route, json={"message": "Параметры уже проверил", "idempotency_key": "repeat-before-2"}).json()
                self.assertEqual(second["client"]["status"], "server_clarification")
            finally:
                app.state.service.close()
            fresh = create_live_app(self.runtime, self.app.state.service.policy, self.audiences, store_path=path)
            fresh_client = TestClient(fresh, base_url="http://127.0.0.1")
            fresh_client.cookies.update(client.cookies)
            try:
                # Both older answered records and new live clarification records restore unchanged.
                self.assertEqual(fresh.state.service.requests[second["request_id"]]["client"], second["client"])
                self.runtime.result_overrides = {"suggested_response": "Уточните модель устройства."}
                third = fresh_client.post(route, json={"message": "Настраиваю роутер", "idempotency_key": "repeat-after-3"})
                self.assertEqual(third.status_code, 200)
                self.assertIn(second["client"]["response"], repr(self.runtime.chat_calls[-1]))
                self.assertEqual(fresh.state.service.queue, {})
            finally:
                fresh.state.service.close()

    def test_explicit_request_to_repeat_instruction_does_not_trigger_loop_guard(self):
        self.runtime.result_overrides = {"suggested_response": "Подробная разрешённая инструкция по настройке: откройте раздел настроек и проверьте выбранное значение."}
        self.send(message="Как настроить?")
        second = self.send(key="repeat-0002", message="Повторите инструкцию, пожалуйста.").json()
        self.assertFalse(second["client"]["escalation"])
        self.assertEqual(second["client"]["response"], self.runtime.result_overrides["suggested_response"])

    def test_client_privacy_and_cookie_isolation(self):
        response = self.send()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"request_id", "created_at", "message", "client"})
        for private in ("raw_model_result", "erp_context", "usage_calls", "trace_events", "policy_version"):
            self.assertNotIn(private, response.text)
        stranger = TestClient(self.app, base_url="http://127.0.0.1")
        self.assertEqual(stranger.get(f"/api/client/conversations/{self.conv}").status_code, 404)
        self.assertEqual(stranger.post(f"/api/client/conversations/{self.conv}/messages", json={"message": "Hi", "idempotency_key": "stranger-1"}).status_code, 404)
        bootstrap = TestClient(self.app, base_url="http://127.0.0.1").get("/api/client/bootstrap")
        self.assertIn("HttpOnly", bootstrap.headers["set-cookie"])
        self.assertIn("SameSite=strict", bootstrap.headers["set-cookie"])

    def test_extra_org_mode_and_length_rejected(self):
        for field in ("org_id", "model_mode", "erp_context"):
            result = self.client.post(f"/api/client/conversations/{self.conv}/messages", json={"message": "hi", "idempotency_key": "abcdefgh", field: "evil"})
            self.assertEqual(result.status_code, 422)
        self.assertEqual(self.send(message="x" * 4001).status_code, 422)
        self.assertEqual(self.send(message="   ").status_code, 422)
        self.app.state.service.conversations[self.conv]["org_id"] = "other"
        self.assertEqual(self.send().status_code, 403)
        self.assertEqual(self.runtime.calls, [])

    def test_idempotency_and_usage_do_not_double_count(self):
        first = self.send()
        self.assertEqual(first.json(), self.send().json())
        self.assertEqual(self.send(message="Other").status_code, 409)
        one = self.client.get("/api/admin/state").json()
        two = self.client.get("/api/admin/state").json()
        self.assertEqual(one["usage"], two["usage"])
        self.assertEqual(one["usage"]["support"]["total_tokens"], 18)
        self.assertEqual(one["usage"]["support"]["api_cost"], 0)
        self.assertEqual(len(self.runtime.calls), 1)

    def test_compare_serialized_same_facts_and_replay(self):
        body = dict(message="Как настроить?", audience="customer", idempotency_key="compare-001")
        first = self.client.post("/api/admin/comparisons", json=body)
        self.assertEqual(first.status_code, 202)
        completed = self.wait_job(first.json()["id"])
        self.assertEqual(completed, self.client.post("/api/admin/comparisons", json=body).json())
        self.assertEqual([x[0] for x in self.runtime.calls], ["base", "fine_tuned"])
        self.assertEqual(self.runtime.calls[0][1], self.runtime.calls[1][1])
        snapshot = self.client.get("/api/admin/state").json()
        self.assertEqual(snapshot["counts"]["requests"], 0)
        self.assertEqual(snapshot["queue"], [])
        self.assertEqual(snapshot["usage"]["comparison"]["total_tokens"], 36)
        self.assertEqual(snapshot["runtime"]["client_mode"], "fine_tuned")

    def test_erp_fallback_queue_and_review_never_sends(self):
        conv = self.client.post("/api/client/conversations", json={"audience": "employee"}).json()["id"]
        response = self.send(conv=conv).json()
        self.assertEqual(response["client"]["status"], "fallback")
        self.assertEqual(self.runtime.calls, [])
        snapshot = self.client.get("/api/admin/state").json()
        ticket = snapshot["queue"][0]
        self.assertEqual(ticket["id"], response["client"]["handoff_id"])
        self.assertEqual(snapshot["usage"]["support"]["calls"], 0)
        result = self.client.post(f'/api/admin/queue/{ticket["id"]}/review', json={"draft": "Локальная редакция", "status": "approved"})
        self.assertEqual(result.status_code, 200)
        history = self.client.get(f"/api/client/conversations/{conv}").json()["messages"]
        self.assertEqual(history[0], response)
        self.assertEqual(self.send(conv=conv).json(), response)
        self.assertEqual(len(self.client.get("/api/admin/state").json()["queue"]), 1)

    def test_failed_generation_null_usage_and_safe_client(self):
        self.runtime.fail = True
        response = self.send()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["client"]["status"], "fallback")
        self.assertNotIn("secret", response.text)
        snapshot = self.client.get("/api/admin/state").json()
        self.assertEqual(snapshot["usage"]["support"]["calls"], 1)
        self.assertIsNone(snapshot["usage"]["support"]["total_tokens"])
        self.assertIsNone(snapshot["usage"]["support"]["api_cost"])
        self.assertFalse(snapshot["usage"]["support"]["complete"])

    def test_host_origin_and_content_type_boundary(self):
        path = "/api/client/conversations"
        body = {"audience": "customer"}
        self.assertEqual(self.client.post(path, json=body, headers={"origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.client.post(path, json=body, headers={"origin": "http://127.0.0.1"}).status_code, 200)
        self.assertEqual(self.client.post(path, json=body, headers={"host": "evil.example"}).status_code, 400)
        self.assertEqual(self.client.post(path, json=body, headers={"sec-fetch-site": "cross-site"}).status_code, 403)
        self.assertEqual(self.client.post(path, content='{"audience":"customer"}', headers={"content-type": "text/plain"}).status_code, 415)

    def test_caps_explicit_no_deletion(self):
        service = self.app.state.service
        service.max_requests = 1
        self.assertEqual(self.send().status_code, 200)
        self.assertEqual(self.send(key="message-0002").status_code, 429)
        self.assertEqual(self.send().status_code, 200)
        self.assertEqual(len(service.requests), 1)

    def test_concurrent_duplicate_and_responsive_snapshot(self):
        self.runtime.block = True
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.send)
            self.assertTrue(self.runtime.entered.wait(3))
            second = pool.submit(self.send)
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertTrue(self.client.get("/api/admin/state").json()["runtime"]["busy"])
            self.runtime.release.set()
            self.assertEqual(first.result().json(), second.result().json())
        self.assertEqual(len(self.runtime.calls), 1)

    @unittest.skip("Historical validation fixture is private; no held-out data in public CI")
    def test_saved_evaluation_uses_actual_validation_metrics(self):
        result = self.client.get("/api/admin/evaluation").json()
        self.assertTrue(result["available"])
        self.assertEqual(result["n"], 150)
        category = next(m for m in result["metrics"] if m["label"] == "category")
        self.assertEqual((category["base"], category["fine_tuned"]), (89, 109))
        report = self.client.get(result["report_url"])
        self.assertEqual(report.status_code, 200)
        self.assertIn("sha256-", report.headers["content-security-policy"])

    def test_business_rules_force_handoff_preserving_raw_and_usage(self):
        cases = [
            ({"erp.payment.status": "paid", "erp.sim.pending_activation_count": 4}, "payment_confirmed_but_access_not_active"),
            ({"erp.access.can_start_connection": False}, "access_change_requires_operator"),
            ({"erp.integration.operator_review_required": True}, "integration_requires_operator"),
            ({"erp.incident.confirmed_mass_outage": True}, "service_outage"),
        ]
        for index, (facts, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                service = self.app.state.service
                service.by_audience["customer"]["erp_context"]["facts"] = {"erp.help": "Настройки", **facts}
                response = self.send(key=f"business-{index:03d}").json()
                self.assertTrue(response["client"]["escalation"])
                self.assertEqual(response["client"]["status"], "human_escalation")
                self.assertEqual(response["client"]["analysis"], dict(category="Settings", priority="Low",
                    sentiment="Neutral", recommended_action="provide_instructions"))
                record = service.requests[response["request_id"]]
                admin = record["admin"]
                self.assertFalse(admin["raw_model_result"]["human_escalation"])
                self.assertEqual(admin["raw_model_result"]["suggested_response"], "Проверьте настройки.")
                self.assertIn("Проверьте настройки.", admin["raw_response"])
                self.assertEqual(admin["server_route"], "human_escalation")
                self.assertEqual(admin["usage_calls"][0]["total_tokens"], 18)
                self.assertEqual(len(admin["usage_calls"]), 1)
                trace = admin["trace_events"][-1]
                self.assertEqual((trace["node"], trace["error"]), ("business_rules", reason))
                self.assertTrue(set(trace["evidence_ids"]).issubset(facts))
                ticket = service.queue[response["client"]["handoff_id"]]
                self.assertEqual(ticket["reason"], reason)
                self.assertEqual(ticket["priority"], "Critical" if reason == "service_outage" else "Low")

    def test_business_rules_do_not_guess_missing_or_wrong_typed_facts(self):
        cases = [{}, {"erp.payment.status": "paid"},
                 {"erp.sim.pending_activation_count": 4},
                 {"erp.payment.status": "unpaid", "erp.sim.pending_activation_count": 4},
                 *[{"erp.payment.status": "paid", "erp.sim.pending_activation_count": value} for value in (0, -1, True, "4", 4.0)],
                 {"erp.access.can_start_connection": "false"}, {"erp.access.can_start_connection": 0},
                 {"erp.integration.operator_review_required": "true"}, {"erp.incident.confirmed_mass_outage": 1}]
        for index, facts in enumerate(cases):
            with self.subTest(facts=facts):
                service = self.app.state.service
                service.by_audience["customer"]["erp_context"]["facts"] = {"erp.help": "Настройки", **facts}
                response = self.send(key=f"no-guess-{index:03d}").json()
                self.assertFalse(response["client"]["escalation"])
                self.assertEqual(service.queue, {})
                admin = service.requests[response["request_id"]]["admin"]
                self.assertFalse(any(e["node"] == "business_rules" for e in admin["trace_events"]))

    def test_business_rules_override_existing_escalation_reason_and_queue_priority(self):
        self.runtime.result_overrides = dict(human_escalation=True, escalation_reason="payment_dispute")
        service = self.app.state.service
        service.by_audience["customer"]["erp_context"]["facts"]["erp.incident.confirmed_mass_outage"] = True
        response = self.send().json()
        admin = service.requests[response["request_id"]]["admin"]
        self.assertEqual(admin["raw_model_result"]["escalation_reason"], "payment_dispute")
        self.assertEqual(admin["raw_model_result"]["priority"], "Low")
        self.assertEqual(admin["trace_events"][-1]["node"], "business_rules")
        ticket = service.queue[response["client"]["handoff_id"]]
        self.assertEqual((ticket["reason"], ticket["priority"]), ("service_outage", "Critical"))

    def test_business_rules_comparison_has_no_queue(self):
        service = self.app.state.service
        service.by_audience["customer"]["erp_context"]["facts"].update(
            {"erp.payment.status": "paid", "erp.sim.pending_activation_count": 4})
        response = self.client.post("/api/admin/comparisons", json=dict(message="Четыре SIM не работают",
            audience="customer", idempotency_key="forced-compare")).json()
        response = self.wait_job(response["id"])
        for mode in ("base", "fine_tuned"):
            self.assertTrue(response[mode]["client"]["escalation"])
            self.assertFalse(response[mode]["admin"]["raw_model_result"]["human_escalation"])
        self.assertEqual(service.queue, {})
        self.assertEqual(service.snapshot()["usage"]["comparison"]["total_tokens"], 36)

    def test_business_rules_preserve_runtime_fallback(self):
        service = self.app.state.service
        service.by_audience["customer"]["erp_context"]["facts"]["erp.incident.confirmed_mass_outage"] = True
        self.runtime.fail = True
        response = self.send().json()
        self.assertEqual(response["client"]["status"], "fallback")
        self.assertIsNone(response["client"]["analysis"])
        admin = service.requests[response["request_id"]]["admin"]
        self.assertFalse(any(e["node"] == "business_rules" for e in admin["trace_events"]))
        self.assertEqual(service.queue[response["client"]["handoff_id"]]["reason"], "model_runtime_error")

    def test_public_audiences_have_neutral_context_independent_of_message(self):
        self.runtime.result_overrides = {"evidence_ids": []}
        policy = self.app.state.service.policy
        app = create_live_app(self.runtime, policy)
        self.addCleanup(app.state.service.close)
        client = TestClient(app, base_url="http://127.0.0.1")
        bootstrap = client.get("/api/client/bootstrap").json()
        self.assertEqual({item["id"] for item in bootstrap["audiences"]}, {"customer", "employee"})
        self.assertNotIn("scenarios", bootstrap)
        for audience in ("customer", "employee"):
            conv = client.post("/api/client/conversations", json={"audience": audience}).json()
            for index, message in enumerate(("SIM не работает, платёж вчера", "ERP не работает, HTTP 401, нет прав")):
                result = client.post(f'/api/client/conversations/{conv["id"]}/messages', json=dict(
                    message=message, idempotency_key=f"neutral-{audience}-{index}"))
                self.assertEqual(result.status_code, 200)
                facts = self.runtime.calls[-1][1]["erp_context"]["facts"]
                self.assertEqual(facts["erp.requester.audience"], audience)
                self.assertTrue(all(key == "erp.requester.audience" or key.startswith("erp.help.") for key in facts))
            self.assertEqual(self.runtime.calls[-1][1]["erp_context"], self.runtime.calls[-2][1]["erp_context"])
        self.assertEqual(client.post("/api/client/conversations", json={"scenario_id": "payment"}).status_code, 422)

    def test_search_pagination_1000_records_and_compact_state(self):
        first = self.send().json()
        service = self.app.state.service
        source = copy.deepcopy(service.requests[first["request_id"]])
        service.requests.clear()
        for index in range(1000):
            record = copy.deepcopy(source)
            record.update(request_id=f"request-{index:04d}", audience="employee" if index % 2 else "customer",
                          message=f"Счёт {index:04d} вопрос")
            service.requests[record["request_id"]] = record
        first_page = self.client.get("/api/admin/requests").json()
        self.assertEqual((first_page["total"], first_page["pages"], len(first_page["items"])), (1000, 50, 20))
        self.assertEqual(first_page["items"][0]["request_id"], "request-0999")
        second_page = self.client.get("/api/admin/requests?page=2").json()
        self.assertFalse({r["request_id"] for r in first_page["items"]} & {r["request_id"] for r in second_page["items"]})
        filtered = self.client.get("/api/admin/requests", params=dict(q="СЧЁТ 00", audience="employee", category="Settings", route="auto_answer")).json()
        self.assertEqual(filtered["total"], 50)
        self.assertTrue(all(r["audience"] == "employee" for r in filtered["items"]))
        exact = self.client.get("/api/admin/requests?q=request-0999").json()
        self.assertEqual(exact["total"], 1)
        self.assertNotIn("raw_response", first_page["items"][0])
        detail = self.client.get("/api/admin/requests/request-0999").json()
        self.assertIn("raw_response", detail["admin"])
        self.assertEqual(self.client.get("/api/admin/requests?page=0").status_code, 422)
        self.assertEqual(self.client.get("/api/admin/requests?page_size=101").status_code, 422)
        state = self.client.get("/api/admin/state").json()
        self.assertNotIn("requests", state)
        self.assertNotIn("comparisons", state)
        self.assertEqual(state["counts"]["requests"], 1000)
        self.assertEqual(len(self.client.get("/api/admin/export").json()["requests"]), 1000)
        stranger = TestClient(self.app, base_url="http://127.0.0.1")
        self.assertEqual(stranger.get("/api/client/bootstrap").json()["conversations"], [])
        self.assertEqual(stranger.get(f"/api/client/conversations/{self.conv}").status_code, 404)

    def test_progressive_comparison_publishes_base_before_ft_and_bounds_queue(self):
        service = self.app.state.service
        original = self.runtime.for_mode
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def mode_runner(mode):
            runner = original(mode)
            if mode == "fine_tuned":
                generate = runner.generate
                def blocked(*args, **kwargs):
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError("Test timeout")
                    return generate(*args, **kwargs)
                runner.generate = blocked
            return runner
        self.runtime.for_mode = mode_runner
        body = dict(message="Вопрос", audience="customer", idempotency_key="progress-001")
        response = self.client.post("/api/admin/comparisons", json=body)
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["id"]
        self.assertTrue(started.wait(3))
        partial = self.client.get(f"/api/admin/comparisons/{job_id}").json()
        self.assertEqual((partial["status"], partial["stage"]), ("running", "fine_tuned"))
        self.assertIsNotNone(partial["base"])
        self.assertIsNone(partial["fine_tuned"])
        replay = self.client.post("/api/admin/comparisons", json=body).json()
        self.assertEqual(replay["id"], job_id)
        self.assertEqual(self.client.post("/api/admin/comparisons", json={**body, "message": "Other"}).status_code, 409)
        service.max_pending_jobs = 1
        self.assertEqual(self.client.post("/api/admin/comparisons", json={**body, "idempotency_key": "progress-002"}).status_code, 429)
        listing = self.client.get("/api/admin/comparisons?q=вопрос&audience=customer").json()
        self.assertEqual(listing["items"][0]["stage"], "fine_tuned")
        self.assertNotIn("base", listing["items"][0])
        release.set()
        finished = self.wait_job(job_id)
        self.assertEqual(finished["status"], "completed")
        self.assertEqual(len(self.runtime.calls), 2)
        self.assertEqual(finished["base"]["erp_context"], finished["fine_tuned"]["erp_context"])

    def test_failed_job_preserves_partial_record_and_unknown_usage(self):
        self.runtime.fail = True
        response = self.client.post("/api/admin/comparisons", json=dict(message="Вопрос", audience="customer", idempotency_key="failed-job-1"))
        job = self.wait_job(response.json()["id"])
        self.assertEqual(job["status"], "failed")
        self.assertIsNotNone(job["base"])
        self.assertIsNone(job["fine_tuned"])
        self.assertNotIn("secret", json.dumps(job))
        usage = self.client.get("/api/admin/state").json()["usage"]["comparison"]
        self.assertEqual(usage["calls"], 1)
        self.assertFalse(usage["complete"])
        self.assertIsNone(usage["total_tokens"])

    def test_admin_record_captures_serving_profile_without_client_disclosure(self):
        response = self.send()
        record = self.client.get("/api/admin/requests/" + response.json()["request_id"]).json()
        self.assertEqual(record["inference_profile"], "bf16")
        self.assertEqual(record["serving_scope"], "serving_only_experiment")
        self.assertNotIn("inference_profile", response.text)
        self.assertNotIn("serving_scope", response.text)
        job_id = self.client.post("/api/admin/comparisons", json=dict(message="Настройки", audience="customer",
            idempotency_key="profile-comparison")).json()["id"]
        job = self.wait_job(job_id)
        for mode in ("base", "fine_tuned"):
            self.assertEqual(job[mode]["inference_profile"], "bf16")
            self.assertEqual(job[mode]["serving_scope"], "serving_only_experiment")

    def test_generic_job_diagnostics_survive_reload_without_exception_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            app = create_live_app(self.runtime, self.app.state.service.policy, self.audiences, store_path=path)
            client = TestClient(app, base_url="http://127.0.0.1")
            service = app.state.service
            def broken_run(*args, **kwargs):
                raise ValueError("TOP_SECRET_TOKEN and private exception details")
            service.run = broken_run
            job_id = client.post("/api/admin/comparisons", json=dict(message="Настройки", audience="customer",
                idempotency_key="diagnostic-failure")).json()["id"]
            service.close()
            job = client.get(f"/api/admin/comparisons/{job_id}").json()
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["failure_type"], "ValueError")
            self.assertRegex(job["diagnostic_id"], r"^[0-9a-f]{32}$")
            self.assertNotIn("TOP_SECRET", json.dumps(job))
            self.assertNotIn("TOP_SECRET", path.read_text(encoding="utf-8"))
            bootstrap = client.get("/api/client/bootstrap").text
            self.assertNotIn("diagnostic_id", bootstrap)
            self.assertNotIn("failure_type", bootstrap)
            fresh = create_live_app(self.runtime, self.app.state.service.policy, self.audiences, store_path=path)
            try:
                restored = fresh.state.service.comparison(job_id)
                self.assertEqual(restored["failure_type"], job["failure_type"])
                self.assertEqual(restored["diagnostic_id"], job["diagnostic_id"])
                self.assertNotIn("TOP_SECRET", json.dumps(restored))
            finally:
                fresh.state.service.close()

    def test_shutdown_waits_owned_job_and_rejects_new_work(self):
        self.runtime.block = True
        body = dict(message="Вопрос", audience="customer", idempotency_key="shutdown-job1")
        self.client.post("/api/admin/comparisons", json=body)
        self.assertTrue(self.runtime.entered.wait(3))
        closed = threading.Event()
        def close():
            self.app.state.service.close()
            closed.set()
        closer = threading.Thread(target=close)
        closer.start()
        try:
            self.assertFalse(closed.wait(0.05))
            response = self.client.post("/api/admin/comparisons", json={**body, "idempotency_key": "shutdown-job2"})
            self.assertEqual(response.status_code, 503)
            self.runtime.release.set()
            self.assertTrue(closed.wait(3))
        finally:
            self.runtime.release.set()
            closer.join(5)
        self.assertEqual(len(self.runtime.calls), 2)

    def test_ft_failure_keeps_completed_base_and_failed_ft_usage(self):
        original = self.runtime.for_mode
        def runner(mode):
            if mode == "fine_tuned":
                self.runtime.fail = True
            return original(mode)
        self.runtime.for_mode = runner
        job_id = self.client.post("/api/admin/comparisons", json=dict(message="Настройки", audience="customer",
            idempotency_key="partial-failure")).json()["id"]
        job = self.wait_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["base"]["client"]["status"], "answered")
        self.assertEqual(job["fine_tuned"]["client"]["status"], "fallback")
        self.assertEqual(job["base"]["admin"]["usage_calls"][0]["total_tokens"], 18)
        usage = self.client.get("/api/admin/state").json()["usage"]["comparison"]
        self.assertEqual(usage["calls"], 2)
        self.assertIsNone(usage["total_tokens"])
        self.assertFalse(usage["complete"])

    def test_persistence_restores_ownership_records_queue_and_idempotency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.json"
            policy = self.app.state.service.policy
            app = create_live_app(self.runtime, policy, self.audiences, store_path=path)
            client = TestClient(app, base_url="http://127.0.0.1")
            client.get("/api/client/bootstrap")
            conv = client.post("/api/client/conversations", json={"audience": "employee"}).json()["id"]
            payload = dict(message="ERP не работает", idempotency_key="persist-msg1")
            response = client.post(f"/api/client/conversations/{conv}/messages", json=payload).json()
            ticket = response["client"]["handoff_id"]
            client.post(f"/api/admin/queue/{ticket}/review", json=dict(draft="Сохранённый черновик", status="approved"))
            compare_payload = dict(message="Настройки", audience="customer", idempotency_key="persist-compare1")
            job_id = client.post("/api/admin/comparisons", json=compare_payload).json()["id"]
            app.state.service.close()
            calls_before = len(self.runtime.calls)
            fresh = create_live_app(self.runtime, policy, self.audiences, store_path=path)
            fresh_client = TestClient(fresh, base_url="http://127.0.0.1")
            fresh_client.cookies.update(client.cookies)
            try:
                self.assertEqual(fresh_client.get(f"/api/client/conversations/{conv}").json()["messages"], [response])
                self.assertEqual(fresh_client.post(f"/api/client/conversations/{conv}/messages", json=payload).json(), response)
                replay = fresh_client.post("/api/admin/comparisons", json=compare_payload).json()
                self.assertEqual((replay["id"], replay["status"]), (job_id, "completed"))
                self.assertEqual(len(self.runtime.calls), calls_before)
                state = fresh_client.get("/api/admin/state").json()
                self.assertEqual(state["queue"][0]["draft"], "Сохранённый черновик")
                stranger = TestClient(fresh, base_url="http://127.0.0.1")
                self.assertEqual(stranger.get(f"/api/client/conversations/{conv}").status_code, 404)
            finally:
                fresh.state.service.close()

    def test_restart_marks_pending_jobs_failed_and_import_preserves_archive(self):
        source = self.send().json()
        archive = self.app.state.service.snapshot(full=True)
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "archive.json"
            store_path = Path(directory) / "store.json"
            archive_path.write_text(json.dumps(archive), encoding="utf-8")
            app = create_live_app(self.runtime, self.app.state.service.policy, self.audiences,
                                  store_path=store_path, archive_path=archive_path)
            service = app.state.service
            self.assertEqual(service.requests[source["request_id"]], archive["requests"][0])
            self.assertEqual(service.conversations, {})
            self.assertEqual(service.owners, set())
            service.comparisons["interrupted"] = dict(id="interrupted", created_at="2026-09-07", message="Вопрос",
                audience="customer", status="running", stage="fine_tuned", base=archive["requests"][0], fine_tuned=None)
            service.comparisons["queued"] = dict(id="queued", created_at="2026-09-07", message="Вопрос",
                audience="customer", status="queued", stage="queued", base=None, fine_tuned=None)
            service.close()
            calls_before = len(self.runtime.calls)
            fresh = create_live_app(self.runtime, self.app.state.service.policy, self.audiences,
                                    store_path=store_path, archive_path=archive_path)
            try:
                self.assertEqual(len(fresh.state.service.requests), 1)
                for job_id in ("interrupted", "queued"):
                    self.assertEqual(fresh.state.service.comparison(job_id)["status"], "failed")
                restored_base = fresh.state.service.comparison("interrupted")["base"]
                self.assertEqual(restored_base.pop("model_history"), [])
                self.assertFalse(restored_base.pop("simulated_handoff"))
                self.assertEqual(restored_base, archive["requests"][0])
                self.assertFalse(fresh.state.service.snapshot()["usage"]["comparison"]["complete"])
                self.assertEqual(len(self.runtime.calls), calls_before)
            finally:
                fresh.state.service.close()


if __name__ == "__main__":
    unittest.main()
