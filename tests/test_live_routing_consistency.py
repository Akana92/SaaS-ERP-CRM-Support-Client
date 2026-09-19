"""The live graph must act on required handoff wording without falsifying raw labels."""
from __future__ import annotations

import time
import unittest

from tests.test_live_app import FakeRuntime
from fastapi.testclient import TestClient
from support.contracts import PolicyDocument, PolicyRule
from support.live_app import create_live_app


class LiveRoutingConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.raw_text = (
            "Справка не подтверждает права этого сотрудника. "
            "Нужен специалист через этот чат; я не могу проверить его доступ."
        )
        self.runtime.result_overrides = dict(suggested_response=self.raw_text)
        audiences = [dict(id=key, label=key, prompt="Доступ", org_id="demo",
                         erp_context=dict(source_status="ok", facts={"erp.help": "Демо"}))
                     for key in ("customer", "employee")]
        self.app = create_live_app(self.runtime, PolicyDocument(version="test", rules=[
            PolicyRule(id="policy.one", text="Передать оператору при необходимости.")]), audiences)
        self.client = TestClient(self.app, base_url="http://127.0.0.1")
        self.client.get("/api/client/bootstrap")

    def tearDown(self):
        self.app.state.service.close()

    def test_required_specialist_creates_one_real_client_ticket_and_preserves_raw(self):
        conv = self.client.post("/api/client/conversations", json={"audience": "employee"}).json()
        body = {"message": "Как получить этот доступ кто его создает в справке его нету",
                "idempotency_key": "access-consistency-001"}
        url = f"/api/client/conversations/{conv['id']}/messages"
        response = self.client.post(url, json=body)
        self.assertEqual(response.status_code, 200)
        row = response.json()
        self.assertTrue(row["client"]["escalation"])
        self.assertEqual(row["client"]["status"], "human_escalation")
        ticket_id = row["client"]["handoff_id"]
        self.assertTrue(ticket_id)
        replay = self.client.post(url, json=body).json()
        self.assertEqual(replay, row)
        state = self.client.get("/api/admin/state").json()
        self.assertEqual(len(state["queue"]), 1)
        self.assertEqual(state["queue"][0]["id"], ticket_id)
        detail = self.client.get("/api/admin/requests/" + row["request_id"]).json()
        self.assertFalse(detail["admin"]["raw_model_result"]["human_escalation"])
        self.assertEqual(detail["admin"]["raw_model_result"]["suggested_response"], self.raw_text)
        self.assertEqual(detail["admin"]["server_route"], "human_escalation")
        self.assertEqual(detail["client"]["analysis"]["category"], "Settings")

    def test_comparison_handoff_is_simulation_and_does_not_create_client_ticket(self):
        response = self.client.post("/api/admin/comparisons", json={
            "message": "Кто выдаст доступ?", "audience": "employee",
            "idempotency_key": "access-comparison-001"})
        self.assertEqual(response.status_code, 202)
        comparison_id = response.json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.client.get("/api/admin/comparisons/" + comparison_id).json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "completed")
        for mode in ("base", "fine_tuned"):
            self.assertEqual(job[mode]["admin"]["server_route"], "human_escalation")
            self.assertTrue(job[mode]["simulated_handoff"])
            self.assertFalse(job[mode]["admin"]["raw_model_result"]["human_escalation"])
            self.assertFalse(job[mode]["client"].get("handoff_id"))
        self.assertEqual(self.client.get("/api/admin/state").json()["queue"], [])


if __name__ == "__main__":
    unittest.main()
