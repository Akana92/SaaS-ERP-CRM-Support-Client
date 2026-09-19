"""Operator queue reads use stored public conversation, never inference."""
import copy
import tempfile
import unittest
from pathlib import Path

from test_live_app import FakeRuntime, create_live_app, TestClient, PolicyDocument, PolicyRule


class LiveQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "live.json"
        self.runtime = FakeRuntime()
        self.policy = PolicyDocument(version="demo", rules=[PolicyRule(id="policy.one", text="Только факты.")])
        self.app = create_live_app(self.runtime, self.policy, store_path=self.path)
        self.service = self.app.state.service
        self.client = TestClient(self.app, base_url="http://127.0.0.1")

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def seed(self, index, *, session="own", priority="High", status="draft"):
        rid = f"request-{index:04}"
        record = dict(request_id=rid, created_at=f"2026-09-17T00:{index // 60:02}:{index % 60:02}Z",
                      audience="employee", message=f"Доступ к ERP {index}",
                      admin=dict(usage_calls=[]),
                      client=dict(response="Передано специалисту", status="human_escalation", escalation=True,
                                  analysis=dict(category="AccountAccess", priority=priority, sentiment="Neutral",
                                                recommended_action="review_access")))
        self.service.requests[rid] = record
        ticket = dict(id=f"ticket-{rid}", request_id=rid, message=record["message"], reason="access",
                      priority=priority, status=status, draft="Draft only")
        self.service.queue[ticket["id"]] = ticket
        conv = self.service.conversations.setdefault(session, dict(id=session, owner=session, org_id="demo",
                                                    audience="employee", title="Access", messages=[]))
        conv["messages"].append(self.service.public_message(record))
        return ticket, record

    def test_thousand_rows_search_filters_paging_and_compact_output(self):
        for i in range(1001):
            self.seed(i, priority="Low" if i % 2 else "High", status="approved" if i % 3 else "draft")
        result = self.client.get("/api/admin/queue").json()
        self.assertEqual((result["total"], result["pages"], len(result["items"])), (1001, 51, 20))
        self.assertEqual(result["items"][0]["request_id"], "request-1000")
        self.assertEqual(set(result["items"][0]), {"id", "request_id", "message", "reason", "priority", "status", "created_at", "audience"})
        self.assertEqual(self.client.get("/api/admin/queue?page=51").json()["items"][0]["request_id"], "request-0000")
        self.assertEqual(self.client.get("/api/admin/queue?page=52").json()["items"], [])
        for q in ("REQUEST-1000", "TICKET-REQUEST-1000", "доступ К erp 1000"):
            self.assertEqual(self.client.get("/api/admin/queue", params={"q": q}).json()["total"], 1)
        result = self.client.get("/api/admin/queue?status=draft&priority=Low&page_size=100").json()
        self.assertTrue(all(r["status"] == "draft" and r["priority"] == "Low" for r in result["items"]))
        self.assertEqual(result["total"], 167)
        compact = self.client.get("/api/admin/state?include_queue=false").json()
        self.assertEqual(compact["counts"]["queue"], 1001)
        self.assertEqual(compact["queue"], [])
        self.assertEqual(len(self.client.get("/api/admin/state").json()["queue"]), 1001)
        self.assertEqual(self.runtime.calls, [])

    def test_query_validation_and_missing_ticket(self):
        for query in ("page=0", "page_size=0", "page_size=101", "status=sent", "priority=urgent", "page=x", "q=" + "a" * 4001):
            with self.subTest(query=query[:30]):
                self.assertEqual(self.client.get("/api/admin/queue?" + query).status_code, 422)
        self.assertEqual(self.client.get("/api/admin/queue/missing").status_code, 404)

    def test_compact_state_omits_queue_without_changing_counts_or_usage(self):
        self.seed(1)
        full = self.client.get("/api/admin/state").json()
        compact = self.client.get("/api/admin/state?include_queue=false").json()
        self.assertEqual(compact["queue"], [])
        self.assertEqual(compact["counts"], full["counts"])
        self.assertEqual(compact["usage"], full["usage"])

    def test_detail_history_is_full_owned_prefix_not_model_context_or_later_turns(self):
        first, _ = self.seed(1)
        ticket, record = self.seed(2)
        self.seed(3)
        self.seed(4, session="foreign")
        record["dialogue_context"] = {"included_request_ids": ["request-0004"]}
        detail = self.client.get("/api/admin/queue/" + ticket["id"]).json()
        self.assertTrue(detail["history_available"])
        self.assertEqual([r["request_id"] for r in detail["conversation"]], [first["request_id"], ticket["request_id"]])
        self.assertEqual(detail["analysis"], record["client"]["analysis"])
        self.assertEqual(detail["audience"], "employee")
        self.assertEqual(detail["created_at"], record["created_at"])
        self.assertEqual(set(detail["conversation"][0]), {"request_id", "created_at", "message", "client"})
        detail["conversation"][0]["message"] = "changed"
        self.assertNotEqual(self.service.conversations["own"]["messages"][0]["message"], "changed")

    def test_orphan_or_missing_handoff_has_no_reconstructed_history(self):
        ticket, record = self.seed(1)
        self.service.conversations["own"]["messages"] = []
        detail = self.client.get("/api/admin/queue/" + ticket["id"]).json()
        self.assertFalse(detail["history_available"])
        self.assertEqual(detail["conversation"], [])
        del self.service.requests[record["request_id"]]
        detail = self.client.get("/api/admin/queue/" + ticket["id"]).json()
        self.assertEqual([detail[k] for k in ("created_at", "audience", "analysis")], [None, None, None])
        self.assertFalse(detail["history_available"])

    def test_ambiguous_conversation_membership_fails_closed(self):
        ticket, record = self.seed(1)
        self.seed(2, session="foreign")
        self.service.conversations["foreign"]["messages"].append(self.service.public_message(record))
        detail = self.client.get("/api/admin/queue/" + ticket["id"]).json()
        self.assertFalse(detail["history_available"])
        self.assertEqual(detail["conversation"], [])

    def test_local_review_persists_without_changing_chat_or_running_model(self):
        ticket, _ = self.seed(1)
        before = copy.deepcopy(self.service.conversations)
        self.service._save()
        response = self.client.post("/api/admin/queue/" + ticket["id"] + "/review",
                                    json={"draft": "Уточните роль сотрудника", "status": "approved"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.conversations, before)
        self.assertEqual(self.runtime.calls, [])
        app = create_live_app(self.runtime, self.policy, store_path=self.path)
        try:
            client = TestClient(app, base_url="http://127.0.0.1")
            detail = client.get("/api/admin/queue/" + ticket["id"]).json()
            self.assertEqual(detail["status"], "approved")
            self.assertEqual(detail["draft"], "Уточните роль сотрудника")
            self.assertTrue(detail["history_available"])
            self.assertEqual(app.state.service.conversations, before)
            self.assertEqual(client.get("/api/admin/state").json()["queue"][0]["status"], "approved")
        finally:
            app.state.service.close()

    def test_real_submission_history_survives_reload_and_excludes_other_turns(self):
        self.runtime.result_overrides = dict(human_escalation=True, escalation_reason="access_change_requires_operator")
        conv = self.client.post("/api/client/conversations", json={"audience": "employee"}).json()["id"]
        def send(index):
            response = self.client.post(f"/api/client/conversations/{conv}/messages", json={
                "message": "Нужен доступ к импорту услуг в ERP", "idempotency_key": f"queue-message-{index}"})
            self.assertEqual(response.status_code, 200)
            return response.json()
        first, handoff, later = send(1), send(2), send(3)
        foreign = TestClient(self.app, base_url="http://127.0.0.1")
        foreign_conv = foreign.post("/api/client/conversations", json={"audience": "employee"}).json()["id"]
        foreign.post(f"/api/client/conversations/{foreign_conv}/messages", json={
            "message": "Не работает интернет", "idempotency_key": "foreign-message"})
        ticket_id = handoff["client"]["handoff_id"]
        result = self.client.get("/api/admin/queue/" + ticket_id).json()
        self.assertTrue(result["history_available"])
        self.assertEqual(result["conversation"], [first, handoff])
        app = create_live_app(self.runtime, self.policy, store_path=self.path)
        try:
            client = TestClient(app, base_url="http://127.0.0.1")
            self.assertEqual(client.get("/api/admin/queue/" + ticket_id).json()["conversation"], [first, handoff])
        finally:
            app.state.service.close()


if __name__ == "__main__":
    unittest.main()
