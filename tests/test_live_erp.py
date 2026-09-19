"""The live fictional connector never resolves an object outside its demo scope."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from support.live_erp import DemoERP


FIXTURE = Path(__file__).resolve().parents[1] / "data/erp/live-objects.json"


def context(audience="customer"):
    return dict(id=audience, org_id="org-demo-01", label="demo", erp_context={
        "source_status": "ok", "facts": {"erp.help.context_scope": "neutral"}})


class LiveERPTests(unittest.TestCase):
    def setUp(self):
        self.erp = DemoERP.from_path(FIXTURE)

    def test_known_objects_have_distinct_facts_and_do_not_mutate_context(self):
        original = context()
        paid, meta = self.erp.resolve(original, "Проверить INV-1001", [])
        unpaid, _ = self.erp.resolve(original, "Проверить INV-1002", [])
        self.assertEqual(paid["erp_context"]["facts"]["erp.payment.status"], "paid")
        self.assertEqual(unpaid["erp_context"]["facts"]["erp.payment.status"], "unpaid")
        self.assertEqual(meta["reference"], "INV-1001")
        self.assertNotEqual(paid["erp_context"]["facts"]["erp.help.context_scope"], "neutral")
        self.assertEqual(original, context())
        paid["erp_context"]["facts"]["erp.payment.status"] = "modified"
        again, _ = self.erp.resolve(original, "INV-1001", [])
        self.assertEqual(again["erp_context"]["facts"]["erp.payment.status"], "paid")

    def test_scope_denials_are_indistinguishable_from_unknown(self):
        unknown = self.erp.resolve(context(), "INV-9999", [])[0]["erp_context"]
        for ref, changes in [("BP-2001", {}), ("INV-1001", {"org_id": "other"}),
                             ("INV-1001", {"principal_id": "other"})]:
            ctx = {**context(), **changes}
            resolved, meta = self.erp.resolve(ctx, ref, [])
            self.assertEqual(resolved["erp_context"], unknown)
            self.assertEqual(meta["status"], "not_found")
            self.assertEqual(meta["resolution"], "not_found")

    def test_reference_boundaries_and_case(self):
        for text in ["xINV-1001", "INV-10010", "INV-1001x", "INV-1001_extra"]:
            resolved, _ = self.erp.resolve(context(), text, [])
            self.assertNotIn("erp.payment.status", resolved["erp_context"]["facts"])
        resolved, _ = self.erp.resolve(context(), "(inv-1001)?", [])
        self.assertEqual(resolved["erp_context"]["facts"]["erp.payment.status"], "paid")

    def test_history_uses_latest_user_reference_only(self):
        history = [{"message": "INV-1001", "client": {"response": "INV-1002"}}]
        resolved, meta = self.erp.resolve(context(), "А сколько активных?", history)
        self.assertEqual(meta["reference"], "INV-1001")
        self.assertEqual(resolved["erp_context"]["facts"]["erp.sim.active_count"], 2)
        history.append({"message": "INV-9999"})
        resolved, _ = self.erp.resolve(context(), "А теперь?", history)
        self.assertEqual(resolved["erp_context"], {"source_status": "not_found", "facts": {}})
        resolved, _ = self.erp.resolve(context(), "Что теперь?", [{"role": "assistant", "message": "INV-1001"}])
        self.assertEqual(resolved, context())

    def test_multiple_references_never_merge_or_fall_back(self):
        for history, text in [([], "INV-1001 и INV-1002"),
                              ([{"message": "INV-1001 и INV-1002"}], "Что со счетом?")]:
            resolved, meta = self.erp.resolve(context(), text, history)
            self.assertEqual(resolved["erp_context"], {"source_status": "not_found", "facts": {}})
            self.assertEqual(meta["resolution"], "ambiguous")
        resolved, _ = self.erp.resolve(context(), "INV-1001, INV-1001", [])
        self.assertEqual(resolved["erp_context"]["source_status"], "ok")

    def test_no_reference_preserves_neutral_server_fields(self):
        ctx = context()
        resolved, meta = self.erp.resolve(ctx, "Оплата прошла, активируй 6 SIM", [])
        self.assertEqual(resolved, ctx)
        self.assertIsNot(resolved, ctx)
        self.assertEqual(meta["resolution"], "general_help")

    def test_access_and_examples_are_audience_scoped(self):
        for ref, allowed in [("BP-2001", False), ("BP-2002", True)]:
            resolved, _ = self.erp.resolve(context("employee"), ref, [])
            self.assertIs(resolved["erp_context"]["facts"]["erp.access.can_start_connection"], allowed)
        self.assertEqual(len(self.erp.examples("customer")), 3)
        self.assertEqual(len(self.erp.examples("employee")), 2)
        self.assertEqual(self.erp.examples("unknown"), [])

    def test_examples_respect_explicit_server_context(self):
        self.assertEqual(self.erp.examples("customer", context=context()), self.erp.examples("customer"))
        for changes in [{"org_id": "foreign"}, {"principal_id": "foreign"}, {"id": "employee"}]:
            self.assertEqual(self.erp.examples("customer", context={**context(), **changes}), [])
        self.assertEqual(self.erp.examples("customer", context={}), [])

    def test_bad_fixtures_fail_closed(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        bad = []
        duplicate = copy.deepcopy(payload)
        duplicate["objects"].append(copy.deepcopy(duplicate["objects"][0]))
        bad.append(duplicate)
        for field, value in [("source_status", "stale"), ("facts", {"secret": "x"}),
                             ("facts", {"erp.bad": ["x"]}), ("principal_id", ""),
                             ("reference", "INV-1001x"), ("audience", "anyone")]:
            variant = copy.deepcopy(payload)
            variant["objects"][0][field] = value
            bad.append(variant)
        for variant in bad:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "objects.json"
                path.write_text(json.dumps(variant), encoding="utf-8")
                with self.assertRaises(ValueError):
                    DemoERP.from_path(path)

    def test_foreign_rows_never_appear_in_resolution_or_examples(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for field, value in [("org_id", "foreign-org"), ("principal_id", "foreign-principal"),
                             ("audience", "employee")]:
            variant = copy.deepcopy(payload)
            variant["objects"][0][field] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "objects.json"
                path.write_text(json.dumps(variant), encoding="utf-8")
                erp = DemoERP.from_path(path)
                resolved, _ = erp.resolve(context(), "INV-1001", [])
                self.assertEqual(resolved["erp_context"], {"source_status": "not_found", "facts": {}})
                self.assertTrue(all("INV-1001" not in example["message"] for example in erp.examples("customer")))

    def test_duplicate_json_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "objects.json"
            path.write_text('{"version":"a","version":"b","objects":[]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                DemoERP.from_path(path)


if __name__ == "__main__":
    unittest.main()
