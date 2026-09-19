from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support.live_store import LiveStore


def state():
    return dict(schema_version=1, owners=["owner"], keys=[], conversations={}, requests={}, comparisons={}, queue={})


class LiveStoreTests(unittest.TestCase):
    def test_atomic_replace_failure_keeps_previous_valid_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            store = LiveStore(path)
            original = state()
            store.save(original)
            updated = {**original, "owners": ["new-owner"]}
            with patch("support.live_store.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    store.save(updated)
            self.assertEqual(store.load(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_corrupt_state_and_unknown_schema_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            for raw in ("{broken", json.dumps({**state(), "schema_version": 99}), json.dumps({**state(), "keys": {}})):
                with self.subTest(raw=raw):
                    path.write_text(raw, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        LiveStore(path).load()
                    self.assertEqual(path.read_text(encoding="utf-8"), raw)

    def test_disabled_store_never_writes_and_unicode_roundtrip(self):
        LiveStore().save(state())
        self.assertIsNone(LiveStore().load())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "state.json"
            payload = state()
            payload["queue"] = {"ticket": {"draft": "Проверить бюджетирование"}}
            store = LiveStore(path)
            store.save(payload)
            self.assertEqual(store.load(), payload)
            self.assertIn("бюджетирование", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
