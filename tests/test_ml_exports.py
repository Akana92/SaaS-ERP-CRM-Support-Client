import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("ml_exports", Path(__file__).parents[1] / "scripts/prepare_ml_exports.py")
exports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exports)


class ExportTests(unittest.TestCase):
    def test_copy_verifies_hash_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source").write_bytes(b"fixture")
            digest = exports.sha256(root / "source")
            exports.copy_checked(root, "source", root / "packet", "nested/file", digest, 7)
            self.assertEqual((root / "packet/nested/file").read_bytes(), b"fixture")
            with self.assertRaises(FileExistsError):
                exports.copy_checked(root, "source", root / "packet", "nested/file", digest)
            with self.assertRaises(ValueError):
                exports.copy_checked(root, "source", root / "packet", "bad", "0" * 64)
            self.assertFalse((root / "packet/bad").exists())
            with self.assertRaises(ValueError):
                exports.check_file(root / "source", digest, 1)

    def test_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for path in ["../other", Path(temporary).absolute()]:
                with self.assertRaises(ValueError):
                    exports.safe_path(temporary, path)

    def test_link_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaises(ValueError):
                    exports.safe_path(temporary, "data")

    def test_train_allowlist_rejects_protected_path_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "allowlist"):
                exports.train_inventory(temporary, {"train_sources": {"data/test/secret.jsonl": "0" * 64}})

    def test_manifest_records_every_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file").write_bytes(b"fixture")
            exports.manifest(root, {"split": "train"})
            receipt = json.loads((root / "manifest.json").read_text())
            self.assertEqual(set(receipt["files"]), {"file"})
            self.assertEqual(receipt["files"]["file"]["sha256"], exports.sha256(root / "file"))

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / exports.DESTINATION
            output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                exports.prepare(root)


if __name__ == "__main__":
    unittest.main()
