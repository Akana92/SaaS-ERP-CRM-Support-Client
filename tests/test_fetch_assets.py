from __future__ import annotations

import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location("fetch_assets", ROOT / "scripts/fetch_assets.py")
assets = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(assets)


def spec(name="weights.bin", content=b"verified"):
    return {"repo_id": "owner/model", "repo_type": "model", "revision": "a" * 40,
            "verification": "sha256", "files": {name: {
                "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}


class FetchAssetsTests(unittest.TestCase):
    def test_verified_local_reused_without_revision_or_download(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "weights.bin").write_bytes(b"verified")
            config = spec()
            config["revision"] = None
            downloader = Mock(side_effect=AssertionError("no network"))
            self.assertEqual(assets.fetch_asset("adapter", config, root, downloader=downloader), "verified local files")
            downloader.assert_not_called()

    def test_wrong_hash_not_replaced_or_downloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "weights.bin").write_bytes(b"tampered")
            downloader = Mock()
            with self.assertRaisesRegex(assets.AssetError, "SHA-256"):
                assets.fetch_asset("adapter", spec(), root, downloader=downloader)
            self.assertEqual((root / "weights.bin").read_bytes(), b"tampered")
            downloader.assert_not_called()

    def test_missing_revision_and_verify_only_do_not_download(self):
        with tempfile.TemporaryDirectory() as folder:
            config = spec()
            config["revision"] = None
            downloader = Mock()
            with self.assertRaisesRegex(assets.AssetError, "not pinned"):
                assets.fetch_asset("adapter", config, Path(folder), downloader=downloader)
            with self.assertRaisesRegex(assets.AssetError, "missing"):
                assets.fetch_asset("adapter", config, Path(folder), verify_only=True, downloader=downloader)
            downloader.assert_not_called()

    def test_download_uses_exact_allowlist_revision_and_token(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def download(**kwargs):
                self.assertEqual(kwargs["allow_patterns"], ["weights.bin"])
                self.assertEqual(kwargs["revision"], "a" * 40)
                self.assertEqual(kwargs["token"], "secret")
                (kwargs["local_dir"] / "weights.bin").write_bytes(b"verified")
            assets.fetch_asset("adapter", spec(), root, token="secret", downloader=download)
            self.assertEqual((root / "weights.bin").read_bytes(), b"verified")

    def test_bad_download_not_installed(self):
        with tempfile.TemporaryDirectory() as folder:
            def download(**kwargs):
                (kwargs["local_dir"] / "weights.bin").write_bytes(b"tampered")
            with self.assertRaises(assets.AssetError):
                assets.fetch_asset("adapter", spec(), Path(folder), downloader=download)
            self.assertFalse((Path(folder) / "weights.bin").exists())

    def test_upstream_errors_do_not_expose_token(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(assets.AssetError) as error:
                assets.fetch_asset("adapter", spec(), Path(folder), token="hf_secret",
                                   downloader=Mock(side_effect=RuntimeError("hf_secret")))
            self.assertNotIn("hf_secret", str(error.exception))
            self.assertTrue(error.exception.__suppress_context__)

    def test_download_failure_categories_redact_upstream_details(self):
        wrapped = RuntimeError("hf_secret")
        wrapped.__cause__ = OSError(errno.ENOSPC, "hf_secret")
        windows_full = OSError("hf_secret")
        windows_full.winerror = 112
        cases = [(wrapped, "Disk full"), (windows_full, "Disk full"),
                 (ConnectionError(errno.ECONNRESET, "hf_secret"), "Connection/network failure"),
                 (TimeoutError("hf_secret"), "Connection/network failure")]
        for status in (401, 403):
            denied = RuntimeError("hf_secret")
            denied.response = types.SimpleNamespace(status_code=status)
            cases.append((denied, f"HTTP {status}"))
        with tempfile.TemporaryDirectory() as folder:
            for failure, category in cases:
                with self.subTest(category=category), self.assertRaises(assets.AssetError) as error:
                    assets.fetch_asset("adapter", spec(), Path(folder), token="hf_secret",
                                       downloader=Mock(side_effect=failure))
                self.assertIn(category, str(error.exception))
                self.assertNotIn("hf_secret", str(error.exception))
                self.assertTrue(error.exception.__suppress_context__)

    def test_windows_main_injects_trust_before_huggingface_import(self):
        trust = types.SimpleNamespace(inject_into_ssl=Mock())
        real_import = __import__
        def guarded_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                trust.inject_into_ssl.assert_called_once_with()
                return types.SimpleNamespace(snapshot_download=Mock(side_effect=ConnectionError("hf_secret")))
            return real_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.json"
            config.write_text(json.dumps({"base": spec(), "adapter": spec()}))
            errors = io.StringIO()
            with patch.object(assets.sys, "platform", "win32"), patch.dict("sys.modules", {"truststore": trust}), \
                 patch("builtins.__import__", side_effect=guarded_import), patch.object(assets, "CONFIG", config), \
                 contextlib.redirect_stderr(errors):
                self.assertEqual(assets.main(["--base-dir", str(Path(folder) / "base")]), 1)
            self.assertIn("Connection/network failure", errors.getvalue())
            self.assertNotIn("hf_secret", errors.getvalue())

    def test_verify_only_and_linux_do_not_import_truststore(self):
        real_import = __import__
        def guarded_import(name, *args, **kwargs):
            if name == "truststore":
                raise AssertionError("Trust store must not be imported")
            return real_import(name, *args, **kwargs)
        for platform, arguments in (("win32", ["--verify-only"]), ("linux", [])):
            with self.subTest(platform=platform), patch.object(assets.sys, "platform", platform), \
                 patch("builtins.__import__", side_effect=guarded_import), \
                 patch.object(assets, "fetch_asset", return_value="verified local files"), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(assets.main(arguments), 0)

    def test_main_filesystem_failures_are_safe(self):
        for failure, category in ((OSError(errno.ENOSPC, "hf_secret"), "Disk full"),
                                  (PermissionError(errno.EACCES, "hf_secret"), "errno 13")):
            errors = io.StringIO()
            with patch.object(assets, "fetch_asset", side_effect=failure), contextlib.redirect_stderr(errors):
                self.assertEqual(assets.main(["--verify-only"]), 1)
            self.assertIn(category, errors.getvalue())
            self.assertNotIn("hf_secret", errors.getvalue())

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("../escape", "/escape", "C:/escape", "..\\escape"):
                with self.subTest(name=name), self.assertRaises(assets.AssetError):
                    assets.fetch_asset("adapter", spec(name), Path(folder), downloader=Mock())

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            try:
                (root / "link").symlink_to(root, target_is_directory=True)
            except OSError:
                self.skipTest("OS does not permit symlink creation")
            with self.assertRaises(assets.AssetError):
                assets.safe_path(root / "link", "weights.bin")

    def test_symlink_guard_without_os_privileges(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaisesRegex(assets.AssetError, "symbolic links"):
                    assets.safe_path(Path(folder), "weights.bin")

    def test_base_index_cannot_reference_unapproved_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                (root / name).write_text("{}")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))
            with self.assertRaisesRegex(assets.AssetError, "unexpected shards"):
                assets.verify_structure("base", root, spec("expected.safetensors"))

    def test_dataset_cannot_overwrite_readme(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(assets.AssetError, "TRAIN"):
                assets.fetch_asset("dataset", spec("README.md"), Path(folder), downloader=Mock())

    def test_dataset_download_only_train_and_preserves_readme(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "README.md").write_text("project")
            name = "data/quality90_v1/train/example.jsonl"
            def download(**kwargs):
                target = kwargs["local_dir"] / name
                target.parent.mkdir(parents=True)
                target.write_bytes(b"verified")
                (kwargs["local_dir"] / "README.md").write_text("HF card")
            assets.fetch_asset("dataset", spec(name), root, downloader=download)
            self.assertEqual((root / "README.md").read_text(), "project")
            self.assertEqual((root / name).read_bytes(), b"verified")

    def test_manifest_pins_all_hashes_and_dataset_is_train_only(self):
        config = json.loads(assets.CONFIG.read_text())
        for key, value in config.items():
            for name, expected in value["files"].items():
                self.assertEqual(len(expected["sha256"]), 64)
                self.assertGreater(expected["bytes"], 0)
                if key == "dataset":
                    self.assertTrue(name.startswith("data/quality90_v1/train/"))
                    self.assertTrue(name.endswith(".jsonl"))

    def test_missing_optional_token_file_uses_default_auth(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(assets, "fetch_asset", return_value="verified local files") as fetch:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = assets.main(["--token-file", str(Path(folder) / "absent"), "--verify-only"])
                self.assertEqual(result, 0)
                self.assertIsNone(fetch.call_args.kwargs["token"])


if __name__ == "__main__":
    unittest.main()
