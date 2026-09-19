"""CPU-only archive boundaries; no model or protected dataset access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("build_submission", Path(__file__).resolve().parents[1] / "scripts/build_submission.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class SubmissionBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.output = self.root / "bundle.zip"
        self.write("README_FOR_SUBMISSION.md", b"Local demo\n")
        self.entries = [{"source": "README_FOR_SUBMISSION.md", "archive": "README.md"}]

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def run_build(self, **kwargs):
        self.config.write_text(json.dumps({"files": self.entries}), encoding="utf-8")
        return builder.build(self.root, self.config, self.output, **kwargs)

    def test_manifest_hash_size_readme_and_determinism(self):
        result = self.run_build()
        with zipfile.ZipFile(self.output) as archive:
            manifest = json.loads(archive.read("MANIFEST.json"))
            self.assertEqual(["MANIFEST.json", "README.md"], archive.namelist())
            for item in manifest["entries"]:
                content = archive.read(item["archive"])
                self.assertEqual(len(content), item["size_bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])
            self.assertFalse(manifest["published"])
            self.assertFalse(manifest["clean_machine_verified"])
            self.assertFalse(manifest["publication_license_cleared"])
            self.assertEqual("README_FOR_SUBMISSION.md", manifest["entries"][0]["source"])
        self.output = self.root / "second.zip"
        self.assertEqual(result["sha256"], self.run_build()["sha256"])

    def test_never_overwrite(self):
        self.output.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            self.run_build()
        self.assertEqual(b"existing", self.output.read_bytes())

    def test_paths_and_aliases_rejected(self):
        for value in ["../secret.txt", "/tmp/foo", "C:/secret", "src/../README.md", "src\\support\\x.py", "scripts/CON.py", "scripts/name. /x.py"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.entries = [{"source": value, "archive": value}]
                self.run_build()
        self.entries = [{"source": "README_FOR_SUBMISSION.md", "archive": "src/support/fake.py"}]
        with self.assertRaises(ValueError):
            self.run_build()

    def test_secrets_weights_raw_and_state_rejected_before_read(self):
        for value in [".env", "scripts/secrets.py", "src/support/credentials.py", "models/model.safetensors", "data/train.jsonl", "artifacts/stage5/live/store.json", "artifacts/stage5/quality90-v1/reviews/raw.json", "src/support/__pycache__/x.pyc"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.entries = [{"source": value, "archive": value}]
                self.run_build()
        self.assertFalse(self.output.exists())

    def test_duplicate_archive_and_source_rejected(self):
        self.entries *= 2
        with self.assertRaises(ValueError):
            self.run_build()

    def test_missing_file(self):
        self.entries.append({"source": "src/support/missing.py", "archive": "src/support/missing.py"})
        with self.assertRaises(FileNotFoundError):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_oversize_content_and_manifest(self):
        for limit in (5, 100):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.run_build(max_bytes=limit)
        self.assertFalse(self.output.exists())

    def test_outside_output_and_config(self):
        self.config.write_text(json.dumps({"files": self.entries}), encoding="utf-8")
        with self.assertRaises(ValueError):
            builder.build(self.root, self.config, self.root.parent / "outside.zip")
        with self.assertRaises(ValueError):
            builder.build(self.root, self.root.parent / "outside.json", self.output)

    def test_only_exact_pinned_metadata_admitted(self):
        source = next(iter(builder.PINS))
        self.write(source, b"not the pinned content")
        self.entries = [{"source": source, "archive": source}]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.run_build()
        self.assertFalse(builder.allowed_source(source.replace("aggregate-summary", "raw-results")))

    def test_symlink_to_outside_or_private_inside_rejected(self):
        self.write("private.json", b"private")
        link = self.root / "src/support/linked.py"
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(self.root / "private.json")
        except OSError:
            self.skipTest("OS does not permit unprivileged symlink creation")
        self.entries = [{"source": "src/support/linked.py", "archive": "src/support/linked.py"}]
        with self.assertRaises(ValueError):
            self.run_build()

    def test_redirected_source_rejected_without_symlink_privilege(self):
        self.write("private.json", b"private")
        source = self.root / "src/support/linked.py"
        original = Path.resolve

        def resolve(path, *args, **kwargs):
            if path == source:
                return self.root / "private.json"
            return original(path, *args, **kwargs)

        self.entries = [{"source": "src/support/linked.py", "archive": "src/support/linked.py"}]
        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ValueError, "redirected"):
                self.run_build()

    def test_case_insensitive_duplicate_archive_rejected(self):
        self.write("src/support/example.py", b"pass\n")
        self.entries = [{"source": p, "archive": p} for p in (
            "src/support/example.py", "src/support/EXAMPLE.py")]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_build()

    def test_operator_template_accepts_only_exact_header(self):
        source = builder.OPERATOR_TEMPLATE
        self.entries = [{"source": source, "archive": source}]
        for index, ending in enumerate(("", "\n", "\r\n")):
            self.write(source, (builder.OPERATOR_HEADER + ending).encode("utf-8-sig"))
            self.output = self.root / f"header-{index}.zip"
            self.run_build()

    def test_operator_observations_and_notes_never_packaged(self):
        source = builder.OPERATOR_TEMPLATE
        self.entries = [{"source": source, "archive": source}]
        for tail in ("\ncase-1,person,manual,erp,40,50,yes,yes,private note\n",
                     "\n# private note\n", "\n\n", ",extra_column\n"):
            content = (builder.OPERATOR_HEADER + tail).encode("utf-8-sig")
            self.write(source, content)
            with self.subTest(tail=tail), self.assertRaisesRegex(ValueError, "contains observations"):
                self.run_build()
            self.assertFalse(self.output.exists())
            self.assertEqual(content, (self.root / source).read_bytes())


if __name__ == "__main__":
    unittest.main()
