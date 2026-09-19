"""Build an explicit, bounded local handoff ZIP. Never publish or bundle weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
OPERATOR_TEMPLATE = "reports/operator-time-study-template.csv"
OPERATOR_HEADER = "case_id,operator_pseudonym,mode,task_family,active_seconds,elapsed_seconds,resolved,correctness_reviewed,notes"
PINS = {
    "artifacts/stage5/quality90-v1/reports/candidate-f-semantic-round06/aggregate-summary.json":
        "762c469663f92d04e6fce59e14886c7eb1e6aa85634b89a9b9d51e544e344318",
    "artifacts/stage5/quality90-v1/verification/candidate-f-final-assessment-verification.json":
        "df7626349a9afc10810f843d22ba46b6ee07affa30bcd88de6ad91abf73f1a29",
    "docs/CANDIDATE_F_RESULTS.md":
        "d88fa9fae0c7c82744a2a17b49ff08707648986fc1607e6805d1882eac00e249",
}
EXTERNAL = {
    "base": {"model_id": "Qwen/Qwen3-4B-Instruct-2507",
             "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
             "expected_path": "models/qwen3_4b"},
    "adapter": {"sha256": "6efc00de7bcecc24872b87de797abd4d9d43ea10e679b23d3b4c13ecb7f7109b",
                "expected_path": "artifacts/stage5/quality90-v1/training/candidate-f-v13-audited-labels/final_adapter",
                "required_files": ["adapter_config.json", "adapter_model.safetensors"]},
    "dataset": {"included": False, "description": "docs/DATASET_METHODS_V13.md"},
}
SPECIAL = {
    "README_FOR_SUBMISSION.md", "README.md", "requirements-stage1.txt",
    "requirements-stage2.txt", "requirements-stage2-lock.txt",
    "configs/models.json", "configs/train-quality90-f-v13-audited-labels.json",
    "configs/submission-files.json", "data/erp/live-audiences.json",
    "data/erp/live-objects.json", "data/policy/employee-telecom-v3.json",
    "src/support/live_input_reference.json", "reports/business-metrics.json",
    "reports/business-metrics.md", "reports/operator-time-study-template.csv",
    "docs/review/LIVE_LATENCY_2026-09-19.md",
}


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Expected a portable relative POSIX path")
    parts = value.split("/")
    if any(p in {"", ".", ".."} or p.endswith((" ", ".")) for p in parts):
        raise ValueError("Invalid relative path")
    if PurePosixPath(value).is_absolute():
        raise ValueError("Absolute paths are forbidden")
    if any(p.lower().split(".")[0] in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))} for p in parts):
        raise ValueError("Reserved device path")
    return value


def allowed_source(value: str) -> bool:
    """Defense in depth; editable allowlist cannot admit state or raw datasets."""
    path = PurePosixPath(value)
    if any(p.startswith(".") or p.lower() in {"__pycache__", "node_modules", "venv", "logs", "history", "secrets", "credentials"} for p in path.parts):
        return False
    if any(word in path.name.lower() for word in ("secret", "credential", "private_key", "token.json")):
        return False
    if value in SPECIAL or value in PINS:
        return True
    return (
        (len(path.parts) == 3 and path.parts[:2] == ("src", "support") and path.suffix == ".py")
        or (len(path.parts) == 4 and path.parts[:3] == ("src", "support", "live_static") and path.suffix in {".html", ".css", ".js"})
        or (len(path.parts) == 2 and path.parts[0] in {"scripts", "tests"} and path.suffix == ".py")
        or (len(path.parts) == 2 and path.parts[0] == "docs" and path.name in {
            "BUSINESS_METRICS.md", "LOCAL_RUN_AND_DELIVERY.md", "DEFENSE_SCRIPT.md",
            "SUBMISSION_CHECKLIST.md", "DATASET_METHODS_V13.md", "DATASET_FINAL_F.md", "DEMO_ERP.md"})
    )


def build(root: Path, config: Path, output: Path, *, max_bytes: int = MAX_BYTES) -> dict:
    root, config, output = root.resolve(), config.resolve(), output.resolve()
    if not config.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("Config and output must stay inside the project root")
    if output.exists():
        raise FileExistsError("Output already exists; choose a new --output path")
    cap = min(max_bytes, MAX_BYTES)
    if cap <= 0:
        raise ValueError("Positive package limit required")
    document = json.loads(config.read_text(encoding="utf-8"))
    entries = document.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Explicit nonempty files list required")
    payload, manifest_entries, names, sources = {}, [], {"manifest.json"}, set()
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"source", "archive"}:
            raise ValueError("Each entry must have source and archive only")
        source, archive = safe_relative(entry["source"]), safe_relative(entry["archive"])
        if not allowed_source(source) or not allowed_source(archive):
            raise ValueError(f"Unsafe submission path: {source} -> {archive}")
        if source != archive and (source, archive) != ("README_FOR_SUBMISSION.md", "README.md"):
            raise ValueError("Only the documented README alias is supported")
        if archive.casefold() in names or source.casefold() in sources:
            raise ValueError("Duplicate source or archive path")
        names.add(archive.casefold())
        sources.add(source.casefold())
        path = (root / source).resolve()
        if not path.is_relative_to(root) or path == output:
            raise ValueError("Source escapes root or includes output")
        if path != root / source:
            raise ValueError("Source links and redirected paths are forbidden")
        if not path.is_file():
            raise FileNotFoundError(source)
        if path.stat().st_size > cap - total:
            raise ValueError("Submission exceeds uncompressed size limit")
        content = path.read_bytes()
        if source == OPERATOR_TEMPLATE:
            try:
                template_text = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError("Operator template must contain only the expected UTF-8 header") from exc
            if template_text not in {OPERATOR_HEADER, OPERATOR_HEADER + "\n", OPERATOR_HEADER + "\r\n"}:
                raise ValueError(
                    "Operator template contains observations or changed columns. "
                    "Preserve filled observations outside the submission; remove this CSV entry "
                    "from a separate submission config, or supply a header-only template in a staging checkout."
                )
        total += len(content)
        if total > cap:
            raise ValueError("Submission exceeds uncompressed size limit")
        digest = hashlib.sha256(content).hexdigest()
        if source in PINS and digest != PINS[source]:
            raise ValueError(f"Pinned metadata identity mismatch: {source}")
        payload[archive] = content
        manifest_entries.append({"source": source, "archive": archive, "size_bytes": len(content), "sha256": digest})
    manifest = {"schema_version": 1, "purpose": "local_handoff",
                "published": False, "clean_machine_verified": False, "publication_license_cleared": False,
                "notice": "Local handoff only. Weights, dataset rows, private live state and logs are excluded. Metadata is not cleared for public publication.",
                "external_artifacts": EXTERNAL, "entries": sorted(manifest_entries, key=lambda e: e["archive"])}
    payload["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if total + len(payload["MANIFEST.json"]) > cap:
        raise ValueError("Manifest exceeds package size limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open("xb") as handle:
            created = True
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive_file:
                for name, content in sorted(payload.items()):
                    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    archive_file.writestr(info, content, compresslevel=9)
        if output.stat().st_size > cap:
            raise ValueError("ZIP exceeds package size limit")
    except Exception:
        if created:
            output.unlink(missing_ok=True)
        raise
    return {"output": str(output), "size_bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "files": len(manifest_entries)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/submission-files.json")
    parser.add_argument("--output", default="artifacts/stage8/capstone-n4-local-demo.zip")
    args = parser.parse_args()
    print(json.dumps(build(ROOT, ROOT / args.config, ROOT / args.output), indent=2))


if __name__ == "__main__":
    main()
