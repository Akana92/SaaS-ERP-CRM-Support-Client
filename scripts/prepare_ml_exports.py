"""Prepare private upload packets for accepted F; never loads a model or evaluates data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat

BASE = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
CONFIG = "configs/train-quality90-f-v13-audited-labels.json"
ADAPTER = "artifacts/stage5/quality90-v1/training/candidate-f-v13-audited-labels/final_adapter"
DESTINATION = "artifacts/stage8/ml-exports"
MODEL_FILES = {
    "adapter_config.json": (1177, "32dacd267e214203e5601d0b2a83234664a1108eebdf17da18733539414a2b61"),
    "adapter_model.safetensors": (132187888, "6efc00de7bcecc24872b87de797abd4d9d43ea10e679b23d3b4c13ecb7f7109b"),
    "added_tokens.json": (735, "79f6ec6fcc423d3a82bfac8b9033b1daac7b7bce06a5f5f441637b480cc605de"),
    "chat_template.jinja": (2690, "0ab115e84509e86db2bb05a63edf63778ae7fe624972cdfa982b1e59053d6710"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "special_tokens_map.json": (644, "57255613bbe23c9497211ca68561ff429a51e871dbaf5a59998fa4c8f7fe168a"),
    "tokenizer_config.json": (5644, "e1ff43043fd7fbe07a7656d301d3c59352984be693c69c4becd4b670f52a374b"),
    "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
TRAIN_NAMES = {
    "erp_workflows_v4_audit_v13", "erp_workflows_colloquial_v4_audit_v13",
    "telecom_support_v2_audit_v13", "telecom_support_colloquial_v3_audit_v13",
    "saas_support_v5", "saas_support_colloquial_v5_audit_v13", "long_context_v2",
    "erp_long_v8_audit_v13", "telecom_long_v9_audit_v13", "saas_long_v9",
    "erp_joint_v12", "telecom_joint_v11_audit_v13", "saas_joint_v11",
}


def safe_path(root, relative):
    """Reject escapes and every link/reparse point, including existing parents."""
    root = Path(root).absolute()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe relative path")
    target = root / relative
    for node in [root, *root.parents, *[root.joinpath(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1)]]:
        if node.is_symlink():
            raise ValueError("Symlinks are forbidden")
        if node.exists() and getattr(node.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024):
            raise ValueError("Reparse points are forbidden")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path outside project")
    return target


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_file(path, expected_hash, expected_size=None):
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("Missing or empty source")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError("Source size mismatch")
    if sha256(path) != expected_hash:
        raise ValueError("Source SHA-256 mismatch")


def copy_checked(root, source, packet, relative, expected_hash, expected_size=None):
    source_path = safe_path(root, source)
    check_file(source_path, expected_hash, expected_size)
    destination = safe_path(packet, relative)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidentally replacing an existing file.
    with source_path.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    check_file(destination, expected_hash, source_path.stat().st_size)
    check_file(source_path, expected_hash, expected_size)
    return destination


def train_inventory(root, config):
    sources = config["train_sources"]
    expected_paths = {f"data/quality90_v1/train/{name}.jsonl" for name in TRAIN_NAMES}
    if set(sources) != expected_paths:
        raise ValueError("Training source allowlist mismatch")
    dialogues = turns = 0
    for relative, expected_hash in sources.items():
        path = safe_path(root, relative)
        check_file(path, expected_hash)
        if path.stat().st_size > 50_000_000:
            raise ValueError("Unexpectedly large training shard")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "pilot_train" or not isinstance(row.get("turns"), list):
                raise ValueError("Not a train dialogue")
            dialogues += 1
            turns += len(row["turns"])
    if (dialogues, turns) != (430, 1113):
        raise ValueError("Training counts mismatch")
    return dialogues, turns


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def manifest(packet, metadata):
    files = {}
    for path in sorted(packet.rglob("*")):
        if path.is_file():
            relative = path.relative_to(packet).as_posix()
            safe_path(packet, relative)
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(packet / "manifest.json", {**metadata, "files": files, "manifest_self_excluded": True})


def prepare(root):
    root = Path(root).absolute()
    output = safe_path(root, DESTINATION)
    if output.exists():
        raise FileExistsError("Export directory exists; refusing overwrite")
    config_path = safe_path(root, CONFIG)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dialogues, turns = train_inventory(root, config)
    for name, (size, digest) in MODEL_FILES.items():
        check_file(safe_path(root, f"{ADAPTER}/{name}"), digest, size)
    license_path = safe_path(root, "models/qwen3_4b/LICENSE")
    license_hash = "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
    check_file(license_path, license_hash, 11343)
    output.mkdir(parents=True, exist_ok=False)
    model = output / "model"
    dataset = output / "dataset"
    model.mkdir()
    dataset.mkdir()
    for name, (size, digest) in MODEL_FILES.items():
        copy_checked(root, f"{ADAPTER}/{name}", model, name, digest, size)
    adapter_config = json.loads((model / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_config.update(base_model_name_or_path=BASE, revision=REVISION)
    write_json(model / "adapter_config.json", adapter_config)
    copy_checked(root, "models/qwen3_4b/LICENSE", model, "BASE_MODEL_LICENSE", license_hash, 11343)
    (model / "README.md").write_text(MODEL_CARD, encoding="utf-8")
    for relative, digest in config["train_sources"].items():
        copy_checked(root, relative, dataset, relative, digest)
    (dataset / "README.md").write_text(DATASET_CARD, encoding="utf-8")
    manifest(model, {"base_model": BASE, "revision": REVISION, "source_directory": ADAPTER,
                     "source_files": {k: {"bytes": v[0], "sha256": v[1]} for k, v in MODEL_FILES.items()},
                     "adapter_config_copy_changes": ["base_model_name_or_path", "revision"]})
    manifest(dataset, {"source_config": CONFIG, "source_config_sha256": sha256(config_path),
                       "dialogues": dialogues, "turns": turns, "split": "train"})
    # Verify frozen originals again after all copy/metadata operations.
    for name, (size, digest) in MODEL_FILES.items():
        check_file(safe_path(root, f"{ADAPTER}/{name}"), digest, size)
    train_inventory(root, config)
    return {"output": str(output), "dialogues": dialogues, "turns": turns,
            "model_bytes": sum(p.stat().st_size for p in model.rglob("*") if p.is_file()),
            "dataset_bytes": sum(p.stat().st_size for p in dataset.rglob("*") if p.is_file())}


MODEL_CARD = """---
language:
- ru
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- support
- experimental
---
# Capstone support: accepted candidate F

Experimental Russian SaaS / fictional ERP / telecom support LoRA adapter.
Published project artifact; project-specific licensing has not been selected.
The upstream Base license does not automatically license the authored adapter or data.
Code and launch tutorial: https://github.com/Akana92/SaaS-ERP-CRM-Support-Client

## Base and training
Base: Qwen/Qwen3-4B-Instruct-2507, revision
`cdbee75f17c01a7cc42f958dc650907174af0554` (download separately; no base weights here).
Clean-base QLoRA NF4, rank 16, alpha 32, dropout .05, all linear layers,
two epochs / 558 optimizer steps. Training set: 430 synthetic train dialogues,
1113 target turns, 13 local shards. Pending human review.
Use the project's pinned runtime and PEFT loader with the adapter files here;
the copied adapter configuration points to the pinned Hub base instead of a local path.
Loading this adapter alone does not reproduce LangGraph, ERP context, or server decisions.

## Evidence and limitations
AI-assessed development comparison: Base 10/100 vs F 49/100 fully successful
dialogues, 225 turns, NF4. Status: QUALIFIED. Success requires the five labels
and semantic response criteria across all turns. This is not human-verified or
final-test accuracy: human review is absent, final sealed test was not run,
`full_success_rate=null`. Live BF16 screenshots are not a new evaluation.
Do not claim 90% quality or production readiness. Real ERP integration, production
authorization, and measured operator time savings are not provided.

## Provenance / license scope
`BASE_MODEL_LICENSE` is the unchanged Apache-2.0 license from the pinned Qwen
base snapshot. It applies to upstream Qwen assets (including copied tokenizer
assets), and is not a license grant for this project's adapter. Upstream source:
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554
`manifest.json` records source and export hashes. Only the COPY of
adapter_config.json changes base_model_name_or_path and revision; originals stay frozen.
No optimizer states, checkpoints, evaluation questions, gold labels, raw reviews,
isolation receipt, or secrets are included. This is not a complete experimental
reproduction bundle. See the code repository's docs/ML_ASSETS.md.
"""

DATASET_CARD = """---
language:
- ru
configs:
- config_name: default
  data_files:
  - split: train
    path: data/quality90_v1/train/*.jsonl
task_categories:
- text-generation
tags:
- synthetic
- support
- train-only
---
# Capstone support: F training shards

Train-only synthetic Russian SaaS / fictional ERP / telecom scenarios:
430 dialogues, 1113 target turns, 13 local JSONL shards. No real company or customer data.
No validation/test split is published here. Files preserve project-relative paths
under data/quality90_v1/train; restore those paths in a checkout to reuse them.
Each dialogue contains context and turn-level target labels/replies. The exact
file list and SHA-256 hashes are in manifest.json and the code repository's
configs/train-quality90-f-v13-audited-labels.json.

## Review and limits
Status: experimental_pending_human_review. AI-assisted annotation audit covered
a sample of 52 dialogues / 155 turns, not the whole corpus. Human verification
is pending. Do not interpret these examples as company policy or verified real facts.
F's 49/100 versus Base 10/100 comes from AI-assessed development evaluation (NF4,
QUALIFIED); no final sealed test was run and full_success_rate remains null.
Protected questions/gold/reviews and the isolation receipt are deliberately absent.
Reproducing the original training gate requires a separately authorized receipt
with SHA-256 1a918393bbc8b1727822d41182b844e3c89451939c7974a862217bc796fa697c.
Its hash is an identifier, not the receipt. This packet is not complete experiment
reproduction. Project-specific licensing has not been selected; publication
does not grant an additional license for project-authored data.
Code and launch tutorial: https://github.com/Akana92/SaaS-ERP-CRM-Support-Client
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), ensure_ascii=False, indent=2))
