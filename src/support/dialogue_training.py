"""Fail-closed CPU integrity checks for the separate dialogue training run."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from support.dialogue_dataset import (
    encode_prepared_turn, load_dialogue_policy, load_dialogues, prepare_turn,
    validate_split_isolation,
)
from support.training import file_sha256, load_model_config, package_versions, stable_json_sha256

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/stage5/dialogue-v1"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _project_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"input must be inside project: {path}")
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_feature_alignment(
    records: list[dict[str, Any]], features: list[dict[str, Any]], probes: list[dict[str, Any]],
) -> list[dict[str, list[int]]]:
    """Join by identity AND order; return only tensor fields for the collator."""
    _require(bool(records), "no training records")
    ids = [r.get("case_id") for r in records]
    for name, rows in (("prepared", records), ("encoded", features), ("probes", probes)):
        row_ids = [r.get("case_id") for r in rows]
        _require(all(isinstance(i, str) and i.strip() for i in row_ids), f"{name}: missing case_id")
        _require(len(row_ids) == len(set(row_ids)), f"{name}: duplicate case_id")
        _require(row_ids == ids, f"{name}: case_id set/order mismatch")
    clean = []
    for record, feature, probe in zip(records, features, probes):
        case_id = record["case_id"]
        _require(record.get("split") == "pilot_train", f"{case_id}: non-training split")
        _require(record.get("history_mode") == "controlled_gold", f"{case_id}: invalid history mode")
        metadata = record.get("metadata", {})
        _require(metadata.get("history_truncated") is False and not metadata.get("omitted_turn_ids"),
                 f"{case_id}: history was truncated")
        _require(record.get("review_status") in {"pending_human_review", "human_approved"},
                 f"{case_id}: unknown review status")
        _require(set(feature) == {"case_id", "input_ids", "attention_mask", "labels"},
                 f"{case_id}: unexpected encoded fields")
        values = {key: feature[key] for key in ("input_ids", "attention_mask", "labels")}
        for key, value in values.items():
            _require(isinstance(value, list) and bool(value) and all(type(v) is int for v in value),
                     f"{case_id}: {key} must be nonempty integer list")
        tokens, attention, labels = (values[k] for k in ("input_ids", "attention_mask", "labels"))
        _require(len(tokens) == len(attention) == len(labels), f"{case_id}: tensor length mismatch")
        _require(all(v >= 0 for v in tokens) and all(v == 1 for v in attention),
                 f"{case_id}: invalid tokens/attention")
        _require(all(label == -100 or label == token for label, token in zip(labels, tokens)),
                 f"{case_id}: labels do not match inputs")
        prompt_count = probe.get("prompt_token_count")
        _require(type(prompt_count) is int and 0 < prompt_count < len(tokens), f"{case_id}: invalid prompt length")
        _require(all(v == -100 for v in labels[:prompt_count]), f"{case_id}: prompt supervision")
        supervised = [v for v in labels if v != -100]
        _require(bool(supervised) and supervised == probe.get("supervised_token_ids"),
                 f"{case_id}: invalid supervision")
        _require(supervised[-1] == probe.get("terminal_eos_token_id"), f"{case_id}: missing supervised EOS")
        _require(len(tokens) == probe.get("total_tokens") and len(supervised) == probe.get("supervised_token_count")
                 and labels.count(-100) == probe.get("ignored_label_count"), f"{case_id}: incorrect probe counts")
        _require(stable_json_sha256(labels) == probe.get("label_sha256"), f"{case_id}: label hash mismatch")
        target_hash = stable_json_sha256(json.loads(record["target_json"]))
        _require(target_hash == record.get("target_sha256") == probe.get("target_sha256"),
                 f"{case_id}: target hash mismatch")
        _require(probe.get("decoded_target") == record["target_json"], f"{case_id}: decoded target mismatch")
        clean.append(values)
    return clean


def prepare_dialogue_training(
    config: dict[str, Any], output_dir: Path, *, allow_pending_review: bool, tokenizer: Any = None,
) -> dict[str, Any]:
    """Rebuild every target from sources without loading model weights or writing files."""
    output = Path(output_dir).resolve()
    _require(output.is_relative_to(ARTIFACTS.resolve()) and output != ARTIFACTS.resolve(),
             "run directory must be below artifacts/stage5/dialogue-v1")
    prepared_dir = _project_path(config["prepared_dir"])
    _require(not output.is_relative_to(prepared_dir) and not prepared_dir.is_relative_to(output),
             "run directory must not overlap the prepared pack")
    paths = {"pilot_train": _project_path(config["train_path"]),
             "development": _project_path(config["development_path"])}
    # This launcher deliberately accepts the new dialogue directory only.
    _require(all(p.parent == ROOT / "data/dialogue_v2_250" for p in paths.values()),
             "only versioned dialogue inputs are allowed; original validation/test are excluded")
    manifest_path = prepared_dir / "manifest.json"
    manifest = _json(manifest_path)
    _require(manifest.get("schema_version") == "dialogue-preparation-v1", "unsupported preparation schema")
    _require(manifest.get("history_mode") == "controlled_gold", "prepared history mode mismatch")
    _require(config["max_length"] == manifest["completion_mask"]["max_length"], "max_length differs from preparation")
    source_hashes: dict[str, str] = {}

    def bind(path: Path, expected: str | None = None) -> str:
        path = _project_path(path)
        digest = file_sha256(path)
        if expected is not None:
            _require(digest == expected, f"hash mismatch: {path.relative_to(ROOT)}")
        source_hashes[str(path.relative_to(ROOT))] = digest
        return digest

    bind(manifest_path)
    for split, path in paths.items():
        _require(path == _project_path(manifest["source_paths"][split]), f"{split}: source path mismatch")
        bind(path, manifest["source_hashes"][split])
    expected_artifacts = {"pilot_train.prepared.jsonl", "pilot_train.encoded.jsonl", "pilot_train.mask_probes.jsonl",
                          "development.controlled.jsonl", "duplication_audit.json"}
    _require(set(manifest["artifact_hashes"]) == expected_artifacts, "preparation artifacts missing or unexpected")
    for name, digest in manifest["artifact_hashes"].items():
        bind(prepared_dir / name, digest)
    for name, digest in manifest["code_hashes"].items():
        bind(_project_path(name), digest)
    frozen_path = ROOT / "artifacts/stage5/dialogue-v1/frozen-check.json"
    bind(frozen_path)
    for item in _json(frozen_path):
        bind(_project_path(item["path"]), item["sha256"])
    dialogues = {split: load_dialogues(path, split) for split, path in paths.items()}
    validate_split_isolation(dialogues["pilot_train"], dialogues["development"])
    _require(len(dialogues["pilot_train"]) == config["expected_dialogues"], "train dialogue count mismatch")
    pending = sum(d.review_status == "pending_human_review" for d in dialogues["pilot_train"])
    _require(not pending or allow_pending_review, "pending_human_review requires explicit experimental mode")
    records = _rows(prepared_dir / "pilot_train.prepared.jsonl")
    features = _rows(prepared_dir / "pilot_train.encoded.jsonl")
    probes = _rows(prepared_dir / "pilot_train.mask_probes.jsonl")
    controlled_dev = _rows(prepared_dir / "development.controlled.jsonl")
    clean = validate_feature_alignment(records, features, probes)
    _require(len(clean) == config["expected_train_rows"] == manifest["completion_mask"]["verified_examples"],
             "train target count mismatch")
    model_config_path = _project_path(config["model_config"])
    bind(model_config_path)
    model_config = load_model_config(model_config_path, config["model_key"])
    model_ref = _project_path(model_config["local_path"])
    _require(model_ref == _project_path(manifest["tokenizer_path"]), "tokenizer/model path mismatch")
    tokenizer_files = {p.name for p in model_ref.iterdir() if p.is_file()
                       and ("token" in p.name or p.name in {"vocab.json", "merges.txt", "chat_template.jinja"})}
    _require(tokenizer_files == set(manifest["tokenizer_hashes"]), "tokenizer file set changed")
    for name, digest in manifest["tokenizer_hashes"].items():
        bind(model_ref / name, digest)
    for path in sorted(model_ref.iterdir()):
        if path.is_file() and (path.suffix == ".safetensors" or path.name in {
            "config.json", "generation_config.json", "model.safetensors.index.json",
        }):
            bind(path)
    initial_adapter = _project_path(config["initial_adapter_path"])
    _require(initial_adapter == ROOT / "artifacts/stage4/portable-full-v8/final_adapter",
             "initial adapter must be the frozen full-v8 adapter")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        bind(initial_adapter / name)
    for name in ("scripts/train_dialogue_pilot.py", "src/support/dialogue_training.py", "scripts/train_main.py",
                 "src/support/training_sessions.py", "src/support/training.py", "src/support/completion_logits.py",
                 "scripts/training_control.py", "src/support/efficient_attention.py",
                 "src/support/contracts.py", "src/support/graph.py",
                 "src/support/modeling.py", "data/policy/employee-telecom-v3.json"):
        bind(ROOT / name)
    policy = load_dialogue_policy()
    _require(stable_json_sha256(policy.model_dump(mode="json")) == manifest["policy_sha256"], "effective policy changed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_ref), local_files_only=True, trust_remote_code=False)

    def count(messages: list[dict[str, str]]) -> int:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    for split, saved in (("pilot_train", records), ("development", controlled_dev)):
        expected_ids = [f"{d.id}:turn-{i + 1}" for d in dialogues[split] for i in range(len(d.turns))]
        _require([r["case_id"] for r in saved] == expected_ids, f"{split}: source turn identity/order mismatch")
        _require(len(saved) == manifest["splits"][split]["turns"]
                 and len(dialogues[split]) == manifest["splits"][split]["dialogues"], f"{split}: manifest counts mismatch")
        position = 0
        for dialogue in dialogues[split]:
            for index in range(len(dialogue.turns)):
                prepared = prepare_turn(dialogue, index, policy, count)
                rebuilt = {"case_id": prepared.case_id, "dialogue_id": dialogue.id, "family_id": dialogue.family_id,
                           "split": split, "review_status": dialogue.review_status, "history_mode": prepared.history_mode,
                           "model_input": prepared.model_input, "chat_messages": prepared.chat_messages,
                           "metadata": prepared.metadata.as_dict(), "target_json": prepared.target_json,
                           "target_sha256": stable_json_sha256(json.loads(prepared.target_json))}
                _require(rebuilt == saved[position], f"{prepared.case_id}: source/prompt reconstruction mismatch")
                _require(not prepared.metadata.omitted_turn_ids, f"{prepared.case_id}: required history lost")
                if split == "pilot_train":
                    encoded = encode_prepared_turn(prepared, tokenizer, max_length=config["max_length"])
                    _require(encoded.to_features() == features[position], f"{prepared.case_id}: re-encoding mismatch")
                    _require(asdict(encoded.mask_probe) == probes[position], f"{prepared.case_id}: rebuilt mask mismatch")
                position += 1
    # Recheck cheap sources after composition, so an edit during preparation fails closed.
    for name, digest in list(source_hashes.items()):
        path = ROOT / name
        if path.suffix != ".safetensors":
            _require(file_sha256(path) == digest, f"input changed during preflight: {name}")
    integrity = {
        "schema_version": "dialogue-training-integrity-v1", "artifact_status": "candidate_not_selected",
        "experimental_pending_review": bool(pending), "pending_dialogues": pending,
        "train_dialogues": len(dialogues["pilot_train"]), "train_targets": len(clean),
        "development_dialogues": len(dialogues["development"]), "development_targets": len(controlled_dev),
        "development_used_for_training": False, "history_mode": "controlled_gold", "history_truncations": 0,
        "max_total_tokens": max(len(f["input_ids"]) for f in clean),
        "case_ids": [r["case_id"] for r in records],
        "effective_prompt_hashes": {r["case_id"]: r["metadata"]["effective_input_sha256"] for r in records},
        "source_hashes": source_hashes, "policy_sha256": manifest["policy_sha256"],
        "tokenizer_chat_template_sha256": stable_json_sha256(tokenizer.chat_template),
        "model": model_config, "initial_adapter": str(initial_adapter.relative_to(ROOT)),
        "config": config, "config_sha256": stable_json_sha256(config),
        "packages": package_versions(["torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors"]),
        "cross_split_text_matches": manifest["duplication_counts"],
        "review_caveat": "AI-reviewed synthetic dialogue draft; no human approval or measured improvement implied",
    }
    reload_example = next(r for r in records if r["metadata"]["seen_turn_count"] > 0)
    return {"features": clean, "source_hashes": source_hashes, "integrity_manifest": integrity,
            "tokenizer": tokenizer, "model_config": model_config, "model_ref": model_ref,
            "initial_adapter": initial_adapter, "reload_example": reload_example, "records": records, "probes": probes}
