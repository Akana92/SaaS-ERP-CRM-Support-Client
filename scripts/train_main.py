from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

try:
    from transformers import TrainerCallback as _TrainerCallback
except Exception:  # pragma: no cover - import is verified when transformers is installed
    _TrainerCallback = object

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support.contracts import DevCase, validate_evidence
from support.training import (
    CompletionOnlyCollator,
    adapter_tensor_snapshot,
    build_reload_generation_record,
    build_training_example,
    build_transformers_load_kwargs,
    cuda_memory_snapshot,
    file_sha256,
    load_model_config,
    load_policy_document,
    measure_token_lengths,
    package_versions,
    release_gpu_memory,
    require_cuda_dtype,
    reset_cuda_peak_memory,
    stable_json_sha256,
    trainable_parameter_counts,
    trainable_parameter_digest,
    verify_adapter_tensors_equal,
    verify_batch_labels_against_probes,
    verify_lora_only_trainable,
)


PROGRESS_FILE = "progress.json"
MANIFEST_FILE = "manifest.json"
MICROBATCH_MEMORY_FILE = "microbatch_memory.jsonl"
COMPLETION_LOGITS_MODEL_KEY = "qwen3_4b"
COMPLETION_LOGITS_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_strict_cases(path: Path, policy_path: Path, *, split: str, expected_rows: int | None = None) -> list[DevCase]:
    policy = load_policy_document(policy_path)
    policy_ids = {rule.id for rule in policy.rules}
    rows = read_jsonl(path)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{path} contains {len(rows)} rows, expected {expected_rows}")
    cases: list[DevCase] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{line_number} id must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id in {split}: {case_id}")
        seen_ids.add(case_id)
        real_split = row.get("split")
        if real_split != split:
            raise ValueError(f"{path}:{line_number} expected only split='{split}', got {real_split!r}")
        if split == "train" and row.get("expected_model_call") is not True:
            raise ValueError(f"{case_id} must have expected_model_call=true for main training")
        policy_refs = row.get("policy_refs")
        if not isinstance(policy_refs, list) or not policy_refs:
            raise ValueError(f"{case_id} policy_refs must be a non-empty list")
        if any(not isinstance(ref, str) or not ref for ref in policy_refs):
            raise ValueError(f"{case_id} policy_refs must contain only non-empty strings")
        if len(policy_refs) != len(set(policy_refs)):
            raise ValueError(f"{case_id} policy_refs must be unique")
        unknown_policy_refs = sorted(set(policy_refs) - policy_ids)
        if unknown_policy_refs:
            raise ValueError(f"{case_id} policy_refs contain unknown ids: {', '.join(unknown_policy_refs)}")
        contract_row = dict(row)
        contract_row["split"] = "development"
        case = DevCase.model_validate(contract_row)
        if case.policy_version != policy.version:
            raise ValueError(
                f"{case.id} policy_version {case.policy_version!r} does not match {policy.version!r}"
            )
        if split == "train" and case.expected is None:
            raise ValueError(f"{case.id} must have expected target for main training")
        if case.expected is not None:
            validate_evidence(case.expected, case.erp_context, policy_ids)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} has no {split} cases")
    return cases


def validation_family_check(validation_path: Path, policy_path: Path, train_cases: Sequence[DevCase]) -> dict[str, Any]:
    validation_cases = load_strict_cases(validation_path, policy_path, split="validation")
    train_families = {case.family_id for case in train_cases}
    validation_families = {case.family_id for case in validation_cases}
    overlap = sorted(train_families & validation_families)
    if overlap:
        raise ValueError(f"train/validation family overlap: {', '.join(overlap[:10])}")
    return {
        "validation_path": str(validation_path),
        "validation_sha256": file_sha256(validation_path),
        "validation_rows": len(validation_cases),
        "validation_family_count": len(validation_families),
        "train_validation_family_overlap": 0,
    }


def review_summary(cases: Sequence[DevCase]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.review_status] = counts.get(case.review_status, 0) + 1
    all_human_approved = set(counts) == {"human_approved"}
    return {
        "all_human_approved": all_human_approved,
        "counts": dict(sorted(counts.items())),
        "pending_human_review_visible": not all_human_approved,
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    relevant = dict(config)
    return stable_json_sha256(relevant)


def training_collator_mode(config: dict[str, Any], model_config: dict[str, Any]) -> str:
    enabled = config.get("completion_logits", False)
    if not isinstance(enabled, bool):
        raise ValueError("completion_logits must be a boolean top-level config flag")
    if not enabled:
        return "completion_only"
    if model_config.get("key") != COMPLETION_LOGITS_MODEL_KEY or model_config.get("model_id") != COMPLETION_LOGITS_MODEL_ID:
        raise ValueError("completion_logits is only supported for frozen Qwen3 4B config qwen3_4b")
    revision = model_config.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.lower()):
        raise ValueError("completion_logits requires a frozen 40-character model revision")
    return "completion_logits"


def validate_training_config(config: dict[str, Any]) -> None:
    training_config = config.get("training", {})
    torch_empty_cache_steps = training_config.get("torch_empty_cache_steps")
    if torch_empty_cache_steps is not None:
        if isinstance(torch_empty_cache_steps, bool) or not isinstance(torch_empty_cache_steps, int) or torch_empty_cache_steps <= 0:
            raise ValueError("training.torch_empty_cache_steps must be a positive integer when provided")
    empty_cache_at_boundary = training_config.get("empty_cache_at_microbatch_boundary", False)
    if not isinstance(empty_cache_at_boundary, bool):
        raise ValueError("training.empty_cache_at_microbatch_boundary must be a boolean when provided")
    efficient_sdpa = training_config.get("memory_efficient_sdpa", False)
    if not isinstance(efficient_sdpa, bool):
        raise ValueError("training.memory_efficient_sdpa must be a boolean when provided")
    if efficient_sdpa and config.get("model_key") != COMPLETION_LOGITS_MODEL_KEY:
        raise ValueError("memory_efficient_sdpa is verified only for qwen3_4b")
    portable_sessions = training_config.get("portable_sessions", False)
    if not isinstance(portable_sessions, bool):
        raise ValueError("training.portable_sessions must be a boolean when provided")
    if portable_sessions:
        if int(training_config.get("save_steps", 0)) != 1:
            raise ValueError("training.portable_sessions requires training.save_steps=1")
        if int(training_config.get("save_total_limit", 0)) != 3:
            raise ValueError("training.portable_sessions requires training.save_total_limit=3")


def build_training_collator(tokenizer: Any, config: dict[str, Any], model_config: dict[str, Any]) -> Any:
    if training_collator_mode(config, model_config) == "completion_logits":
        from support.completion_logits import CompletionLogitsCollator

        return CompletionLogitsCollator(tokenizer)
    return CompletionOnlyCollator(tokenizer)


def source_hashes(
    project_root: Path,
    config_path: Path,
    *,
    include_completion_logits: bool = False,
    include_training_sessions: bool = False,
    include_efficient_attention: bool = False,
) -> dict[str, str]:
    paths = [
        Path("scripts/train_main.py"),
        Path("src/support/training.py"),
        Path("src/support/modeling.py"),
        Path("src/support/prompting.py"),
        config_path.relative_to(project_root),
    ]
    if include_completion_logits:
        paths.insert(2, Path("src/support/completion_logits.py"))
    if include_training_sessions:
        paths.insert(2, Path("src/support/training_sessions.py"))
    if include_efficient_attention:
        paths.insert(2, Path("src/support/efficient_attention.py"))
    return {str(path).replace("\\", "/"): file_sha256(project_root / path) for path in paths}


def build_manifest(
    *,
    project_root: Path,
    config_path: Path,
    config: dict[str, Any],
    model_config: dict[str, Any],
    train_path: Path,
    validation_check: dict[str, Any],
    policy_path: Path,
    train_cases: Sequence[DevCase],
    length_report: Any,
    batch_label_probe: list[dict[str, Any]],
    experimental_draft: bool,
    phase: str,
) -> dict[str, Any]:
    from support.prompting import PROMPT_HASH, PROMPT_VERSION

    collator_mode = training_collator_mode(config, model_config)
    total_training_tokens = sum(item.total_tokens for item in length_report.items)
    supervised_tokens = sum(item.supervised_token_count for item in length_report.items)
    return {
        "version": config["version"],
        "artifact_status": config["artifact_status"],
        "phase": phase,
        "created_at_unix": int(time.time()),
        "experimental_draft": experimental_draft,
        "human_review": review_summary(train_cases),
        "data": {
            "train_path": str(train_path),
            "train_sha256": file_sha256(train_path),
            "train_rows": len(train_cases),
            "train_family_count": len({case.family_id for case in train_cases}),
            "expected_model_call_true": True,
            **validation_check,
            "test_dataset_read": False,
        },
        "policy": {
            "path": str(policy_path),
            "sha256": file_sha256(policy_path),
            "version": load_policy_document(policy_path).version,
        },
        "model": {
            "key": model_config["key"],
            "model_id": model_config["model_id"],
            "revision": model_config["revision"],
            "local_path": model_config["local_path"],
            "config_path": str(project_root / config["model_config"]),
            "config_sha256": file_sha256(project_root / config["model_config"]),
            "selected_config": model_config,
            "selected_config_sha256": stable_json_sha256(model_config),
        },
        "prompt": {
            "version": PROMPT_VERSION,
            "sha256": PROMPT_HASH,
        },
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
            "fingerprint": config_fingerprint(config),
        },
        "source_sha256": source_hashes(
            project_root,
            config_path,
            include_completion_logits=collator_mode == "completion_logits",
            include_training_sessions=bool(config["training"].get("portable_sessions", False)),
            include_efficient_attention=bool(config["training"].get("memory_efficient_sdpa", False)),
        ),
        "training_sessions": {
            "portable_sessions_enabled": bool(config["training"].get("portable_sessions", False)),
            "pause_request_file": "PAUSE_REQUEST",
            "checkpoint_receipt": "checkpoint_integrity.json",
            "checkpoint_complete_marker": "checkpoint_complete.json",
        },
        "attention": {
            "memory_efficient_sdpa": bool(config["training"].get("memory_efficient_sdpa", False)),
            "training_attn_implementation": "sdpa_repeat_kv"
            if bool(config["training"].get("memory_efficient_sdpa", False))
            else config["qlora"].get("attn_implementation", "sdpa"),
        },
        "training_collator": {
            "mode": collator_mode,
            "completion_logits_enabled": collator_mode == "completion_logits",
            "preflight_label_probe_mode": "completion_only",
        },
        "length_report": {
            "example_count": length_report.example_count,
            "max_prompt_tokens": length_report.max_prompt_tokens,
            "max_supervised_tokens": length_report.max_supervised_tokens,
            "max_total_tokens": length_report.max_total_tokens,
            "max_length": length_report.max_length,
            "total_training_tokens": total_training_tokens,
            "supervised_tokens": supervised_tokens,
        },
        "batch_label_probe": batch_label_probe,
        "training": config["training"],
        "qlora": config["qlora"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
        "versions": package_versions(["torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets"]),
    }


def inspect_collator_labels(tokenizer: Any, encoded: Sequence[Any], *, batch_size: int) -> list[dict[str, Any]]:
    collator = CompletionOnlyCollator(tokenizer, return_tensors=False)
    inspected: list[dict[str, Any]] = []
    for index in range(0, len(encoded), batch_size):
        chunk = encoded[index : index + batch_size]
        batch = collator([item.to_features() for item in chunk])
        inspected.extend(verify_batch_labels_against_probes(batch, tokenizer, [item.mask_probe for item in chunk]))
    return inspected


def prepare_run(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    preflight_only: bool,
    experimental_draft: bool,
    resume_from_checkpoint: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path, list[DevCase], list[Any], dict[str, Any]]:
    config = read_json(config_path)
    validate_training_config(config)
    model_config = load_model_config(project_root / config["model_config"], config.get("model_key"))
    training_collator_mode(config, model_config)
    train_path = project_root / config["train_path"]
    validation_path = project_root / config["validation_path"]
    policy_path = project_root / config["policy_path"]
    train_cases = load_strict_cases(
        train_path,
        policy_path,
        split="train",
        expected_rows=int(config["expected_train_rows"]),
    )
    validation_check = validation_family_check(validation_path, policy_path, train_cases)
    if not review_summary(train_cases)["all_human_approved"] and not (preflight_only or experimental_draft):
        raise ValueError("training uses pending_human_review rows; pass --experimental-draft to run a provisional experiment")

    existing_manifest = output_dir / MANIFEST_FILE
    if output_dir.exists():
        if preflight_only:
            raise FileExistsError(f"output dir already exists: {output_dir}")
        if not existing_manifest.exists():
            raise FileExistsError(f"output dir already exists without manifest: {output_dir}")
        manifest = read_json(existing_manifest)
        validate_resume_manifest(manifest, config, model_config, train_path, validation_check, policy_path, project_root, config_path)
        if resume_from_checkpoint is None:
            if manifest.get("phase") != "preflight_only":
                raise FileExistsError(f"output dir already exists and is not a preflight run: {output_dir}")
        else:
            validate_checkpoint_contents(
                output_dir,
                resume_from_checkpoint,
                expected_total_steps=int(config["expected_optimizer_steps"]),
                portable_sessions=bool(config["training"].get("portable_sessions", False)),
            )
    else:
        if resume_from_checkpoint is not None:
            raise FileNotFoundError(f"cannot resume missing output dir: {output_dir}")
        output_dir.mkdir(parents=True)

    model_ref = project_root / model_config["local_path"]
    return config, model_config, train_path, validation_path, policy_path, train_cases, [], {
        "model_ref": str(model_ref),
        "validation_check": validation_check,
    }


def validate_resume_manifest(
    manifest: dict[str, Any],
    config: dict[str, Any],
    model_config: dict[str, Any],
    train_path: Path,
    validation_check: dict[str, Any],
    policy_path: Path,
    project_root: Path,
    config_path: Path,
) -> None:
    from support.prompting import PROMPT_HASH, PROMPT_VERSION

    if manifest.get("config", {}).get("fingerprint") != config_fingerprint(config):
        raise ValueError("existing run manifest config fingerprint does not match")
    if manifest.get("config", {}).get("sha256") != file_sha256(config_path):
        raise ValueError("existing run manifest config sha256 does not match")
    if manifest.get("data", {}).get("train_sha256") != file_sha256(train_path):
        raise ValueError("existing run manifest train sha256 does not match")
    if manifest.get("data", {}).get("validation_sha256") != validation_check["validation_sha256"]:
        raise ValueError("existing run manifest validation sha256 does not match")
    if manifest.get("policy", {}).get("sha256") != file_sha256(policy_path):
        raise ValueError("existing run manifest policy sha256 does not match")
    if manifest.get("model", {}).get("config_sha256") != file_sha256(project_root / config["model_config"]):
        raise ValueError("existing run manifest model config sha256 does not match")
    if manifest.get("model", {}).get("key") != model_config["key"]:
        raise ValueError("existing run manifest model key does not match")
    if manifest.get("model", {}).get("model_id") != model_config["model_id"]:
        raise ValueError("existing run manifest model id does not match")
    if manifest.get("model", {}).get("revision") != model_config["revision"]:
        raise ValueError("existing run manifest model revision does not match")
    if manifest.get("model", {}).get("local_path") != model_config["local_path"]:
        raise ValueError("existing run manifest model local_path does not match")
    if manifest.get("model", {}).get("selected_config") != model_config:
        raise ValueError("existing run manifest selected model config does not match")
    if manifest.get("model", {}).get("selected_config_sha256") != stable_json_sha256(model_config):
        raise ValueError("existing run manifest selected model config sha256 does not match")
    if manifest.get("prompt", {}).get("version") != PROMPT_VERSION:
        raise ValueError("existing run manifest prompt version does not match")
    if manifest.get("prompt", {}).get("sha256") != PROMPT_HASH:
        raise ValueError("existing run manifest prompt sha256 does not match")
    if manifest.get("source_sha256") != source_hashes(
        project_root,
        config_path,
        include_completion_logits=training_collator_mode(config, model_config) == "completion_logits",
        include_training_sessions=bool(config["training"].get("portable_sessions", False)),
        include_efficient_attention=bool(config["training"].get("memory_efficient_sdpa", False)),
    ):
        raise ValueError("existing run manifest source hashes do not match")


def validate_checkpoint_path(output_dir: Path, checkpoint: Path | None) -> str | None:
    if checkpoint is None:
        return None
    resolved_output = output_dir.resolve()
    resolved_checkpoint = checkpoint.resolve()
    trainer_dir = (resolved_output / "_trainer").resolve()
    if resolved_checkpoint.parent != trainer_dir or not resolved_checkpoint.name.startswith("checkpoint-"):
        raise ValueError("--resume-from-checkpoint must point to this run's _trainer/checkpoint-*")
    if not checkpoint.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
    return str(resolved_checkpoint)


def validate_checkpoint_contents(
    output_dir: Path,
    checkpoint: Path | None,
    *,
    expected_total_steps: int,
    portable_sessions: bool = False,
) -> str | None:
    checkpoint_path = validate_checkpoint_path(output_dir, checkpoint)
    if checkpoint_path is None:
        return None
    checkpoint_dir = Path(checkpoint_path)
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
    missing = [name for name in required if not (checkpoint_dir / name).exists()]
    has_rng = (checkpoint_dir / "rng_state.pth").exists() or any(checkpoint_dir.glob("rng_state_*.pth"))
    if not has_rng:
        missing.append("rng_state.pth")
    if missing:
        raise FileNotFoundError(f"resume checkpoint is incomplete; missing: {', '.join(missing)}")
    trainer_state = read_json(checkpoint_dir / "trainer_state.json")
    global_step = trainer_state.get("global_step")
    if not isinstance(global_step, int) or global_step <= 0:
        raise ValueError("resume checkpoint trainer_state.global_step must be a positive integer")
    suffix = checkpoint_dir.name.rsplit("-", 1)[-1]
    if not suffix.isdigit() or int(suffix) != global_step:
        raise ValueError("resume checkpoint step does not match trainer_state.global_step")
    if global_step > expected_total_steps:
        raise ValueError("resume checkpoint global_step exceeds expected total steps")
    if portable_sessions:
        from support.training_sessions import validate_complete_checkpoint

        validate_complete_checkpoint(checkpoint_dir, expected_total_steps=expected_total_steps)
    return checkpoint_path


def write_progress(output_dir: Path, *, status: str, step: int, total: int, started: float, **extra: Any) -> None:
    payload = {
        "status": status,
        "step": step,
        "total": total,
        "epoch": extra.pop("epoch", None),
        "elapsed_seconds": round(time.time() - started, 3),
        "loss": extra.pop("loss", None),
        **extra,
    }
    atomic_write_json(output_dir / PROGRESS_FILE, payload)


def read_progress(output_dir: Path) -> dict[str, Any]:
    path = output_dir / PROGRESS_FILE
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def cuda_memory_counters() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"cuda_available": False}
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {"cuda_available": False}
    stats = cuda.memory_stats()
    return {
        "cuda_available": True,
        "memory_allocated": int(cuda.memory_allocated()),
        "memory_reserved": int(cuda.memory_reserved()),
        "max_memory_allocated": int(cuda.max_memory_allocated()),
        "max_memory_reserved": int(cuda.max_memory_reserved()),
        "inactive_split_bytes.all.current": int(stats.get("inactive_split_bytes.all.current", 0)),
    }


def clear_cuda_cache_with_precounters() -> dict[str, Any]:
    before = cuda_memory_counters()
    payload = {
        "pre_clear_memory_allocated": before.get("memory_allocated"),
        "pre_clear_memory_reserved": before.get("memory_reserved"),
        "pre_clear_max_memory_allocated": before.get("max_memory_allocated"),
        "pre_clear_max_memory_reserved": before.get("max_memory_reserved"),
        "pre_clear_inactive_split_bytes.all.current": before.get("inactive_split_bytes.all.current"),
    }
    if not before.get("cuda_available"):
        return payload
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        payload["cache_clear_error"] = True
    return payload


class ProgressCallback(_TrainerCallback):
    def __init__(
        self,
        output_dir: Path,
        started: float,
        total_steps: int,
        *,
        memory_provider: Any | None = None,
        time_provider: Any | None = None,
        allocation_config_provider: Any | None = None,
        clear_cache_at_microbatch_boundary: bool = False,
        cache_clearer: Any | None = None,
    ):
        self.output_dir = output_dir
        self.started = started
        self.total_steps = total_steps
        self.memory_provider = memory_provider or cuda_memory_counters
        self.time_provider = time_provider or time.time
        self.allocation_config_provider = allocation_config_provider or (
            lambda: os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        )
        self.clear_cache_at_microbatch_boundary = clear_cache_at_microbatch_boundary
        self.cache_clearer = cache_clearer or clear_cuda_cache_with_precounters

    def append_memory_event(self, event: str, state: Any) -> None:
        pre_clear = {}
        if self.clear_cache_at_microbatch_boundary:
            pre_clear = self.cache_clearer()
        counters = self.memory_provider()
        timestamp = self.time_provider()
        payload = {
            "timestamp": timestamp,
            "timeelapsed": round(timestamp - self.started, 3),
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "epoch": float(state.epoch) if getattr(state, "epoch", None) is not None else None,
            "event": event,
            "cuda_available": bool(counters.get("cuda_available", False)),
            "memory_allocated": counters.get("memory_allocated"),
            "memory_reserved": counters.get("memory_reserved"),
            "max_memory_allocated": counters.get("max_memory_allocated"),
            "max_memory_reserved": counters.get("max_memory_reserved"),
            "inactive_split_bytes.all.current": counters.get("inactive_split_bytes.all.current"),
            "pytorch_cuda_alloc_conf": self.allocation_config_provider(),
            "cache_cleared_at_microbatch_boundary": self.clear_cache_at_microbatch_boundary,
            **pre_clear,
        }
        append_jsonl(self.output_dir / MICROBATCH_MEMORY_FILE, payload)

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self.append_memory_event("train_begin", state)

    def on_substep_end(self, args, state, control, **kwargs):  # noqa: ANN001
        self.append_memory_event("substep_end", state)

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        self.append_memory_event("step_end", state)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            raise RuntimeError("trainer reported non-finite loss")
        write_progress(
            self.output_dir,
            status="training",
            step=int(state.global_step),
            total=self.total_steps,
            epoch=float(state.epoch) if state.epoch is not None else None,
            started=self.started,
            loss=float(loss) if loss is not None else None,
        )


def run_main_training(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    preflight_only: bool = False,
    experimental_draft: bool = False,
    resume_from_checkpoint: Path | None = None,
    session_seconds: float | None = None,
) -> int:
    started = time.time()
    may_write_failure_progress = False
    try:
        config = read_json(config_path)
        if preflight_only:
            experimental_draft = False
        config, model_config, train_path, _validation_path, policy_path, train_cases, _unused, prepared = prepare_run(
            project_root=project_root,
            config_path=config_path,
            output_dir=output_dir,
            preflight_only=preflight_only,
            experimental_draft=experimental_draft,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        may_write_failure_progress = True
        if not review_summary(train_cases)["all_human_approved"] and not (preflight_only or experimental_draft):
            raise ValueError("training uses pending_human_review rows; pass --experimental-draft to run a provisional experiment")

        write_progress(output_dir, status="loading", step=0, total=int(config["expected_optimizer_steps"]), started=started)

        from transformers import AutoTokenizer

        tokenizer_kwargs, _model_kwargs = build_transformers_load_kwargs(
            revision=model_config["revision"],
            dtype=None,
            quantization_config=None,
        )
        tokenizer = AutoTokenizer.from_pretrained(prepared["model_ref"], **tokenizer_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        policy = load_policy_document(policy_path)
        examples = [build_training_example(case, policy) for case in train_cases]
        encoded, length_report = measure_token_lengths(tokenizer, examples, max_length=int(config["max_length"]))
        batch_label_probe = inspect_collator_labels(
            tokenizer,
            encoded,
            batch_size=int(config["training"]["per_device_train_batch_size"]),
        )
        manifest = build_manifest(
            project_root=project_root,
            config_path=config_path,
            config=config,
            model_config=model_config,
            train_path=train_path,
            validation_check=prepared["validation_check"],
            policy_path=policy_path,
            train_cases=train_cases,
            length_report=length_report,
            batch_label_probe=batch_label_probe[:10],
            experimental_draft=experimental_draft,
            phase="preflight_only" if preflight_only else "training_initialized",
        )
        if resume_from_checkpoint is None:
            atomic_write_json(output_dir / MANIFEST_FILE, manifest)
            atomic_write_json(
                output_dir / "length_counts.json",
                {
                    "items": [asdict(item) for item in length_report.items],
                    "total_training_tokens": manifest["length_report"]["total_training_tokens"],
                    "supervised_tokens": manifest["length_report"]["supervised_tokens"],
                },
            )
        if preflight_only:
            write_progress(
                output_dir,
                status="completed",
                step=0,
                total=int(config["expected_optimizer_steps"]),
                started=started,
                phase="preflight_only",
            )
            return 0

        return train_gpu(
            config=config,
            model_config=model_config,
            tokenizer=tokenizer,
            encoded=encoded,
            output_dir=output_dir,
            started=started,
            resume_from_checkpoint=resume_from_checkpoint,
            session_seconds=session_seconds,
            first_case=train_cases[0],
            policy=policy,
            model_ref=Path(prepared["model_ref"]),
        )
    except Exception as exc:
        known_progress = read_progress(output_dir)
        if may_write_failure_progress and output_dir.exists():
            write_progress(
                output_dir,
                status="failed",
                step=int(known_progress.get("step", 0) or 0),
                total=int(config.get("expected_optimizer_steps", known_progress.get("total", 0)) or 0),
                epoch=known_progress.get("epoch"),
                loss=known_progress.get("loss"),
                started=started,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise


def train_gpu(
    *,
    config: dict[str, Any],
    model_config: dict[str, Any],
    tokenizer: Any,
    encoded: Sequence[Any],
    output_dir: Path,
    started: float,
    resume_from_checkpoint: Path | None,
    session_seconds: float | None,
    first_case: DevCase,
    policy: Any,
    model_ref: Path,
) -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    dtype = require_cuda_dtype(torch)
    reset_cuda_peak_memory(torch)
    set_seed(int(config["seed"]))
    trainer_dir = output_dir / config["trainer_subdir"]
    adapter_dir = output_dir / config["adapter_subdir"]
    checkpoint = validate_checkpoint_path(output_dir, resume_from_checkpoint)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["qlora"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bool(config["qlora"]["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=dtype,
    )
    _tokenizer_kwargs, model_kwargs = build_transformers_load_kwargs(
        revision=model_config["revision"],
        dtype=dtype,
        quantization_config=quantization_config,
    )
    trainer = None
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_ref), **model_kwargs)
        if config["training"].get("memory_efficient_sdpa", False):
            from support.efficient_attention import enable_memory_efficient_sdpa

            attention_info = enable_memory_efficient_sdpa(model)
            atomic_write_json(output_dir / "attention_backend.json", attention_info)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=int(config["qlora"]["r"]),
            lora_alpha=int(config["qlora"]["lora_alpha"]),
            lora_dropout=float(config["qlora"]["lora_dropout"]),
            bias=config["qlora"]["bias"],
            task_type=config["qlora"]["task_type"],
            target_modules=config["qlora"]["target_modules"],
        )
        model = get_peft_model(model, lora_config)
        lora_trainability = verify_lora_only_trainable(model)
        counts_before = trainable_parameter_counts(model)
        digest_before = trainable_parameter_digest(model)

        dataset = Dataset.from_list([item.to_features() for item in encoded])
        collator = build_training_collator(tokenizer, config, model_config)
        bf16 = dtype == torch.bfloat16
        args = TrainingArguments(
            output_dir=str(trainer_dir),
            max_steps=int(config["training"]["max_steps"]),
            num_train_epochs=float(config["training"]["num_train_epochs"]),
            per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
            learning_rate=float(config["training"]["learning_rate"]),
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            warmup_ratio=float(config["training"]["warmup_ratio"]),
            optim=config["training"]["optim"],
            logging_steps=int(config["training"]["logging_steps"]),
            save_strategy="steps",
            save_steps=int(config["training"]["save_steps"]),
            save_total_limit=int(config["training"]["save_total_limit"]),
            report_to=[],
            dataloader_num_workers=int(config["training"]["dataloader_num_workers"]),
            remove_unused_columns=bool(config["training"]["remove_unused_columns"]),
            gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
            fp16=torch.cuda.is_available() and not bf16,
            bf16=bf16,
            seed=int(config["seed"]),
            torch_empty_cache_steps=config["training"].get("torch_empty_cache_steps"),
        )
        callbacks: list[Any] = [
            ProgressCallback(
                output_dir,
                started,
                int(config["expected_optimizer_steps"]),
                clear_cache_at_microbatch_boundary=bool(
                    config["training"].get("empty_cache_at_microbatch_boundary", False)
                ),
            )
        ]
        portable_callback = None
        portable_sessions = bool(config["training"].get("portable_sessions", False))
        if portable_sessions:
            from support.training_sessions import PortableSessionCallback

            portable_callback = PortableSessionCallback(
                output_dir=output_dir,
                total_steps=int(config["expected_optimizer_steps"]),
                enabled=True,
                resumed_checkpoint=checkpoint,
                session_seconds=session_seconds,
            )
            callbacks.append(portable_callback)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=callbacks,
        )
        initial_step = 0
        initial_epoch = None
        if resume_from_checkpoint is not None:
            initial_state_path = resume_from_checkpoint / "trainer_state.json"
            if initial_state_path.exists():
                initial_state = read_json(initial_state_path)
                initial_step = int(initial_state.get("global_step", 0) or 0)
                initial_epoch = initial_state.get("epoch")
        write_progress(
            output_dir,
            status="training",
            step=initial_step,
            total=int(config["expected_optimizer_steps"]),
            epoch=initial_epoch,
            started=started,
        )
        finalization_recovery = False
        train_result = None
        if portable_sessions and checkpoint is not None and initial_step == int(config["expected_optimizer_steps"]):
            from support.training_sessions import load_verified_checkpoint_adapter

            load_verified_checkpoint_adapter(model, Path(checkpoint), device=str(args.device))
            from transformers import TrainerState

            trainer.state = TrainerState.load_from_json(str(Path(checkpoint) / "trainer_state.json"))
            finalization_recovery = True
        else:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
        if not losses or not all(math.isfinite(float(loss)) for loss in losses):
            raise RuntimeError("training did not report finite loss values")
        optimizer_steps = int(trainer.state.global_step)
        if portable_sessions and portable_callback is not None and portable_callback.pause_requested:
            from support.training_sessions import latest_complete_checkpoint

            checkpoint_path = latest_complete_checkpoint(output_dir, expected_total_steps=int(config["expected_optimizer_steps"]))
            if checkpoint_path is None:
                raise RuntimeError("portable training paused without a verified complete checkpoint")
            checkpoint_step = int(checkpoint_path.name.rsplit("-", 1)[-1])
            if checkpoint_step != optimizer_steps:
                raise RuntimeError("portable training paused but latest complete checkpoint does not match optimizer step")
            manifest = read_json(output_dir / MANIFEST_FILE)
            manifest["phase"] = "paused"
            manifest["portable_session"] = {
                "status": "paused",
                "optimizer_steps": optimizer_steps,
                "expected_optimizer_steps": config["expected_optimizer_steps"],
                "latest_complete_checkpoint": str(checkpoint_path),
            }
            atomic_write_json(output_dir / MANIFEST_FILE, manifest)
            write_progress(
                output_dir,
                status="paused",
                step=optimizer_steps,
                total=int(config["expected_optimizer_steps"]),
                epoch=float(trainer.state.epoch) if trainer.state.epoch is not None else None,
                started=started,
                loss=float(losses[-1]),
                latest_complete_checkpoint=str(checkpoint_path),
            )
            return 0
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError(
                f"training completed {optimizer_steps} optimizer steps, expected {config['expected_optimizer_steps']}"
            )
        digest_after = trainable_parameter_digest(model)
        if digest_after == digest_before:
            raise RuntimeError("trainable LoRA parameter digest did not change")
        final_epoch = float(trainer.state.epoch) if trainer.state.epoch is not None else None
        final_adapter_tensors = adapter_tensor_snapshot(model)
        training_cuda_memory = cuda_memory_snapshot(torch)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        del trainer
        trainer = None
        del model
        model = None
        release_gpu_memory(torch)

        model_input = __import__("support.contracts", fromlist=["model_input_from_case"]).model_input_from_case(first_case, policy)
        generation_check = adapter_reload_generation_check(
            config=model_config,
            adapter_path=adapter_dir,
            model_input=model_input,
            expected_adapter_tensors=final_adapter_tensors,
        )
        report = {
            "artifact_status": config["artifact_status"],
            "adapter_path": str(adapter_dir),
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": config["expected_optimizer_steps"],
            "losses": losses,
            "train_loss": getattr(train_result, "training_loss", None),
            "finalization_recovery": finalization_recovery,
            "trainable_parameter_counts_before": counts_before,
            "lora_trainability": lora_trainability,
            "trainable_digest_before": digest_before,
            "trainable_digest_after": digest_after,
            "training_peak_cuda_memory_bytes": training_cuda_memory["peak_cuda_memory_bytes"],
            "training_reserved_cuda_memory_bytes": training_cuda_memory["reserved_cuda_memory_bytes"],
            "generation_check": generation_check,
        }
        atomic_write_json(output_dir / "training_report.json", report)
        manifest = read_json(output_dir / MANIFEST_FILE)
        manifest["phase"] = "completed"
        manifest["adapter"] = {
            "path": str(adapter_dir),
            "trainable_digest_after": digest_after,
            "optimizer_steps": optimizer_steps,
            "reload_generation_status": generation_check,
        }
        atomic_write_json(output_dir / MANIFEST_FILE, manifest)
        write_progress(
            output_dir,
            status="completed",
            step=optimizer_steps,
            total=int(config["expected_optimizer_steps"]),
            epoch=final_epoch,
            started=started,
            loss=float(losses[-1]),
            adapter_path=str(adapter_dir),
        )
        return 0
    finally:
        trainer = None
        model = None
        release_gpu_memory(torch)


def adapter_reload_generation_check(
    *,
    config: dict[str, Any],
    adapter_path: Path,
    model_input: dict[str, Any],
    expected_adapter_tensors: dict[str, Any],
) -> dict[str, Any]:
    from support.modeling import LocalModelRunner

    runner = LocalModelRunner(config, adapter_path=str(adapter_path))
    try:
        runner.load()
        counts = trainable_parameter_counts(runner.model)
        digest = trainable_parameter_digest(runner.model)
        verify_adapter_tensors_equal(expected_adapter_tensors, runner.model)
        generated = runner.generate(
            model_input,
            request_id="stage4-main-reload",
            node="model_call",
            max_new_tokens=512,
        )
        return build_reload_generation_record(
            model_input=model_input,
            generated=generated,
            adapter_tensors_match=True,
            trainable_counts=counts,
            trainable_digest=digest,
            reload_peak_cuda_memory_bytes=getattr(generated, "peak_memory_allocated_bytes", None),
            reload_reserved_cuda_memory_bytes=getattr(generated, "memory_reserved_bytes", None),
        )
    finally:
        runner.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or run the Stage 4 main local QLoRA candidate.")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--config", default="configs/train-main.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--experimental-draft", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--session-seconds", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    config_path = (project_root / args.config).resolve()
    config = read_json(config_path)
    output_dir = (project_root / (args.output or config["output_dir"])).resolve()
    checkpoint = Path(args.resume_from_checkpoint).resolve() if args.resume_from_checkpoint else None
    if args.session_seconds is not None and (not math.isfinite(args.session_seconds) or args.session_seconds <= 0):
        raise ValueError("--session-seconds must be a positive number of seconds")
    exit_code = run_main_training(
        project_root=project_root,
        config_path=config_path,
        output_dir=output_dir,
        preflight_only=bool(args.preflight_only),
        experimental_draft=bool(args.experimental_draft),
        resume_from_checkpoint=checkpoint,
        session_seconds=args.session_seconds,
    )
    progress = read_progress(output_dir)
    print(json.dumps({"status": progress.get("status", "ok"), "output_dir": str(output_dir)}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
