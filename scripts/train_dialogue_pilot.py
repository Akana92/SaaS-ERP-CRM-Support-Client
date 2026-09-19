"""Isolated continued-SFT pilot; --preflight-only never loads model weights.

The initial full-v8 adapter is read-only input. Every checkpoint and final
adapter belongs to this new run. A step-limited session keeps the full scheduler
horizon; --resume restores only a verified checkpoint inside that same run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from support.training import (
    adapter_tensor_snapshot, build_reload_generation_record, build_transformers_load_kwargs,
    cuda_memory_snapshot, file_sha256, package_versions, release_gpu_memory,
    require_cuda_dtype, stable_json_sha256, trainable_parameter_counts,
    trainable_parameter_digest, verify_adapter_tensors_equal, verify_lora_only_trainable,
)
from support.training_sessions import (
    PortableSessionCallback, latest_complete_checkpoint, load_verified_checkpoint_adapter,
    validate_complete_checkpoint,
)
from train_main import (
    ProgressCallback, atomic_write_json, append_jsonl, build_training_collator,
    validate_checkpoint_path, validate_training_config, write_progress,
)
from training_control import assert_no_unmanaged_training, gpu_lock, prevent_idle_sleep


MAX_SESSION_SECONDS = 20 * 3600


def validate_config(config: dict[str, Any], *, allow_multiple_epochs: bool = False) -> None:
    validate_training_config(config)
    training = config["training"]
    for name in ("expected_train_rows", "expected_dialogues", "expected_optimizer_steps", "max_length", "seed"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("max_steps", "per_device_train_batch_size", "gradient_accumulation_steps", "save_steps", "save_total_limit"):
        value = training[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{name} must be a positive integer")
    if training["max_steps"] != config["expected_optimizer_steps"]:
        raise ValueError("optimizer horizon must match training.max_steps")
    per_step = training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]
    epochs = training["num_train_epochs"]
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("training.num_train_epochs must be a positive integer")
    if not allow_multiple_epochs and epochs != 1:
        raise ValueError("pilot must cover exactly one epoch at the configured batch/accumulation")
    if config["expected_optimizer_steps"] != epochs * math.ceil(config["expected_train_rows"] / per_step):
        raise ValueError("optimizer horizon must cover every configured epoch at the batch/accumulation")
    if not training.get("portable_sessions") or training["save_steps"] != 1 or training["save_total_limit"] != 3:
        raise ValueError("pilot requires every-step portable checkpoints and three retained checkpoints")
    if not config.get("completion_logits") or config["model_key"] != "qwen3_4b":
        raise ValueError("pilot requires the existing Qwen3 completion-logits path")
    if training.get("remove_unused_columns") is not False:
        raise ValueError("tensor features must reach the completion collator unchanged")
    if training.get("group_by_length") is not True or training.get("logging_nan_inf_filter") is not False:
        raise ValueError("pilot requires longest-first length grouping and unfiltered loss logging")
    if config["qlora"].get("load_in_4bit") is not True or config["qlora"].get("bnb_4bit_quant_type") != "nf4":
        raise ValueError("pilot requires NF4 initial model loading")
    validate_session_seconds(config["session_seconds"], config["pause_reserve_seconds"])


def validate_session_seconds(seconds: float, reserve: float) -> None:
    if isinstance(seconds, bool) or not math.isfinite(seconds) or not 0 < seconds <= MAX_SESSION_SECONDS:
        raise ValueError("session must be finite, positive and no longer than 20 hours")
    if isinstance(reserve, bool) or not math.isfinite(reserve) or not 0 < reserve < seconds:
        raise ValueError("pause reserve must be positive and smaller than session duration")


def checked_run_dir(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    allowed = (ROOT / "artifacts/stage5/dialogue-v1/training").resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError("run-dir must be a child of artifacts/stage5/dialogue-v1/training")
    return path


class PilotSessionCallback(PortableSessionCallback):
    """Pause after additional steps without changing Trainer.max_steps."""

    def __init__(self, *, pause_after_steps: int | None, initial_step: int, **kwargs: Any):
        super().__init__(**kwargs)
        if pause_after_steps is not None and (isinstance(pause_after_steps, bool) or pause_after_steps <= 0):
            raise ValueError("pause-after-steps must be positive")
        self.pause_step = initial_step + pause_after_steps if pause_after_steps is not None else None
        self.initial_step = initial_step

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        # Trainer restores PEFT + optimizer/scheduler before this callback. Verify
        # the actual restored model, not merely the checkpoint files on disk.
        if self.resumed_checkpoint:
            checkpoint = Path(self.resumed_checkpoint)
            if int(state.global_step) != self.initial_step:
                raise ValueError("Trainer resumed at the wrong global step")
            verify_loaded_adapter(kwargs["model"], checkpoint)
            verify_lora_only_trainable(kwargs["model"])
            optimizer = kwargs.get("optimizer")
            if optimizer is None or not optimizer.state:
                raise ValueError("resumed optimizer state is empty")
            import torch
            saved_scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
            scheduler = kwargs.get("lr_scheduler")
            if scheduler is None or scheduler.state_dict() != saved_scheduler:
                raise ValueError("resumed scheduler differs from checkpoint")
            self._receipt("resume_state_verified", state, adapter_tensors_match=True,
                          optimizer_state_present=True, scheduler_state_match=True)
        return super().on_train_begin(args, state, control, **kwargs)

    def _should_pause(self, state: Any) -> bool:
        step = int(getattr(state, "global_step", 0) or 0)
        if step >= self.total_steps:
            return False
        return super()._should_pause(state) or (self.pause_step is not None and step >= self.pause_step)


class PilotProgressCallback(ProgressCallback):
    """Reject nonfinite gradients/weights before writing a complete checkpoint."""

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        import torch
        for name, parameter in kwargs["model"].named_parameters():
            if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError(f"nonfinite gradient before optimizer step: {name}")

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        import torch
        for name, parameter in kwargs["model"].named_parameters():
            if parameter.requires_grad and not torch.isfinite(parameter).all().item():
                raise RuntimeError(f"nonfinite adapter weight before checkpoint: {name}")
        super().on_step_end(args, state, control, **kwargs)

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
        for field in ("loss", "grad_norm"):
            value = (logs or {}).get(field)
            if value is not None and not math.isfinite(float(value)):
                raise RuntimeError(f"nonfinite training {field}")
        super().on_log(args, state, control, logs=logs, **kwargs)


def probe_sampler(trainer: Any, prepared: dict[str, Any]) -> dict[str, Any]:
    import torch
    before = torch.random.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        order = list(iter(trainer._get_train_sampler()))
    if not torch.equal(before, torch.random.get_rng_state()):
        raise RuntimeError("sampler probe changed CPU random state")
    features = prepared["features"]
    if sorted(order) != list(range(len(features))):
        raise RuntimeError("sampler does not cover each training target once")
    lengths = [len(row["input_ids"]) for row in features]
    if lengths[order[0]] != max(lengths):
        raise RuntimeError("length-grouped sampler did not put a longest sample first")
    return {"scope": "unconsumed sampler probe; RNG restored; resume skips already consumed batches",
            "first_index": order[0], "first_case_id": prepared["records"][order[0]]["case_id"],
            "first_tokens": lengths[order[0]], "maximum_tokens": max(lengths),
            "all_targets_once": True, "rng_preserved": True, "probe_order_sha256": stable_json_sha256(order)}


@contextmanager
def session_deadline(seconds: float):
    """Last-resort process exit if a GPU step/save cannot reach the soft pause.

    This may lose the unfinished step, never rewrites a verified checkpoint,
    and releases OS locks on process exit. Normal stopping uses the callback.
    """
    finished = threading.Event()

    def enforce():
        if not finished.wait(seconds):
            os._exit(124)

    watcher = threading.Thread(target=enforce, name="pilot-session-deadline", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        watcher.join(timeout=1)


def select_checkpoint(run_dir: Path, resume: str | None, total_steps: int) -> Path | None:
    if resume is None:
        return None
    checkpoint = latest_complete_checkpoint(run_dir, expected_total_steps=total_steps) if resume == "latest" else (ROOT / resume).resolve()
    if checkpoint is None:
        raise ValueError("no verified checkpoint in this pilot run")
    validate_checkpoint_path(run_dir, checkpoint)
    validate_complete_checkpoint(checkpoint, expected_total_steps=total_steps)
    return checkpoint


def run_identity(config: dict[str, Any], config_path: Path, prepared: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "dialogue-training-run-v1", "config": config,
        "config_sha256": file_sha256(config_path),
        "source_hashes": prepared["source_hashes"],
        "integrity_manifest": prepared["integrity_manifest"],
        "versions": package_versions(["torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets"]),
    }


def verify_loaded_adapter(model: Any, adapter_path: Path) -> None:
    """Compare PEFT-normalized tensors against the immutable adapter on disk."""
    from peft import get_peft_model_state_dict, load_peft_weights

    expected = load_peft_weights(str(adapter_path), device="cpu", local_files_only=True)
    actual = get_peft_model_state_dict(model)
    if set(actual) != set(expected):
        raise ValueError("loaded initial adapter tensor keys do not match")
    for name, value in expected.items():
        if not actual[name].detach().float().cpu().equal(value.detach().float().cpu()):
            raise ValueError(f"loaded initial adapter differs: {name}")


def reload_smoke(model_config: dict[str, Any], adapter_dir: Path, expected: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    from support.live_precision import LocalPrecisionRunner

    runner = LocalPrecisionRunner(model_config, adapter_path=str(adapter_dir), inference_profile="nf4")
    try:
        runner.load()
        verify_adapter_tensors_equal(expected, runner.model)
        generated = runner.generate_chat(example["chat_messages"], example["model_input"],
                                         request_id="dialogue-pilot-reload", max_new_tokens=512)
        record = build_reload_generation_record(
            model_input=example["model_input"], generated=generated, adapter_tensors_match=True,
            trainable_counts=trainable_parameter_counts(runner.model), trainable_digest=trainable_parameter_digest(runner.model),
            reload_peak_cuda_memory_bytes=getattr(generated, "peak_memory_allocated_bytes", None),
            reload_reserved_cuda_memory_bytes=getattr(generated, "memory_reserved_bytes", None),
        )
        record.update(case_id=example["case_id"], chat_messages=example["chat_messages"],
                      effective_input_sha256=stable_json_sha256(example["chat_messages"]))
        return record
    finally:
        runner.close()


def train(config: dict[str, Any], prepared: dict[str, Any], run_dir: Path, manifest: dict[str, Any],
          checkpoint: Path | None, *, pause_after_steps: int | None, seconds: float, reserve: float) -> int:
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainerState, TrainingArguments, set_seed

    started = time.time()
    total = config["expected_optimizer_steps"]
    initial_step = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))["global_step"] if checkpoint else 0
    manifest["phase"] = "loading"
    atomic_write_json(run_dir / "manifest.json", manifest)
    write_progress(run_dir, status="loading", step=initial_step, total=total, started=started)
    append_jsonl(run_dir / "session_launches.jsonl", {
        "started_at": started, "hard_deadline_at": started + seconds, "soft_pause_at": started + seconds - reserve,
        "initial_step": initial_step, "pause_after_steps": pause_after_steps, "total_steps": total,
        "resume": str(checkpoint) if checkpoint else None, "hard_timeout_exit_code": 124,
    })
    model = trainer = None
    try:
        dtype = require_cuda_dtype(torch)
        torch.cuda.reset_peak_memory_stats()
        set_seed(config["seed"])
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=config["qlora"]["bnb_4bit_use_double_quant"],
                                         bnb_4bit_compute_dtype=dtype)
        _, model_kwargs = build_transformers_load_kwargs(revision=prepared["model_config"]["revision"], dtype=dtype,
                                                         quantization_config=quantization)
        model = AutoModelForCausalLM.from_pretrained(str(prepared["model_ref"]), **model_kwargs)
        if config["training"].get("memory_efficient_sdpa"):
            from support.efficient_attention import enable_memory_efficient_sdpa
            atomic_write_json(run_dir / "attention_backend.json", enable_memory_efficient_sdpa(model))
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = PeftModel.from_pretrained(model, str(prepared["initial_adapter"]), is_trainable=True, local_files_only=True)
        verify_loaded_adapter(model, Path(prepared["initial_adapter"]))
        trainability = verify_lora_only_trainable(model)
        initial_digest = trainable_parameter_digest(model)
        manifest["initial_adapter_verified"] = True
        atomic_write_json(run_dir / "manifest.json", manifest)
        tc = config["training"]
        tokenizer = prepared["tokenizer"]
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        args = TrainingArguments(
            output_dir=str(run_dir / "_trainer"), max_steps=tc["max_steps"], num_train_epochs=tc["num_train_epochs"],
            per_device_train_batch_size=tc["per_device_train_batch_size"], gradient_accumulation_steps=tc["gradient_accumulation_steps"],
            learning_rate=tc["learning_rate"], lr_scheduler_type=tc["lr_scheduler_type"], warmup_ratio=tc["warmup_ratio"], optim=tc["optim"],
            logging_steps=tc["logging_steps"], save_strategy="steps", save_steps=1, save_total_limit=3,
            dataloader_num_workers=0, remove_unused_columns=False, gradient_checkpointing=tc["gradient_checkpointing"],
            bf16=dtype == torch.bfloat16, fp16=dtype != torch.bfloat16, seed=config["seed"], report_to=[],
            torch_empty_cache_steps=tc.get("torch_empty_cache_steps"),
            group_by_length=tc["group_by_length"], logging_nan_inf_filter=tc["logging_nan_inf_filter"],
        )
        portable = PilotSessionCallback(output_dir=run_dir, total_steps=total, enabled=True,
                                        resumed_checkpoint=str(checkpoint) if checkpoint else None,
                                        session_seconds=seconds-reserve, pause_after_steps=pause_after_steps, initial_step=initial_step)
        portable.started_at = started  # Count model loading against the soft deadline as well.
        trainer = Trainer(
            model=model, args=args, train_dataset=Dataset.from_list(prepared["features"]),
            data_collator=build_training_collator(tokenizer, config, prepared["model_config"]),
            callbacks=[PilotProgressCallback(run_dir, started, total, clear_cache_at_microbatch_boundary=tc.get("empty_cache_at_microbatch_boundary", False)), portable],
        )
        atomic_write_json(run_dir / "sampler_probe.json", probe_sampler(trainer, prepared))
        write_progress(run_dir, status="training", step=initial_step, total=total, started=started)
        if checkpoint and initial_step == total:
            load_verified_checkpoint_adapter(model, checkpoint, device=str(args.device))
            trainer.state = TrainerState.load_from_json(str(checkpoint / "trainer_state.json"))
        else:
            trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        step = int(trainer.state.global_step)
        losses = [float(row["loss"]) for row in trainer.state.log_history if "loss" in row]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise RuntimeError("training must report finite loss")
        last = latest_complete_checkpoint(run_dir, expected_total_steps=total)
        if last is None or int(last.name.rsplit("-", 1)[1]) != step:
            raise RuntimeError("training ended without an exact-step verified checkpoint")
        if step < total:
            if not portable.pause_requested:
                raise RuntimeError("training stopped early without a portable pause")
            manifest.update(phase="paused", global_step=step, latest_checkpoint=str(last))
            atomic_write_json(run_dir / "manifest.json", manifest)
            write_progress(run_dir, status="paused", step=step, total=total, started=started, loss=losses[-1], latest_complete_checkpoint=str(last))
            return 0
        if step != total:
            raise RuntimeError("training exceeded the fixed optimizer horizon")
        final_digest = trainable_parameter_digest(model)
        if final_digest == initial_digest:
            raise RuntimeError("adapter did not change from the initial full-v8")
        expected = adapter_tensor_snapshot(model)
        peak = cuda_memory_snapshot(torch)
        adapter_dir = run_dir / "final_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        trainer = None
        model = None
        release_gpu_memory(torch)
        generation = reload_smoke(prepared["model_config"], adapter_dir, expected, prepared["reload_example"])
        report = {"artifact_status": config["artifact_status"], "optimizer_steps": step, "losses": losses,
                  "initial_adapter": str(prepared["initial_adapter"]), "initial_digest": initial_digest,
                  "final_digest": final_digest, "lora_trainability": trainability, "training_memory": peak,
                  "generation_check": generation, "promotion": False}
        atomic_write_json(run_dir / "training_report.json", report)
        manifest.update(phase="completed", global_step=step, final_adapter=str(adapter_dir), promotion=False)
        atomic_write_json(run_dir / "manifest.json", manifest)
        write_progress(run_dir, status="completed", step=step, total=total, started=started, adapter_path=str(adapter_dir))
        return 0
    except BaseException as exc:
        manifest.update(phase="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(run_dir / "manifest.json", manifest)
        progress = run_dir / "progress.json"
        previous = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else {}
        write_progress(run_dir, status="failed", step=int(previous.get("step", initial_step)), total=total, started=started, error=manifest["error"])
        raise
    finally:
        trainer = model = None
        release_gpu_memory(torch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train-dialogue-pilot.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--allow-pending-review", action="store_true")
    parser.add_argument("--pause-after-steps", type=int)
    parser.add_argument("--session-seconds", type=float)
    parser.add_argument("--pause-reserve-seconds", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    run_dir = checked_run_dir(args.run_dir or config["run_dir"])
    seconds = args.session_seconds if args.session_seconds is not None else config["session_seconds"]
    reserve = args.pause_reserve_seconds if args.pause_reserve_seconds is not None else config["pause_reserve_seconds"]
    validate_session_seconds(seconds, reserve)
    if args.pause_after_steps is not None and args.pause_after_steps <= 0:
        raise ValueError("pause-after-steps must be positive")
    if args.preflight_only and args.resume:
        raise ValueError("preflight-only cannot resume training")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from support.dialogue_training import prepare_dialogue_training

    with gpu_lock(run_dir / ".run.lock"):
        manifest_path = run_dir / "manifest.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        if existing is None and any(path.name != ".run.lock" for path in run_dir.iterdir()):
            raise ValueError("refusing nonempty run directory without a pilot manifest")
        prepared = prepare_dialogue_training(config, run_dir, allow_pending_review=args.allow_pending_review)
        identity = run_identity(config, config_path, prepared)
        fingerprint = stable_json_sha256(identity)
        if existing is not None:
            if existing.get("fingerprint") != fingerprint or existing.get("identity") != identity:
                raise ValueError("run fingerprint changed; start a new run instead of resume")
            if not args.preflight_only and not args.resume and existing.get("phase") != "preflight_only":
                raise ValueError("existing training run requires explicit --resume")
        elif args.resume:
            raise ValueError("resume requires an existing pilot manifest")
        checkpoint = select_checkpoint(run_dir, args.resume, config["expected_optimizer_steps"])
        manifest = existing or {"identity": identity, "fingerprint": fingerprint, "phase": "preflight_only", "promotion": False}
        atomic_write_json(run_dir / "preflight.json", prepared["integrity_manifest"])
        if existing is None:
            atomic_write_json(manifest_path, manifest)
        if args.preflight_only:
            print(json.dumps({"status": "preflight_pass", "features": len(prepared["features"]), "fingerprint": fingerprint, "run_dir": str(run_dir)}))
            return 0
        with gpu_lock(ROOT / "artifacts/stage4/portable-gpu.lock"), prevent_idle_sleep(), session_deadline(seconds):
            assert_no_unmanaged_training()
            return train(config, prepared, run_dir, manifest, checkpoint, pause_after_steps=args.pause_after_steps, seconds=seconds, reserve=reserve)


if __name__ == "__main__":
    raise SystemExit(main())
