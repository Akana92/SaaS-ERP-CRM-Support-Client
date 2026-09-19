from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - import availability is covered by train runtime
    TrainerCallback = object  # type: ignore[assignment]


PAUSE_REQUEST_FILE = "PAUSE_REQUEST"
SESSION_RECEIPTS_FILE = "session_receipts.jsonl"
CHECKPOINT_RECEIPT_FILE = "checkpoint_integrity.json"
CHECKPOINT_COMPLETE_FILE = "checkpoint_complete.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_step(checkpoint_dir: Path) -> int:
    suffix = checkpoint_dir.name.rsplit("-", 1)[-1]
    if not checkpoint_dir.name.startswith("checkpoint-") or not suffix.isdigit():
        raise ValueError(f"invalid checkpoint directory name: {checkpoint_dir.name}")
    step = int(suffix)
    if step <= 0:
        raise ValueError(f"invalid checkpoint step: {checkpoint_dir.name}")
    return step


def _checkpoint_required_files(checkpoint_dir: Path) -> list[Path]:
    required_names = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "adapter_model.safetensors",
        "adapter_config.json",
    ]
    required = [checkpoint_dir / name for name in required_names]
    rng_files = sorted(checkpoint_dir.glob("rng_state*.pth"))
    if rng_files:
        required.extend(rng_files)
    else:
        required.append(checkpoint_dir / "rng_state.pth")
    return required


def checkpoint_status(checkpoint_dir: Path) -> dict[str, Any]:
    missing: list[str] = []
    empty: list[str] = []
    hashes: dict[str, str] = {}
    for path in _checkpoint_required_files(checkpoint_dir):
        if not path.exists():
            missing.append(path.name)
        elif path.stat().st_size <= 0:
            empty.append(path.name)
        else:
            hashes[path.name] = _sha256(path)
    trainer_state: dict[str, Any] | None = None
    global_step: int | None = None
    state_path = checkpoint_dir / "trainer_state.json"
    if state_path.exists() and state_path.stat().st_size > 0:
        trainer_state = _read_json(state_path)
        value = trainer_state.get("global_step")
        if isinstance(value, int):
            global_step = value
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "missing": sorted(set(missing)),
        "empty": sorted(set(empty)),
        "global_step": global_step,
        "hashes": dict(sorted(hashes.items())),
    }


def write_checkpoint_receipt(checkpoint_dir: Path, *, created_at: float | None = None) -> dict[str, Any]:
    status = checkpoint_status(checkpoint_dir)
    if status["missing"] or status["empty"]:
        raise FileNotFoundError(
            f"checkpoint is incomplete; missing={status['missing']} empty={status['empty']}"
        )
    step_from_name = _checkpoint_step(checkpoint_dir)
    if status["global_step"] != step_from_name:
        raise ValueError("checkpoint trainer_state.global_step does not match directory suffix")
    # Flush the saved training state before publishing its complete marker.
    # A sudden laptop shutdown must not leave only a durable marker in front of
    # still-buffered optimizer/adapter data.
    for path in _checkpoint_required_files(checkpoint_dir):
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    receipt = {
        "schema": "capstone-portable-checkpoint-v1",
        "created_at_unix": created_at if created_at is not None else time.time(),
        "checkpoint": checkpoint_dir.name,
        "global_step": status["global_step"],
        "hashes": status["hashes"],
    }
    _atomic_write_json(checkpoint_dir / CHECKPOINT_RECEIPT_FILE, receipt)
    _atomic_write_json(
        checkpoint_dir / CHECKPOINT_COMPLETE_FILE,
        {
            "schema": "capstone-portable-checkpoint-complete-v1",
            "checkpoint": checkpoint_dir.name,
            "global_step": status["global_step"],
            "receipt_sha256": _sha256(checkpoint_dir / CHECKPOINT_RECEIPT_FILE),
            "completed_at_unix": receipt["created_at_unix"],
        },
    )
    return receipt


def read_checkpoint_receipt(checkpoint_dir: Path) -> dict[str, Any]:
    return _read_json(checkpoint_dir / CHECKPOINT_RECEIPT_FILE)


def validate_complete_checkpoint(checkpoint_dir: Path, *, expected_total_steps: int | None = None) -> dict[str, Any]:
    complete_path = checkpoint_dir / CHECKPOINT_COMPLETE_FILE
    receipt_path = checkpoint_dir / CHECKPOINT_RECEIPT_FILE
    if not complete_path.exists() or not receipt_path.exists():
        raise FileNotFoundError("checkpoint complete marker or receipt is missing")
    complete = _read_json(complete_path)
    receipt = _read_json(receipt_path)
    if receipt.get("schema") != "capstone-portable-checkpoint-v1":
        raise ValueError("checkpoint receipt schema mismatch")
    if complete.get("schema") != "capstone-portable-checkpoint-complete-v1":
        raise ValueError("checkpoint complete marker schema mismatch")
    if complete.get("checkpoint") != checkpoint_dir.name or receipt.get("checkpoint") != checkpoint_dir.name:
        raise ValueError("checkpoint marker name mismatch")
    if complete.get("receipt_sha256") != _sha256(receipt_path):
        raise ValueError("checkpoint receipt hash mismatch")
    status = checkpoint_status(checkpoint_dir)
    if status["missing"] or status["empty"]:
        raise FileNotFoundError(
            f"checkpoint is incomplete; missing={status['missing']} empty={status['empty']}"
        )
    global_step = receipt.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise ValueError("checkpoint receipt global_step must be a positive integer")
    if complete.get("global_step") != global_step:
        raise ValueError("checkpoint complete marker global_step mismatch")
    if status["global_step"] != global_step:
        raise ValueError("checkpoint global_step mismatch")
    if _checkpoint_step(checkpoint_dir) != global_step:
        raise ValueError("checkpoint step mismatch")
    if expected_total_steps is not None and global_step > expected_total_steps:
        raise ValueError("checkpoint global_step exceeds expected total steps")
    receipt_hashes = receipt.get("hashes")
    if not isinstance(receipt_hashes, dict) or set(receipt_hashes) != set(status["hashes"]):
        raise ValueError("checkpoint receipt hash set mismatch")
    for name, expected_hash in receipt_hashes.items():
        path = checkpoint_dir / name
        if not path.exists() or _sha256(path) != expected_hash:
            raise ValueError(f"checkpoint hash mismatch for {name}")
    return receipt


def choose_latest_verified_checkpoint(
    trainer_dir: Path, *, expected_total_steps: int | None = None
) -> Path | None:
    stepped_candidates: list[tuple[int, Path]] = []
    for path in trainer_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            stepped_candidates.append((_checkpoint_step(path), path))
        except ValueError:
            continue
    candidates = [path for _step, path in sorted(stepped_candidates, key=lambda item: item[0], reverse=True)]
    for checkpoint in candidates:
        try:
            validate_complete_checkpoint(checkpoint, expected_total_steps=expected_total_steps)
        except Exception:
            continue
        return checkpoint
    return None


def latest_complete_checkpoint(output_dir: Path, *, expected_total_steps: int | None = None) -> Path | None:
    return choose_latest_verified_checkpoint(output_dir / "_trainer", expected_total_steps=expected_total_steps)


def _normalized_adapter_snapshot(model: Any) -> dict[str, Any]:
    import torch

    snapshot: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        normalized = name.replace(".default.", ".")
        snapshot[normalized] = parameter.detach().float().cpu().clone()
    if not snapshot:
        raise ValueError("no adapter tensors found after checkpoint load")
    return snapshot


def load_verified_checkpoint_adapter(model: Any, checkpoint_dir: Path, *, device: str | None = None) -> dict[str, Any]:
    validate_complete_checkpoint(checkpoint_dir)
    from peft import load_peft_weights, set_peft_model_state_dict

    checkpoint_weights = load_peft_weights(str(checkpoint_dir), device=device)
    set_peft_model_state_dict(model, checkpoint_weights)
    actual = _normalized_adapter_snapshot(model)
    expected = {name: tensor.detach().float().cpu().clone() for name, tensor in checkpoint_weights.items()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(f"loaded adapter missing tensors: {', '.join(missing)}")
    if extra:
        raise ValueError(f"loaded adapter has unexpected tensors: {', '.join(extra)}")
    for name, expected_tensor in expected.items():
        if not actual[name].equal(expected_tensor):
            raise ValueError(f"loaded adapter tensor mismatch: {name}")
    return actual


class PortableSessionCallback(TrainerCallback):
    def __init__(
        self,
        *,
        output_dir: Path,
        total_steps: int,
        enabled: bool,
        resumed_checkpoint: str | None = None,
        session_seconds: float | None = None,
        time_provider: Any | None = None,
    ):
        self.output_dir = output_dir
        self.total_steps = total_steps
        self.enabled = enabled
        self.resumed_checkpoint = resumed_checkpoint
        self.session_seconds = session_seconds
        self.time_provider = time_provider or time.time
        self.started_at = float(self.time_provider())
        self.pause_requested = False

    def _receipt(self, event: str, state: Any, **extra: Any) -> None:
        if not self.enabled:
            return
        now = float(self.time_provider())
        _append_jsonl(
            self.output_dir / SESSION_RECEIPTS_FILE,
            {
                "event": event,
                "timestamp": now,
                "global_step": int(getattr(state, "global_step", 0) or 0),
                "epoch": getattr(state, "epoch", None),
                "resumed_checkpoint": self.resumed_checkpoint,
                "session_elapsed_seconds": round(now - self.started_at, 3),
                **extra,
            },
        )

    def _pause_file_exists(self) -> bool:
        return (self.output_dir / PAUSE_REQUEST_FILE).exists()

    def _consume_pause_file(self) -> None:
        path = self.output_dir / PAUSE_REQUEST_FILE
        if path.exists():
            path.unlink()

    def _session_expired(self) -> bool:
        if self.session_seconds is None:
            return False
        return float(self.time_provider()) - self.started_at >= self.session_seconds

    def _should_pause(self, state: Any) -> bool:
        if not self.enabled:
            return False
        step = int(getattr(state, "global_step", 0) or 0)
        if step >= self.total_steps:
            return False
        return self._pause_file_exists() or self._session_expired()

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._receipt("session_start", state)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self._should_pause(state):
            self.pause_requested = True
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not self.enabled:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        receipt = write_checkpoint_receipt(checkpoint_dir)
        self._receipt("checkpoint_saved", state, checkpoint=str(checkpoint_dir), receipt_global_step=receipt["global_step"])
        if self._should_pause(state) or self.pause_requested:
            self.pause_requested = True
            control.should_training_stop = True
            self._consume_pause_file()
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._receipt(
            "session_end",
            state,
            status="paused" if self.pause_requested else "completed",
        )
        return control
