from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Protocol

from support.contracts import ModelCallUsage
from support.live_precision import LocalPrecisionRunner, PRECISION_SCOPE, normalize_inference_profile
from support.modeling import ModelRun
from support.prompting import build_messages


FULL_V8_ADAPTER_MODEL_SHA256 = "66307918a51f69fc0b1390e4e47e8346b5756cf0f6bcf69061305380c90e395d"
Mode = Literal["base", "fine_tuned"]


class RunnerLike(Protocol):
    adapter_id: str | None
    model: Any

    def load(self) -> "RunnerLike":
        ...

    def generate(
        self,
        model_input: dict[str, Any],
        request_id: str,
        node: str = "model_call",
        max_new_tokens: int = 512,
    ) -> ModelRun:
        ...

    def generate_chat(
        self,
        chat_messages: list[dict[str, str]],
        model_input: dict[str, Any],
        request_id: str,
        node: str = "model_call",
        max_new_tokens: int = 512,
    ) -> ModelRun:
        ...

    def close(self) -> None:
        ...


class DualModeRuntime:
    """Serve Base and FT calls through one loaded PEFT runner.

    The runtime keeps the full-v8 adapter loaded once. Base calls run inside
    PEFT's disable_adapter context and are serialized with FT calls so adapter
    state cannot overlap between concurrent requests.
    """

    def __init__(
        self,
        config: dict[str, Any],
        adapter_path: str | Path,
        *,
        precision: str = "nf4",
        runner_factory=None,
        expected_adapter_model_sha256: str = FULL_V8_ADAPTER_MODEL_SHA256,
    ):
        self._config = dict(config)
        self._adapter_path = Path(adapter_path)
        self._inference_profile = normalize_inference_profile(precision)
        self._runner_factory = runner_factory
        self._expected_adapter_model_sha256 = expected_adapter_model_sha256
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._runner: RunnerLike | None = None
        self._loaded = False
        self._busy = False
        self._active_mode: Mode | None = None

    def load(self) -> "DualModeRuntime":
        with self._lock:
            if self._loaded:
                return self
            self._validate_config()
            self._validate_adapter_files()
            runner = self._create_runner()
            try:
                runner.load()
                self._validate_loaded_peft_status(runner)
            except Exception:
                _close_quietly(runner)
                raise
            self._runner = runner
            self._set_status(loaded=True)
            return self

    def for_mode(self, mode: str) -> "ModeRunner":
        if mode not in {"base", "fine_tuned"}:
            raise ValueError("mode must be base or fine_tuned")
        return ModeRunner(self, mode)  # type: ignore[arg-type]

    def close(self) -> None:
        with self._lock:
            runner = self._runner
            self._runner = None
            self._set_status(loaded=False, busy=False, active_mode=None)
            if runner is not None:
                runner.close()

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "loaded": bool(self._loaded),
                "busy": bool(self._busy),
                "active_mode": self._active_mode,
                "inference_profile": self._inference_profile,
                "precision_scope": PRECISION_SCOPE,
                "serving_optimizations": self._serving_optimizations(),
            }

    def count_prompt_tokens(self, model_input: dict[str, Any]) -> int:
        with self._state_lock:
            loaded = bool(self._loaded)
        if self._runner is None or not loaded:
            raise RuntimeError("DualModeRuntime.load() must be called before count_prompt_tokens()")
        return self.count_chat_prompt_tokens(build_messages(model_input))

    def count_chat_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        with self._state_lock:
            loaded = bool(self._loaded)
        runner = self._runner
        if runner is None or not loaded:
            raise RuntimeError("DualModeRuntime.load() must be called before count_chat_prompt_tokens()")
        tokenizer = getattr(runner, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(runner, "_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("loaded runner does not expose tokenizer for prompt token counting")
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return _token_count(input_ids)

    def _set_status(
        self,
        *,
        loaded: bool | None = None,
        busy: bool | None = None,
        active_mode: Mode | None = None,
    ) -> None:
        with self._state_lock:
            if loaded is not None:
                self._loaded = loaded
            if busy is not None:
                self._busy = busy
            self._active_mode = active_mode

    def _serving_optimizations(self) -> dict[str, Any]:
        data = {"logits_to_keep": 1, "empty_cache_after_call": True}
        runner = self._runner
        runner_data = getattr(runner, "serving_optimizations", None)
        if isinstance(runner_data, dict):
            data.update(runner_data)
        return data

    def _clear_cuda_cache_after_call(self) -> None:
        torch = _maybe_import_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(
        self,
        mode: Mode,
        model_input: dict[str, Any],
        request_id: str,
        node: str = "model_call",
        max_new_tokens: int = 512,
        chat_messages: list[dict[str, str]] | None = None,
    ) -> ModelRun:
        with self._lock:
            if self._runner is None or not self.status()["loaded"]:
                raise RuntimeError("DualModeRuntime.load() must be called before generate()")
            self._set_status(busy=True, active_mode=mode)
            try:
                if mode == "base":
                    model = getattr(self._runner, "model", None)
                    disable_adapter = getattr(model, "disable_adapter", None)
                    if disable_adapter is None:
                        raise RuntimeError("loaded model does not expose PEFT disable_adapter()")
                    with disable_adapter():
                        run = self._call_runner(
                            model_input,
                            request_id,
                            node=node,
                            max_new_tokens=max_new_tokens,
                            chat_messages=chat_messages,
                        )
                    return _as_base_run(run)
                return self._call_runner(
                    model_input,
                    request_id,
                    node=node,
                    max_new_tokens=max_new_tokens,
                    chat_messages=chat_messages,
                )
            finally:
                try:
                    self._clear_cuda_cache_after_call()
                finally:
                    self._set_status(busy=False, active_mode=None)

    def _create_runner(self) -> RunnerLike:
        adapter_path = str(self._adapter_path)
        if self._runner_factory is not None:
            return self._runner_factory(self._config, adapter_path)
        return LocalPrecisionRunner(self._config, adapter_path, inference_profile=self._inference_profile)

    def _call_runner(
        self,
        model_input: dict[str, Any],
        request_id: str,
        *,
        node: str,
        max_new_tokens: int,
        chat_messages: list[dict[str, str]] | None,
    ) -> ModelRun:
        if self._runner is None:
            raise RuntimeError("DualModeRuntime.load() must be called before generate()")
        if chat_messages is None:
            return self._runner.generate(model_input, request_id, node=node, max_new_tokens=max_new_tokens)
        generate_chat = getattr(self._runner, "generate_chat", None)
        if generate_chat is None:
            raise RuntimeError("loaded runner does not support chat_messages")
        return generate_chat(chat_messages, model_input, request_id, node=node, max_new_tokens=max_new_tokens)

    def _validate_config(self) -> None:
        if self._config.get("key") != "qwen3_4b":
            raise ValueError("DualModeRuntime requires selected model key qwen3_4b")

    def _validate_adapter_files(self) -> None:
        config_path = self._adapter_path / "adapter_config.json"
        weights_path = self._adapter_path / "adapter_model.safetensors"
        if not config_path.is_file():
            raise ValueError(f"missing adapter_config.json: {config_path}")
        if not weights_path.is_file():
            raise ValueError(f"missing adapter_model.safetensors: {weights_path}")
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        _require_adapter_value(adapter_config, "peft_type", "LORA")
        _require_adapter_value(adapter_config, "bias", "none")
        _require_adapter_value(adapter_config, "lora_bias", False)
        _require_adapter_value(adapter_config, "modules_to_save", None)
        if adapter_config.get("merged_adapters"):
            raise ValueError("adapter_config.merged_adapters must be empty or absent")
        actual = _sha256_file(weights_path)
        if actual != self._expected_adapter_model_sha256:
            raise ValueError(
                "adapter_model.safetensors sha256 mismatch: "
                f"expected {self._expected_adapter_model_sha256}, got {actual}"
            )

    def _validate_loaded_peft_status(self, runner: RunnerLike) -> None:
        model = getattr(runner, "model", None)
        get_model_status = getattr(model, "get_model_status", None)
        if get_model_status is None:
            raise RuntimeError("PEFT adapter status is unavailable")
        status = get_model_status()
        enabled = getattr(status, "enabled", None)
        merged_adapters = getattr(status, "merged_adapters", [])
        if enabled is not True or merged_adapters:
            raise RuntimeError("PEFT adapter status must be enabled and unmerged")


class ModeRunner:
    def __init__(self, runtime: DualModeRuntime, mode: Mode):
        self._runtime = runtime
        self.mode = mode

    def generate(
        self,
        model_input: dict[str, Any],
        request_id: str,
        node: str = "model_call",
        max_new_tokens: int = 512,
        chat_messages: list[dict[str, str]] | None = None,
    ) -> ModelRun:
        return self._runtime._generate(
            self.mode,
            model_input,
            request_id,
            node=node,
            max_new_tokens=max_new_tokens,
            chat_messages=chat_messages,
        )


def _as_base_run(run: ModelRun) -> ModelRun:
    usage_data = run.usage.model_dump(mode="json")
    usage_data["mode"] = "base"
    usage_data["adapter_id"] = None
    usage = ModelCallUsage.model_validate(usage_data)
    return replace(run, usage=usage)


def _require_adapter_value(adapter_config: dict[str, Any], key: str, expected: Any) -> None:
    actual = adapter_config.get(key)
    if actual != expected:
        raise ValueError(f"adapter_config.{key} must be {expected!r}, got {actual!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maybe_import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _token_count(token_tensor) -> int:
    shape = getattr(token_tensor, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(token_tensor[0])


def _close_quietly(runner: RunnerLike) -> None:
    try:
        runner.close()
    except Exception:
        pass
