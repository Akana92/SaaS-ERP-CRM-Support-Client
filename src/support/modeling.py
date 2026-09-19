from __future__ import annotations

import gc
import json
import sys
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from support.contracts import ERPContext, ModelCallUsage, ModelResult, validate_evidence
from support.prompting import build_messages


CONTEXT_LIMIT_TOKENS = 8192


@dataclass(frozen=True)
class ModelRun:
    raw_text: str
    result: ModelResult | None
    usage: ModelCallUsage
    error: str | None
    peak_memory_allocated_bytes: int | None = None
    memory_reserved_bytes: int | None = None


def parse_model_result(raw_text: str, erp_context: ERPContext, policy_ids: set[str]) -> ModelResult:
    try:
        payload = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    try:
        result = ModelResult.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"schema validation failed: {exc}") from exc
    validate_evidence(result, erp_context, policy_ids)
    return result


class LocalModelRunner:
    def __init__(self, config: dict[str, Any], adapter_path: str | None = None):
        self.config = dict(config)
        self.model_id = _required_str(self.config, "model_id")
        self.revision = _required_str(self.config, "revision")
        self.local_path = _required_str(self.config, "local_path")
        self.key = _required_str(self.config, "key")
        if self.key not in {"qwen3_4b", "qwen25_15b"}:
            raise ValueError("config key must be qwen3_4b or qwen25_15b")
        if len(self.revision) != 40:
            raise ValueError("revision must be a 40-character commit hash")
        self.adapter_id = str(adapter_path) if adapter_path is not None else None
        self.mode = "fine_tuned" if self.adapter_id is not None else "base"
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._generation_lock = threading.Lock()
        self.load_seconds: float | None = None

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def load(self, torch_module=None) -> "LocalModelRunner":
        start = time.perf_counter()
        torch = torch_module if torch_module is not None else _import_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the local model runtime")
        dtype = torch.bfloat16 if _cuda_bf16_supported(torch) else torch.float16
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.local_path,
            revision=self.revision,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.local_path,
            revision=self.revision,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            quantization_config=quantization_config,
            dtype=dtype,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        )
        if self.adapter_id is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_id, local_files_only=True, is_trainable=False)
        self._model = model.eval()
        self._loaded = True
        self.load_seconds = time.perf_counter() - start
        return self

    def generate(
        self,
        model_input: dict[str, Any],
        request_id: str,
        node: str = "model_call",
        max_new_tokens: int = 512,
    ) -> ModelRun:
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("LocalModelRunner.load() must be called before generate()")
        call_id = str(uuid.uuid4())
        try:
            messages = build_messages(model_input)
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception as exc:
            usage = self._usage(
                call_id=call_id,
                request_id=request_id,
                node=node,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                latency_ms=None,
                complete=False,
            )
            return ModelRun(raw_text="", result=None, usage=usage, error=str(exc))
        input_token_count = _token_count(input_ids)
        if input_token_count + max_new_tokens > CONTEXT_LIMIT_TOKENS:
            usage = self._usage(
                call_id=call_id,
                request_id=request_id,
                node=node,
                input_tokens=input_token_count,
                output_tokens=None,
                total_tokens=None,
                latency_ms=None,
                complete=False,
            )
            return ModelRun(raw_text="", result=None, usage=usage, error="context limit exceeded")
        try:
            input_ids = input_ids.to(self._model.device)
        except Exception as exc:
            usage = self._usage(
                call_id=call_id,
                request_id=request_id,
                node=node,
                input_tokens=input_token_count,
                output_tokens=None,
                total_tokens=None,
                latency_ms=None,
                complete=False,
            )
            return ModelRun(raw_text="", result=None, usage=usage, error=str(exc))
        with self._generation_lock:
            torch = _maybe_import_torch()
            _cuda_reset_peak(torch)
            _cuda_synchronize(torch)
            started = time.perf_counter()
            try:
                with _inference_context(torch):
                    output_ids = self._model.generate(
                        input_ids,
                        attention_mask=input_ids.new_ones(input_ids.shape),
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
                        eos_token_id=getattr(self._tokenizer, "eos_token_id", None),
                    )
            except Exception as exc:  # runtime errors after prompt construction must keep known input usage
                _cuda_synchronize(torch)
                usage = self._usage(
                    call_id=call_id,
                    request_id=request_id,
                    node=node,
                    input_tokens=input_token_count,
                    output_tokens=None,
                    total_tokens=None,
                    latency_ms=None,
                    complete=False,
                )
                return ModelRun(raw_text="", result=None, usage=usage, error=str(exc), **_memory_snapshot(torch))
            _cuda_synchronize(torch)
            latency_ms = max(0, round((time.perf_counter() - started) * 1000))
            memory_stats = _memory_snapshot(torch)
        suffix_ids = output_ids[:, input_token_count:]
        output_token_count = _token_count(suffix_ids)
        raw_text = self._tokenizer.decode(suffix_ids[0], skip_special_tokens=False)
        validation_text = self._tokenizer.decode(
            _without_terminal_stop_token(suffix_ids[0], self._tokenizer, self._model),
            skip_special_tokens=False,
        )
        usage = self._usage(
            call_id=call_id,
            request_id=request_id,
            node=node,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            total_tokens=input_token_count + output_token_count,
            latency_ms=latency_ms,
            complete=True,
        )
        try:
            erp_context = ERPContext.model_validate(model_input["erp_context"])
            policy_ids = {rule["id"] for rule in model_input["policy_rules"]}
            result = parse_model_result(validation_text, erp_context, policy_ids)
        except Exception as exc:
            return ModelRun(raw_text=raw_text, result=None, usage=usage, error=str(exc), **memory_stats)
        return ModelRun(raw_text=raw_text, result=result, usage=usage, error=None, **memory_stats)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        torch = _maybe_import_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def _usage(
        self,
        *,
        call_id: str,
        request_id: str,
        node: str,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        latency_ms: int | None,
        complete: bool,
    ) -> ModelCallUsage:
        return ModelCallUsage(
            call_id=call_id,
            request_id=request_id,
            node=node,
            model_id=self.model_id,
            revision=self.revision,
            adapter_id=self.adapter_id,
            mode=self.mode,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            api_cost=0.0 if complete else None,
            currency="USD",
            complete=complete,
        )


def _required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.{key} must be a non-empty string")
    return value


def _token_count(token_tensor) -> int:
    shape = getattr(token_tensor, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(token_tensor[0])


def _without_terminal_stop_token(token_ids, tokenizer, model):
    ids = _token_id_list(token_ids)
    stop_ids = _stop_token_ids(tokenizer, model)
    if ids and ids[-1] in stop_ids:
        return ids[:-1]
    return token_ids


def _token_id_list(token_ids) -> list[int]:
    if hasattr(token_ids, "tolist"):
        return [int(item) for item in token_ids.tolist()]
    return [int(item) for item in token_ids]


def _stop_token_ids(tokenizer, model) -> set[int]:
    values = []
    for owner in (tokenizer, getattr(tokenizer, "generation_config", None), getattr(model, "generation_config", None)):
        if owner is not None and hasattr(owner, "eos_token_id"):
            values.append(getattr(owner, "eos_token_id"))
    stop_ids = set()
    for value in values:
        if isinstance(value, int):
            stop_ids.add(value)
        elif isinstance(value, (list, tuple, set)):
            stop_ids.update(int(item) for item in value if isinstance(item, int))
    return stop_ids


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for LocalModelRunner.load()") from exc
    return torch


def _maybe_import_torch():
    if "torch" not in sys.modules:
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda_bf16_supported(torch) -> bool:
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(probe and probe())


def _cuda_synchronize(torch) -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_reset_peak(torch) -> None:
    if torch is not None and torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()


def _inference_context(torch):
    if torch is None or not hasattr(torch, "inference_mode"):
        return nullcontext()
    return torch.inference_mode()


def _memory_snapshot(torch) -> dict[str, int | None]:
    if torch is None or not torch.cuda.is_available():
        return {"peak_memory_allocated_bytes": None, "memory_reserved_bytes": None}
    return {
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
    }
