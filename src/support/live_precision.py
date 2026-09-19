from __future__ import annotations

import functools
import time
from typing import Any, Literal

from support.live_chat import generate_chat_for_runner
from support.modeling import LocalModelRunner


InferenceProfile = Literal["nf4", "bf16", "fp16"]
PRECISION_SCOPE = "serving_only_experiment"
SUPPORTED_INFERENCE_PROFILES = {"nf4", "bf16", "fp16"}
SUPPORTED_ATTENTION_BACKENDS = {"sdpa", "sdpa_repeat_kv"}


def normalize_attention_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError("live attention backend must be one of: sdpa, sdpa_repeat_kv")
    return backend


def normalize_inference_profile(value: str) -> InferenceProfile:
    profile = str(value).strip().lower()
    if profile not in SUPPORTED_INFERENCE_PROFILES:
        raise ValueError("inference profile must be one of: nf4, bf16, fp16")
    return profile  # type: ignore[return-value]


class LocalPrecisionRunner(LocalModelRunner):
    """Serving-only precision variants for the live demo runtime.

    The frozen LocalModelRunner.generate/parser/usage path is inherited. Only
    model loading changes so root can measure NF4 vs full-precision serving in
    separate one-profile processes.
    """

    precision_scope = PRECISION_SCOPE

    def __init__(self, config: dict[str, Any], adapter_path: str | None = None, *, inference_profile: str = "nf4"):
        self.inference_profile = normalize_inference_profile(inference_profile)
        self.attention_backend = normalize_attention_backend(config.get("live_attention_backend", "sdpa"))
        self.serving_optimizations = {"logits_to_keep": 1, "attn_implementation": "sdpa"}
        super().__init__(config, adapter_path=adapter_path)

    def load(self, torch_module=None) -> "LocalPrecisionRunner":
        # Each load creates a new SDPA model, including after close().
        # Applied-backend metadata belongs to that model, not to the runner lifetime.
        self.serving_optimizations = {"logits_to_keep": 1, "attn_implementation": "sdpa"}
        if self.inference_profile == "nf4":
            super().load(torch_module=torch_module)
            self.enable_serving_optimizations()
            return self

        start = time.perf_counter()
        torch = torch_module if torch_module is not None else _import_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the local model runtime")
        if self.inference_profile == "bf16" and not _cuda_bf16_supported(torch):
            raise RuntimeError("BF16 inference profile requires CUDA BF16 support")
        dtype = torch.bfloat16 if self.inference_profile == "bf16" else torch.float16
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.enable_serving_optimizations()
        return self

    def enable_serving_optimizations(self) -> None:
        if self._model is None:
            return
        if (self.attention_backend == "sdpa_repeat_kv"
                and self.serving_optimizations["attn_implementation"] != self.attention_backend):
            from support.efficient_attention import enable_memory_efficient_sdpa

            attention = enable_memory_efficient_sdpa(self._model)
            self.serving_optimizations.update(attention)
        generate = getattr(self._model, "generate", None)
        if generate is None or getattr(generate, "_capstone_logits_to_keep", False):
            return

        @functools.wraps(generate)
        def generate_with_logits_to_keep(*args, **kwargs):
            kwargs.setdefault("logits_to_keep", 1)
            return generate(*args, **kwargs)

        generate_with_logits_to_keep._capstone_logits_to_keep = True
        self._model.generate = generate_with_logits_to_keep

    def generate_chat(
        self,
        chat_messages: list[dict[str, str]],
        model_input: dict[str, Any],
        request_id: str,
        node: str = "model_call",
        max_new_tokens: int = 512,
    ):
        return generate_chat_for_runner(
            self,
            chat_messages,
            model_input,
            request_id,
            node=node,
            max_new_tokens=max_new_tokens,
        )


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for LocalPrecisionRunner.load()") from exc
    return torch


def _cuda_bf16_supported(torch) -> bool:
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(probe and probe())
