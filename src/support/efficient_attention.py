from __future__ import annotations

from typing import Any


ATTENTION_IMPLEMENTATION = "sdpa_repeat_kv"


def sdpa_repeat_kv_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """SDPA attention that expands grouped KV before dispatch.

    PyTorch 2.6 on this Windows laptop dispatches Qwen3 GQA with enable_gqa=True
    to the slower math attention path. Repeating KV first preserves the
    attention result and lets SDPA select the efficient kernel.
    """
    import torch
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    num_key_value_groups = int(getattr(module, "num_key_value_groups", 1) or 1)
    if num_key_value_groups > 1:
        key = repeat_kv(key, num_key_value_groups)
        value = repeat_kv(value, num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = bool(is_causal.item())

    if getattr(query, "is_cuda", False):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        context = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    else:
        from contextlib import nullcontext

        context = nullcontext()

    with context:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
    return attn_output.transpose(1, 2).contiguous(), None


def register_sdpa_repeat_kv_attention() -> str:
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register(ATTENTION_IMPLEMENTATION, sdpa_repeat_kv_attention_forward)
    AttentionMaskInterface.register(ATTENTION_IMPLEMENTATION, AttentionMaskInterface._global_mapping["sdpa"])
    return ATTENTION_IMPLEMENTATION


def enable_memory_efficient_sdpa(model: Any) -> dict[str, Any]:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "qwen3":
        raise ValueError("memory efficient SDPA backend is verified only for Qwen3")
    implementation = register_sdpa_repeat_kv_attention()
    setter = getattr(model, "set_attn_implementation", None)
    if not callable(setter):
        raise TypeError("model does not support dynamic attention implementation")
    setter(implementation)
    return {
        "enabled": True,
        "attn_implementation": implementation,
        "reason": "repeat grouped KV before SDPA dispatch",
        "cuda_kernel_policy": "efficient_attention_only",
    }
