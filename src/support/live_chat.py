from __future__ import annotations

import time
import uuid
from typing import Any

from support.contracts import ERPContext
from support.modeling import (
    CONTEXT_LIMIT_TOKENS,
    ModelRun,
    _cuda_reset_peak,
    _cuda_synchronize,
    _inference_context,
    _maybe_import_torch,
    _memory_snapshot,
    _token_count,
    _without_terminal_stop_token,
    parse_model_result,
)


def generate_chat_for_runner(
    runner,
    chat_messages: list[dict[str, str]],
    model_input: dict[str, Any],
    request_id: str,
    *,
    node: str = "model_call",
    max_new_tokens: int = 512,
) -> ModelRun:
    """Generate from already-built chat messages using an already-loaded LocalModelRunner."""

    if not getattr(runner, "_loaded", False) or runner.model is None or runner.tokenizer is None:
        raise RuntimeError("LocalModelRunner.load() must be called before generate_chat()")
    call_id = str(uuid.uuid4())
    try:
        input_ids = runner.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception as exc:
        usage = runner._usage(
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
        usage = runner._usage(
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
        input_ids = input_ids.to(runner.model.device)
    except Exception as exc:
        usage = runner._usage(
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
    with runner._generation_lock:
        torch = _maybe_import_torch()
        _cuda_reset_peak(torch)
        _cuda_synchronize(torch)
        started = time.perf_counter()
        try:
            with _inference_context(torch):
                output_ids = runner.model.generate(
                    input_ids,
                    attention_mask=input_ids.new_ones(input_ids.shape),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(runner.tokenizer, "eos_token_id", None),
                    eos_token_id=getattr(runner.tokenizer, "eos_token_id", None),
                    logits_to_keep=1,
                )
        except Exception as exc:
            _cuda_synchronize(torch)
            usage = runner._usage(
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
    raw_text = runner.tokenizer.decode(suffix_ids[0], skip_special_tokens=False)
    validation_text = runner.tokenizer.decode(
        _without_terminal_stop_token(suffix_ids[0], runner.tokenizer, runner.model),
        skip_special_tokens=False,
    )
    usage = runner._usage(
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
        return ModelRun(raw_text=raw_text, result=None, usage=usage, error="model_output_invalid: " + str(exc), **memory_stats)
    return ModelRun(raw_text=raw_text, result=result, usage=usage, error=None, **memory_stats)
