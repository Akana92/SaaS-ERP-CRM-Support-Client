from __future__ import annotations

import argparse
import gc
import io
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .contracts import DevCase, ModelResult, PolicyDocument, model_input_from_case

ChatMessage = dict[str, str]
BuildMessages = Callable[[dict[str, Any]], list[ChatMessage]]


@dataclass(frozen=True)
class TrainingExample:
    case_id: str
    family_id: str
    review_status: str
    messages: list[ChatMessage]
    target_json: str
    model_input_sha256: str
    target_sha256: str


@dataclass(frozen=True)
class MaskProbe:
    case_id: str
    prompt_token_count: int
    supervised_token_count: int
    total_tokens: int
    ignored_label_count: int
    target_sha256: str
    decoded_target: str
    supervised_token_ids: list[int]
    terminal_eos_token_id: int
    label_sha256: str


@dataclass(frozen=True)
class EncodedExample:
    case_id: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    mask_probe: MaskProbe

    def to_features(self) -> dict[str, list[int] | str]:
        return {
            "case_id": self.case_id,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }


@dataclass(frozen=True)
class LengthItem:
    case_id: str
    prompt_token_count: int
    supervised_token_count: int
    total_tokens: int


@dataclass(frozen=True)
class LengthReport:
    example_count: int
    max_prompt_tokens: int
    max_supervised_tokens: int
    max_total_tokens: int
    max_length: int
    items: list[LengthItem]


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_document(path: Path) -> PolicyDocument:
    return PolicyDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_development_cases(path: Path, sample_limit: int | None = None) -> list[DevCase]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [DevCase.model_validate(row) for row in rows]
    model_cases = [case for case in cases if case.expected_model_call]
    if sample_limit is not None:
        model_cases = model_cases[:sample_limit]
    if not model_cases:
        raise ValueError("no model-call development cases found")
    return model_cases


def model_result_target_json(result: ModelResult) -> str:
    return stable_json_dumps(result.model_dump(mode="json"))


def default_build_messages(model_input: dict[str, Any]) -> list[ChatMessage]:
    from support.prompting import build_messages

    return build_messages(model_input)


def build_training_example(
    case: DevCase,
    policy_document: PolicyDocument,
    build_messages: BuildMessages | None = None,
) -> TrainingExample:
    if case.expected is None:
        raise ValueError(f"{case.id} has no assistant target")
    builder = build_messages or default_build_messages
    model_input = model_input_from_case(case, policy_document)
    messages = builder(model_input)
    target_json = model_result_target_json(case.expected)
    return TrainingExample(
        case_id=case.id,
        family_id=case.family_id,
        review_status=case.review_status,
        messages=messages,
        target_json=target_json,
        model_input_sha256=stable_json_sha256(model_input),
        target_sha256=stable_json_sha256(case.expected.model_dump(mode="json")),
    )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _eos_token_ids(tokenizer: Any) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id for completion-only EOS supervision")
    if isinstance(eos_token_id, int):
        return {int(eos_token_id)}
    return {int(token_id) for token_id in eos_token_id}


def _render_prompt_and_full(tokenizer: Any, messages: Sequence[ChatMessage], target_json: str) -> tuple[str, str]:
    prompt_text = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": target_json}],
        tokenize=False,
    )
    if not isinstance(prompt_text, str) or not isinstance(full_text, str):
        raise TypeError("tokenizer chat template must render text when tokenize=False")
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template did not preserve prompt prefix for completion-only labels")
    return prompt_text, full_text


def encode_completion_only(
    tokenizer: Any,
    messages: Sequence[ChatMessage],
    target_json: str,
    *,
    case_id: str = "adhoc",
    max_length: int = 8192,
) -> EncodedExample:
    prompt_text, full_text = _render_prompt_and_full(tokenizer, messages, target_json)
    prompt_ids = _token_ids(tokenizer, prompt_text)
    full_ids = _token_ids(tokenizer, full_text)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenized prompt is not aligned with tokenized full sample")
    if len(full_ids) > max_length:
        raise ValueError(f"{case_id} exceeds max_length {max_length}: {len(full_ids)} tokens")
    completion_ids = full_ids[len(prompt_ids) :]
    target_ids = _token_ids(tokenizer, target_json)
    if completion_ids[: len(target_ids)] != target_ids:
        raise ValueError(f"{case_id} target JSON is not aligned at the start of the assistant completion")
    if not target_ids:
        raise ValueError(f"{case_id} has no supervised target tokens")
    suffix_ids = completion_ids[len(target_ids) :]
    eos_token_ids = _eos_token_ids(tokenizer)
    eos_suffix_index = next((index for index, token_id in enumerate(suffix_ids) if token_id in eos_token_ids), None)
    if eos_suffix_index is None:
        raise ValueError(f"{case_id} assistant completion has no EOS token after target JSON")
    terminal_eos_token_id = suffix_ids[eos_suffix_index]
    supervised_token_ids = [*target_ids, terminal_eos_token_id]
    suffix_after_eos = len(suffix_ids) - eos_suffix_index - 1
    labels = (
        [-100] * len(prompt_ids)
        + target_ids
        + [-100] * eos_suffix_index
        + [terminal_eos_token_id]
        + [-100] * suffix_after_eos
    )
    decoded_target = tokenizer.decode(supervised_token_ids, skip_special_tokens=True)
    if decoded_target != target_json:
        raise ValueError(f"{case_id} decoded supervised target does not match target JSON exactly")
    return EncodedExample(
        case_id=case_id,
        input_ids=full_ids,
        attention_mask=[1] * len(full_ids),
        labels=labels,
        mask_probe=MaskProbe(
            case_id=case_id,
            prompt_token_count=len(prompt_ids),
            supervised_token_count=len(supervised_token_ids),
            total_tokens=len(full_ids),
            ignored_label_count=labels.count(-100),
            target_sha256=stable_json_sha256(json.loads(target_json)),
            decoded_target=decoded_target,
            supervised_token_ids=supervised_token_ids,
            terminal_eos_token_id=terminal_eos_token_id,
            label_sha256=stable_json_sha256(labels),
        ),
    )


def measure_token_lengths(
    tokenizer: Any,
    examples: Sequence[TrainingExample],
    *,
    max_length: int = 8192,
) -> tuple[list[EncodedExample], LengthReport]:
    encoded = [
        encode_completion_only(
            tokenizer,
            example.messages,
            example.target_json,
            case_id=example.case_id,
            max_length=max_length,
        )
        for example in examples
    ]
    items = [
        LengthItem(
            case_id=item.case_id,
            prompt_token_count=item.mask_probe.prompt_token_count,
            supervised_token_count=item.mask_probe.supervised_token_count,
            total_tokens=item.mask_probe.total_tokens,
        )
        for item in encoded
    ]
    return encoded, LengthReport(
        example_count=len(items),
        max_prompt_tokens=max(item.prompt_token_count for item in items),
        max_supervised_tokens=max(item.supervised_token_count for item in items),
        max_total_tokens=max(item.total_tokens for item in items),
        max_length=max_length,
        items=items,
    )


class CompletionOnlyCollator:
    def __init__(
        self,
        tokenizer: Any,
        label_pad_token_id: int = -100,
        *,
        return_tensors: bool = True,
    ) -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", 0)
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append([*feature["input_ids"], *([self.pad_token_id] * pad_len)])
            attention_mask.append([*feature["attention_mask"], *([0] * pad_len)])
            labels.append([*feature["labels"], *([self.label_pad_token_id] * pad_len)])
        if not self.return_tensors:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        import torch

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_model_config(config_path: Path, model_key: str | None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = config.get("candidates", [])
    selected_key = model_key or config.get("selected_key")
    if selected_key is None:
        if len(candidates) != 1:
            raise ValueError("--model-key is required when selected_key is null and multiple candidates exist")
        selected_key = candidates[0]["key"]
    for candidate in candidates:
        if candidate.get("key") == selected_key:
            return candidate
    raise ValueError(f"unknown model key: {selected_key}")


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_transformers_load_kwargs(
    *,
    revision: str,
    dtype: Any,
    quantization_config: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tokenizer_kwargs = {
        "revision": revision,
        "local_files_only": True,
        "trust_remote_code": False,
        "use_fast": True,
    }
    model_kwargs = {
        "revision": revision,
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "quantization_config": quantization_config,
        "torch_dtype": dtype,
        "device_map": {"": "cuda:0"},
        "attn_implementation": "sdpa",
    }
    return tokenizer_kwargs, model_kwargs


def build_sft_config_kwargs(
    *,
    output_dir: str,
    steps: int,
    max_length: int,
    bf16: bool,
) -> dict[str, Any]:
    return {
        "output_dir": output_dir,
        "max_steps": steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "optim": "adamw_torch",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "fp16": not bf16,
        "bf16": bf16,
        "seed": 42,
        "packing": False,
        "max_length": max_length,
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }


def _as_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _batch_rows(value: Any) -> list[list[int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[int(item) for item in row] for row in value]


def verify_batch_labels_against_probes(
    batch: dict[str, Any],
    tokenizer: Any,
    probes: Sequence[MaskProbe],
) -> list[dict[str, Any]]:
    expected_by_label_hash: dict[str, list[MaskProbe]] = {}
    for probe in probes:
        expected_by_label_hash.setdefault(probe.label_sha256, []).append(probe)
    inspected: list[dict[str, Any]] = []
    label_rows = _batch_rows(batch["labels"])
    if "attention_mask" in batch:
        attention_rows = _batch_rows(batch["attention_mask"])
    else:
        attention_rows = [[1] * len(labels) for labels in label_rows]
    for labels, attention_mask in zip(label_rows, attention_rows):
        active_labels = [label for label, attention in zip(labels, attention_mask) if attention]
        label_sha256 = stable_json_sha256(active_labels)
        matching_probes = expected_by_label_hash.get(label_sha256, [])
        if not matching_probes:
            raise ValueError("trainer batch label mask not found in mask probes")
        probe = matching_probes.pop(0)
        supervised_ids = [token_id for token_id in active_labels if token_id != -100]
        if not supervised_ids:
            raise ValueError("trainer batch contains no supervised labels")
        if supervised_ids != probe.supervised_token_ids:
            raise ValueError("trainer batch supervised token ids do not match mask probe")
        decoded = tokenizer.decode(supervised_ids, skip_special_tokens=True)
        if decoded != probe.decoded_target:
            raise ValueError("trainer batch decoded supervised target does not match mask probe")
        if len(supervised_ids) != probe.supervised_token_count:
            raise ValueError("trainer batch supervised token count does not match mask probe")
        inspected.append(
            {
                "case_id": probe.case_id,
                "supervised_token_count": len(supervised_ids),
                "terminal_eos_token_id": probe.terminal_eos_token_id,
                "target_sha256": probe.target_sha256,
                "label_sha256": probe.label_sha256,
            }
        )
    return inspected


def verify_trainer_batches_against_probes(
    trainer: Any,
    tokenizer: Any,
    probes: Sequence[MaskProbe],
) -> list[dict[str, Any]]:
    expected_case_ids = {probe.case_id for probe in probes}
    inspected: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for batch in trainer.get_train_dataloader():
        batch_inspected = verify_batch_labels_against_probes(batch, tokenizer, probes)
        inspected.extend(batch_inspected)
        seen_case_ids.update(item["case_id"] for item in batch_inspected)
        if seen_case_ids == expected_case_ids:
            break
    missing = sorted(expected_case_ids - seen_case_ids)
    if missing:
        raise ValueError(f"missing inspected trainer cases: {', '.join(missing)}")
    return inspected


def verify_generation_check_fields(
    *,
    generation_complete: bool,
    json_schema_evidence_valid: bool,
    controlled_failure: bool,
    adapter_tensors_match: bool,
    error: str | None,
) -> dict[str, Any]:
    if not adapter_tensors_match:
        raise ValueError("adapter tensors did not match after reload")
    if json_schema_evidence_valid and not generation_complete:
        raise ValueError("generation evidence cannot be schema-valid when generation is incomplete")
    if not generation_complete and not controlled_failure:
        raise ValueError("generation evidence must be complete or a controlled failure")
    if controlled_failure and not error:
        raise ValueError("controlled failure requires an error")
    return {
        "generation_complete": generation_complete,
        "json_schema_evidence_valid": json_schema_evidence_valid,
        "controlled_failure": controlled_failure,
        "adapter_tensors_match": adapter_tensors_match,
        "error": error,
    }


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def cuda_memory_snapshot(torch: Any) -> dict[str, int | None]:
    if torch is None or not torch.cuda.is_available():
        return {"peak_cuda_memory_bytes": None, "reserved_cuda_memory_bytes": None}
    return {
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_cuda_memory_bytes": int(torch.cuda.memory_reserved()),
    }


def build_reload_generation_record(
    *,
    model_input: dict[str, Any],
    generated: Any,
    adapter_tensors_match: bool,
    trainable_counts: dict[str, int] | None,
    trainable_digest: str | None,
    reload_peak_cuda_memory_bytes: int | None,
    reload_reserved_cuda_memory_bytes: int | None,
) -> dict[str, Any]:
    result = getattr(generated, "result", None)
    usage = getattr(generated, "usage", None)
    error = getattr(generated, "error", None)
    evidence = verify_generation_check_fields(
        generation_complete=bool(getattr(usage, "complete", False)),
        json_schema_evidence_valid=result is not None,
        controlled_failure=error is not None,
        adapter_tensors_match=adapter_tensors_match,
        error=error,
    )
    return {
        "adapter_reload": True,
        "trainable_counts": trainable_counts,
        "trainable_digest": trainable_digest,
        "model_input_sha256": stable_json_sha256(model_input),
        "raw_text": getattr(generated, "raw_text", None),
        "result": _jsonable(result),
        "usage": _jsonable(usage),
        "reload_peak_cuda_memory_bytes": reload_peak_cuda_memory_bytes,
        "reload_reserved_cuda_memory_bytes": reload_reserved_cuda_memory_bytes,
        **evidence,
    }


def is_adapter_parameter_name(name: str) -> bool:
    lowered = name.lower()
    return "lora_" in lowered or ".lora" in lowered or "modules_to_save" in lowered


def verify_lora_only_trainable(model: Any) -> dict[str, Any]:
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    ]
    if not trainable_names:
        raise ValueError("no trainable LoRA parameters found")
    unexpected = [name for name in trainable_names if not is_adapter_parameter_name(name)]
    if unexpected:
        raise ValueError(f"non-LoRA trainable parameters found: {', '.join(unexpected)}")
    return {
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": len(trainable_names),
    }


def adapter_tensor_snapshot(model: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if not is_adapter_parameter_name(name):
            continue
        tensor = parameter.detach().float().cpu().clone()
        snapshot[name] = tensor
    if not snapshot:
        raise ValueError("no adapter tensors found")
    return snapshot


def verify_adapter_tensors_equal(expected: dict[str, Any], model: Any) -> None:
    actual = adapter_tensor_snapshot(model)
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"missing reloaded adapter tensors: {', '.join(missing)}")
    extra = sorted(set(actual) - set(expected))
    if extra:
        raise ValueError(f"unexpected reloaded adapter tensors: {', '.join(extra)}")
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        if hasattr(actual_tensor, "equal"):
            equal = bool(actual_tensor.equal(expected_tensor))
        else:
            equal = actual_tensor == expected_tensor
        if not equal:
            raise ValueError(f"reloaded adapter tensor mismatch: {name}")


def trainable_parameter_digest(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        buffer = io.BytesIO()
        torch.save(parameter.detach().float().cpu(), buffer)
        digest.update(buffer.getvalue())
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return digest.hexdigest()


def trainable_parameter_counts(model: Any) -> dict[str, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"trainable": trainable, "total": total}


def require_cuda_dtype(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 2 QLoRA smoke training")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def release_gpu_memory(torch: Any) -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def reset_cuda_peak_memory(torch: Any) -> None:
    if torch is not None and torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()


def run_smoke_training(args: argparse.Namespace) -> dict[str, Any]:
    from support.prompting import PROMPT_HASH, PROMPT_VERSION
    import torch

    dtype = require_cuda_dtype(torch)
    reset_cuda_peak_memory(torch)

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    set_seed(42)

    started = time.time()
    project_root = Path(args.project_root)
    cases_path = project_root / args.cases
    policy_path = project_root / args.policy
    output_dir = project_root / args.output
    report_dir = project_root / args.report_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_config = load_model_config(project_root / args.config, args.model_key)
    policy_document = load_policy_document(policy_path)
    cases = load_development_cases(cases_path, sample_limit=args.sample_limit)
    examples = [build_training_example(case, policy_document) for case in cases]

    model_ref = model_config.get("local_path") or model_config["model_id"]
    tokenizer_kwargs, model_kwargs = build_transformers_load_kwargs(
        revision=model_config["revision"],
        dtype=dtype,
        quantization_config=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_ref, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded, length_report = measure_token_lengths(tokenizer, examples, max_length=args.max_length)
    features = [item.to_features() for item in encoded]
    mask_probe = [asdict(item.mask_probe) for item in encoded]
    (report_dir / "mask_probe.json").write_text(
        json.dumps(mask_probe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    _, model_kwargs = build_transformers_load_kwargs(
        revision=model_config["revision"],
        dtype=dtype,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    lora_trainability = verify_lora_only_trainable(model)
    counts_before = trainable_parameter_counts(model)
    digest_before = trainable_parameter_digest(model)

    dataset = Dataset.from_list(features)
    collator = CompletionOnlyCollator(tokenizer)
    bf16 = dtype == torch.bfloat16
    training_args = TrainingArguments(
        output_dir=str(output_dir / "_trainer"),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available() and not bf16,
        bf16=bf16,
        seed=42,
    )
    trainer_kind = "transformers.Trainer"
    trainer_decision = (
        "Using pretokenized completion-only labels with Transformers Trainer because the smoke "
        "path must preserve inspected labels exactly after chat-template rendering."
    )
    try:
        from trl import SFTConfig, SFTTrainer

        sft_args = SFTConfig(**build_sft_config_kwargs(
            output_dir=str(output_dir / "_trainer"),
            steps=args.steps,
            max_length=args.max_length,
            bf16=bf16,
        ))
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )
        trainer_kind = "trl.SFTTrainer"
        trainer_decision = "TRL SFTTrainer accepted the pretokenized completion-only dataset and collator."
    except Exception as exc:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )
        trainer_decision += f" TRL SFTTrainer construction was blocked: {type(exc).__name__}: {exc}"

    batch_label_probe = verify_trainer_batches_against_probes(
        trainer,
        tokenizer,
        [item.mask_probe for item in encoded],
    )
    train_result = trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    if not losses or not all(loss == loss and loss != float("inf") and loss != float("-inf") for loss in losses):
        raise RuntimeError("training did not report finite loss values")
    optimizer_steps = int(trainer.state.global_step)
    if optimizer_steps != args.steps:
        raise RuntimeError(f"training completed {optimizer_steps} optimizer steps, expected {args.steps}")
    digest_after = trainable_parameter_digest(model)
    if digest_after == digest_before:
        raise RuntimeError("trainable LoRA parameter digest did not change")
    final_adapter_tensors = adapter_tensor_snapshot(model)
    training_cuda_memory = cuda_memory_snapshot(torch)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    del trainer
    del model
    release_gpu_memory(torch)

    generation_check = _adapter_reload_generation_check(
        config=model_config,
        adapter_path=output_dir,
        model_input=model_input_from_case(cases[0], policy_document),
        expected_adapter_tensors=final_adapter_tensors,
    )

    report = {
        "purpose": "stage2_qlora_smoke",
        "artifact_status": "smoke_not_final",
        "model_config": model_config,
        "prompt_version": PROMPT_VERSION,
        "prompt_template_hash": PROMPT_HASH,
        "rendered_messages_sha256": {example.case_id: stable_json_sha256(example.messages) for example in examples},
        "source_sha256": {name: file_sha256(project_root / "src/support" / name)
                          for name in ["prompting.py", "modeling.py", "training.py"]},
        "seed": 42,
        "steps_requested": args.steps,
        "optimizer_steps": optimizer_steps,
        "losses": losses,
        "train_loss": getattr(train_result, "training_loss", None),
        "trainable_parameter_counts": counts_before,
        "lora_trainability": lora_trainability,
        "trainable_digest_before": digest_before,
        "trainable_digest_after": digest_after,
        "length_report": asdict(length_report),
        "batch_label_probe": batch_label_probe,
        "data": {
            "cases_path": str(cases_path),
            "cases_sha256": file_sha256(cases_path),
            "policy_path": str(policy_path),
            "policy_sha256": file_sha256(policy_path),
            "example_count": len(examples),
            "case_ids": [example.case_id for example in examples],
            "review_statuses": sorted({example.review_status for example in examples}),
        },
        "versions": package_versions(["torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"]),
        "trainer_kind": trainer_kind,
        "trainer_decision": trainer_decision,
        "generation_check": generation_check,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "training_peak_cuda_memory_bytes": training_cuda_memory["peak_cuda_memory_bytes"],
            "training_reserved_cuda_memory_bytes": training_cuda_memory["reserved_cuda_memory_bytes"],
            "reload_peak_cuda_memory_bytes": generation_check["reload_peak_cuda_memory_bytes"],
            "reload_reserved_cuda_memory_bytes": generation_check["reload_reserved_cuda_memory_bytes"],
            "duration_seconds": round(time.time() - started, 3),
        },
    }
    manifest = {
        "artifact_status": "smoke_not_final",
        "adapter_path": str(output_dir),
        "base_model_id": model_config["model_id"],
        "base_revision": model_config.get("revision"),
        "adapter_digest_after_training": digest_after,
        "report_path": str(report_dir / "training_report.json"),
        "future_base_bench_rule": "Use a fresh base model load without this adapter for future Base benchmarks.",
    }
    (report_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "adapter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "adapter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _adapter_reload_generation_check(
    *,
    config: dict[str, Any],
    adapter_path: Path,
    model_input: dict[str, Any],
    expected_adapter_tensors: dict[str, Any],
) -> dict[str, Any]:
    from support.modeling import LocalModelRunner

    runner = LocalModelRunner(config, adapter_path=str(adapter_path)).load()
    try:
        counts = None
        digest = None
        adapter_tensors_match = False
        if getattr(runner, "model", None) is not None:
            counts = trainable_parameter_counts(runner.model)
            digest = trainable_parameter_digest(runner.model)
            verify_adapter_tensors_equal(expected_adapter_tensors, runner.model)
            adapter_tensors_match = True
        generated = runner.generate(
            model_input,
            request_id="stage2-smoke-reload",
            node="model_call",
            max_new_tokens=512,
        )
        return build_reload_generation_record(
            model_input=model_input,
            generated=generated,
            adapter_tensors_match=adapter_tensors_match,
            trainable_counts=counts,
            trainable_digest=digest,
            reload_peak_cuda_memory_bytes=getattr(generated, "peak_memory_allocated_bytes", None),
            reload_reserved_cuda_memory_bytes=getattr(generated, "memory_reserved_bytes", None),
        )
    finally:
        runner.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded Stage 2 QLoRA smoke training.")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--config", default="configs/models.json")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--cases", default="data/development/cases.jsonl")
    parser.add_argument("--policy", default="data/policy/demo-v1.json")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", default="adapters/stage2-smoke")
    parser.add_argument("--report-dir", default="artifacts/stage2/training")
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_smoke_training(args)
    print(json.dumps({"report": str(Path(args.project_root) / args.report_dir / "training_report.json"), "summary": report}, ensure_ascii=False, indent=2))
    return 0
