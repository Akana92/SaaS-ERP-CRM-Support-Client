from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from support.app import DEFAULT_MESSAGES, load_model_config, scenario_context
from support.graph import invoke_support_graph
from support.live_precision import SUPPORTED_INFERENCE_PROFILES, normalize_inference_profile
from support.live_runtime import DualModeRuntime, FULL_V8_ADAPTER_MODEL_SHA256, _sha256_file
from support.prompting import PROMPT_HASH, PROMPT_VERSION, load_policy
from training_control import assert_no_unmanaged_training, gpu_lock


SCHEMA_VERSION = "live-runtime-benchmark-v1"
DEFAULT_SCENARIOS = ["payment", "access", "integration"]


def _inside_project(path: str | Path) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("benchmark paths must stay inside this project")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _prepare_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"benchmark output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _sync_cuda(torch) -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak(torch) -> None:
    if torch is not None and torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()


def _gpu_snapshot(torch) -> dict[str, int | None]:
    if torch is None or not torch.cuda.is_available():
        return {
            "memory_allocated_bytes": None,
            "memory_reserved_bytes": None,
            "max_memory_allocated_bytes": None,
            "max_memory_reserved_bytes": None,
        }
    return {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _dtype_summary(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    named_parameters = getattr(model, "named_parameters", None)
    if named_parameters is None:
        return counts
    with contextlib.suppress(Exception):
        for name, parameter in named_parameters():
            dtype = str(getattr(parameter, "dtype", "unknown"))
            prefix = "adapter" if "lora_" in name or "modules_to_save" in name else "base_or_other"
            key = f"{prefix}:{dtype}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _assert_no_live_server_process() -> dict[str, Any] | None:
    if os.name != "nt":
        return None
    script = r'''$ErrorActionPreference='Stop'
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
Where-Object { $_.CommandLine -match '(?i)(-m\s+support\.app|src[\\/]support[\\/]app\.py|scripts[\\/](benchmark_live_runtime|live_demo)\.py)' } |
Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress'''
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        detail = ((result.stderr or "") + (result.stdout or ""))[-600:]
        raise RuntimeError("Cannot verify live demo processes; benchmark refused. " + detail.strip())
    rows = _decode_process_rows(result.stdout)
    suspects = _external_live_processes(rows, own_pid=os.getpid())
    if suspects:
        raise RuntimeError("A live runtime/demo benchmark process is already running: " + json.dumps(suspects, ensure_ascii=False))
    return {"status": "verified"}


def _decode_process_rows(stdout: str) -> list[dict[str, Any]]:
    text = stdout.strip()
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _external_live_processes(rows: list[dict[str, Any]], *, own_pid: int) -> list[dict[str, Any]]:
    process_rows: dict[int, dict[str, Any]] = {}
    parents: dict[int, int | None] = {}
    for row in rows:
        pid = _optional_int(row.get("ProcessId"))
        if pid is None:
            continue
        process_rows[pid] = row
        parents[pid] = _optional_int(row.get("ParentProcessId"))
    own_tree = _related_process_ids(own_pid, parents)
    suspects = []
    for pid, row in sorted(process_rows.items()):
        if pid in own_tree:
            continue
        suspects.append({
            "ProcessId": pid,
            "ParentProcessId": parents.get(pid),
            "CommandLine": str(row.get("CommandLine") or ""),
        })
    return suspects


def _related_process_ids(own_pid: int, parents: dict[int, int | None]) -> set[int]:
    ancestors = {own_pid}
    current = own_pid
    seen = set()
    while current in parents and current not in seen:
        seen.add(current)
        parent = parents.get(current)
        if parent is None:
            break
        ancestors.add(parent)
        current = parent

    descendants = {own_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return ancestors | descendants


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scenario_ids(limit: int) -> list[str]:
    if limit < 1 or limit > len(DEFAULT_SCENARIOS):
        raise ValueError(f"cases must be between 1 and {len(DEFAULT_SCENARIOS)}")
    return DEFAULT_SCENARIOS[:limit]


def run_benchmark(
    *,
    profile: str,
    output: Path,
    config_path: Path,
    model_key: str,
    adapter_path: Path,
    policy_path: Path,
    cases: int,
) -> int:
    profile = normalize_inference_profile(profile)
    scenarios = _scenario_ids(cases)
    _prepare_output(output)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

    config = load_model_config(config_path, model_key)
    policy = load_policy(policy_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "precision_scope": "serving_only_experiment",
        "serving_optimizations": {"logits_to_keep": 1, "empty_cache_after_call": True},
        "model_key": model_key,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "adapter_path": str(adapter_path),
        "adapter_model_sha256": _sha256_file(adapter_path / "adapter_model.safetensors"),
        "expected_adapter_model_sha256": FULL_V8_ADAPTER_MODEL_SHA256,
        "policy_path": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "scenarios": scenarios,
        "calls_planned": len(scenarios) * 2,
        "max_new_tokens": 512,
        "created_at_unix": time.time(),
    }
    _write_json(output / "manifest.json", manifest)
    records_path = output / "records.jsonl"
    summary_path = output / "summary.json"
    torch = _import_torch()
    runtime = DualModeRuntime(config, adapter_path, precision=profile).load()
    try:
        _write_json(output / "runtime_status_loaded.json", runtime.status())
        dtype_summary = _dtype_summary(getattr(getattr(runtime, "_runner", None), "model", None))
        records = []
        for call_index, scenario_id in enumerate(scenarios, start=1):
            for mode in ("base", "fine_tuned"):
                request_id = f"live-bench-{profile}-{scenario_id}-{mode}"
                _reset_peak(torch)
                _sync_cuda(torch)
                started = time.perf_counter()
                error = None
                try:
                    admin, client = invoke_support_graph(
                        runtime.for_mode(mode),
                        policy,
                        customer_message=DEFAULT_MESSAGES[scenario_id],
                        erp_context=scenario_context(scenario_id),
                        request_id=request_id,
                        session_id="live-runtime-benchmark",
                        scenario_id=scenario_id,
                    )
                except Exception as exc:
                    admin = None
                    client = None
                    error = f"{type(exc).__name__}: {exc}"
                _sync_cuda(torch)
                wall_ms = round((time.perf_counter() - started) * 1000)
                usage = admin.usage_calls[0].model_dump(mode="json") if admin and admin.usage_calls else None
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "profile": profile,
                    "precision_scope": "serving_only_experiment",
                    "scenario_id": scenario_id,
                    "message": DEFAULT_MESSAGES[scenario_id],
                    "erp_context": scenario_context(scenario_id).model_dump(mode="json"),
                    "mode": mode,
                    "cold": call_index == 1 and mode == "base",
                    "request_id": request_id,
                    "wall_ms": wall_ms,
                    "tokens_per_second": _tokens_per_second(usage, wall_ms),
                    "usage": usage,
                    "server_route": admin.server_route if admin else None,
                    "client_status": client.status if client else None,
                    "model_error": error or (None if admin is None else (admin.trace_events[-1].error if admin.trace_events and admin.trace_events[-1].status == "failed" else None)),
                    "raw_model_result": admin.raw_model_result.model_dump(mode="json") if admin and admin.raw_model_result else None,
                    "contract_valid": bool(admin and admin.raw_model_result),
                    "gpu": _gpu_snapshot(torch),
                    "runtime_status_after_call": runtime.status(),
                    "dtype_summary": dtype_summary,
                }
                _append_jsonl(records_path, record)
                records.append(record)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "profile": profile,
            "records": len(records),
            "errors": sum(1 for item in records if item["model_error"]),
            "invalid_results": sum(1 for item in records if not item["contract_valid"]),
            "dtype_summary": dtype_summary,
            "runtime_status_final": runtime.status(),
            "completed_at_unix": time.time(),
        }
        _write_json(summary_path, summary)
        return 0 if summary["errors"] == 0 and summary["invalid_results"] == 0 else 2
    finally:
        runtime.close()


def _tokens_per_second(usage: dict[str, Any] | None, wall_ms: int) -> float | None:
    if not usage or wall_ms <= 0:
        return None
    output_tokens = usage.get("output_tokens")
    if not isinstance(output_tokens, int) or output_tokens < 0:
        return None
    return output_tokens / (wall_ms / 1000.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark one live runtime precision profile in one GPU process.")
    parser.add_argument("--profile", required=True, choices=sorted(SUPPORTED_INFERENCE_PROFILES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/models.json")
    parser.add_argument("--model-key", default="qwen3_4b")
    parser.add_argument("--adapter", type=Path, default=ROOT / "artifacts/stage4/portable-full-v8/final_adapter")
    parser.add_argument("--policy", type=Path, default=ROOT / "data/policy/employee-telecom-v3.json")
    parser.add_argument("--cases", type=int, default=2)
    args = parser.parse_args(argv)

    output = _inside_project(args.output)
    config_path = _inside_project(args.config)
    adapter_path = _inside_project(args.adapter)
    policy_path = _inside_project(args.policy)
    with gpu_lock(ROOT / "artifacts/stage4/portable-gpu.lock"):
        process_check = assert_no_unmanaged_training()
        live_check = _assert_no_live_server_process()
        code = run_benchmark(
            profile=args.profile,
            output=output,
            config_path=config_path,
            model_key=args.model_key,
            adapter_path=adapter_path,
            policy_path=policy_path,
            cases=args.cases,
        )
    print(json.dumps({"status": "finished", "exit_code": code, "output": str(output), "process_check": process_check, "live_check": live_check}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
