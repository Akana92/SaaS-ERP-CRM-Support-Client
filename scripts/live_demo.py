"""Local stage5 server lifecycle. No training, downloads or external API calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from training_control import assert_no_unmanaged_training, gpu_lock, read_json, write_json
from support.live_precision import normalize_attention_backend

OUTPUT = ROOT / "artifacts/stage5/live"
STATUS = OUTPUT / "server.json"
STOP = OUTPUT / "stop.json"
LOCK = ROOT / "artifacts/stage4/portable-gpu.lock"
ADAPTER_PATHS = {
    "quality-f": ROOT / "artifacts/stage5/quality90-v1/training/candidate-f-v13-audited-labels/final_adapter",
    "full-v8": ROOT / "artifacts/stage4/portable-full-v8/final_adapter",
    "dialogue-250": ROOT / "artifacts/stage5/dialogue-v1/training/pilot-250-v2/final_adapter",
}


def serving_info(adapter_profile: str) -> dict:
    from support.live_runtime import FULL_V8_ADAPTER_MODEL_SHA256

    if adapter_profile not in ADAPTER_PATHS:
        raise ValueError("Unknown adapter profile")
    expected_sha = {
        "quality-f": "6efc00de7bcecc24872b87de797abd4d9d43ea10e679b23d3b4c13ecb7f7109b",
        "full-v8": FULL_V8_ADAPTER_MODEL_SHA256,
        "dialogue-250": "1c1fc4295f0c400b7338a21d7c9a305c345b07cc82c1a2e0bb294a3085964991",
    }[adapter_profile]
    return {"adapter_profile": adapter_profile,
            "adapter_path": str(ADAPTER_PATHS[adapter_profile].resolve()),
            "adapter_model_sha256": expected_sha,
            "candidate_status": ("selected_for_local_demo" if adapter_profile == "quality-f"
                                 else "candidate_not_selected"),
            "client_mode": "fine_tuned"}


def offline_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", GRADIO_ANALYTICS_ENABLED="False",
               LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false",
               TOKENIZERS_PARALLELISM="false")
    return env


def assert_port_free(port: int) -> None:
    with socket.socket() as probe:
        # Exclusive bind prevents Windows SO_REUSEADDR from hiding an occupied port.
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", port))


def assert_no_legacy_inference() -> None:
    if os.name != "nt":
        return
    command = r'''$ErrorActionPreference='Stop'
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
Where-Object { $_.CommandLine -match '(?i)(-m\s+support\.app(\s|$)|scripts[\\/](inference_smoke|benchmark_models)\.py(\s|"))' } |
Select-Object ProcessId | ConvertTo-Json -Compress'''
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or result.stdout.strip():
        raise RuntimeError("Cannot confirm that legacy inference is stopped; GPU launch refused.")


def serve(port: int, run_id: str, precision: str = "bf16", adapter_profile: str = "quality-f",
          attention: str = "sdpa_repeat_kv") -> None:
    attention = normalize_attention_backend(attention)
    os.environ.update(offline_environment())
    from support.contracts import PolicyDocument
    from support.live_runtime import DualModeRuntime
    from support.live_app import create_live_app
    from support.live_service import CLIENT_MODEL_MODE
    import uvicorn

    selected = serving_info(adapter_profile)
    assert_port_free(port)
    # Shared OS lock is released even if this process crashes; no stale PID bypass.
    with gpu_lock(LOCK):
        assert_no_unmanaged_training()
        assert_no_legacy_inference()
        config_doc = read_json(ROOT / "configs/models.json")
        config = next(c for c in config_doc["candidates"] if c["key"] == "qwen3_4b")
        config["local_path"] = str((ROOT / config["local_path"]).resolve())
        config["live_attention_backend"] = attention
        policy = PolicyDocument.model_validate(read_json(ROOT / "data/policy/employee-telecom-v3.json"))
        record = {"run_id": run_id, "pid": os.getpid(), "port": port,
                  "started_at": time.time(), "status": "loading", "client_mode": CLIENT_MODEL_MODE,
                  "precision": precision, "attention": attention, **selected}
        write_json(STATUS, record)
        runtime = DualModeRuntime(config, selected["adapter_path"], precision=precision,
                                  expected_adapter_model_sha256=selected["adapter_model_sha256"])
        finished = threading.Event()
        try:
            runtime.load()
            app = create_live_app(runtime, policy, store_path=OUTPUT / "store.json",
                                  archive_path=OUTPUT / "pre-ux-update-snapshot.json", serving_info=selected)
            server = uvicorn.Server(uvicorn.Config(app,
                                                   host="127.0.0.1", port=port, workers=1,
                                                   access_log=False, log_level="info"))

            def watch_stop() -> None:
                while not finished.wait(1):
                    if server.started and record["status"] == "starting_server":
                        record.update(status="running", ready_at=time.time())
                        write_json(STATUS, record)
                    try:
                        request = read_json(STOP, {})
                    except (OSError, ValueError):
                        continue
                    if request.get("run_id") == run_id:
                        record.update(status="stopping")
                        write_json(STATUS, record)
                        server.should_exit = True
                        return

            record.update(status="starting_server", loaded_at=time.time())
            write_json(STATUS, record)
            threading.Thread(target=watch_stop, daemon=True, name="local-demo-stop").start()
            server.run()
            record.update(status="stopped", ended_at=time.time())
        except BaseException:
            record.update(status="failed", ended_at=time.time())
            raise
        finally:
            finished.set()
            runtime.close()
            write_json(STATUS, record)


def start(port: int, precision: str = "bf16", adapter_profile: str = "quality-f",
          attention: str = "sdpa_repeat_kv") -> dict:
    attention = normalize_attention_backend(attention)
    selected = serving_info(adapter_profile)
    assert_port_free(port)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    with (OUTPUT / f"{run_id}.stdout.log").open("w", encoding="utf-8") as stdout, \
            (OUTPUT / f"{run_id}.stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen([sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                                    "serve", "--port", str(port), "--run-id", run_id,
                                    "--adapter-profile", adapter_profile,
                                    "--attention", attention, "--precision", precision],
                                   cwd=ROOT, env=offline_environment(), stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    receipt = {"run_id": run_id, "launcher_pid": process.pid, "status": "dispatched",
               "status_file": str(STATUS), "port": port, "precision": precision,
               "attention": attention, **selected}
    write_json(OUTPUT / "launch.json", receipt)
    return receipt


def stop() -> dict:
    record = read_json(STATUS, {})
    if record.get("status") in {None, "stopped", "failed"}:
        return {"status": "no_active_run_record"}
    write_json(STOP, {"run_id": record["run_id"], "requested_at": time.time()})
    return {"status": "stop_requested", "run_id": record["run_id"],
            "note": "The current inference request finishes before shutdown; check status before transport."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "serve", "status", "stop"))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--adapter-profile", choices=tuple(ADAPTER_PATHS), default="quality-f",
                        help="Adapter for this local demo; does not change frozen evaluation or weights.")
    parser.add_argument("--precision", choices=("nf4", "bf16", "fp16"), default="bf16",
                        help="Live serving precision; does not alter the frozen NF4 evaluation.")
    parser.add_argument("--attention", choices=("sdpa", "sdpa_repeat_kv"), default="sdpa_repeat_kv",
                        help="Live attention backend; use sdpa to restore the original serving path.")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be 1024..65535")
    if args.action == "serve":
        serve(args.port, args.run_id or uuid.uuid4().hex, args.precision, args.adapter_profile, args.attention)
        return
    result = start(args.port, args.precision, args.adapter_profile, args.attention) if args.action == "start" else stop() if args.action == "stop" else read_json(STATUS, {})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
