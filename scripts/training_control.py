"""Local training supervisor: bounded sessions, one GPU owner, verified resume."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_CONFIG = "configs/train-portable-full-v8.json"
_WINDOWS_JOB_HANDLE = None


def _set_thread_execution_state(flags):
    import ctypes
    from ctypes import wintypes

    function = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
    function.argtypes = [wintypes.DWORD]
    function.restype = wintypes.DWORD
    if not function(flags):
        raise ctypes.WinError(ctypes.get_last_error())


@contextlib.contextmanager
def prevent_idle_sleep():
    """Keep only this active supervisor awake; manual sleep/lid close still work.

    Microsoft SetThreadExecutionState: ES_CONTINUOUS | ES_SYSTEM_REQUIRED.
    No display requirement or persistent Windows power-plan change.
    """
    enabled = os.name == "nt"
    if enabled:
        _set_thread_execution_state(0x80000001)
    try:
        yield enabled
    finally:
        if enabled:
            _set_thread_execution_state(0x80000000)


def bind_supervisor_lifetime():
    """Windows kills this supervisor's future children if its job handle is closed.

    See Microsoft Job Objects: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. The unnamed,
    non-inheritable handle is held until OS process exit, never passed to a child.
    """
    global _WINDOWS_JOB_HANDLE
    if os.name != "nt" or _WINDOWS_JOB_HANDLE is not None:
        return
    import ctypes
    from ctypes import wintypes

    class BasicLimit(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_working", ctypes.c_size_t),
                    ("max_working", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [("basic", BasicLimit), ("io", IoCounters),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = ExtendedLimit()
    info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    _WINDOWS_JOB_HANDLE = job


def assert_no_unmanaged_training():
    if os.name != "nt":
        return None
    script = r'''$ErrorActionPreference='Stop'
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
Where-Object { $_.CommandLine -match '(?i)scripts[\\/]((train_main|train_smoke|benchmark_validation)\.py)(\s|\")' } |
Select-Object ProcessId | ConvertTo-Json -Compress'''
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode != 0:
        detail = ((result.stderr or "") + (result.stdout or ""))[-600:]
        raise RuntimeError("Cannot verify other trainer processes; GPU launch refused. " + detail.strip())
    if result.stdout.strip():
        raise RuntimeError("A training/evaluation process is already running outside this supervisor: " + result.stdout.strip())
    return {"status": "verified"}


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def inside_project(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Training paths must stay inside this project")
    return path


def checked_hours(value):
    hours = float(value)
    if not math.isfinite(hours) or not 0 < hours <= 20:
        raise ValueError("Session must be greater than zero and at most20 hours")
    return hours


def session_allowance(hours, used_seconds, total_hours=16):
    checked_hours(hours)
    if not math.isfinite(used_seconds) or used_seconds < 0:
        raise ValueError("Invalid cumulative training time")
    return max(0.0, min(hours * 3600, total_hours * 3600 - used_seconds))


@contextlib.contextmanager
def gpu_lock(path: Path):
    """OS lock survives no process: a stale lock file cannot block tomorrow's run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another managed GPU training session is already running") from exc
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def resume_checkpoint(output: Path, config: dict):
    from support.training_sessions import latest_complete_checkpoint, validate_complete_checkpoint

    progress = read_json(output / "progress.json", {})
    if progress.get("status") == "completed" and progress.get("phase") != "preflight_only":
        raise RuntimeError("Training has already completed; do not start it again")
    checkpoint = latest_complete_checkpoint(output, expected_total_steps=int(config["expected_optimizer_steps"]))
    rejected = []
    for candidate in (output / "_trainer").glob("checkpoint-*"):
        if candidate.is_dir():
            try:
                validate_complete_checkpoint(candidate, expected_total_steps=int(config["expected_optimizer_steps"]))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                rejected.append({"checkpoint": str(candidate), "error": f"{type(exc).__name__}: {exc}"})
    if rejected:
        write_json(output / "checkpoint_selection.json", {"selected": str(checkpoint) if checkpoint else None,
                   "rejected": rejected, "checked_at": time.time()})
    manifest = read_json(output / "manifest.json", {})
    if manifest and manifest.get("phase") != "preflight_only" and checkpoint is None:
        raise RuntimeError("This run started but has no verified complete checkpoint; refusing to restart silently")
    return checkpoint


def child_command(config_path, checkpoint, allowed_seconds):
    # Reserve time to finish the current optimizer step before the hard stop.
    grace = min(600, allowed_seconds * .1)
    command = [sys.executable, "-u", "-B", str(ROOT / "scripts/train_main.py"),
               "--project-root", str(ROOT), "--config", str(config_path),
               "--experimental-draft", "--session-seconds", str(max(1, allowed_seconds - grace))]
    if checkpoint is not None:
        command += ["--resume-from-checkpoint", str(checkpoint)]
    return command


def stop_owned_child(process):
    """Only the process tree created by this supervisor; never search/kill other apps."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode and process.poll() is None:
            raise RuntimeError("Could not stop the owned training process at its deadline")
    else:
        process.kill()
    process.wait(timeout=30)


def readable_status(output: Path, data: dict) -> str:
    progress, supervisor, budget = data["training"], data["supervisor"], data["budget"]
    state = supervisor.get("status") or progress.get("status", "not_started")
    if state == "running":
        try:
            with gpu_lock(ROOT / "artifacts/stage4/portable-gpu.lock"):
                state = "interrupted"
        except RuntimeError:
            pass
    if progress.get("phase") == "preflight_only" and not supervisor:
        state = "ready"
    names = {"running": "Обучение идёт", "training": "Обучение идёт", "paused": "Пауза — можно выключать ноутбук",
             "completed": "Обучение и проверка адаптера завершены", "failed": "Ошибка — смотрите журналы",
             "interrupted": "Прервано — следующий запуск проверит сохранения", "not_started": "Ещё не запускалось",
             "ready": "Подготовлено к запуску", "loading": "Загрузка модели"}
    used = max(0, float(budget.get("used_seconds", 0)))
    checkpoints = []
    for marker in (output / "_trainer").glob("checkpoint-*/checkpoint_complete.json"):
        complete = read_json(marker, {})
        if isinstance(complete.get("global_step"), int):
            checkpoints.append(complete["global_step"])
    lines = [f"Состояние: {names.get(state, state)}",
             f"Шаги: {progress.get('step', 0)} / {progress.get('total', '?')}",
             f"Последняя записанная точка: {'checkpoint-' + str(max(checkpoints)) if checkpoints else 'ещё нет'}",
             f"Использовано обучение: {used / 3600:.2f} ч из {budget.get('total_budget_hours', 16):g} ч"]
    if data.get("pause_requested"):
        lines.append("Пауза запрошена. Дождитесь сохранения и состояния «Пауза».")
    if supervisor.get("forced_reason"):
        lines.append("Сработал ограничитель времени/скорости. Предыдущие сохранения остались на диске.")
    lines.append(f"Журналы: {output / 'session_logs'}")
    return "\n".join(lines)


def supervised_run(config_path: Path, hours: float):
    config = read_json(config_path)
    output = inside_project(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if config["training"].get("portable_sessions") is not True:
        raise ValueError("The supervisor requires portable_sessions=true")
    with gpu_lock(ROOT / "artifacts/stage4/portable-gpu.lock"), prevent_idle_sleep() as awake:
        process_check = assert_no_unmanaged_training()
        checkpoint = resume_checkpoint(output, config)
        ledger_path = output / "control_budget.json"
        ledger = read_json(ledger_path, {"used_seconds": 0.0, "total_budget_hours": 16.0})
        total_hours = float(ledger["total_budget_hours"])
        if not 0 < total_hours <= 16:
            raise ValueError("Cumulative training budget must not exceed16 hours")
        used = float(ledger["used_seconds"])
        allowance = session_allowance(hours, used, total_hours)
        if allowance <= 0:
            raise RuntimeError("Training budget exhausted; checkpoint preserved for evaluation/review")
        pause_path = output / "PAUSE_REQUEST"
        if pause_path.exists():
            archive = output / "session_logs"
            archive.mkdir(exist_ok=True)
            pause_path.replace(archive / f"pause-request-{uuid.uuid4().hex}.json")
        session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        logs = output / "session_logs"
        logs.mkdir(exist_ok=True)
        command = child_command(config_path, checkpoint, allowance)
        bind_supervisor_lifetime()
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        allocator = config.get("runtime", {}).get("pytorch_cuda_alloc_conf", "max_split_size_mb:128")
        if allocator is None:
            environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            environment["PYTORCH_CUDA_ALLOC_CONF"] = str(allocator)
        started = time.time()
        record = {"session_id": session_id, "status": "running", "started_at": started,
                  "supervisor_pid": os.getpid(), "resumed_from": str(checkpoint) if checkpoint else None,
                  "allowed_seconds": allowance, "command": command, "allocator": environment.get("PYTORCH_CUDA_ALLOC_CONF"),
                  "idle_sleep_prevention": awake,
                  "process_check": process_check}
        status_path = output / "control_status.json"
        with (logs / f"{session_id}.stdout.log").open("w", encoding="utf-8") as stdout, \
                (logs / f"{session_id}.stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            record["training_launcher_pid"] = process.pid
            write_json(status_path, record)
            write_json(logs / f"{session_id}.json", record)
            no_progress_since = None
            forced_reason = None
            try:
                while process.poll() is None:
                    now = time.time()
                    elapsed = max(0, now - started)
                    ledger.update(used_seconds=used + elapsed, updated_at=now, session_id=session_id)
                    write_json(ledger_path, ledger)
                    if elapsed >= allowance:
                        forced_reason = "session_deadline" if used + elapsed < total_hours * 3600 else "training_budget_exhausted"
                        stop_owned_child(process)
                        break
                    events = output / "microbatch_memory.jsonl"
                    last_event = max(started, events.stat().st_mtime if events.exists() else started)
                    if now - last_event >= 900:
                        if no_progress_since is None:
                            no_progress_since = now
                            write_json(pause_path, {"reason": "slow_progress_budget_15_minutes", "requested_at": now})
                        elif now - no_progress_since >= 120:
                            forced_reason = "slow_progress_budget_cutoff"
                            stop_owned_child(process)
                            break
                    else:
                        no_progress_since = None
                    time.sleep(5)
            finally:
                # A monitor exception must not leave its GPU child running unbounded.
                if process.poll() is None:
                    stop_owned_child(process)
                ended = time.time()
                ledger.update(used_seconds=used + max(0, ended - started), updated_at=ended)
                write_json(ledger_path, ledger)
                progress = read_json(output / "progress.json", {})
                record.update(ended_at=ended, elapsed_seconds=max(0, ended - started),
                              returncode=process.returncode, forced_reason=forced_reason,
                              status="interrupted" if forced_reason else (progress.get("status", "failed") if process.returncode == 0 else "failed"),
                              last_reported_step=progress.get("step"))
                write_json(status_path, record)
                write_json(logs / f"{session_id}.json", record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Start/resume, pause or inspect local portable training.")
    parser.add_argument("action", choices=["start", "supervise", "pause", "status"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--hours", type=checked_hours, default=4.0)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)
    config_path = inside_project(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    config = read_json(config_path)
    output = inside_project(config["output_dir"])
    if args.action == "pause":
        write_json(output / "PAUSE_REQUEST", {"reason": "user_pause", "requested_at": time.time()})
        result = {"status": "pause_requested", "message": "Wait for status=paused before shutdown; the current optimizer step will be saved."}
    elif args.action == "status":
        result = {"training": read_json(output / "progress.json", {}),
                  "supervisor": read_json(output / "control_status.json", {}),
                  "budget": read_json(output / "control_budget.json", {}),
                  "pause_requested": (output / "PAUSE_REQUEST").exists()}
    elif args.action == "supervise":
        result = supervised_run(config_path, args.hours)
    else:
        output.mkdir(parents=True, exist_ok=True)
        # Check synchronously; the supervisor repeats checks under its OS lock.
        with gpu_lock(ROOT / "artifacts/stage4/portable-gpu.lock"):
            process_check = assert_no_unmanaged_training()
            resume_checkpoint(output, config)
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "supervise", "--config", str(config_path), "--hours", str(args.hours)]
        with (output / "supervisor.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                       env={**os.environ, "PYTHONUTF8": "1"})
        result = {"status": "supervisor_started", "pid": process.pid, "session_hours": args.hours,
                  "process_check": process_check,
                  "message": "Check Training-Status.cmd for accepted run or startup errors."}
    if args.action == "status" and not args.json:
        print(readable_status(output, result))
    elif args.action == "pause" and not args.json:
        print("Пауза запрошена. Дождитесь сохранения текущего шага и состояния «Пауза» в Training-Status.cmd.")
    elif args.action == "start" and not args.json:
        print(f"Запрос запуска/продолжения отправлен. Сеанс до {args.hours:g} ч. Статус: Training-Status.cmd.")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
