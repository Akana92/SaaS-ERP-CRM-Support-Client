"""Explicit pause/resume controller for isolated quality90 training runs."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training_control import bind_supervisor_lifetime, gpu_lock, read_json, write_json

ALLOWED = ROOT / "artifacts/stage5/quality90-v1/training"
SERIES = ALLOWED.parent
STATE = "quality_control.json"
LATCH = "USER_PAUSED.json"
PAUSE = "PAUSE_REQUEST"


def checked_run(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    if path == ALLOWED.resolve() or not path.is_relative_to(ALLOWED.resolve()):
        raise ValueError("run-dir must be a child of artifacts/stage5/quality90-v1/training")
    return path


def checked_config(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to((ROOT / "configs").resolve()):
        raise ValueError("config must be inside this project's configs directory")
    return path


def active_pointer() -> dict:
    pointer = read_json(SERIES / "active_training.json", {})
    if pointer:
        return {"run_dir": checked_run(pointer["run_dir"]), "config": checked_config(pointer["config"])}
    return {}


def evaluation_state() -> dict:
    """Never trust terminal text without verifying the actual evaluator exit."""
    try:
        pointer = read_json(SERIES / "active_evaluation.json", {})
        if not pointer:
            return {"active": False, "identity_error": None}
        if pointer.get("schema_version") != "quality90-active-evaluation-v1":
            raise ValueError("Unknown active evaluation schema")
        path = (ROOT / pointer["run_dir"]).resolve()
        allowed = (SERIES / "evaluation").resolve()
        if path == allowed or not path.is_relative_to(allowed):
            raise ValueError("Evaluation path must stay inside this series evaluation directory")
        if type(pointer.get("pid")) is not int or pointer["pid"] <= 0:
            raise ValueError("Evaluator PID is invalid")
        return {"active": alive(pointer), "identity_error": None, "run_dir": str(path)}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"active": False, "identity_error": f"Evaluator: {exc}"}


def shared_gpu_state() -> dict:
    try:
        with gpu_lock(ROOT / "artifacts/stage4/portable-gpu.lock"):
            return {"idle": True, "identity_error": None}
    except (RuntimeError, BlockingIOError):
        return {"idle": False, "identity_error": None}
    except OSError as exc:
        return {"idle": False, "identity_error": f"Shared GPU lock: {exc}"}


def process_identity(pid: int) -> str | None:
    """PID plus OS creation time prevents confusing a recycled PID with our worker."""
    if os.name != "nt":
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            return None
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:
            return None
        raise OSError("Cannot verify worker process identity")
    try:
        values = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            raise OSError("Cannot read worker creation time")
        # An exited process can remain queryable while another process owns a handle.
        exit_code = wintypes.DWORD()
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise OSError("Cannot verify worker exit")
        if exit_code.value != 259:
            return None
        return str((values[0].dwHighDateTime << 32) | values[0].dwLowDateTime)
    finally:
        kernel.CloseHandle(handle)


def alive(record: dict) -> bool:
    if not record.get("pid") or not record.get("identity"):
        raise ValueError("Worker identity is incomplete; safe stop is unverified")
    return process_identity(record["pid"]) == record["identity"]


def claim_supervisor(record: dict) -> dict:
    """Bind only self or the verified direct Windows venv launcher parent."""
    own_pid = os.getpid()
    own_identity = process_identity(own_pid)
    if record.get("pid") == own_pid and own_identity and record.get("identity") == own_identity:
        return record
    if (os.name == "nt" and record.get("pid") == os.getppid()
            and own_identity and alive(record)):
        return {**record, "pid": own_pid, "identity": own_identity,
                "launcher": {"pid": record["pid"], "identity": record["identity"]}}
    raise RuntimeError("Supervisor identity mismatch")


def checkpoint(run: Path, total: int | None = None) -> Path | None:
    from support.training_sessions import latest_complete_checkpoint
    return latest_complete_checkpoint(run, expected_total_steps=total)


def status(run: Path) -> dict:
    record = read_json(run / STATE, {})
    progress = read_json(run / "progress.json", {})
    unknown = None
    active = False
    try:
        if record:
            active = alive(record)
            if record.get("launcher"):
                active = alive(record["launcher"]) or active
            if record.get("child"):
                active = alive(record["child"]) or active
        elif (run / "manifest.json").exists() and read_json(run / "manifest.json", {}).get("phase") != "preflight_only":
            unknown = "Run was not launched by this controller"
    except (OSError, ValueError) as exc:
        unknown = str(exc)
    saved = None
    if not active and not unknown:
        saved = checkpoint(run)
    no_training_started = bool(
        record.get("status") == "user_paused_before_launch" and not record.get("child")
        and read_json(run / "manifest.json", {}).get("phase") in (None, "preflight_only")
        and progress.get("status") not in ("loading", "training", "failed", "paused", "completed")
        and not active and not unknown)
    evaluation = evaluation_state()
    gpu = shared_gpu_state()
    unknown = unknown or evaluation["identity_error"] or gpu["identity_error"]
    all_idle = not active and not evaluation["active"] and gpu["idle"] and not unknown
    return {"active": active or evaluation["active"] or not gpu["idle"],
            "training_active": active, "evaluation": evaluation, "gpu_idle": gpu["idle"],
            "identity_error": unknown, "run_dir": str(run),
            "pause_requested": (run / LATCH).exists() or (SERIES / LATCH).exists(), "training": progress,
            "checkpoint": str(saved) if saved else None,
            "no_training_started": no_training_started,
            "safe_to_travel": bool((no_training_started or (record and saved)) and all_idle)}


def pause(run: Path) -> dict:
    with gpu_lock(SERIES / "series-launch.lock"):
        # Resolve again under the launch lock: a new candidate may have started
        # between the CLI default lookup and this pause request.
        run = active_pointer().get("run_dir", run)
        run.mkdir(parents=True, exist_ok=True)
        with supervisor_launch_lock(run):
            receipt = {"requested_at": time.time(), "reason": "user_pause", "run_dir": str(run)}
            evaluation = evaluation_state()
            if evaluation["active"] and not evaluation["identity_error"]:
                receipt["evaluation_run_dir"] = evaluation["run_dir"]
            write_json(SERIES / LATCH, receipt)
            write_json(run / LATCH, receipt)
            (run / PAUSE).write_text("user_pause\n", encoding="utf-8")
            if receipt.get("evaluation_run_dir"):
                (Path(receipt["evaluation_run_dir"]) / "pause.request").write_text("user_pause\n", encoding="utf-8")
    return status(run)


def resume_evaluation() -> dict:
    """Explicit user acknowledgement only; never launch or silently resume GPU work."""
    with gpu_lock(SERIES / "series-launch.lock"):
        receipt = read_json(SERIES / LATCH, {})
        evaluation = evaluation_state()
        if (not receipt.get("evaluation_run_dir") or evaluation["identity_error"]
                or not evaluation.get("run_dir")
                or (ROOT / receipt["evaluation_run_dir"]).resolve() != Path(evaluation["run_dir"])):
            raise RuntimeError("Evaluation pause does not match a verified active evaluation pointer")
        gpu = shared_gpu_state()
        if evaluation["active"] or not gpu["idle"] or gpu["identity_error"]:
            raise RuntimeError("Wait for evaluator and shared GPU work to stop before acknowledging resume")
        pointer = active_pointer()
        training_run = checked_run(receipt["run_dir"]) if receipt.get("run_dir") else pointer.get("run_dir")
        if pointer and training_run != pointer["run_dir"]:
            raise RuntimeError("Paused training context no longer matches the active candidate")
        run_latch = None
        if training_run:
            current = status(training_run)
            if current["active"] or current["identity_error"]:
                raise RuntimeError("Training worker exit is unverified")
            run_latch = training_run / LATCH
            local_pause = read_json(run_latch, {})
            if local_pause and local_pause.get("evaluation_run_dir") != receipt["evaluation_run_dir"]:
                raise RuntimeError("An unrelated training pause must remain in place")
        # Keep evaluation pause.request: evaluator --resume consumes it using
        # its normal verified resume path. Only remove the matching user latch.
        if run_latch:
            run_latch.unlink(missing_ok=True)
        (SERIES / LATCH).unlink()
        metadata = read_json(SERIES / "active_evaluation.json", {})
        return {"status": "evaluation_resume_authorized", "started": False,
                "evaluation_run_dir": evaluation["run_dir"],
                "resume_command": metadata.get("resume_command"),
                "manifest": str(Path(evaluation["run_dir"]) / "manifest.json"),
                "message": "Пауза проверки снята по явной команде. Проверка не запущена; продолжите тот же evaluator с --resume."}


def resume_series() -> dict:
    """Acknowledge a completed candidate's user pause; explicit user command only."""
    with gpu_lock(SERIES / "series-launch.lock"):
        pointer_path = SERIES / "active_training.json"
        series_latch = SERIES / LATCH
        if not pointer_path.is_file() or not series_latch.is_file():
            raise RuntimeError("Completed series resume requires an active pointer and series pause")
        pointer_bytes = pointer_path.read_bytes()
        series_bytes = series_latch.read_bytes()
        pointer = active_pointer()
        series_pause = read_json(series_latch, {})
        if not isinstance(series_pause, dict) or not series_pause.get("run_dir"):
            raise RuntimeError("Series pause target is unknown")
        run = checked_run(series_pause["run_dir"])
        if pointer.get("run_dir") != run:
            raise RuntimeError("Series pause does not match the active candidate")
        with gpu_lock(run / "quality-control.lock"):
            run_latch = run / LATCH
            local_pause = read_json(run_latch, {})
            for paused in (series_pause, local_pause):
                if (not isinstance(paused, dict) or not paused.get("run_dir")
                        or "evaluation_run_dir" in paused or paused.get("reason") != "user_pause"
                        or type(paused.get("requested_at")) not in (int, float)
                        or checked_run(paused["run_dir"]) != run):
                    raise RuntimeError("Unknown, unrelated or evaluation pause must remain in place")
            if local_pause != series_pause:
                raise RuntimeError("Series and run pause receipts do not match")
            paths = [pointer_path, series_latch, run_latch, run / PAUSE,
                     run / "manifest.json", run / "progress.json", run / STATE,
                     SERIES / "active_evaluation.json"]
            if any(path.is_symlink() for path in paths):
                raise RuntimeError("Controller metadata must not be symbolic links")
            frozen = {path: path.read_bytes() if path.exists() else None for path in paths}
            if frozen[pointer_path] != pointer_bytes or frozen[series_latch] != series_bytes:
                raise RuntimeError("Pause identity changed during validation")
            if frozen[run / PAUSE] not in (None, b"user_pause\n", b"user_pause\r\n"):
                raise RuntimeError("An unrelated PAUSE_REQUEST must remain in place")
            manifest = read_json(run / "manifest.json", {})
            current = status(run)
            if (current.get("active") is not False or current.get("identity_error", "missing") is not None
                    or current.get("safe_to_travel") is not True
                    or current.get("training", {}).get("status") != "completed"
                    or manifest.get("phase") != "completed"):
                raise RuntimeError("Resume-series requires verified inactive completed training")
            evaluation = current.get("evaluation", {})
            if evaluation.get("active") or evaluation.get("identity_error"):
                raise RuntimeError("Evaluation exit is unverified")
            if evaluation.get("run_dir"):
                evaluation_run = Path(evaluation["run_dir"]).resolve()
                allowed_evaluation = (SERIES / "evaluation").resolve()
                if evaluation_run == allowed_evaluation or not evaluation_run.is_relative_to(allowed_evaluation):
                    raise RuntimeError("Evaluation pointer is outside this series")
                if any((evaluation_run / name).exists() for name in ("pause.request", LATCH)):
                    raise RuntimeError("An independent evaluation pause must remain in place")
            saved = checkpoint(run)
            if (saved is None or not current.get("checkpoint")
                    or Path(current["checkpoint"]).resolve() != saved.resolve()
                    or not saved.resolve().is_relative_to(run.resolve())):
                raise RuntimeError("Completed checkpoint is not verified")
            # Keep GPU work excluded between the status probe and acknowledgement.
            with gpu_lock(ROOT / "artifacts/stage4/portable-gpu.lock"):
                for path, before in frozen.items():
                    if (path.read_bytes() if path.exists() else None) != before:
                        raise RuntimeError("Controller state changed during resume-series validation")
                removed = [series_latch, run_latch] + ([run / PAUSE] if frozen[run / PAUSE] is not None else [])
                receipt_path = SERIES / f"resume-series-{uuid.uuid4().hex}.json"
                receipt = {"schema_version": "quality90-resume-series-v1", "action": "resume-series",
                           "acknowledged_at": time.time(), "run_dir": str(run), "started": False,
                           "checkpoint": str(saved), "series_pause": series_pause, "run_pause": local_pause,
                           "active_pointer_unchanged": True, "continuous_window_reset": False,
                           "authorization": "Explicit new user continuation required; never automatic heartbeat",
                           "removed_files": [{"path": str(path),
                                              "sha256": hashlib.sha256(frozen[path]).hexdigest(),
                                              "original_bytes_base64": base64.b64encode(frozen[path]).decode("ascii")}
                                             for path in removed],
                           "verified_metadata_sha256": {str(path): hashlib.sha256(raw).hexdigest() if raw is not None else None
                                                        for path, raw in frozen.items()}}
                # Persist the exact originals before clearing matching latches.
                with receipt_path.open("x", encoding="utf-8") as handle:
                    json.dump(receipt, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                for path in removed:
                    path.unlink()
    return {"status": "series_resume_authorized", "started": False, "run_dir": str(run),
            "receipt": str(receipt_path),
            "message": "Пауза завершённого кандидата подтверждена. Процессы не запущены; окно сеанса не изменено."}


def command(config: Path, run: Path, resume: bool, pause_after_steps: int | None = None) -> list[str]:
    result = [sys.executable, "-u", "-B", str(ROOT / "scripts/train_quality_candidate.py"),
              "--config", str(config), "--run-dir", str(run), "--allow-pending-review"]
    if pause_after_steps is not None:
        if pause_after_steps <= 0:
            raise ValueError("pause-after-steps must be positive")
        result += ["--pause-after-steps", str(pause_after_steps)]
    if resume:
        result += ["--resume", "latest"]
    return result


@contextmanager
def supervisor_launch_lock(run: Path):
    """Parent writes our identity while holding this lock after Popen returns."""
    deadline = time.monotonic() + 10
    while True:
        lock = gpu_lock(run / "quality-control.lock")
        try:
            lock.__enter__()
            break
        except (RuntimeError, BlockingIOError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for launcher identity receipt")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def launch(config: Path, run: Path, *, resume: bool, pause_after_steps: int | None = None) -> dict:
    if pause_after_steps is not None and pause_after_steps <= 0:
        raise ValueError("pause-after-steps must be positive")
    config = checked_config(config)
    run = checked_run(run)
    settings = read_json(config)
    if not settings:
        raise ValueError("Training config is missing")
    run.mkdir(parents=True, exist_ok=True)
    with gpu_lock(SERIES / "series-launch.lock"), gpu_lock(run / "quality-control.lock"):
        pointer = active_pointer()
        if not pointer:
            legacy_config = ROOT / "configs/train-quality90-v1.json"
            legacy_run = read_json(legacy_config, {}).get("run_dir")
            if legacy_run:
                legacy_run = checked_run(legacy_run)
                if (legacy_run / LATCH).exists():
                    pointer = {"run_dir": legacy_run, "config": legacy_config}
        series_pause = read_json(SERIES / LATCH)
        if series_pause:
            if series_pause.get("evaluation_run_dir"):
                raise RuntimeError("На паузе проверка модели. Напишите в задаче «продолжи проверку». Пауза сохранена; обучение заново не запускается.")
            if not series_pause.get("run_dir"):
                raise RuntimeError("User paused the series; pause target is unverified and cannot be cleared")
            paused_run = checked_run(series_pause["run_dir"])
            if not resume or paused_run != run or (pointer and pointer["run_dir"] != run):
                raise RuntimeError("User paused the series; explicitly resume the same paused candidate")
        if pointer and pointer["run_dir"] != run:
            previous = status(pointer["run_dir"])
            if previous["active"] or previous["identity_error"]:
                raise RuntimeError("Another candidate is active or its exit is unverified")
            if (pointer["run_dir"] / LATCH).exists():
                raise RuntimeError("User paused the previous candidate; explicitly resume that candidate")
        current = status(run)
        resume_checkpoint = resume
        if current["active"] or current["identity_error"]:
            raise RuntimeError("Worker active or its exit is unverified")
        if current["training"].get("status") == "completed":
            raise RuntimeError("Обучение завершено. Если на паузе проверка модели, напишите в задаче «продолжи проверку». Этот кандидат заново не запускается.")
        if resume:
            if checkpoint(run, settings["expected_optimizer_steps"]) is None:
                manifest = read_json(run / "manifest.json", {})
                prior = read_json(run / STATE, {})
                if manifest.get("phase") not in (None, "preflight_only") or prior.get("child"):
                    raise RuntimeError("Resume requires a verified complete checkpoint")
                # Pause before the first worker launch: explicitly release the
                # user's latch, but there is no optimizer state to resume yet.
                resume_checkpoint = False
        elif (run / LATCH).exists():
            raise RuntimeError("User paused this run; only explicit resume may clear the pause")
        elif (run / "manifest.json").exists() and read_json(run / "manifest.json", {}).get("phase") != "preflight_only":
            raise RuntimeError("Existing run requires explicit resume")
        if not (ROOT / "scripts/train_quality_candidate.py").exists():
            raise RuntimeError("Candidate trainer is not installed yet")
        # Clear only after all validation; restore latch if process launch fails.
        latched = read_json(run / LATCH)
        try:
            if resume:
                (run / LATCH).unlink(missing_ok=True)
                (run / PAUSE).unlink(missing_ok=True)
                (SERIES / LATCH).unlink(missing_ok=True)
            with (run / "quality-supervisor.log").open("ab") as log:
                worker = subprocess.Popen(
                    [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "_run",
                     "--config", str(config), "--run-dir", str(run)] + (["--resume-worker"] if resume_checkpoint else [])
                    + (["--pause-after-steps", str(pause_after_steps)] if pause_after_steps else []),
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    env={**os.environ, "PYTHONUTF8": "1"})
                write_json(run / STATE, {"pid": worker.pid, "identity": process_identity(worker.pid),
                                        "status": "starting", "started_at": time.time()})
                write_json(SERIES / "active_training.json", {"run_dir": str(run), "config": str(config)})
        except Exception:
            if series_pause:
                write_json(SERIES / LATCH, series_pause)
            if latched:
                write_json(run / LATCH, latched)
                (run / PAUSE).write_text("user_pause\n", encoding="utf-8")
            raise
    return {"status": "launch_requested", "pid": worker.pid, "safe_to_travel": False}


def supervise(config: Path, run: Path, resume: bool, pause_after_steps: int | None = None) -> int:
    bind_supervisor_lifetime()
    with supervisor_launch_lock(run):
        state = read_json(run / STATE, {})
        state = claim_supervisor(state)
        write_json(run / STATE, state)
        if (run / LATCH).exists() or (SERIES / LATCH).exists():
            state["status"] = "user_paused_before_launch"
            write_json(run / STATE, state)
            return 0
        with (run / "quality-trainer.log").open("ab") as log:
            child = subprocess.Popen(command(config, run, resume, pause_after_steps), cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        state.update(status="running", child={"pid": child.pid, "identity": process_identity(child.pid)})
        write_json(run / STATE, state)
    code = child.wait()
    state.update(status="stopped", exit_code=code, stopped_at=time.time())
    write_json(run / STATE, state)
    return code


def wait_for_pause(run: Path) -> dict:
    """Human-facing wait; Ctrl+C cancels only this wait, never the pause latch."""
    while True:
        result = status(run)
        if result.get("safe_to_travel"):
            return result
        if result.get("identity_error"):
            return {**result, "wait_error": result["identity_error"]}
        if not result.get("active"):
            return {**result, "wait_error": "Процесс не работает, но целое сохранение не подтверждено. Проверьте журналы."}
        progress = result.get("training", {})
        phase = "проверка модели" if result.get("evaluation", {}).get("active") else progress.get('status', 'starting')
        print(f"Ожидаем сохранения и завершения процесса. Шаг {progress.get('step', '?')} / "
              f"{progress.get('total', '?')}; состояние: {phase}", flush=True)
        time.sleep(2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "resume", "resume-evaluation", "resume-series", "pause", "status", "_run"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", help="Defaults to run_dir in the selected config")
    parser.add_argument("--resume-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pause-after-steps", type=int)
    parser.add_argument("--wait", action="store_true", help="Wait for a verified safe pause (pause only)")
    args = parser.parse_args(argv)
    if args.wait and args.action != "pause":
        parser.error("--wait is only supported for pause")
    if args.action == "resume-series":
        if args.config is not None or args.run_dir is not None or args.resume_worker or args.pause_after_steps is not None:
            parser.error("resume-series only acknowledges the exact active completed candidate; no overrides allowed")
        print(json.dumps(resume_series(), ensure_ascii=False, indent=2))
        return 0
    if args.action == "resume-evaluation":
        print(json.dumps(resume_evaluation(), ensure_ascii=False, indent=2))
        return 0
    pointer = active_pointer() if args.config is None and args.run_dir is None else {}
    args.config = checked_config(args.config or pointer.get("config") or ROOT / "configs/train-quality90-v1.json")
    value = args.run_dir or pointer.get("run_dir")
    if value is None:
        value = read_json(args.config, {}).get("run_dir")
        if not value:
            if args.action == "status":
                print(json.dumps({"status": "not_configured", "safe_to_travel": False,
                                  "message": "Конфиг и активный кандидат ещё не подготовлены."}, ensure_ascii=False))
                return 0
            parser.error("config has no run_dir; provide --run-dir explicitly")
    run = checked_run(value)
    if args.action == "_run":
        return supervise(args.config, run, args.resume_worker, args.pause_after_steps)
    result = (pause(run) if args.action == "pause" else status(run) if args.action == "status"
              else launch(args.config, run, resume=args.action == "resume", pause_after_steps=args.pause_after_steps))
    if args.wait:
        try:
            result = wait_for_pause(Path(result.get("run_dir", run)))
        except KeyboardInterrupt:
            print("Ожидание прервано. Запрос паузы сохранён; проверьте status перед поездкой.")
            return 130
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("safe_to_travel"):
        if result.get("no_training_started"):
            print("Обучение ещё не начиналось, процесс остановлен. Сохранять шаги не требуется; можно закрывать ноутбук.")
        else:
            print("Сохранение проверено, процесс остановлен. Можно закрывать ноутбук.")
    elif args.action == "pause":
        print("Пауза запрошена. Дождитесь завершения шага и safe_to_travel: true в status.")
    return 1 if result.get("wait_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
