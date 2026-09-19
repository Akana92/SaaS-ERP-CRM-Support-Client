"""Plan/run resumable raw-model Quality90 validation; sealed test is explicitly gated."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from support.dialogue_dataset import load_dialogue_policy
from support.quality_evaluation import (atomic_json, authorize_cases, bind_final_gate,
    blind_review_packet, digest, evaluate, file_sha, load_cases, read_records, scoped_path, summarize,
    validate_final_gate)

CODE = ("src/support/quality_evaluation.py", "scripts/evaluate_quality_candidate.py",
        "src/support/dialogue_dataset.py", "src/support/dialogue_evaluation.py",
        "src/support/live_dialogue.py", "src/support/live_chat.py", "src/support/live_precision.py",
        "src/support/modeling.py", "src/support/contracts.py", "src/support/prompting.py",
        "src/support/graph.py")


def series_path():
    return ROOT / "artifacts/stage5/quality90-v1"


def validate_evaluation_release(cases_path, purpose):
    """Check root-approved v2 metadata without reading any case/model contents.

    Receipt evaluation_release_v2.json schema:
      {schema: quality90-evaluation-release-v1, approved: true,
       files: {rubric: {path, sha256}, manifest: {path, sha256},
               isolation: {path, sha256}, correction_reviews: [{path, sha256}, ...]},
       approved_splits: {validation: {path, sha256}, final_test: {path, sha256}}}.
    All paths are project-relative or absolute inside ROOT. Required artifacts
    have canonical v2 paths. At least two distinct correction-review files are
    byte-hashed only; their private text is never parsed here. Split metadata is
    checked here; selected case bytes are hashed only after authorize_cases has
    enforced the independent final-test gate. The other split is never opened.
    """
    release_path = series_path() / "evaluation_release_v2.json"
    if not release_path.is_file():
        raise ValueError("Approved evaluation v2 release is missing; keep evaluation on hold")
    raw = release_path.read_bytes()
    release = json.loads(raw)
    if (not isinstance(release, dict) or release.get("schema") != "quality90-evaluation-release-v1"
            or release.get("approved") is not True):
        raise ValueError("Evaluation v2 release is not explicitly approved")

    def entry(value, *, expected=None, check_hash=False):
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ValueError("Release file entry requires path and SHA256")
        sha = value.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Release file SHA256 must be lowercase hexadecimal")
        path = scoped_path(ROOT / value["path"], ROOT)
        if expected and path != (ROOT / expected).resolve():
            raise ValueError("Release artifact does not match its canonical v2 path")
        if check_hash and (not path.is_file() or file_sha(path) != sha):
            raise ValueError(f"Released artifact SHA256 mismatch: {path.relative_to(ROOT)}")
        return path, sha

    files = release.get("files")
    if not isinstance(files, dict):
        raise ValueError("Release must list immutable supporting artifacts")
    for role, expected in {"rubric": "docs/QUALITY90_EVALUATION_PROTOCOL.md",
                           "manifest": "data/quality90_v1/evaluation_manifest_v2.json",
                           "isolation": "data/quality90_v1/isolation_manifest_v2.json"}.items():
        entry(files.get(role), expected=expected, check_hash=True)
    reviews = files.get("correction_reviews")
    if not isinstance(reviews, list) or len(reviews) < 2:
        raise ValueError("Release requires independent validation and final-test correction reviews")
    review_paths = []
    for review in reviews:
        path, _ = entry(review)
        if path.suffix != ".json":
            raise ValueError("Correction review must be a JSON report, not case/model contents")
        review_paths.append(path)
    if len(set(review_paths)) != len(review_paths):
        raise ValueError("Correction-review paths must be distinct")
    for review in reviews:
        entry(review, check_hash=True)
    splits = release.get("approved_splits")
    if not isinstance(splits, dict) or set(splits) != {"validation", "final_test"}:
        raise ValueError("Release must approve exactly validation and final_test")
    approved = {name: entry(splits[name], expected=path) for name, path in {
        "validation": "data/quality90_v1/validation/v2/development.jsonl",
        "final_test": "data/quality90_v1/sealed/v2/final_test.jsonl"}.items()}
    if purpose not in approved or (ROOT / cases_path).resolve() != approved[purpose][0]:
        raise ValueError("Requested cases are not the exact approved v2 split for this purpose")
    import hashlib
    return {"evaluation_release_sha256": hashlib.sha256(raw).hexdigest(),
            "cases_sha256": approved[purpose][1]}


def evaluation_session_budget(max_hours, *, now=None):
    """Carry earlier training time into evaluation; automatic resumes never reset it."""
    now = time.time() if now is None else now
    path = series_path() / "continuous_session.json"
    if not path.is_file():
        raise ValueError("A verified continuous session window is required before evaluation")
    window = json.loads(path.read_text(encoding="utf-8"))
    names = ("started_at_unix", "soft_stop_at_unix", "hard_limit_at_unix", "maximum_hours")
    if (window.get("schema") != "quality90-continuous-session-v1"
            or any(type(window.get(name)) not in {int, float} or not math.isfinite(window[name])
                   for name in names)):
        raise ValueError("Invalid continuous session window")
    start, soft, hard, maximum = (window[name] for name in names)
    if (not 0 < max_hours <= 20 or not math.isfinite(max_hours) or maximum != 20
            or not start <= now or not start < soft < hard <= start + 20 * 3600):
        raise ValueError("Invalid or extended continuous session limits")
    hard_seconds = min(max_hours * 3600, hard - now)
    soft_seconds = min(hard_seconds - 60, soft - now)
    if soft_seconds <= 0:
        raise ValueError("Session budget exhausted; pause until explicit continuation after travel")
    return {"window_sha256": file_sha(path), "measured_at_unix": now,
            "hard_seconds": hard_seconds, "soft_seconds": soft_seconds,
            "hard_stop_at_unix": now + hard_seconds, "soft_stop_at_unix": now + soft_seconds}


def evaluation_pause_requested(output):
    return (series_path() / "USER_PAUSED.json").exists() or (output / "pause.request").exists()


@contextmanager
def active_evaluation(output, *, resume=False, resume_command=None):
    """Publish exact OS ownership before GPU work; terminal state is not exit proof."""
    from quality_training_control import process_identity, alive
    from training_control import gpu_lock
    series = series_path()
    output = scoped_path(output, series / "evaluation")
    pointer = series / "active_evaluation.json"
    owner = None
    with gpu_lock(series / "series-launch.lock"):
        if (series / "USER_PAUSED.json").exists():
            raise RuntimeError("User paused the series; evaluator cannot clear or bypass that pause")
        if pointer.exists():
            previous = json.loads(pointer.read_text(encoding="utf-8"))
            if previous.get("schema_version") != "quality90-active-evaluation-v1":
                raise RuntimeError("Existing evaluator ownership is unverified")
            scoped_path(previous["run_dir"], series / "evaluation")
            if alive(previous):
                raise RuntimeError("An evaluator is still active; wait for its actual process exit")
        own_identity = process_identity(os.getpid())
        if not own_identity:
            raise RuntimeError("Cannot verify evaluator process identity")
        pause = output / "pause.request"
        if pause.exists():
            if not resume:
                raise RuntimeError("Evaluation pause is durable; explicit resume required")
            os.replace(pause, output / f"pause-consumed-{time.time_ns()}.request")
        owner = {"schema_version": "quality90-active-evaluation-v1", "run_dir": str(output),
                 "pid": os.getpid(), "identity": own_identity, "status": "starting",
                 "started_at_unix": time.time(), "updated_at_unix": time.time()}
        if resume_command is not None:
            owner["resume_command"] = list(resume_command)
        atomic_json(pointer, owner)

    def update(status):
        with gpu_lock(series / "series-launch.lock"):
            current = json.loads(pointer.read_text(encoding="utf-8"))
            if (current.get("pid"), current.get("identity"), current.get("run_dir")) != (
                    owner["pid"], owner["identity"], owner["run_dir"]):
                raise RuntimeError("Evaluator ownership changed; refusing to overwrite another process")
            owner.update(status=status, updated_at_unix=time.time())
            atomic_json(pointer, owner)
    try:
        yield update
    except BaseException:
        update("failed")
        raise
    finally:
        # Keep PID/creation time in place. Controller verifies process death;
        # even status=completed is not permission to move an active process.
        if owner["status"] not in {"paused", "completed", "failed"}:
            update("failed")


def source_model_hashes(model_path):
    """Fresh content fingerprints, including every shard and tokenizer input."""
    model_path = Path(model_path).resolve()
    paths = sorted(path for path in model_path.iterdir()
                   if path.is_file() and path.suffix in {".json", ".safetensors", ".jinja", ".txt"})
    if not any(path.suffix == ".safetensors" for path in paths):
        raise ValueError("local base model has no safetensors weights")
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for name in set(index["weight_map"].values()):
            shard = scoped_path(model_path / name, model_path)
            if shard not in paths:
                raise ValueError("model index references a missing or unsupported shard")
    return {path.name: file_sha(path) for path in paths}


def resume_command(args, output):
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "run",
               "--cases", str(args.cases), "--adapter", str(args.adapter.resolve()),
               "--output", str(output), "--mode", args.mode, "--history", args.history,
               "--profile", args.profile, "--purpose", args.purpose,
               "--max-hours", str(args.max_hours), "--resume"]
    if args.final_gate:
        command += ["--final-gate", str(args.final_gate)]
    return command


def build_identity(args, policy, config_path):
    from support.app import load_model_config
    config = load_model_config(config_path, "qwen3_4b")
    return {"schema_version": "quality90-run-v1", "purpose": args.purpose,
            "cases_sha256": file_sha(args.cases), "mode": args.mode, "history": args.history,
            "profile": args.profile, "max_new_tokens": 512, "do_sample": False,
            "config_sha256": file_sha(config_path), "model_key": "qwen3_4b",
            "base_model_sha256": source_model_hashes(ROOT / config["local_path"]),
            "runtime_versions": {package: version(package) for package in
                                 ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
            "adapter_model_sha256": file_sha(args.adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": file_sha(args.adapter / "adapter_config.json"),
            "policy_sha256": digest(policy.model_dump(mode="json")),
            "rubric_sha256": file_sha(ROOT / "docs/QUALITY90_EVALUATION_PROTOCOL.md"),
            "code_sha256": {name: file_sha(ROOT / name) for name in CODE}}


def save_report(output, cases, rows, identity, reviews_path=None):
    reviews = []
    if reviews_path:
        reviews = [json.loads(line) for line in reviews_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = summarize(cases, rows, reviews, rubric_sha256=identity["rubric_sha256"])
    report["identity_sha256"] = digest(identity)
    for key in ("mode", "history", "adapter_model_sha256"):
        report[key] = identity[key]
    report["reviews_sha256"] = file_sha(reviews_path) if reviews_path else None
    atomic_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "report", "pause"), nargs="?", default="plan")
    parser.add_argument("--cases", type=Path, default=ROOT / "data/quality90_v1/validation/v2/development.jsonl")
    parser.add_argument("--purpose", choices=("validation", "final_test"), default="validation")
    parser.add_argument("--mode", choices=("base", "fine_tuned"), default="fine_tuned")
    parser.add_argument("--history", choices=("free", "controlled"), default="free")
    parser.add_argument("--profile", choices=("nf4", "bf16"), default="nf4")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--reviews", type=Path, help="JSONL semantic sidecars; no automatic judge")
    parser.add_argument("--final-gate", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 20:
        parser.error("0 < max-hours <= 20 required")
    output = scoped_path(args.output, ROOT / "artifacts/stage5/quality90-v1/evaluation")
    if args.action == "pause":
        if not (output / "manifest.json").is_file():
            parser.error("pause requires an existing Quality90 run")
        from training_control import gpu_lock
        with gpu_lock(series_path() / "series-launch.lock"):
            receipt = {"requested_at": time.time(), "reason": "user_pause", "evaluation_run_dir": str(output)}
            training_pointer = series_path() / "active_training.json"
            if training_pointer.exists():
                training = json.loads(training_pointer.read_text(encoding="utf-8"))
                receipt["run_dir"] = str(scoped_path(training["run_dir"], series_path() / "training"))
            latch = series_path() / "USER_PAUSED.json"
            if not latch.exists():
                atomic_json(latch, receipt)
            try:
                with (output / "pause.request").open("x", encoding="utf-8") as handle:
                    handle.write("User requested safe pause after the current generation.\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError:
                pass
        print("Pause requested; wait for Training Status safe_to_travel=true with verified process exit.")
        return 0
    if args.action == "run":
        if (series_path() / "evaluation_hold.json").exists():
            raise RuntimeError("Evaluation is on hold for independent gold audit; do not bypass or clear it automatically")
        from training_control import gpu_lock
        with gpu_lock(series_path() / "series-launch.lock"):
            if (series_path() / "USER_PAUSED.json").exists():
                raise RuntimeError("User paused the series; explicit controller acknowledgement required")
    if args.adapter is None:
        parser.error("--adapter required to bind both Base and candidate to the same comparison")
    release = validate_evaluation_release(args.cases, args.purpose)
    gate = None
    if args.final_gate:
        args.final_gate = scoped_path(args.final_gate, ROOT / "data/quality90_v1/sealed")
        gate = json.loads(args.final_gate.read_text(encoding="utf-8"))
    args.cases = authorize_cases(args.cases, ROOT, args.purpose, gate)
    if file_sha(args.cases) != release["cases_sha256"]:
        raise ValueError("Approved evaluation split SHA256 changed; do not evaluate modified cases")
    policy = load_dialogue_policy()
    config_path = ROOT / "configs/models.json"
    identity = build_identity(args, policy, config_path)
    identity["evaluation_release_sha256"] = release["evaluation_release_sha256"]
    if args.purpose == "final_test":
        validate_final_gate(args.final_gate, identity)
        if args.action == "plan":
            print(json.dumps({"purpose": "final_test", "sealed_contents_read": False, "identity": identity}, indent=2))
            return 0
    cases = load_cases(args.cases, policy)
    if args.purpose == "final_test" and len(cases) < 200:
        raise ValueError("final test requires at least 200 complete cases")
    if args.action == "plan":
        print(json.dumps({"cases": len(cases), "turns": sum(len(case.turns) for case in cases),
                          "identity": identity, "loads_model": False}, ensure_ascii=False, indent=2))
        return 0
    from training_control import gpu_lock
    output.mkdir(parents=True, exist_ok=True)
    with gpu_lock(output / ".run.lock"):
        manifest = output / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8"))["identity"] != identity:
                raise ValueError("immutable identity changed; use a new validation run")
            if args.action == "run" and not args.resume:
                raise ValueError("existing run requires --resume")
        elif args.action == "report" or args.resume:
            raise ValueError("no run manifest to resume/report")
        else:
            if any(path.name != ".run.lock" for path in output.iterdir()):
                raise ValueError("unrecognized existing work in output")
            if args.purpose == "final_test":
                # Serialize all Base/candidate slots so two processes cannot
                # create conflicting one-time bindings.
                with gpu_lock(args.final_gate.with_suffix(".lock")):
                    bind_final_gate(args.final_gate, identity, output)
            atomic_json(manifest, {"identity": identity, "created_at_unix": time.time()})
        if args.purpose == "final_test":
            with gpu_lock(args.final_gate.with_suffix(".lock")):
                bind_final_gate(args.final_gate, identity, output)
        rows = read_records(output, cases, args.mode, args.history)
        for row in rows:
            packet_path = output / "review-packets" / f"{row['review_key']}.json"
            if not packet_path.exists():
                # Repair a crash after the durable result but before its derived
                # blind packet, without repeating inference or changing the row.
                atomic_json(packet_path, blind_review_packet(row))
        if args.action == "report":
            print(json.dumps(save_report(output, cases, rows, identity, args.reviews), indent=2))
            return 0
        if len(rows) == sum(len(case.turns) for case in cases):
            save_report(output, cases, rows, identity, args.reviews)
            print("All generations already saved; no model loaded.")
            return 0
        from live_demo import offline_environment, assert_no_legacy_inference
        from benchmark_live_runtime import _assert_no_live_server_process
        from training_control import assert_no_unmanaged_training, prevent_idle_sleep
        from train_dialogue_pilot import session_deadline
        from support.app import load_model_config
        from support.live_precision import LocalPrecisionRunner
        os.environ.update(offline_environment())
        budget_started = time.monotonic()
        budget = evaluation_session_budget(args.max_hours)
        with active_evaluation(output, resume=args.resume, resume_command=resume_command(args, output)) as update_owner:
            deadline = budget_started + budget["soft_seconds"]
            atomic_json(output / f"session-budget-{time.time_ns()}.json", budget)
            runner, status = None, "failed"
            try:
                with (gpu_lock(ROOT / "artifacts/stage4/portable-gpu.lock"),
                      prevent_idle_sleep(), session_deadline(max(.001,
                          budget_started + budget["hard_seconds"] - time.monotonic()))):
                    if evaluation_pause_requested(output) or time.monotonic() >= deadline:
                        status = "paused"
                        return 0
                    assert_no_unmanaged_training()
                    assert_no_legacy_inference()
                    _assert_no_live_server_process()
                    config = load_model_config(config_path, "qwen3_4b")
                    config["local_path"] = str((ROOT / config["local_path"]).resolve())
                    runner = LocalPrecisionRunner(config, str(args.adapter) if args.mode == "fine_tuned" else None,
                                                  inference_profile=args.profile)
                    with closing(runner):
                        atomic_json(output / "status.json", {"status": "loading", "completed_turns": len(rows)})
                        update_owner("running")
                        runner.load()
                        def save(index, row):
                            atomic_json(output / "results" / f"{index:05d}.json", row)
                            atomic_json(output / "review-packets" / f"{row['review_key']}.json", blind_review_packet(row))
                            atomic_json(output / "status.json", {"status": "running", "completed_turns": index + 1})
                            print(json.dumps({"completed": index + 1, "dialogue_id": row["dialogue_id"]}), flush=True)
                        rows, status = evaluate(runner, cases, policy, args.mode, args.history, saved=rows,
                            on_result=save, should_pause=lambda: evaluation_pause_requested(output) or time.monotonic() >= deadline)
                    runner = None
            finally:
                rows = read_records(output, cases, args.mode, args.history)
                save_report(output, cases, rows, identity, args.reviews)
                atomic_json(output / "status.json", {"status": status, "completed_turns": len(rows),
                                                     "ended_at_unix": time.time()})
                update_owner(status)
            print(json.dumps({"status": status, "output": str(output)}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
