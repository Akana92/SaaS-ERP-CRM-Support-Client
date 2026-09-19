"""Quality90 raw-model evaluation. No training loader or automatic semantic judge."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from support.contracts import validate_evidence
from support.dialogue_dataset import Dialogue, prepare_turn
from support.dialogue_evaluation import FIELDS, planned_steps, validate_prefix
from support.graph import FALLBACK_RESPONSE, _client_status, _route_from_model

SEMANTIC_FIELDS = ("understandable", "useful_next_step", "uses_context", "no_unnecessary_repeat",
                   "grounded", "policy_and_handoff_consistent")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def scoped_path(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError(f"path must be inside {parent}")
    return path


def authorize_cases(path, root, purpose, gate=None):
    """Reject training and sealed reads before opening any case contents."""
    if purpose not in {"validation", "final_test"}:
        raise ValueError("unsupported evaluation purpose")
    path = scoped_path(path, Path(root) / "data/quality90_v1" / ("sealed" if purpose == "final_test" else purpose))
    if purpose == "final_test":
        if not gate or gate.get("schema_version") != "quality90-final-gate-v1" or gate.get("approved") is not True:
            raise ValueError("sealed final test requires an approved final-gate manifest")
        if Path(gate["cases_path"]).resolve() != path or gate["cases_sha256"] != file_sha(path):
            raise ValueError("final gate dataset mismatch")
    return path


def load_cases(path, policy):
    cases = [Dialogue.model_validate(json.loads(line))
             for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("empty or duplicate case IDs")
    for case in cases:
        if case.split != "development":
            raise ValueError("evaluation never accepts training rows")
        for turn in case.turns:
            validate_evidence(turn.expected, turn.erp_context, [rule.id for rule in policy.rules])
    return cases


def raw_history(row, message):
    result = row["raw_model_result"]
    # A failed raw call remains a failed scored turn. The placeholder only permits
    # further fixed scenario turns; no server guard can turn it into a win.
    if result is None:
        response, status = FALLBACK_RESPONSE, "fallback"
    else:
        from support.contracts import ModelResult
        model = ModelResult.model_validate(result)
        response, status = model.suggested_response, _client_status(_route_from_model(model))
    return {"request_id": row["request_id"], "message": message,
            "client": {"response": response, "status": status}}


def evaluate(runner, cases, policy, mode, history, *, saved=(), on_result=None,
             should_pause=lambda: False, max_new_tokens=512):
    if history not in {"free", "controlled"}:
        raise ValueError("invalid history mode")
    steps = planned_steps(cases, [mode])
    rows = list(saved)
    validate_prefix(rows, steps, history)
    for row in rows:
        verify_record(row)
    counter = lambda messages: len(runner.tokenizer.apply_chat_template(messages, add_generation_prompt=True))
    index = 0
    for case in cases:
        previous = []
        for turn_index, turn in enumerate(case.turns):
            if should_pause():
                return rows, "paused"
            prepared = prepare_turn(case, turn_index, policy, counter,
                                    history=previous if history == "free" else None)
            if index < len(rows):
                row = rows[index]
                if (row["input_messages"] != prepared.chat_messages
                        or row["input_sha256"] != prepared.metadata.effective_input_sha256
                        or row["expected"] != turn.expected.model_dump(mode="json")):
                    raise ValueError("saved prefix prompt or expected result changed")
            else:
                request_id = f"quality90-{digest([case.id, turn_index])[:24]}"
                call = runner.generate_chat(prepared.chat_messages, prepared.model_input,
                                            request_id, max_new_tokens=max_new_tokens)
                row = {"schema_version": "quality90-result-v1", "mode": mode, "history_mode": history,
                       "dialogue_id": case.id, "family_id": case.family_id, "turn_index": turn_index,
                       "request_id": request_id, "input_messages": prepared.chat_messages,
                       "input_sha256": prepared.metadata.effective_input_sha256,
                       "expected": turn.expected.model_dump(mode="json"),
                       "required_points": list(turn.required_points),
                       "forbidden_behaviors": list(turn.forbidden_behaviors),
                       "raw_response": call.raw_text,
                       "raw_model_result": call.result.model_dump(mode="json") if call.result else None,
                       "usage": call.usage.model_dump(mode="json"), "error": call.error}
                row["record_sha256"] = digest(row)
                row["review_key"] = digest(["blind-review", row["record_sha256"]])
                if on_result:
                    on_result(index, row)
                rows.append(row)
            previous.append(raw_history(row, turn.message))
            index += 1
    return rows, "completed"


def verify_record(row):
    body = {key: value for key, value in row.items() if key not in {"record_sha256", "review_key"}}
    if row.get("record_sha256") != digest(body):
        raise ValueError("saved record integrity mismatch")
    if row.get("review_key") != digest(["blind-review", row["record_sha256"]]):
        raise ValueError("saved review key mismatch")


def read_records(output, cases, mode, history):
    paths = sorted((Path(output) / "results").glob("*.json"))
    if [p.name for p in paths] != [f"{index:05d}.json" for index in range(len(paths))]:
        raise ValueError("result files are not a contiguous prefix")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    validate_prefix(rows, planned_steps(cases, [mode]), history)
    for row in rows:
        verify_record(row)
    return rows


def blind_review_packet(row):
    """No mode, adapter, model usage or desired overall pass rate reaches judge."""
    return {"review_key": row["review_key"], "record_sha256": row["record_sha256"],
            "input_messages": row["input_messages"], "raw_response": row["raw_response"],
            "required_points": row["required_points"], "forbidden_behaviors": row["forbidden_behaviors"],
            "criteria": list(SEMANTIC_FIELDS)}


def summarize(cases, rows, reviews=(), *, rubric_sha256):
    lookup = {}
    valid_keys = {row["review_key"] for row in rows}
    for review in reviews:
        key = review["review_key"]
        if key in lookup or key not in valid_keys:
            raise ValueError("duplicate or unknown semantic review")
        if (review.get("rubric_sha256") != rubric_sha256 or review.get("kind") not in {"ai", "human"}
                or not str(review.get("reviewer", "")).strip() or not str(review.get("notes", "")).strip()):
            raise ValueError("review requires matching rubric, kind, reviewer and explanation")
        if set(review.get("criteria", {})) != set(SEMANTIC_FIELDS):
            raise ValueError("semantic review criteria incomplete")
        if any(value is not None and type(value) is not bool for value in review["criteria"].values()):
            raise ValueError("criteria values must be bool or null")
        lookup[key] = review
    by_turn = {(row["dialogue_id"], row["turn_index"]): row for row in rows}
    if len(by_turn) != len(rows):
        raise ValueError("cannot combine modes or duplicate turn records in one score")
    expected_keys = {(case.id, index) for case in cases for index in range(len(case.turns))}
    if set(by_turn) - expected_keys:
        raise ValueError("records contain unplanned case turns")
    correct = dict.fromkeys((*FIELDS, "all_five", "contract_valid"), 0)
    success = unknown = 0
    kinds = set()
    for case in cases:
        states = []
        for index in range(len(case.turns)):
            row = by_turn.get((case.id, index))
            if row is None:
                states.append(None)
                continue
            verify_record(row)
            if row["expected"] != case.turns[index].expected.model_dump(mode="json"):
                raise ValueError("record gold differs from the frozen case")
            raw = row["raw_model_result"]
            valid = raw is not None and row["error"] is None and row["usage"].get("complete") is True
            correct["contract_valid"] += int(valid)
            labels = valid and all(raw[field] == row["expected"][field] for field in FIELDS)
            for field in FIELDS:
                correct[field] += int(valid and raw[field] == row["expected"][field])
            correct["all_five"] += int(labels)
            review = lookup.get(row["review_key"])
            if review:
                if review.get("record_sha256") != row["record_sha256"]:
                    raise ValueError("semantic review targets a different answer")
                kinds.add(review["kind"])
            checks = list(review["criteria"].values()) if review else [None]
            states.append(False if not labels or False in checks else None if None in checks else True)
        success += int(all(value is True for value in states))
        unknown += int(False not in states and None in states)
    complete = len(rows) == sum(len(case.turns) for case in cases)
    fully_reviewed = len(lookup) == len(rows) and all(
        all(value is not None for value in review["criteria"].values()) for review in lookup.values())
    status = "human_verified" if complete and fully_reviewed and kinds == {"human"} else "ai_assessed" if complete and fully_reviewed and kinds == {"ai"} else "incomplete_or_mixed_review"
    rate = success / len(cases)
    return {"schema_version": "quality90-summary-v1", "planned_cases": len(cases),
            "planned_turns": sum(len(case.turns) for case in cases), "completed_turns": len(rows),
            "successful_cases": success, "unknown_cases": unknown,
            "full_success_rate": rate if complete and fully_reviewed else None,
            "confirmed_success_lower_bound": rate, "correct_turns": correct,
            "generation_complete": complete, "assessment": status, "rubric_sha256": rubric_sha256,
            "provisional_90": complete and fully_reviewed and rate >= .9,
            "human_verified_90": status == "human_verified" and rate >= .9}


def validate_final_gate(gate_path, identity):
    """Validate the gate without reading sealed cases or mutating its binding."""
    gate_path = Path(gate_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != "quality90-final-gate-v1" or gate.get("approved") is not True:
        raise ValueError("final gate is not approved")
    frozen = {key: value for key, value in identity.items() if key not in {"mode", "history"}}
    if gate.get("identity") != frozen:
        raise ValueError("final gate identity mismatch")
    report_path = Path(gate["validation_summary_path"])
    if file_sha(report_path) != gate["validation_summary_sha256"]:
        raise ValueError("validation summary changed")
    summary = json.loads(report_path.read_text(encoding="utf-8"))
    if (summary.get("provisional_90") is not True or summary.get("planned_cases", 0) < 100
            or summary.get("generation_complete") is not True
            or summary.get("history") != "free" or summary.get("mode") != "fine_tuned"
            or summary.get("adapter_model_sha256") != identity.get("adapter_model_sha256")
            or summary.get("rubric_sha256") != identity.get("rubric_sha256")
            or summary.get("assessment") not in {"ai_assessed", "human_verified"}
            or not isinstance(summary.get("full_success_rate"), (int, float))
            or not .9 <= summary["full_success_rate"] <= 1):
        raise ValueError("final test needs complete >=90% development assessment on >=100 cases")
    return gate


def bind_final_gate(gate_path, identity, output):
    """One immutable candidate/protocol binding, one output per mode/history slot."""
    gate_path = Path(gate_path)
    validate_final_gate(gate_path, identity)
    binding_path = gate_path.with_suffix(".binding.json")
    slot = f"{identity['mode']}:{identity['history']}"
    if binding_path.exists():
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if binding["gate_sha256"] != file_sha(gate_path):
            raise ValueError("a consumed final gate cannot be replaced")
    else:
        binding = {"gate_sha256": file_sha(gate_path), "outputs": {}}
    prior = binding["outputs"].get(slot)
    if prior is not None and prior != str(Path(output).resolve()):
        raise ValueError("final evaluation already bound to another output; resume it")
    binding["outputs"][slot] = str(Path(output).resolve())
    atomic_json(binding_path, binding)
