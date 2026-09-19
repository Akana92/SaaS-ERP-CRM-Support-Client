"""Development dialogue evaluation; never score server substitutions as model wins."""
from __future__ import annotations

import copy
from collections import Counter
from typing import Callable

from support.dialogue_dataset import prepare_turn, public_history


FIELDS = ("category", "priority", "sentiment", "recommended_action", "human_escalation")
HISTORY_MODES = ("controlled", "free")


def planned_steps(dialogues, modes):
    if not modes or len(modes) != len(set(modes)) or set(modes) - {"base", "fine_tuned"}:
        raise ValueError("modes must contain unique base/fine_tuned values")
    return [(mode, dialogue.id, index) for mode in modes for dialogue in dialogues
            for index in range(len(dialogue.turns))]


def validate_prefix(records, steps, history_mode):
    if len(records) > len(steps):
        raise ValueError("saved results exceed planned steps")
    for index, row in enumerate(records):
        if (row["mode"], row["dialogue_id"], row["turn_index"]) != steps[index]:
            raise ValueError("saved results are not the exact planned prefix")
        if row["history_mode"] != history_mode:
            raise ValueError("cannot mix history modes")


def evaluate_dialogues(service, dialogues, modes, history_mode, *,
                       saved_records=None, on_result: Callable | None = None,
                       should_pause: Callable[[], bool] | None = None):
    """Run through the live service without opening HTTP or writing its user store.

    Controlled history uses prior public gold responses. Free history uses only
    this model's previous actual public responses, including server fallbacks.
    The latter measures the existing assistant pipeline, not raw-model-only chat.
    """
    if history_mode not in HISTORY_MODES:
        raise ValueError("history_mode must be controlled or free")
    if any(dialogue.split != "development" for dialogue in dialogues):
        raise ValueError("dialogue evaluation accepts development only")
    steps = planned_steps(dialogues, modes)
    records = copy.deepcopy(saved_records or [])
    validate_prefix(records, steps, history_mode)
    lookup = {dialogue.id: dialogue for dialogue in dialogues}
    if len(lookup) != len(dialogues):
        raise ValueError("duplicate dialogue IDs")
    histories = {}
    for row in records:
        histories.setdefault((row["mode"], row["dialogue_id"]), []).append(row["record"])

    for mode, dialogue_id, turn_index in steps[len(records):]:
        if should_pause is not None and should_pause():
            return records, "paused"
        dialogue = lookup[dialogue_id]
        turn = dialogue.turns[turn_index]
        history = (public_history(dialogue, turn_index) if history_mode == "controlled"
                   else copy.deepcopy(histories.get((mode, dialogue_id), [])))
        prepared = prepare_turn(dialogue, turn_index, service.policy,
                                service.runtime.count_chat_prompt_tokens, history=history)
        context = {"id": dialogue.audience, "erp_context": turn.erp_context.model_dump(mode="json")}
        record = service.run(turn.message, context, f"dev-{mode}-{dialogue.id}", mode, history=history)
        actual_hash = record.get("dialogue_context", {}).get("effective_input_sha256")
        expected_hash = prepared.metadata.effective_input_sha256
        row = {
            "schema_version": "dialogue-evaluation-v1", "split": "development",
            "mode": mode, "history_mode": history_mode,
            "dialogue_id": dialogue.id, "family_id": dialogue.family_id,
            "turn_index": turn_index, "review_status": "pending_human_review",
            "input_messages": prepared.chat_messages,
            "input_sha256": expected_hash, "input_matches_live": actual_hash == expected_hash,
            "expected": turn.expected.model_dump(mode="json"),
            "required_points": list(turn.required_points),
            "forbidden_behaviors": list(turn.forbidden_behaviors),
            "human_review": {"uses_known_details": None, "useful_next_step": None,
                             "unnecessary_repeat": None, "unsupported_claim": None,
                             "notes": ""},
            "record": record,
        }
        if on_result is not None:
            on_result(len(records), row)
        records.append(row)
        histories.setdefault((mode, dialogue_id), []).append(record)
        if not row["input_matches_live"]:
            # Persist the actual result before failing; never silently score a
            # different input than the one the review report claims was used.
            raise ValueError("prepared prompt differs from the actual live prompt")
    return records, "completed"


def summarize(records, planned_count):
    modes = {}
    for mode in ("base", "fine_tuned"):
        rows = [row for row in records if row["mode"] == mode]
        if not rows:
            continue
        valid = 0
        correct = Counter()
        routes = Counter()
        errors = Counter()
        total_tokens = 0
        known_token_calls = 0
        unknown_token_calls = 0
        latency_ms = 0
        latency_calls = 0
        for row in rows:
            admin = row["record"]["admin"]
            result = admin.get("raw_model_result")
            if result is not None and row["input_matches_live"]:
                valid += 1
                for field in FIELDS:
                    correct[field] += result.get(field) == row["expected"][field]
                correct["all_five"] += all(result.get(f) == row["expected"][f] for f in FIELDS)
            routes[admin["server_route"]] += 1
            errors.update({event["error"] for event in admin.get("trace_events", []) if event.get("error")})
            for usage in admin.get("usage_calls", []):
                if usage.get("total_tokens") is None:
                    unknown_token_calls += 1
                else:
                    total_tokens += usage["total_tokens"]
                    known_token_calls += 1
                if usage.get("latency_ms") is not None:
                    latency_ms += usage["latency_ms"]
                    latency_calls += 1
        modes[mode] = {"turns": len(rows), "valid_model_results": valid,
                       "correct": {field: correct[field] for field in (*FIELDS, "all_five")},
                       "accuracy_denominator": len(rows), "server_routes": dict(routes),
                       "errors": dict(errors), "known_total_tokens": total_tokens,
                       "known_token_calls": known_token_calls, "unknown_token_calls": unknown_token_calls,
                       "mean_model_latency_ms": latency_ms / latency_calls if latency_calls else None,
                       "human_review": "pending_human_review", "response_quality": None,
                       "hallucination_rate": None}
    return {"schema_version": "dialogue-evaluation-summary-v1", "split": "development",
            "planned_turns": planned_count, "completed_turns": len(records),
            "history_modes": sorted({row["history_mode"] for row in records}), "modes": modes,
            "notes": ["Invalid or missing model results count as incorrect, not as removed samples.",
                      "Server handoff and repeat guards are separate from raw model predictions.",
                      "Semantic quality and hallucination rate require human review.",
                      "Controlled histories are gold; free histories include actual public server fallbacks.",
                      "Authored fixture ERP facts are controlled inputs, not retrieved live knowledge."]}
