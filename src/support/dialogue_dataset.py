"""CPU-only dialogue fixtures and controlled-history preparation.

Gold history is a teacher-forced diagnostic, not free-running model evaluation.
Pass actual previous public responses as ``history`` for free-running evaluation.
Neither mode includes historical gold classifications in the prompt.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from support.contracts import (
    BLOCKING_ERP_STATUSES, ERPContext, ModelResult, NonBlankStr, PolicyDocument,
    StrictContract, validate_evidence,
)
from support.graph import FALLBACK_RESPONSE, _client_status, _route_from_model
from support.live_dialogue import (
    DialogueCompositionMetadata, TokenCounter, compose_dialogue_messages,
    create_live_dialogue_policy,
)
from support.training import EncodedExample, encode_completion_only, load_policy_document, model_result_target_json

ROOT = Path(__file__).resolve().parents[2]
DialogueSplit = Literal["pilot_train", "development"]


class DialogueTurn(StrictContract):
    message: NonBlankStr
    erp_context: ERPContext
    expected: ModelResult
    required_points: list[NonBlankStr] = Field(min_length=1)
    forbidden_behaviors: list[NonBlankStr] = Field(min_length=1)
    note: NonBlankStr

    @model_validator(mode="after")
    def model_call_available(self) -> "DialogueTurn":
        if self.erp_context.source_status in BLOCKING_ERP_STATUSES:
            raise ValueError("blocking ERP statuses skip model calls and cannot have a model target")
        return self


class Dialogue(StrictContract):
    id: NonBlankStr
    family_id: NonBlankStr
    split: DialogueSplit
    audience: Literal["employee", "customer"]
    source: Literal["synthetic_authored"]
    review_status: Literal["pending_human_review", "human_approved"]
    focus: list[NonBlankStr] = Field(min_length=1)
    turns: list[DialogueTurn] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def consistent_audience(self) -> "Dialogue":
        for turn in self.turns:
            audience = turn.erp_context.facts.get("erp.requester.audience")
            if audience is not None and audience != self.audience:
                raise ValueError("erp.requester.audience does not match dialogue audience")
        return self


def load_dialogue_policy() -> PolicyDocument:
    return create_live_dialogue_policy(load_policy_document(ROOT / "data/policy/employee-telecom-v3.json"))


def load_dialogues(path: str | Path, expected_split: DialogueSplit | None = None) -> list[Dialogue]:
    if expected_split is not None and expected_split not in {"pilot_train", "development"}:
        raise ValueError("unsupported expected split")
    policy = load_dialogue_policy()
    result: list[Dialogue] = []
    ids: set[str] = set()
    family_splits: dict[str, str] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = Dialogue.model_validate(json.loads(line))
            if expected_split is not None and item.split != expected_split:
                raise ValueError(f"expected split {expected_split}, got {item.split}")
            if item.id in ids:
                raise ValueError(f"duplicate dialogue ID {item.id}")
            if item.family_id in family_splits and family_splits[item.family_id] != item.split:
                raise ValueError(f"family crosses splits: {item.family_id}")
            for turn in item.turns:
                validate_evidence(turn.expected, turn.erp_context, [rule.id for rule in policy.rules])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        ids.add(item.id)
        family_splits[item.family_id] = item.split
        result.append(item)
    if not result:
        raise ValueError(f"empty dialogue file: {path}")
    return result


def validate_split_isolation(train: list[Dialogue], development: list[Dialogue]) -> None:
    if any(d.split != "pilot_train" for d in train) or any(d.split != "development" for d in development):
        raise ValueError("split isolation violation")
    ids = [d.id for d in train + development]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dialogue IDs across input sets")
    overlap = {d.family_id for d in train} & {d.family_id for d in development}
    if overlap:
        raise ValueError(f"families cross splits: {sorted(overlap)}")


def public_history(dialogue: Dialogue, turn_index: int) -> list[dict[str, Any]]:
    if not 0 <= turn_index < len(dialogue.turns):
        raise IndexError("turn_index outside dialogue")
    history = []
    for index, turn in enumerate(dialogue.turns[:turn_index]):
        route = _route_from_model(turn.expected)
        history.append({
            "request_id": f"{dialogue.id}:turn-{index + 1}", "message": turn.message,
            "client": {"response": FALLBACK_RESPONSE if route == "human_escalation" else turn.expected.suggested_response,
                       "status": _client_status(route)},
        })
    return history


@dataclass(frozen=True)
class PreparedTurn:
    case_id: str
    split: DialogueSplit
    history_mode: str
    model_input: dict[str, Any]
    chat_messages: list[dict[str, str]]
    metadata: DialogueCompositionMetadata
    target_json: str


def prepare_turn(
    dialogue: Dialogue, turn_index: int, policy: PolicyDocument, token_counter: TokenCounter,
    history: list[dict[str, Any]] | None = None,
) -> PreparedTurn:
    if not 0 <= turn_index < len(dialogue.turns):
        raise IndexError("turn_index outside dialogue")
    policy = create_live_dialogue_policy(policy)
    turn = dialogue.turns[turn_index]
    validate_evidence(turn.expected, turn.erp_context, [rule.id for rule in policy.rules])
    controlled = history is None
    if controlled:
        history = public_history(dialogue, turn_index)
    else:
        if len(history) != turn_index:
            raise ValueError("explicit history must contain exactly the preceding turns")
        # Only public response/status are accepted; caller cannot inject gold labels.
        cleaned = []
        for index, previous in enumerate(history):
            if previous.get("message") != dialogue.turns[index].message:
                raise ValueError("explicit history must match preceding fixture messages")
            client = previous.get("client", {})
            if not isinstance(client, dict) or not isinstance(client.get("response"), str) or not client["response"].strip():
                raise ValueError("explicit history requires a complete public response")
            if client.get("status") not in {"answered", "needs_information", "human_escalation", "fallback"}:
                raise ValueError("explicit history requires a valid public status")
            request_id = previous.get("request_id")
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("explicit history requires a nonblank public request_id")
            cleaned.append({"request_id": request_id, "message": previous["message"],
                            "client": {"response": client["response"], "status": client["status"]}})
        history = cleaned
    model_input = {
        "customer_message": turn.message, "erp_context": turn.erp_context.model_dump(mode="json"),
        "policy_version": policy.version, "policy_rules": [r.model_dump(mode="json") for r in policy.rules],
    }
    messages, metadata = compose_dialogue_messages(model_input, history, token_counter)
    return PreparedTurn(f"{dialogue.id}:turn-{turn_index + 1}", dialogue.split,
                        "controlled_gold" if controlled else "free_running", model_input, messages, metadata,
                        model_result_target_json(turn.expected))


def encode_prepared_turn(prepared: PreparedTurn, tokenizer: Any, *, max_length: int = 8192) -> EncodedExample:
    if prepared.split != "pilot_train":
        raise ValueError("only pilot_train may be encoded as training features")
    if prepared.history_mode != "controlled_gold":
        raise ValueError("training packing requires controlled gold history")
    if prepared.metadata.omitted_turn_ids:
        raise ValueError("training packing would omit required prior context")
    return encode_completion_only(tokenizer, prepared.chat_messages, prepared.target_json,
                                  case_id=prepared.case_id, max_length=max_length)


def duplication_audit(train: list[Dialogue], development: list[Dialogue], threshold: float = 0.85) -> dict[str, Any]:
    """Audit only the new two splits; similarity is a review flag, not a verdict."""
    validate_split_isolation(train, development)
    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
        return " ".join(re.findall(r"\w+", re.sub(r"\d+", "#", text)))

    exact, normalized, similar = [], [], []
    for left in train:
        for right in development:
            for li, lt in enumerate(left.turns):
                for ri, rt in enumerate(right.turns):
                    pair = {"pilot_train": f"{left.id}:turn-{li + 1}", "development": f"{right.id}:turn-{ri + 1}"}
                    a, b = normalize(lt.message), normalize(rt.message)
                    if lt.message == rt.message:
                        exact.append(pair)
                    if a == b:
                        normalized.append(pair)
                    score = SequenceMatcher(None, a, b, autojunk=False).ratio()
                    if score >= threshold:
                        similar.append({**pair, "similarity": round(score, 4)})
    return {"scope": "new pilot_train vs development user turns only", "similarity_method": "NFKC casefold punctuation/numeric normalization + SequenceMatcher",
            "threshold": threshold, "exact": exact, "normalized": normalized, "similar": similar,
            "human_review_required": True}
