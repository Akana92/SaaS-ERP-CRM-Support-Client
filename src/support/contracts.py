from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Annotated


NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

Category = Literal[
    "Bug",
    "Plans",
    "Settings",
    "AccountAccess",
    "Integration",
    "Payment",
    "ServiceIncident",
    "Other",
]
Priority = Literal["Low", "Medium", "High", "Critical"]
Sentiment = Literal["Positive", "Neutral", "Negative"]
RecommendedAction = Literal[
    "provide_instructions",
    "explain_plan",
    "check_payment",
    "review_access",
    "troubleshoot_integration",
    "request_information",
    "escalate_human",
]
EscalationReason = Literal[
    "payment_confirmed_but_access_not_active",
    "payment_dispute",
    "access_change_requires_operator",
    "security_risk",
    "service_outage",
    "integration_requires_operator",
    "bug_requires_operator",
    "missing_or_conflicting_facts",
    "untrusted_instruction",
    "model_output_invalid",
    "model_runtime_error",
    "erp_unavailable",
    "erp_stale",
    "erp_forbidden",
]
ModelMode = Literal["base", "fine_tuned"]
SourceStatus = Literal["ok", "not_found", "unavailable", "stale", "conflict", "forbidden"]
ServerRoute = Literal["auto_answer", "request_information", "human_escalation"]
ClientStatus = Literal["answered", "needs_information", "human_escalation", "fallback"]
ScenarioId = Literal[
    "information",
    "access",
    "payment",
    "integration",
    "incident",
    "uncertain",
]
TraceStatus = Literal["ok", "skipped", "failed"]
ReviewStatus = Literal["pending_human_review", "human_approved"]
DataSource = Literal["synthetic_authored"]
Split = Literal["development"]
FactValue = str | bool | int | float
BLOCKING_ERP_STATUSES = {"unavailable", "stale", "conflict", "forbidden"}


class StrictContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )


class ModelResult(StrictContract):
    category: Category
    priority: Priority
    sentiment: Sentiment
    recommended_action: RecommendedAction
    suggested_response: NonBlankStr
    human_escalation: bool
    escalation_reason: EscalationReason | None
    evidence_ids: list[NonBlankStr]

    @model_validator(mode="after")
    def validate_escalation_reason(self) -> "ModelResult":
        if self.human_escalation and self.escalation_reason is None:
            raise ValueError("escalation_reason is required when human_escalation is true")
        if not self.human_escalation and self.escalation_reason is not None:
            raise ValueError("escalation_reason must be null when human_escalation is false")
        if self.recommended_action == "escalate_human" and not self.human_escalation:
            raise ValueError("recommended_action escalate_human requires human_escalation true")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique")
        return self


class ERPContext(StrictContract):
    source_status: SourceStatus
    facts: dict[NonBlankStr, FactValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_facts(self) -> "ERPContext":
        for evidence_id, value in self.facts.items():
            if not evidence_id.startswith("erp."):
                raise ValueError("ERP fact evidence ids must start with 'erp.'")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("ERP numeric facts must be finite")
        if self.source_status != "ok" and self.facts:
            raise ValueError("ERP facts must be empty unless source_status is ok")
        return self


class SupportRequest(StrictContract):
    message: NonBlankStr


class AdminDemoRequest(StrictContract):
    message: NonBlankStr
    scenario_id: ScenarioId
    model_mode: ModelMode


class PolicyRule(StrictContract):
    id: NonBlankStr
    text: NonBlankStr


class PolicyDocument(StrictContract):
    version: NonBlankStr
    rules: list[PolicyRule]

    @model_validator(mode="after")
    def validate_rule_ids(self) -> "PolicyDocument":
        rule_ids = [rule.id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("policy rule ids must be unique")
        return self


class ModelCallUsage(StrictContract):
    call_id: NonBlankStr
    request_id: NonBlankStr
    node: NonBlankStr
    model_id: NonBlankStr
    revision: NonBlankStr
    adapter_id: NonBlankStr | None
    mode: ModelMode
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: int | None
    api_cost: float | None
    currency: Literal["USD"]
    complete: bool

    @model_validator(mode="after")
    def validate_usage(self) -> "ModelCallUsage":
        usage_fields = [
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.latency_ms,
            self.api_cost,
        ]
        if self.complete and any(value is None for value in usage_fields):
            raise ValueError("complete calls must have token, latency, and cost values")
        if not self.complete and self.total_tokens == 0:
            raise ValueError("unknown incomplete usage must be null, not zero")
        for name in ("input_tokens", "output_tokens", "total_tokens", "latency_ms"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.api_cost is not None:
            if not math.isfinite(self.api_cost) or self.api_cost < 0:
                raise ValueError("api_cost must be a finite non-negative value")
            if self.api_cost != 0:
                raise ValueError("local stage1 contract only allows zero API cost")
        if None not in (self.input_tokens, self.output_tokens, self.total_tokens):
            if self.input_tokens + self.output_tokens != self.total_tokens:
                raise ValueError("total_tokens must equal input_tokens + output_tokens")
        if self.mode == "base" and self.adapter_id is not None:
            raise ValueError("base mode must not include adapter_id")
        if self.mode == "fine_tuned" and self.adapter_id is None:
            raise ValueError("fine_tuned mode requires adapter_id")
        return self


class ClientAnalysis(StrictContract):
    category: Category
    priority: Priority
    sentiment: Sentiment
    recommended_action: RecommendedAction


class ClientResponse(StrictContract):
    response: NonBlankStr
    status: ClientStatus
    escalation: bool
    handoff_id: NonBlankStr | None = None
    analysis: ClientAnalysis | None = None

    @classmethod
    def from_model_result(
        cls,
        result: ModelResult,
        status: ClientStatus,
        *,
        server_response: NonBlankStr | None = None,
        handoff_id: NonBlankStr | None = None,
    ) -> "ClientResponse":
        if status in {"human_escalation", "fallback"} and server_response is None:
            raise ValueError("server_response is required for escalation and fallback responses")
        response_text = server_response if server_response is not None else result.suggested_response
        analysis = None
        if status != "fallback":
            analysis = ClientAnalysis(
                category=result.category,
                priority=result.priority,
                sentiment=result.sentiment,
                recommended_action=result.recommended_action,
            )
        return cls(
            response=response_text,
            status=status,
            escalation=result.human_escalation,
            handoff_id=handoff_id,
            analysis=analysis,
        )

    @model_validator(mode="after")
    def validate_status(self) -> "ClientResponse":
        if self.status == "answered":
            if self.escalation or self.handoff_id is not None or self.analysis is None:
                raise ValueError("answered responses require analysis and no escalation or handoff_id")
            if self.analysis.recommended_action in {"request_information", "escalate_human"}:
                raise ValueError("answered responses require an answer action")
        elif self.status == "needs_information":
            if self.escalation or self.handoff_id is not None or self.analysis is None:
                raise ValueError("needs_information responses require analysis and no escalation or handoff_id")
            if self.analysis.recommended_action != "request_information":
                raise ValueError("needs_information requires request_information action")
        elif self.status == "human_escalation":
            if not self.escalation:
                raise ValueError("human_escalation status requires escalation true")
        elif self.status == "fallback":
            if self.analysis is not None:
                raise ValueError("fallback responses must not include model analysis")
            if self.handoff_id is not None and not self.escalation:
                raise ValueError("fallback with handoff_id requires escalation true")
        return self


class TraceEvent(StrictContract):
    node: NonBlankStr
    description: NonBlankStr
    request_id: NonBlankStr
    status: TraceStatus
    duration_ms: int = Field(ge=0)
    error: NonBlankStr | None = None
    call_id: NonBlankStr | None = None
    token_count: int | None = Field(default=None, ge=0)
    evidence_ids: list[NonBlankStr] = Field(default_factory=list)


class AdminResponse(StrictContract):
    request_id: NonBlankStr
    raw_model_result: ModelResult | None
    raw_response: str | None = None
    server_route: ServerRoute
    response: NonBlankStr
    usage_calls: list[ModelCallUsage] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_usage_and_trace(self) -> "AdminResponse":
        call_ids = [call.call_id for call in self.usage_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("usage call_id values must be unique")
        for call in self.usage_calls:
            if call.request_id != self.request_id:
                raise ValueError("usage request_id must match admin response request_id")
        for event in self.trace_events:
            if event.request_id != self.request_id:
                raise ValueError("trace request_id must match admin response request_id")
            if event.call_id is not None and event.call_id not in call_ids:
                raise ValueError("trace call_id must refer to a real usage call_id")
        complete_calls = [call for call in self.usage_calls if call.complete]
        if self.raw_model_result is not None and not complete_calls:
            raise ValueError("raw_model_result requires at least one complete usage call")
        if self.server_route in {"auto_answer", "request_information"}:
            if self.raw_model_result is None:
                raise ValueError("auto_answer and request_information require raw_model_result")
            if not complete_calls:
                raise ValueError("auto_answer and request_information require at least one complete usage call")
            if self.server_route == "auto_answer":
                if self.raw_model_result.human_escalation:
                    raise ValueError("auto_answer requires a non-escalating raw model result")
                if self.raw_model_result.recommended_action in {"request_information", "escalate_human"}:
                    raise ValueError("auto_answer requires an answer action")
            if self.server_route == "request_information":
                if self.raw_model_result.human_escalation:
                    raise ValueError("request_information requires a non-escalating raw model result")
                if self.raw_model_result.recommended_action != "request_information":
                    raise ValueError("request_information route requires request_information action")
        if self.raw_model_result is None:
            if self.server_route != "human_escalation":
                raise ValueError("raw_model_result can be null only for human_escalation")
            if not any(event.status in {"skipped", "failed"} for event in self.trace_events):
                raise ValueError("raw null human escalation requires skipped or failed trace evidence")
            if self.usage_calls and not any(
                event.status == "failed" and event.call_id in call_ids for event in self.trace_events
            ):
                raise ValueError("raw null usage calls require failed trace evidence for the call")
        return self


class SupportState(StrictContract):
    request_id: NonBlankStr
    session_id: NonBlankStr
    scenario_id: ScenarioId
    customer_message: NonBlankStr
    model_mode: ModelMode
    erp_context: ERPContext | None = None
    policy_version: NonBlankStr | None = None
    policy_refs: list[NonBlankStr] = Field(default_factory=list)
    raw_response: str | None = None
    model_result: ModelResult | None = None
    server_route: ServerRoute | None = None
    check_reasons: list[NonBlankStr] = Field(default_factory=list)
    usage_calls: list[ModelCallUsage] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())


class DevCase(StrictContract):
    id: NonBlankStr
    family_id: NonBlankStr
    split: Split
    source: DataSource
    review_status: ReviewStatus
    scenario_id: ScenarioId
    customer_message: NonBlankStr
    org_id: NonBlankStr
    erp_context: ERPContext
    policy_version: NonBlankStr
    policy_refs: list[NonBlankStr] = Field(default_factory=list)
    expected: ModelResult | None
    expected_route: ServerRoute
    expected_model_call: bool
    fallback_response: NonBlankStr | None = None
    notes_for_reviewer: NonBlankStr
    covered_risks: list[NonBlankStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expected_and_fallback(self) -> "DevCase":
        blocked_erp = self.erp_context.source_status in BLOCKING_ERP_STATUSES
        if blocked_erp:
            if self.expected_model_call:
                raise ValueError("blocking ERP statuses must skip the model call")
            if self.expected is not None:
                raise ValueError("blocking ERP statuses require expected null")
            if self.expected_route != "human_escalation":
                raise ValueError("blocking ERP statuses require human_escalation route")
            if self.fallback_response is None:
                raise ValueError("blocking ERP statuses require fallback_response")
        if self.expected_model_call and self.expected is None:
            raise ValueError("expected is required when expected_model_call is true")
        if not self.expected_model_call and self.expected is not None:
            raise ValueError("expected must be null when expected_model_call is false")
        if not self.expected_model_call and self.fallback_response is None:
            raise ValueError("fallback_response is required when no model call is expected")
        if self.expected_model_call and self.fallback_response is not None:
            raise ValueError("model-call development cases must not include fallback_response")
        if self.expected is not None:
            if self.expected_route == "auto_answer":
                if self.expected.human_escalation:
                    raise ValueError("auto_answer route requires non-escalating expected result")
                if self.expected.recommended_action in {"request_information", "escalate_human"}:
                    raise ValueError("auto_answer route requires an answer action")
            if self.expected_route == "request_information":
                if self.expected.human_escalation:
                    raise ValueError("request_information route requires non-escalating expected result")
                if self.expected.recommended_action != "request_information":
                    raise ValueError("request_information route requires request_information action")
            if self.expected_route == "human_escalation":
                if not self.expected.human_escalation:
                    raise ValueError("human_escalation route requires escalating expected result")
        return self


def validate_evidence(
    result: ModelResult,
    erp_context: ERPContext,
    policy_ids: set[str] | list[str] | tuple[str, ...],
) -> None:
    allowed_ids = set(erp_context.facts) | set(policy_ids)
    unknown = [evidence_id for evidence_id in result.evidence_ids if evidence_id not in allowed_ids]
    if unknown:
        raise ValueError(f"unknown evidence ids: {', '.join(unknown)}")


def model_input_from_case(case: DevCase, policy_document: PolicyDocument) -> dict[str, Any]:
    if not case.expected_model_call:
        raise ValueError("model input is not available for no-LLM development cases")
    if case.policy_version != policy_document.version:
        raise ValueError("policy version mismatch")
    return {
        "customer_message": case.customer_message,
        "erp_context": case.erp_context.model_dump(mode="json"),
        "policy_version": case.policy_version,
        "policy_rules": [rule.model_dump(mode="json") for rule in policy_document.rules],
    }


SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    ModelResult,
    ERPContext,
    SupportRequest,
    AdminDemoRequest,
    PolicyRule,
    PolicyDocument,
    ModelCallUsage,
    ClientAnalysis,
    ClientResponse,
    TraceEvent,
    AdminResponse,
    SupportState,
    DevCase,
)


def export_json_schemas(output_dir: str | Path) -> list[Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in SCHEMA_MODELS:
        path = target / f"{model.__name__}.schema.json"
        schema_text = json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)
        path.write_text(schema_text + "\n", encoding="utf-8")
        written.append(path)
    return written
