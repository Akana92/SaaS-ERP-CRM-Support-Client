from __future__ import annotations

import time
from typing import Any, Literal, Protocol, TypedDict

from support.contracts import (
    BLOCKING_ERP_STATUSES,
    AdminResponse,
    ClientResponse,
    ClientStatus,
    ERPContext,
    ModelCallUsage,
    ModelResult,
    PolicyDocument,
    ScenarioId,
    ServerRoute,
    TraceEvent,
    validate_evidence,
)


FALLBACK_RESPONSE = "Нужна проверка оператором."


class ModelRunLike(Protocol):
    raw_text: str | None
    result: ModelResult | None
    usage: ModelCallUsage | None
    error: str | None


class SupportRunner(Protocol):
    def generate(
        self,
        model_input: dict[str, Any],
        request_id: str,
        node: str = "model_call",
        max_new_tokens: int = 512,
    ) -> ModelRunLike:
        ...


class SupportGraphState(TypedDict, total=False):
    request_id: str
    session_id: str
    scenario_id: ScenarioId
    customer_message: str
    erp_context: dict[str, Any]
    policy_version: str
    policy_rules: list[dict[str, str]]
    raw_response: str | None
    model_result: dict[str, Any] | None
    server_route: ServerRoute
    response: str
    check_reasons: list[str]
    usage_calls: list[dict[str, Any]]
    trace_events: list[dict[str, Any]]
    model_error: str | None


def _duration_ms(started_at: float) -> int:
    return max(0, round((time.perf_counter() - started_at) * 1000))


def _trace(
    state: SupportGraphState,
    *,
    node: str,
    description: str,
    status: Literal["ok", "skipped", "failed"],
    started_at: float,
    error: str | None = None,
    call_id: str | None = None,
    token_count: int | None = None,
    evidence_ids: list[str] | None = None,
) -> None:
    event = TraceEvent(
        node=node,
        description=description,
        request_id=state["request_id"],
        status=status,
        duration_ms=_duration_ms(started_at),
        error=error,
        call_id=call_id,
        token_count=token_count,
        evidence_ids=evidence_ids or [],
    )
    state.setdefault("trace_events", []).append(event.model_dump(mode="json"))


def _policy_ids(policy: PolicyDocument) -> set[str]:
    return {rule.id for rule in policy.rules}


def _model_input(state: SupportGraphState) -> dict[str, Any]:
    return {
        "customer_message": state["customer_message"],
        "erp_context": state["erp_context"],
        "policy_version": state["policy_version"],
        "policy_rules": state["policy_rules"],
    }


def _server_response_for_error(reason: str) -> str:
    if reason == "model_output_invalid":
        return "Модель не вернула корректный результат. Нужна проверка оператором."
    if reason == "model_runtime_error":
        return "Модель временно недоступна. Нужна проверка оператором."
    return FALLBACK_RESPONSE


def _route_from_model(result: ModelResult) -> ServerRoute:
    if result.human_escalation:
        return "human_escalation"
    if result.recommended_action == "request_information":
        return "request_information"
    return "auto_answer"


def _client_status(route: ServerRoute) -> ClientStatus:
    if route == "request_information":
        return "needs_information"
    if route == "human_escalation":
        return "human_escalation"
    return "answered"


def _prepare_context_node(policy: PolicyDocument):
    def prepare_context(state: SupportGraphState) -> SupportGraphState:
        started_at = time.perf_counter()
        erp_context = ERPContext.model_validate(state["erp_context"])
        state["erp_context"] = erp_context.model_dump(mode="json")
        state["policy_version"] = policy.version
        state["policy_rules"] = [rule.model_dump(mode="json") for rule in policy.rules]
        state.setdefault("usage_calls", [])
        state.setdefault("trace_events", [])
        state.setdefault("check_reasons", [])
        _trace(
            state,
            node="prepare_context",
            description="Собран безопасный контекст политики и демо-ERP.",
            status="ok",
            started_at=started_at,
            evidence_ids=list(erp_context.facts),
        )
        if erp_context.source_status in BLOCKING_ERP_STATUSES:
            state["server_route"] = "human_escalation"
            state["response"] = FALLBACK_RESPONSE
            state["check_reasons"].append(f"erp_{erp_context.source_status}")
            _trace(
                state,
                node="model_call",
                description="Модель не вызвана из-за заблокированного статуса ERP.",
                status="skipped",
                started_at=started_at,
                error=f"erp_{erp_context.source_status}",
            )
        return state

    return prepare_context


def _should_call_model(state: SupportGraphState) -> Literal["model_call", "route"]:
    if state.get("server_route") == "human_escalation":
        return "route"
    return "model_call"


def _model_call_node(runner: SupportRunner):
    def model_call(state: SupportGraphState) -> SupportGraphState:
        started_at = time.perf_counter()
        run = runner.generate(_model_input(state), state["request_id"], node="model_call", max_new_tokens=512)
        if run.raw_text is not None:
            state["raw_response"] = run.raw_text
        if run.usage is not None:
            state.setdefault("usage_calls", []).append(run.usage.model_dump(mode="json"))
        if run.result is not None:
            state["model_result"] = run.result.model_dump(mode="json")
        if run.error:
            state["model_error"] = run.error
            state["check_reasons"].append(
                "model_output_invalid" if "json" in run.error.lower() or "valid" in run.error.lower() else "model_runtime_error"
            )
            _trace(
                state,
                node="model_call",
                description="Модель вызвана, но результат не прошёл runtime-проверку.",
                status="failed",
                started_at=started_at,
                error=state["check_reasons"][-1],
                call_id=run.usage.call_id if run.usage is not None else None,
                token_count=run.usage.total_tokens if run.usage is not None else None,
            )
        else:
            _trace(
                state,
                node="model_call",
                description="Модель вернула структурированный результат.",
                status="ok",
                started_at=started_at,
                call_id=run.usage.call_id if run.usage is not None else None,
                token_count=run.usage.total_tokens if run.usage is not None else None,
                evidence_ids=run.result.evidence_ids if run.result is not None else [],
            )
        return state

    return model_call


def _check_result_node(policy: PolicyDocument):
    def check_result(state: SupportGraphState) -> SupportGraphState:
        started_at = time.perf_counter()
        if state.get("model_error") or state.get("model_result") is None:
            if not state.get("check_reasons"):
                state["check_reasons"].append("model_runtime_error")
            reason = state["check_reasons"][-1]
            state["server_route"] = "human_escalation"
            state["response"] = _server_response_for_error(reason)
            _trace(
                state,
                node="check_result",
                description="Сервер направил обращение оператору после ошибки модели.",
                status="failed",
                started_at=started_at,
                error=reason,
            )
            return state

        erp_context = ERPContext.model_validate(state["erp_context"])
        result = ModelResult.model_validate(state["model_result"])
        try:
            validate_evidence(result, erp_context, _policy_ids(policy))
        except ValueError:
            state["check_reasons"].append("model_output_invalid")
            state["server_route"] = "human_escalation"
            state["response"] = _server_response_for_error("model_output_invalid")
            _trace(
                state,
                node="check_result",
                description="Серверная проверка отклонила ссылки на источники.",
                status="failed",
                started_at=started_at,
                error="model_output_invalid",
                evidence_ids=result.evidence_ids,
            )
            return state

        state["server_route"] = _route_from_model(result)
        state["response"] = FALLBACK_RESPONSE if state["server_route"] == "human_escalation" else result.suggested_response
        _trace(
            state,
            node="check_result",
            description="Серверные проверки результата прошли.",
            status="ok",
            started_at=started_at,
            evidence_ids=result.evidence_ids,
        )
        return state

    return check_result


def _route_node(state: SupportGraphState) -> SupportGraphState:
    started_at = time.perf_counter()
    state.setdefault("server_route", "human_escalation")
    state.setdefault("response", FALLBACK_RESPONSE)
    _trace(
        state,
        node="route",
        description="Сформирован итоговый маршрут обращения.",
        status="ok",
        started_at=started_at,
    )
    return state


def build_support_graph(runner: SupportRunner, policy: PolicyDocument):
    try:
        from langgraph.graph import END, START, StateGraph
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when dependency is absent
        raise RuntimeError("LangGraph is required for Stage 2 support graph") from exc

    graph = StateGraph(SupportGraphState)
    graph.add_node("prepare_context", _prepare_context_node(policy))
    graph.add_node("model_call", _model_call_node(runner))
    graph.add_node("check_result", _check_result_node(policy))
    graph.add_node("route", _route_node)
    graph.add_edge(START, "prepare_context")
    graph.add_conditional_edges("prepare_context", _should_call_model, {"model_call": "model_call", "route": "route"})
    graph.add_edge("model_call", "check_result")
    graph.add_edge("check_result", "route")
    graph.add_edge("route", END)
    return graph.compile()


def _to_admin_response(state: SupportGraphState) -> AdminResponse:
    model_result = ModelResult.model_validate(state["model_result"]) if state.get("model_result") else None
    return AdminResponse(
        request_id=state["request_id"],
        raw_model_result=model_result,
        raw_response=state.get("raw_response"),
        server_route=state.get("server_route", "human_escalation"),
        response=state.get("response") or FALLBACK_RESPONSE,
        usage_calls=[ModelCallUsage.model_validate(call) for call in state.get("usage_calls", [])],
        trace_events=[TraceEvent.model_validate(event) for event in state.get("trace_events", [])],
    )


def _to_client_response(admin: AdminResponse) -> ClientResponse:
    if admin.raw_model_result is None:
        return ClientResponse(response=admin.response, status="fallback", escalation=True, handoff_id=None, analysis=None)
    return ClientResponse.from_model_result(
        admin.raw_model_result,
        status=_client_status(admin.server_route),
        server_response=admin.response if admin.server_route == "human_escalation" else None,
        handoff_id=None,
    )


def invoke_support_graph(
    runner: SupportRunner,
    policy: PolicyDocument,
    *,
    customer_message: str,
    erp_context: ERPContext,
    request_id: str,
    session_id: str,
    scenario_id: ScenarioId,
) -> tuple[AdminResponse, ClientResponse]:
    compiled = build_support_graph(runner, policy)
    state: SupportGraphState = {
        "request_id": request_id,
        "session_id": session_id,
        "scenario_id": scenario_id,
        "customer_message": customer_message,
        "erp_context": erp_context.model_dump(mode="json"),
        "trace_events": [],
        "usage_calls": [],
        "check_reasons": [],
        "raw_response": None,
        "model_result": None,
        "model_error": None,
    }
    result_state = compiled.invoke(state)
    admin = _to_admin_response(result_state)
    return admin, _to_client_response(admin)
