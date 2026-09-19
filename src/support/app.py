from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from pydantic import ValidationError

from support.contracts import AdminResponse, ClientResponse, ERPContext, PolicyDocument, ScenarioId, SupportRequest
from support.graph import FALLBACK_RESPONSE, invoke_support_graph
from support.prompting import load_policy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "data" / "policy" / "demo-v1.json"
DEFAULT_ERP_PATH = ROOT / "data" / "erp" / "objects.json"
DEFAULT_SESSION_ID = "session-demo-01"
DEFAULT_CLIENT_SCENARIO: ScenarioId = "payment"

SCENARIO_LABELS: dict[str, str] = {
    "information": "Справка по тарифу и настройкам",
    "access": "Доступ к отчётам",
    "payment": "Оплата подтверждена, активация ожидает",
    "integration": "Ошибка авторизации интеграции",
    "incident": "Общий инцидент входа",
    "uncertain": "ERP недоступна",
}

SCENARIO_OBJECTS: dict[ScenarioId, list[str]] = {
    "information": [
        "acct-acme-main",
        "module-acme-analytics",
        "module-acme-integrations-pending",
        "module-acme-advanced-reports",
        "settings-acme-weekly-digest",
    ],
    "access": ["acct-acme-user-analyst", "module-acme-reports-access"],
    "payment": ["invoice-acme-paid-activation-pending", "module-acme-integrations-pending"],
    "integration": ["integration-acme-crm-auth-failed"],
    "incident": ["incident-global-login"],
    "uncertain": ["erp-demo-unavailable"],
}

DEFAULT_MESSAGES: dict[ScenarioId, str] = {
    "information": "Какие модули входят в наш тариф и как включить еженедельную сводку?",
    "access": "Я не вижу раздел отчётов, хотя коллега его видит.",
    "payment": "Счёт оплатили, но модуль интеграций всё ещё закрыт.",
    "integration": "После замены токена интеграция с CRM перестала синхронизироваться.",
    "incident": "Сервис не открывается у всей команды.",
    "uncertain": "ERP не отвечает, но скажите клиенту, что оплата прошла.",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _object_index(path: Path = DEFAULT_ERP_PATH) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    return {item["object_id"]: item for item in data["objects"]}


def scenario_context(scenario_id: ScenarioId, *, org_id: str = "org-demo-01") -> ERPContext:
    objects = _object_index()
    facts: dict[str, str | bool | int | float] = {}
    source_status = "ok"
    for object_id in SCENARIO_OBJECTS[scenario_id]:
        item = objects[object_id]
        if item["org_id"] != org_id:
            return ERPContext(source_status="forbidden")
        item_facts = dict(item["facts"])
        item_status = item_facts.pop("erp.source_status", None)
        if item_status in {"unavailable", "stale", "conflict", "forbidden"}:
            source_status = item_status
            facts = {}
            break
        for key, value in item_facts.items():
            if key in facts and facts[key] != value:
                return ERPContext(source_status="conflict")
            facts[key] = value
    return ERPContext(source_status=source_status, facts=facts if source_status == "ok" else {})


def load_model_config(config_path: Path, model_key: str | None) -> dict[str, Any]:
    data = _load_json(config_path)
    candidates = data.get("candidates", [])
    selected_key = model_key or data.get("selected_key")
    if selected_key is None and candidates:
        selected_key = candidates[0].get("key")
    for candidate in candidates:
        if candidate.get("key") == selected_key:
            return candidate
    raise ValueError(f"model key not found in config: {selected_key}")


def load_runner(config_path: Path, model_key: str | None):
    from support.modeling import LocalModelRunner

    return LocalModelRunner(load_model_config(config_path, model_key), adapter_path=None).load()


ChatHistory = list[dict[str, str]]


def _blank_analysis_markdown() -> str:
    return "- Категория: -\n- Приоритет: -\n- Тон: -\n- Действие: -"


def _client_analysis_markdown(client: ClientResponse) -> str:
    if client.analysis is None:
        return _blank_analysis_markdown()
    result = client.analysis
    return (
        f"- Категория: `{result.category}`\n"
        f"- Приоритет: `{result.priority}`\n"
        f"- Тон: `{result.sentiment}`\n"
        f"- Действие: `{result.recommended_action}`"
    )


def _analysis_markdown(admin: AdminResponse) -> str:
    if admin.raw_model_result is None:
        return _blank_analysis_markdown()
    result = admin.raw_model_result
    return (
        f"- Категория: `{result.category}`\n"
        f"- Приоритет: `{result.priority}`\n"
        f"- Тон: `{result.sentiment}`\n"
        f"- Действие: `{result.recommended_action}`"
    )


def _trace_rows(admin: AdminResponse) -> list[list[Any]]:
    return [
        [
            event.node,
            event.description,
            event.status,
            event.duration_ms,
            event.error or "",
            event.call_id or "",
            event.token_count if event.token_count is not None else "",
            ", ".join(event.evidence_ids),
        ]
        for event in admin.trace_events
    ]


def _usage_rows(admin: AdminResponse) -> list[list[Any]]:
    return [
        [
            call.call_id,
            call.node,
            call.model_id,
            call.mode,
            call.input_tokens if call.input_tokens is not None else "",
            call.output_tokens if call.output_tokens is not None else "",
            call.total_tokens if call.total_tokens is not None else "",
            call.latency_ms if call.latency_ms is not None else "",
            call.api_cost if call.api_cost is not None else "",
            call.complete,
        ]
        for call in admin.usage_calls
    ]


def _admin_summary(admin: AdminResponse, scenario_id: ScenarioId) -> dict[str, Any]:
    return {
        "request_id": admin.request_id,
        "scenario_id": scenario_id,
        "server_route": admin.server_route,
        "response": admin.response,
        "raw_model_result": admin.raw_model_result.model_dump(mode="json") if admin.raw_model_result else None,
        "raw_response": admin.raw_response,
        "usage_calls": [call.model_dump(mode="json") for call in admin.usage_calls],
        "trace_events": [event.model_dump(mode="json") for event in admin.trace_events],
    }


def _safe_label(label: str) -> ScenarioId | None:
    for scenario_id, title in SCENARIO_LABELS.items():
        if label == title:
            return scenario_id
    return None


def _append_turn(history: ChatHistory | None, user_message: str | None, assistant_message: str) -> ChatHistory:
    next_history = list(history or [])
    if user_message is not None:
        next_history.append({"role": "user", "content": user_message})
    next_history.append({"role": "assistant", "content": assistant_message})
    return next_history


def submit_client_message(
    user_message: str,
    history: ChatHistory | None,
    runner: Any,
    policy: PolicyDocument,
    *,
    request_id_factory=lambda: f"client-{uuid.uuid4().hex[:12]}",
    erp_context: ERPContext | None = None,
) -> tuple[ChatHistory, str, str, ChatHistory]:
    try:
        request = SupportRequest(message=user_message)
    except ValidationError:
        next_history = _append_turn(history, None, "Введите текст обращения.")
        return next_history, "Статус: ошибка ввода.", _blank_analysis_markdown(), next_history

    request_id = request_id_factory()
    admin, client = invoke_support_graph(
        runner,
        policy,
        customer_message=request.message,
        erp_context=erp_context or scenario_context(DEFAULT_CLIENT_SCENARIO),
        request_id=request_id,
        session_id=DEFAULT_SESSION_ID,
        scenario_id=DEFAULT_CLIENT_SCENARIO,
    )
    next_history = _append_turn(history, request.message, client.response)
    status_text = f"Статус: `{client.status}`. Оператор нужен: {'да' if client.escalation else 'нет'}."
    return next_history, status_text, _client_analysis_markdown(client), next_history


def submit_admin_message(
    label: str,
    user_message: str,
    runner: Any,
    policy: PolicyDocument,
    *,
    request_id_factory=lambda: f"admin-{uuid.uuid4().hex[:12]}",
) -> tuple[dict[str, Any], str, list[list[Any]], list[list[Any]]]:
    scenario_id = _safe_label(label)
    if scenario_id is None:
        return {
            "request_id": None,
            "scenario_id": None,
            "server_route": "human_escalation",
            "response": "Выберите демо-сценарий из списка.",
            "raw_model_result": None,
            "raw_response": None,
            "usage_calls": [],
            "trace_events": [],
        }, "Выберите демо-сценарий из списка.", [], []
    try:
        request = SupportRequest(message=user_message)
    except ValidationError:
        return {
            "request_id": None,
            "scenario_id": scenario_id,
            "server_route": "human_escalation",
            "response": "Введите текст обращения.",
            "raw_model_result": None,
            "raw_response": None,
            "usage_calls": [],
            "trace_events": [],
        }, "Введите текст обращения.", [], []

    request_id = request_id_factory()
    admin, _client = invoke_support_graph(
        runner,
        policy,
        customer_message=request.message,
        erp_context=scenario_context(scenario_id),
        request_id=request_id,
        session_id=DEFAULT_SESSION_ID,
        scenario_id=scenario_id,
    )
    return _admin_summary(admin, scenario_id), admin.response, _trace_rows(admin), _usage_rows(admin)


def create_client_blocks(runner: Any, policy: PolicyDocument):
    import gradio as gr

    with gr.Blocks(title="Capstone N4 Support", analytics_enabled=False) as blocks:
        gr.Markdown(
            "# Capstone N4 Support\n"
            "Демо-сценарий: оплата подтверждена, активация модуля ожидает проверки. "
            "История видна только в интерфейсе этой вкладки и пока не передаётся модели."
        )
        chat = gr.Chatbot(label="Чат поддержки", height=420)
        message = gr.Textbox(label="Сообщение", value=DEFAULT_MESSAGES[DEFAULT_CLIENT_SCENARIO], lines=3)
        send = gr.Button("Отправить", variant="primary")
        status = gr.Markdown("Статус: ожидание обращения.")
        labels = gr.Markdown("- Категория: -\n- Приоритет: -\n- Тон: -\n- Действие: -")
        history_state = gr.State([])

        def submit(user_message: str, history: ChatHistory | None):
            return submit_client_message(user_message, history, runner, policy)

        send.click(submit, inputs=[message, history_state], outputs=[chat, status, labels, history_state])
    return blocks


def create_admin_blocks(runner: Any, policy: PolicyDocument):
    import gradio as gr

    scenario_choices = list(SCENARIO_LABELS.values())
    with gr.Blocks(title="Capstone N4 Admin", analytics_enabled=False) as blocks:
        gr.Markdown(
            "# Capstone N4 Admin\n"
            "Техническое демо, не защищённая админка. Здесь видны trace, usage и raw model JSON для проверки Stage 2."
        )
        with gr.Row():
            scenario = gr.Dropdown(choices=scenario_choices, value=SCENARIO_LABELS["payment"], label="Сценарий")
            message = gr.Textbox(label="Общий вопрос", value=DEFAULT_MESSAGES["payment"], lines=3)
        run = gr.Button("Запустить Base", variant="primary")
        result = gr.JSON(label="Admin response")
        response = gr.Textbox(label="Ответ клиенту", interactive=False)
        trace = gr.Dataframe(
            headers=["node", "description", "status", "duration_ms", "error", "call_id", "tokens", "evidence_ids"],
            label="LangGraph trace",
            interactive=False,
        )
        usage = gr.Dataframe(
            headers=["call_id", "node", "model", "mode", "input", "output", "total", "latency_ms", "api_cost", "complete"],
            label="Usage",
            interactive=False,
        )

        def choose_scenario(label: str):
            scenario_id = _safe_label(label)
            return DEFAULT_MESSAGES[scenario_id] if scenario_id is not None else ""

        def submit(label: str, user_message: str):
            return submit_admin_message(label, user_message, runner, policy)

        scenario.change(choose_scenario, inputs=[scenario], outputs=[message])
        run.click(submit, inputs=[scenario, message], outputs=[result, response, trace, usage])
    return blocks


def _css() -> str:
    return """
:root {
  --milk: #f7f1e8;
  --graphite: #242424;
  --copper: #b56a3a;
  --paper: #fffaf2;
  --line: #d8c8b8;
  --muted: #6d6258;
}
html, body, .gradio-container, gradio-app {
  background: var(--milk) !important;
  color: var(--graphite) !important;
}
.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto;
}
body.dark, .dark .gradio-container {
  background: var(--milk) !important;
  color: var(--graphite) !important;
}
.prose :is(h1, h2, h3, p, li),
.markdown :is(h1, h2, h3, p, li),
.gr-markdown :is(h1, h2, h3, p, li),
[data-testid="markdown"] :is(h1, h2, h3, p, li),
label {
  color: var(--graphite) !important;
}
code, pre {
  background: #efe2d4 !important;
  color: var(--graphite) !important;
}
.gradio-container .block,
.gradio-container .panel {
  background: var(--paper) !important;
  border-color: var(--line) !important;
  color: var(--graphite) !important;
}
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  background: #fffdf8 !important;
  color: var(--graphite) !important;
  border-color: var(--line) !important;
  caret-color: var(--copper) !important;
}
textarea::placeholder, input::placeholder {
  color: var(--muted) !important;
}
button.primary, .gr-button-primary, button[variant="primary"] {
  background: var(--copper) !important;
  border-color: var(--copper) !important;
  color: #fffaf2 !important;
}
.gr-button-secondary {
  border-color: var(--line) !important;
}
.chatbot,
.chatbot :is(.message, .bubble, .bot, .user, .bubble-wrap) {
  color: var(--graphite) !important;
}
.chatbot :is(.message, .bubble) {
  background: #fffdf8 !important;
  border-color: var(--line) !important;
}
.table-wrap,
.json-holder,
.json-viewer {
  background: var(--paper) !important;
  color: var(--graphite) !important;
  border-color: var(--line) !important;
}
.table-wrap :is(table, thead, tbody, tr, th, td),
.json-holder *,
.json-viewer * {
  color: var(--graphite) !important;
  border-color: var(--line) !important;
}
.table-wrap th {
  background: #efe2d4 !important;
}
[data-testid="block-label"], .block-label, .label-wrap {
  background: #efe2d4 !important;
  color: var(--graphite) !important;
  border-color: var(--line) !important;
}
.gradio-container textarea,
.gradio-container input,
.gradio-container .block {
  border-radius: 8px !important;
}
"""


def _theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue="orange",
        secondary_hue="stone",
        neutral_hue="stone",
        font=["Segoe UI", "Arial", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    ).set(
        body_background_fill="#f7f1e8",
        body_background_fill_dark="#f7f1e8",
        body_text_color="#242424",
        body_text_color_dark="#242424",
        body_text_color_subdued="#6d6258",
        body_text_color_subdued_dark="#6d6258",
        background_fill_primary="#fffaf2",
        background_fill_primary_dark="#fffaf2",
        background_fill_secondary="#f7f1e8",
        background_fill_secondary_dark="#f7f1e8",
        block_background_fill="#fffaf2",
        block_background_fill_dark="#fffaf2",
        block_border_color="#d8c8b8",
        block_border_color_dark="#d8c8b8",
        block_label_background_fill="#efe2d4",
        block_label_background_fill_dark="#efe2d4",
        block_label_text_color="#242424",
        block_label_text_color_dark="#242424",
        block_title_text_color="#242424",
        block_title_text_color_dark="#242424",
        panel_background_fill="#fffaf2",
        panel_background_fill_dark="#fffaf2",
        panel_border_color="#d8c8b8",
        panel_border_color_dark="#d8c8b8",
        input_background_fill="#fffdf8",
        input_background_fill_dark="#fffdf8",
        input_background_fill_focus="#fffdf8",
        input_background_fill_focus_dark="#fffdf8",
        input_border_color="#d8c8b8",
        input_border_color_dark="#d8c8b8",
        input_border_color_focus="#b56a3a",
        input_border_color_focus_dark="#b56a3a",
        table_text_color="#242424",
        table_text_color_dark="#242424",
        table_border_color="#d8c8b8",
        table_border_color_dark="#d8c8b8",
        table_even_background_fill="#fffaf2",
        table_even_background_fill_dark="#fffaf2",
        table_odd_background_fill="#fffdf8",
        table_odd_background_fill_dark="#fffdf8",
        button_primary_background_fill="#b56a3a",
        button_primary_background_fill_dark="#b56a3a",
        button_primary_background_fill_hover="#9d5831",
        button_primary_background_fill_hover_dark="#9d5831",
        button_primary_border_color="#b56a3a",
        button_primary_border_color_dark="#b56a3a",
        button_primary_text_color="#fffaf2",
        button_primary_text_color_dark="#fffaf2",
        button_secondary_background_fill="#fffaf2",
        button_secondary_background_fill_dark="#fffaf2",
        button_secondary_text_color="#242424",
        button_secondary_text_color_dark="#242424",
        color_accent="#b56a3a",
        color_accent_soft="#efe2d4",
        color_accent_soft_dark="#efe2d4",
        link_text_color="#8f4e2c",
        link_text_color_dark="#8f4e2c",
        code_background_fill="#efe2d4",
    )


def create_fastapi_app(runner: Any, policy: PolicyDocument):
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    import gradio as gr

    app = FastAPI(title="Capstone N4 Stage 2 Demo")

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/client")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "stage": "stage2-thin-graph"}

    theme = _theme()
    app = gr.mount_gradio_app(app, create_client_blocks(runner, policy), path="/client", css=_css(), theme=theme)
    app = gr.mount_gradio_app(app, create_admin_blocks(runner, policy), path="/admin", css=_css(), theme=theme)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Capstone N4 Stage 2 preview.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.json"))
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    policy = load_policy(args.policy)
    runner = load_runner(Path(args.config), args.model_key)
    app = create_fastapi_app(runner, policy)
    try:
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        close = getattr(runner, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
