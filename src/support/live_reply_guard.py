"""Stop observed reply loops in live chat without changing model predictions."""
from __future__ import annotations

from difflib import SequenceMatcher
import re

from support.contracts import AdminResponse, ClientResponse, TraceEvent
from support.graph import FALLBACK_RESPONSE
from support.live_contracts import LiveAdminResponse, LiveClientResponse


REPEAT_SIMILARITY = 0.90


def preserve_handoff_explanation(admin, client):
    """Keep a validated model handoff's wording without undoing server overrides."""
    result = admin.raw_model_result
    checked = any(event.node == "check_result" and event.status == "ok" and event.error is None
                  for event in admin.trace_events)
    failed = any(event.status == "failed" for event in admin.trace_events)
    if (result is None or not result.human_escalation or not checked or failed
            or admin.server_route != "human_escalation"
            or client.status != "human_escalation" or not client.escalation
            or admin.response != FALLBACK_RESPONSE or client.response != FALLBACK_RESPONSE
            or result.suggested_response == FALLBACK_RESPONSE):
        return admin, client
    payload = admin.model_dump(mode="json")
    payload["response"] = result.suggested_response
    payload["trace_events"].append(TraceEvent(
        node="handoff_explanation", request_id=admin.request_id, status="ok", duration_ms=0,
        description="В клиентском ответе сохранено объяснение модели после проверки результата. Маршрут передачи специалисту и исходные метки не изменены.",
    ).model_dump(mode="json"))
    return (AdminResponse.model_validate(payload),
            client.model_copy(update={"response": result.suggested_response}))


def _explicit_handoff(text):
    # Only direct assertions count. Quoted examples and conditional offers are
    # not evidence that this response requires an actual handoff.
    unquoted = re.sub(r'«[^»]*»|“[^”]*”|"[^"\n]*"|`[^`]*`|\'[^\'\n]*\'', "", text.casefold())
    for sentence in re.findall(r"[^.!?\n]+[.!?]?", unquoted):
        if sentence.endswith("?") or re.search(
                r"\b(?:если|когда|возможно|можно|может|могут|хотите|например|предположим|допустим)\b|"
                r"\bне\s+(?:думаю|считаю|уверен[а]?|могу\s+сказать)\b|"
                r"\bпри\s+(?:необходимости|желании|условии)\b|\bв случае\b", sentence):
            continue
        for clause in re.split(r"[,;:]", sentence):
            if re.search(r"\b(?:не|ни)\b", clause):
                continue
            required = re.search(
                r"\b(?:нужен|необходим|требуется)\s+(?:специалист|оператор)\b|"
                r"\b(?:специалист|оператор)\s+(?:нужен|необходим)\b|"
                r"\b(?:нужна|необходима|требуется)\s+(?:проверка|помощь)\s+(?:специалиста|специалистом|оператора|оператором)\b",
                clause)
            performed = re.search(
                r"\b(?:передаю|направляю|передаём|передаем|направляем)\s+(?:обращение|запрос|вопрос)\s+(?:специалисту|оператору)\b|"
                r"\b(?:обращение|запрос|вопрос)\s+(?:передан|передано|направлен|направлено)\s+(?:специалисту|оператору)\b",
                clause)
            if required or performed:
                return True
    return False


def apply_handoff_guard(admin, client):
    """Align a direct public handoff assertion with routing, retaining raw output."""
    if (admin.raw_model_result is None or admin.raw_model_result.human_escalation
            or client.escalation or client.status == "fallback"
            or admin.server_route == "human_escalation" or not _explicit_handoff(client.response)):
        return admin, client
    response = "Для проверки нужен специалист. Обращение передано оператору вместе с перепиской."
    payload = admin.model_dump(mode="json")
    payload.update(server_route="human_escalation", response=response)
    payload["trace_events"].append(TraceEvent(
        node="handoff_guard", request_id=admin.request_id, status="ok", duration_ms=0,
        error="explicit_handoff_not_routed",
        description="Ответ явно требует или объявляет передачу специалисту, но модель не включила эскалацию. Сервер направил обращение оператору; исходный ответ и метки сохранены.",
    ).model_dump(mode="json"))
    safe_client = ClientResponse(response=response, status="human_escalation", escalation=True,
                                 analysis=client.analysis)
    return AdminResponse.model_validate(payload), safe_client


def _normalise(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def apply_reply_guard(admin, client, history, current_message):
    if not history or client.escalation or client.status == "fallback" or admin.raw_model_result is None:
        return admin, client
    # Repeating an instruction on an explicit request is a useful answer, not a loop.
    if re.search(r"(?<!не )\b(?:повтори|повторите|напомни|напомните)\b|\bможно\s+ещ[её]\s+раз\b", current_message.casefold()):
        return admin, client
    proposed = _normalise(client.response)
    if len(proposed) < 60:
        return admin, client
    matched = None
    for previous in reversed(history[-3:]):
        prior_client = previous.get("client", {})
        prior = _normalise(prior_client.get("response", ""))
        if not prior or prior_client.get("escalation"):
            continue
        similarity = SequenceMatcher(None, proposed, prior, autojunk=False).ratio()
        if similarity >= REPEAT_SIMILARITY:
            matched = previous
            break
    if matched is None:
        return admin, client

    response = "Пока не удалось продвинуться с решением. Уточните, пожалуйста, что осталось непонятным или что произошло после последнего действия."
    payload = admin.model_dump(mode="json")
    payload.update(server_route="server_clarification", response=response)
    payload["trace_events"].append(TraceEvent(
        node="response_guard", request_id=admin.request_id, status="ok", duration_ms=0,
        error="repeated_previous_response",
        description="Модель повторила предыдущий ответ. Сервер запросил уточнение без передачи оператору; исходный ответ и метки модели сохранены.",
    ).model_dump(mode="json"))
    safe_client = LiveClientResponse(response=response, status="server_clarification", escalation=False,
                                    analysis=client.analysis)
    return LiveAdminResponse.model_validate(payload), safe_client
