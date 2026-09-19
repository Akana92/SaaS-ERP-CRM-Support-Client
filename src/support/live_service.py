"""Local support orchestration; experiment source files stay unchanged."""
from __future__ import annotations

import copy
import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

from support.contracts import ClientAnalysis, ERPContext, ModelCallUsage, TraceEvent
from support.live_contracts import LiveAdminResponse as AdminResponse, LiveClientResponse as ClientResponse
from support.graph import invoke_support_graph
from support.live_store import LiveStore
from support.live_dialogue import DialogueBudgetError, DialogueRunner, create_live_dialogue_policy
from support.live_reply_guard import apply_handoff_guard, apply_reply_guard, preserve_handoff_explanation
from support.live_input_guard import check_input_status, evaluate_input

CLIENT_MODEL_MODE = "fine_tuned"


def identifier():
    return uuid.uuid4().hex


def now():
    return datetime.now(timezone.utc).isoformat()


class GuardedRunner:
    def __init__(self, runner, mode):
        self.runner, self.mode = runner, mode

    def generate(self, model_input, request_id, **kwargs):
        try:
            return self.runner.generate(model_input, request_id, **kwargs)
        except Exception:
            # The attempt happened, but no trustworthy token/cost observation exists.
            usage = ModelCallUsage(
                call_id=identifier(), request_id=request_id, node="model_call",
                model_id="local-runtime", revision="unavailable",
                adapter_id="unavailable" if self.mode == "fine_tuned" else None,
                mode=self.mode, input_tokens=None, output_tokens=None, total_tokens=None,
                latency_ms=None, api_cost=None, currency="USD", complete=False,
            )
            return SimpleNamespace(raw_text=None, result=None, usage=usage, error="model_runtime_error")


class ContextRunner:
    """Use only the server-owned conversation snapshot; never accept history from POST."""

    def __init__(self, runtime, mode, history):
        self.delegate = DialogueRunner(GuardedRunner(runtime.for_mode(mode), mode), history,
                                      runtime.count_chat_prompt_tokens)
        self.history_size = len(history)
        self.budget_error = False

    def generate(self, model_input, request_id, **kwargs):
        try:
            return self.delegate.generate(model_input, request_id, **kwargs)
        except DialogueBudgetError:
            # No model call happened, so do not invent a usage record for a GPU attempt.
            self.budget_error = True
            return SimpleNamespace(raw_text=None, result=None, usage=None, error="context_budget_exceeded")

    def metadata(self):
        if self.delegate.last_metadata is not None:
            return {**self.delegate.last_metadata.as_dict(), "status": "included"}
        return dict(status="budget_exceeded" if self.budget_error else "not_used",
                    source_message_count=self.history_size, included_request_ids=[],
                    omitted_request_ids=[], history_truncated=False)


def usage_total(records):
    calls = {}
    for record in records:
        for call in record["admin"]["usage_calls"]:
            calls.setdefault(call["call_id"], call)
    result = {"currency": "USD", "calls": len(calls), "complete": all(c["complete"] for c in calls.values())}
    for field in ("input_tokens", "output_tokens", "total_tokens", "api_cost"):
        values = [call[field] for call in calls.values()]
        result[field] = sum(values) if all(value is not None for value in values) else None
    return result


def apply_business_rules(admin, client, erp_context):
    """Enforce trusted ERP handoff requirements without rewriting model output."""
    if erp_context["source_status"] != "ok" or admin.raw_model_result is None or client.status == "fallback":
        return admin, client
    facts = erp_context["facts"]
    pending = facts.get("erp.sim.pending_activation_count")
    rules = [
        (facts.get("erp.incident.confirmed_mass_outage") is True,
         "service_outage", ["erp.incident.confirmed_mass_outage"],
         "ERP подтверждает массовую недоступность сервиса: требуется оператор; приоритет очереди — Critical, независимо от метки модели."),
        (facts.get("erp.payment.status") == "paid" and type(pending) is int and pending > 0,
         "payment_confirmed_but_access_not_active", ["erp.payment.status", "erp.sim.pending_activation_count"],
         "ERP подтверждает оплату и наличие SIM-карт в ожидании активации: требуется проверка оператором."),
        (facts.get("erp.access.can_start_connection") is False,
         "access_change_requires_operator", ["erp.access.can_start_connection"],
         "ERP подтверждает отсутствие права запуска подключения: доступ проверяет оператор."),
        (facts.get("erp.integration.operator_review_required") is True,
         "integration_requires_operator", ["erp.integration.operator_review_required"],
         "ERP указывает обязательную проверку интеграции оператором."),
    ]
    matched = [rule for rule in rules if rule[0]]
    if not matched:
        return admin, client
    safe_response = {
        "payment_confirmed_but_access_not_active": "Оплата подтверждена, но часть SIM ещё ожидает активации. Причину должен проверить специалист.",
        "access_change_requires_operator": "Текущий доступ не разрешает запуск подключения. Нужна проверка прав специалистом.",
        "integration_requires_operator": "По данным демо-ERP интеграция требует проверки специалистом.",
        "service_outage": "В демо-ERP подтверждён массовый сбой. Нужна проверка специалистом.",
    }[matched[0][1]]
    payload = admin.model_dump(mode="json")
    payload.update(server_route="human_escalation", response=safe_response)
    for _, reason, evidence, description in matched:
        payload["trace_events"].append(TraceEvent(
            node="business_rules", description=description + " Исходные метки и ответ модели сохранены без изменений.",
            request_id=admin.request_id, status="ok", duration_ms=0, error=reason, evidence_ids=evidence,
        ).model_dump(mode="json"))
    raw = admin.raw_model_result
    safe_client = ClientResponse(response=safe_response, status="human_escalation", escalation=True,
        analysis=ClientAnalysis(category=raw.category, priority=raw.priority,
                                sentiment=raw.sentiment, recommended_action=raw.recommended_action))
    return AdminResponse.model_validate(payload), safe_client


class LiveService:
    def __init__(self, runtime, policy, audiences, *, store_path=None, archive_path=None, serving_info=None, erp=None):
        self.runtime, self.policy = runtime, create_live_dialogue_policy(policy)
        self.erp = erp
        self.serving_info = copy.deepcopy({key: serving_info[key] for key in
            ("adapter_profile", "adapter_path", "adapter_model_sha256", "client_mode")
            if serving_info is not None and key in serving_info})
        self.audiences = copy.deepcopy(audiences)
        self.by_audience = {item["id"]: item for item in self.audiences}
        if set(self.by_audience) != {"customer", "employee"} or len(self.audiences) != 2:
            raise ValueError("Exactly customer and employee audiences are required")
        for item in self.audiences:
            ERPContext.model_validate(item["erp_context"])
            if not item.get("org_id"):
                raise ValueError("Audience organization is required")
        self.state_lock = threading.RLock()
        self.work_lock = threading.Lock()
        self.owners, self.conversations = set(), {}
        self.requests, self.comparisons, self.queue, self.keys = {}, {}, {}, {}
        self.input_states, self.pending_submissions = {}, {}
        self.max_owners, self.max_conversations = 2000, 5000
        self.max_requests, self.max_comparisons, self.max_pending_jobs = 10000, 5000, 8
        self.store = LiveStore(store_path)
        self.closing = False
        self._restore()
        if archive_path is not None:
            self._import_archive(archive_path)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="support-compare")

    def _save(self):
        self.store.save(dict(schema_version=1, owners=sorted(self.owners), conversations=self.conversations,
            requests=self.requests, comparisons=self.comparisons, queue=self.queue,
            input_states=self.input_states,
            keys=[dict(key=list(key), fingerprint=value[0], target=value[1]) for key, value in self.keys.items()]))

    def _restore(self):
        saved = self.store.load()
        if saved is None:
            return
        self.owners = set(saved["owners"])
        self.input_states = saved.get("input_states", {})
        for name in ("conversations", "requests", "comparisons", "queue"):
            setattr(self, name, saved[name])
        self.keys = {tuple(item["key"]): (item["fingerprint"], item["target"]) for item in saved["keys"]}
        for job in self.comparisons.values():
            if job.get("status") in {"queued", "running"}:
                job["unobserved_attempt"] = job["status"] == "running"
                job.update(status="failed", stage="failed", error="Запуск прерван перезапуском сервера. Автоматический повтор не выполнялся.")
        self._save()

    def _import_archive(self, path):
        path = Path(path)
        if not path.exists():
            return
        archive = json.loads(path.read_text(encoding="utf-8"))
        # Original records are copied verbatim; no ownership or conversations are reconstructed.
        for record in archive["requests"]:
            AdminResponse.model_validate(record["admin"])
            ClientResponse.model_validate(record["client"])
            self.requests.setdefault(record["request_id"], copy.deepcopy(record))
        for comparison in archive["comparisons"]:
            for mode in ("base", "fine_tuned"):
                if comparison.get(mode):
                    AdminResponse.model_validate(comparison[mode]["admin"])
            self.comparisons.setdefault(comparison["id"], copy.deepcopy(comparison))
        for ticket in archive["queue"]:
            self.queue.setdefault(ticket["id"], copy.deepcopy(ticket))
        self._save()

    def close(self):
        with self.state_lock:
            self.closing = True
        # Finish owned work before the launcher unloads the single GPU model.
        self.executor.shutdown(wait=True)
        with self.state_lock:
            self._save()

    def owner(self, candidate):
        with self.state_lock:
            if candidate in self.owners:
                return candidate
            self.capacity(len(self.owners), self.max_owners)
            token = identifier()
            self.owners.add(token)
            self._save()
            return token

    @staticmethod
    def capacity(count, maximum):
        if count >= maximum:
            raise HTTPException(429, "Лимит локального хранилища или очереди достигнут. Сохранённая история остаётся доступной.")

    def audience(self, audience_id):
        if audience_id not in self.by_audience:
            raise HTTPException(422, "Выберите клиента или сотрудника.")
        return self.by_audience[audience_id]

    def conversation(self, owner, conversation_id):
        conv = self.conversations.get(conversation_id)
        if conv is None or conv["owner"] != owner:
            raise HTTPException(404, "Беседа не найдена.")
        if self.audience(conv["audience"])["org_id"] != conv["org_id"]:
            raise HTTPException(403, "Контекст организации не совпадает.")
        return conv

    @staticmethod
    def public_conversation(conv):
        return copy.deepcopy({key: conv[key] for key in ("id", "title", "audience", "messages")})

    @staticmethod
    def public_message(record):
        return copy.deepcopy({k: record[k] for k in ("request_id", "created_at", "message", "client")})

    def bootstrap(self, owner):
        with self.state_lock:
            audiences = [{k: item[k] for k in ("id", "label", "prompt")} for item in self.audiences]
            if self.erp is not None:
                for item in audiences:
                    item["examples"] = self.erp.examples(item["id"], context=self.audience(item["id"]))
            return {"audiences": audiences,
                    "conversations": [{k: c[k] for k in ("id", "title", "audience")}
                                      for c in self.conversations.values() if c["owner"] == owner]}

    def create_conversation(self, owner, audience_id):
        with self.state_lock:
            self.capacity(len(self.conversations), self.max_conversations)
            audience = self.audience(audience_id)
            conv = dict(id=identifier(), owner=owner, org_id=audience["org_id"],
                        title="Новая беседа", audience=audience_id, messages=[])
            self.conversations[conv["id"]] = conv
            self._save()
            return self.public_conversation(conv)

    def replay(self, key, fingerprint):
        previous = self.keys.get(key)
        if previous:
            if previous[0] != fingerprint:
                raise HTTPException(409, "Ключ уже использован для другого сообщения.")
            return previous[1]
        return None

    def run(self, message, context, session_id, mode, *, history=None):
        request_id = identifier()
        runtime_status = self.runtime.status()
        erp_lookup = None
        if self.erp is not None:
            context, erp_lookup = self.erp.resolve(context, message, history or [])
        # ScenarioId is an internal legacy graph contract; it does not classify the message.
        scenario_id = context.get("scenario_id", "information")
        context_runner = ContextRunner(self.runtime, mode, history or [])
        admin, client = invoke_support_graph(
            context_runner, self.policy,
            customer_message=message, erp_context=ERPContext.model_validate(context["erp_context"]),
            request_id=request_id, session_id=session_id, scenario_id=scenario_id,
        )
        dialogue_context = context_runner.metadata()
        if dialogue_context["status"] == "included":
            count = len(dialogue_context["included_request_ids"])
            description = (f"Модель получила {count} предыдущих обменов из этой беседы. " if count
                           else "Новый вопрос: предыдущих реплик в запросе нет. ")
            if dialogue_context.get("history_truncated"):
                description += "Часть истории исключена по лимиту контекста; полная беседа сохранена."
            status, error = "ok", None
        elif context_runner.budget_error:
            description = "Текущее сообщение не помещается в лимит контекста. Модель не вызывалась."
            status, error = "failed", "context_budget_exceeded"
        else:
            description = "История не передавалась: граф завершился до вызова модели."
            status, error = "skipped", None
        payload = admin.model_dump(mode="json")
        if erp_lookup is not None:
            source_status = context["erp_context"]["source_status"]
            reference = erp_lookup.get("reference")
            if source_status == "ok" and reference:
                erp_description = f"Прочитаны факты вымышленной ERP для {reference} в рамках учебного профиля."
            elif source_status == "ok":
                erp_description = "Номер объекта не указан: доступна только общая учебная справка."
            else:
                erp_description = "Нет доступных фактов для указанного номера в учебном профиле; чужие данные не передавались модели."
            payload["trace_events"].insert(0, TraceEvent(node="erp_lookup", description=erp_description,
                request_id=request_id, status="ok", duration_ms=0,
                evidence_ids=list(context["erp_context"]["facts"])).model_dump(mode="json"))
        payload["trace_events"].insert(1, TraceEvent(node="dialogue_context", description=description,
            request_id=request_id, status=status, duration_ms=0, error=error).model_dump(mode="json"))
        if context_runner.budget_error:
            response = "Текущее сообщение слишком длинное для локальной модели. Обращение передано оператору."
            payload["response"] = response
            client = client.model_copy(update={"response": response})
            for event in payload["trace_events"]:
                if event["node"] == "model_call":
                    event.update(status="skipped", description="Модель не вызвана: превышен лимит входного контекста.",
                                 error="context_budget_exceeded")
        admin = AdminResponse.model_validate(payload)
        admin, client = apply_business_rules(admin, client, context["erp_context"])
        admin, client = preserve_handoff_explanation(admin, client)
        admin, client = apply_handoff_guard(admin, client)
        admin, client = apply_reply_guard(admin, client, history or [], message)
        record = dict(request_id=request_id, created_at=now(), message=message, audience=context["id"],
                    dialogue_context=dialogue_context,
                    inference_profile=runtime_status.get("inference_profile"),
                    serving_scope=runtime_status.get("precision_scope"),
                    policy_version=self.policy.version, erp_context=copy.deepcopy(context["erp_context"]),
                    admin=admin.model_dump(mode="json"), client=client.model_dump(mode="json"))
        if erp_lookup is not None:
            record["erp_lookup"] = copy.deepcopy(erp_lookup)
        if self.serving_info:
            record["serving_info"] = {**copy.deepcopy(self.serving_info), "mode": mode}
            if mode == "base":
                # The profile identifies the comparison pair, not an applied Base adapter.
                record["serving_info"].update(adapter_path=None, adapter_model_sha256=None)
        return record

    def submit(self, owner, conversation_id, message, key):
        cache_key = ("support", owner, conversation_id, key)
        with self.state_lock:
            self.conversation(owner, conversation_id)
            previous = self.replay(cache_key, message)
            if previous is not None:
                return self.public_message(self.requests[previous])
            decision = evaluate_input(self.input_states.setdefault(owner, {}), message,
                                      conversation_id + ":" + key, time.time())
            self._save()
            if not decision["allowed"]:
                status = 429 if decision["blocked"] else 409 if decision["code"] == "idempotency_conflict" else 422
                raise HTTPException(status, detail=decision)
            self.pending_submissions.setdefault(cache_key, dict(status="queued", created_at=now()))
        try:
            return self._submit_accepted(owner, conversation_id, message, key)
        except Exception:
            with self.state_lock:
                self.pending_submissions.pop(cache_key, None)
            raise
        finally:
            with self.state_lock:
                if cache_key in self.keys:
                    self.pending_submissions.pop(cache_key, None)

    def input_status(self, owner):
        with self.state_lock:
            value = self.input_states.setdefault(owner, {})
            before = copy.deepcopy(value)
            decision = check_input_status(value, time.time())
            if value != before:
                self._save()
            return decision

    def submission_status(self, owner, conversation_id, key):
        with self.state_lock:
            self.conversation(owner, conversation_id)
            cache_key = ("support", owner, conversation_id, key)
            saved = self.keys.get(cache_key)
            if saved is not None:
                record = self.requests[saved[1]]
                return dict(status="completed", request_id=record["request_id"], created_at=record["created_at"])
            return copy.deepcopy(self.pending_submissions.get(cache_key, dict(status="not_found", created_at=None)))

    def _submit_accepted(self, owner, conversation_id, message, key):
        with self.work_lock:
            with self.state_lock:
                conv = self.conversation(owner, conversation_id)
                cache_key = ("support", owner, conversation_id, key)
                previous = self.replay(cache_key, message)
                if previous is not None:
                    return self.public_message(self.requests[previous])
                self.capacity(len(self.requests), self.max_requests)
                context = copy.deepcopy(self.audience(conv["audience"]))
                history = copy.deepcopy(conv["messages"])
                self.pending_submissions[cache_key]["status"] = "running"
            record = self.run(message, context, conversation_id, CLIENT_MODEL_MODE, history=history)
            with self.state_lock:
                if record["client"]["escalation"]:
                    ticket_id = "ticket-" + record["request_id"]
                    record["client"]["handoff_id"] = ticket_id
                    raw = record["admin"]["raw_model_result"]
                    errors = [e["error"] for e in record["admin"]["trace_events"] if e["error"]]
                    forced = [e["error"] for e in record["admin"]["trace_events"]
                              if e["node"] == "business_rules" and e["error"]]
                    self.queue[ticket_id] = dict(id=ticket_id, request_id=record["request_id"], message=message,
                        reason=(forced[0] if forced else None) or (raw or {}).get("escalation_reason") or (errors[-1] if errors else "human_escalation"),
                        priority="Critical" if "service_outage" in forced else (raw or {}).get("priority"),
                        status="draft", draft=record["client"]["response"])
                self.requests[record["request_id"]] = record
                public = self.public_message(record)
                if not conv["messages"]:
                    conv["title"] = message[:80]
                conv["messages"].append(public)
                self.keys[cache_key] = (message, record["request_id"])
                self._save()
                return copy.deepcopy(public)

    def enqueue_comparison(self, owner, message, audience_id, key, parent_comparison_id=None):
        with self.state_lock:
            if self.closing:
                raise HTTPException(503, "Сервер завершает текущие задачи.")
            cache_key = ("compare", owner, key)
            fingerprint = [message, audience_id]
            if parent_comparison_id is not None:
                fingerprint.append(parent_comparison_id)
            previous = self.replay(cache_key, fingerprint)
            if previous is not None:
                return self._public_job(self.comparisons[previous])
            chain = []
            if parent_comparison_id is not None:
                if parent_comparison_id not in self.comparisons:
                    raise HTTPException(404, "Сравнение не найдено.")
                chain = self._comparison_chain(parent_comparison_id)
                if any(turn.get("status", "completed") != "completed" or
                       not all(turn.get(mode) for mode in ("base", "fine_tuned")) for turn in chain):
                    raise HTTPException(409, "Продолжение доступно только после успешного завершения сравнения.")
                self._validate_comparison_history(chain)
                if chain[-1].get("audience") != audience_id:
                    raise HTTPException(409, "Аудитория диалога сравнения не может изменяться.")
                if any(j.get("parent_comparison_id") == parent_comparison_id for j in self.comparisons.values()):
                    raise HTTPException(409, "У сравнения уже есть продолжение. Выберите последний ход.")
            self.capacity(len(self.comparisons), self.max_comparisons)
            pending = sum(job.get("status") in {"queued", "running"} for job in self.comparisons.values())
            self.capacity(pending, self.max_pending_jobs)
            context = copy.deepcopy(self.audience(audience_id))
            comparison_id = identifier()
            job = dict(id=comparison_id, created_at=now(), message=message, audience=audience_id,
                       status="queued", stage="queued", base=None, fine_tuned=None, error=None,
                       parent_comparison_id=parent_comparison_id,
                       dialogue_id=chain[0]["id"] if chain else comparison_id, turn_index=len(chain))
            if self.serving_info:
                job["serving_info"] = copy.deepcopy(self.serving_info)
            self.comparisons[comparison_id] = job
            self.keys[cache_key] = (fingerprint, comparison_id)
            self._save()
            response = self._public_job(job)
            self.executor.submit(self._execute_comparison, comparison_id, context)
            return response

    def _execute_comparison(self, comparison_id, context):
        with self.work_lock:
            try:
                for mode in ("base", "fine_tuned"):
                    with self.state_lock:
                        job = self.comparisons[comparison_id]
                        job.update(status="running", stage=mode)
                        self._save()
                        message = job["message"]
                        chain = self._comparison_chain(comparison_id)
                        history = [self.public_message(turn[mode]) for turn in chain[:-1]]
                    record = self.run(message, context, job["dialogue_id"], mode, history=history)
                    record["simulated_handoff"] = bool(record["client"]["escalation"])
                    with self.state_lock:
                        job[mode] = record
                        self._save()
                        if any(event["error"] == "model_runtime_error" for event in record["admin"]["trace_events"]):
                            job.update(status="failed", stage="failed", error="Модель не завершила генерацию. Частичный результат и наблюдаемый расход сохранены.")
                            self._save()
                            return
                with self.state_lock:
                    job.update(status="completed", stage="completed")
                    self._save()
            except Exception as exc:
                with self.state_lock:
                    job = self.comparisons[comparison_id]
                    job.update(status="failed", stage="failed", unobserved_attempt=True,
                               failure_type=type(exc).__name__, diagnostic_id=identifier(),
                               error="Не удалось завершить сравнение. Сохранённые результаты доступны; автоматического повтора нет.")
                    self._save()

    def _validate_comparison_history(self, chain):
        """Fail closed on malformed saved sources before scheduling any model call."""
        seen = set()
        prior = {"base": set(), "fine_tuned": set()}
        try:
            for turn in chain:
                for mode in ("base", "fine_tuned"):
                    record = turn[mode]
                    request_id = record["request_id"]
                    if not isinstance(request_id, str) or request_id in seen:
                        raise ValueError("duplicate request")
                    seen.add(request_id)
                    if record["message"] != turn["message"] or record.get("audience") != turn.get("audience"):
                        raise ValueError("mismatched source")
                    ClientResponse.model_validate(record["client"])
                    admin = AdminResponse.model_validate(record["admin"])
                    if admin.request_id != request_id:
                        raise ValueError("mismatched request")
                    if any(call.mode != mode for call in admin.usage_calls):
                        raise ValueError("mismatched model")
                    self._included_comparison_ids(record, prior[mode])
                    prior[mode].add(request_id)
                    self.public_message(record)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(409, "Повреждены сохранённые ответы диалога сравнения.") from exc

    @staticmethod
    def _included_comparison_ids(record, prior_ids):
        context = record.get("dialogue_context")
        # Old single-turn archives have no history metadata.
        if context is None and not prior_ids:
            return []
        included = context.get("included_request_ids") if isinstance(context, dict) else None
        if (not isinstance(included, list) or any(not isinstance(key, str) for key in included)
                or len(set(included)) != len(included) or any(key not in prior_ids for key in included)):
            raise HTTPException(409, "Повреждены ссылки на историю модели в сравнении.")
        return included

    def _comparison_chain(self, comparison_id):
        chain, seen = [], set()
        current = comparison_id
        while current is not None:
            if not isinstance(current, str) or current in seen or current not in self.comparisons:
                raise HTTPException(409, "Повреждена цепочка сравнения.")
            seen.add(current)
            turn = self.comparisons[current]
            if turn.get("id") != current:
                raise HTTPException(409, "Повреждён идентификатор сравнения.")
            chain.append(turn)
            current = turn.get("parent_comparison_id")
        chain.reverse()
        for index, turn in enumerate(chain):
            if index and ("dialogue_id" not in turn or "turn_index" not in turn):
                raise HTTPException(409, "Отсутствуют данные продолжения сравнения.")
            if (turn.get("audience") != chain[0].get("audience") or
                    turn.get("dialogue_id", chain[0]["id"]) != chain[0]["id"] or
                    turn.get("turn_index", index) != index):
                raise HTTPException(409, "Контекст цепочки сравнения не совпадает.")
            if index < len(chain) - 1 and (turn.get("status", "completed") != "completed" or
                    not all(turn.get(mode) for mode in ("base", "fine_tuned"))):
                raise HTTPException(409, "Предыдущий ход сравнения не завершён.")
        return chain

    def _public_job(self, job):
        chain = self._comparison_chain(job["id"])
        turns = []
        prior = {"base": {}, "fine_tuned": {}}
        for turn in chain:
            public = {key: copy.deepcopy(turn.get(key)) for key in
                      ("id", "message", "audience", "created_at", "base", "fine_tuned")}
            public["status"] = turn.get("status", "completed")
            if "serving_info" in turn:
                public["serving_info"] = copy.deepcopy(turn["serving_info"])
            for mode in prior:
                record = public[mode]
                if record:
                    included = self._included_comparison_ids(record, prior[mode])
                    record["model_history"] = [copy.deepcopy(prior[mode][key]) for key in included]
                    record["simulated_handoff"] = bool(record["client"]["escalation"])
                    prior[mode][record["request_id"]] = self.public_message(record)
            turns.append(public)
        result = copy.deepcopy(job)
        result.setdefault("audience", None)
        result.setdefault("status", "completed")
        result.setdefault("stage", "completed")
        result.setdefault("error", None)
        result.update(dialogue_id=chain[0]["id"], turn_index=len(chain)-1,
                      parent_comparison_id=job.get("parent_comparison_id"), turns=turns,
                      base=turns[-1]["base"], fine_tuned=turns[-1]["fine_tuned"])
        latest, visited = job["id"], set()
        while True:
            children = [c for c in self.comparisons.values() if c.get("parent_comparison_id") == latest]
            if not children:
                break
            if len(children) != 1 or latest in visited:
                raise HTTPException(409, "Повреждены продолжения сравнения.")
            visited.add(latest)
            latest = children[0]["id"]
        result["latest_comparison_id"] = latest
        return result

    def comparison(self, comparison_id):
        with self.state_lock:
            if comparison_id not in self.comparisons:
                raise HTTPException(404, "Сравнение не найдено.")
            return self._public_job(self.comparisons[comparison_id])

    def request(self, request_id):
        with self.state_lock:
            if request_id not in self.requests:
                raise HTTPException(404, "Обращение не найдено.")
            result = copy.deepcopy(self.requests[request_id])
            included = result.get("dialogue_context", {}).get("included_request_ids", [])
            result["model_history"] = [self.public_message(self.requests[prior_id])
                                       for prior_id in included if prior_id in self.requests]
            return result

    @staticmethod
    def _page(items, page, page_size):
        total = len(items)
        return dict(items=copy.deepcopy(items[(page - 1) * page_size:page * page_size]),
                    total=total, page=page, page_size=page_size, pages=math.ceil(total / page_size))

    def list_requests(self, *, q="", audience=None, category=None, route=None, page=1, page_size=20):
        with self.state_lock:
            items = []
            for record in reversed(list(self.requests.values())):
                analysis = (record["admin"].get("raw_model_result") or {})
                if q.casefold() not in (record["message"] + " " + record["request_id"]).casefold():
                    continue
                if audience and record.get("audience") != audience:
                    continue
                if category and analysis.get("category") != category:
                    continue
                if route and record["admin"]["server_route"] != route:
                    continue
                item = dict(request_id=record["request_id"], created_at=record["created_at"], message=record["message"],
                    audience=record.get("audience"), analysis={k: analysis[k] for k in ("category", "priority", "sentiment", "recommended_action")} if analysis else None,
                    server_route=record["admin"]["server_route"])
                if "serving_info" in record:
                    item["serving_info"] = copy.deepcopy(record["serving_info"])
                items.append(item)
            return self._page(items, page, page_size)

    def list_comparisons(self, *, q="", audience=None, page=1, page_size=20):
        with self.state_lock:
            items = []
            for record in reversed(list(self.comparisons.values())):
                if q.casefold() not in (record["message"] + " " + record["id"]).casefold():
                    continue
                if audience and record.get("audience") != audience:
                    continue
                item = dict(id=record["id"], created_at=record["created_at"], message=record["message"],
                    audience=record.get("audience"), status=record.get("status", "completed"),
                    stage=record.get("stage", "completed"))
                if "serving_info" in record:
                    item["serving_info"] = copy.deepcopy(record["serving_info"])
                items.append(item)
            return self._page(items, page, page_size)

    def list_queue(self, *, q="", status=None, priority=None, page=1, page_size=20):
        with self.state_lock:
            items = []
            for ticket in self.queue.values():
                if q.casefold() not in " ".join(ticket[k] for k in ("message", "id", "request_id")).casefold():
                    continue
                if status and ticket["status"] != status:
                    continue
                if priority and ticket.get("priority") != priority:
                    continue
                record = self.requests.get(ticket["request_id"], {})
                items.append({**{k: ticket.get(k) for k in ("id", "request_id", "message", "reason", "priority", "status")},
                              "created_at": record.get("created_at"), "audience": record.get("audience")})
            # Imported undated tickets follow dated ones; IDs break timestamp ties.
            items.sort(key=lambda item: (item["created_at"] or "", item["id"]), reverse=True)
            return self._page(items, page, page_size)

    def queue_ticket(self, ticket_id):
        with self.state_lock:
            if ticket_id not in self.queue:
                raise HTTPException(404, "Карточка не найдена.")
            ticket = self.queue[ticket_id]
            record = self.requests.get(ticket["request_id"], {})
            history = []
            # Requests have no owner/session binding; only an exact, unique stored
            # conversation membership proves which public thread belongs to this ticket.
            matches = [(conv, index) for conv in self.conversations.values()
                       for index, message in enumerate(conv["messages"])
                       if message["request_id"] == ticket["request_id"]] if record else []
            if len(matches) == 1:
                conversation, index = matches[0]
                history = [self.public_message(message) for message in conversation["messages"][:index + 1]]
            return copy.deepcopy({**ticket, "created_at": record.get("created_at"),
                                  "audience": record.get("audience"),
                                  "analysis": record.get("client", {}).get("analysis"),
                                  "history_available": bool(history), "conversation": history})

    def snapshot(self, *, full=False, include_queue=True):
        with self.state_lock:
            requests, comparisons = list(self.requests.values()), list(self.comparisons.values())
            comparison_usage = usage_total([c[m] for c in comparisons for m in ("base", "fine_tuned") if c.get(m)])
            if any(c.get("unobserved_attempt") for c in comparisons):
                comparison_usage.update(complete=False, input_tokens=None, output_tokens=None, total_tokens=None, api_cost=None)
            result = copy.deepcopy(dict(counts=dict(requests=len(requests), comparisons=len(comparisons), queue=len(self.queue)),
                queue=list(self.queue.values()) if include_queue else [],
                usage=dict(support=usage_total(requests), comparison=comparison_usage)))
            if full:
                result.update(requests=copy.deepcopy(requests), comparisons=copy.deepcopy(comparisons))
        result["runtime"] = {**self.runtime.status(), **copy.deepcopy(self.serving_info), "client_mode": CLIENT_MODEL_MODE,
                             "candidate_status": ("selected_for_local_demo" if self.serving_info.get("adapter_profile") == "quality-f"
                                                  else "candidate_not_selected"), "demo_model_selected_by_user": True}
        return result

    def review(self, ticket_id, draft, status):
        with self.state_lock:
            if ticket_id not in self.queue:
                raise HTTPException(404, "Карточка не найдена.")
            self.queue[ticket_id].update(draft=draft, status=status)
            self._save()
            return copy.deepcopy(self.queue[ticket_id])
