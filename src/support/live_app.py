"""Same-origin localhost API and static front, without import-time GPU work."""
from __future__ import annotations

import base64
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware

from support.live_service import LiveService
from support.contracts import Category, Priority, ServerRoute
from support import live_evaluation

ROOT = Path(__file__).resolve().parents[2]
COOKIE = "capstone_demo_owner"
REPORT = ROOT / "artifacts/stage4/validation-comparison-full-v8"
Message = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
Key = Annotated[str, StringConstraints(min_length=8, max_length=80)]
Audience = Literal["customer", "employee"]


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ConversationBody(Body):
    audience: Audience


class MessageBody(Body):
    message: Message
    idempotency_key: Key


class CompareBody(MessageBody):
    audience: Audience
    parent_comparison_id: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None


class ReviewBody(Body):
    draft: Message
    status: Literal["draft", "approved"]


def create_live_app(runtime, policy, audiences=None, *, store_path=None, archive_path=None, serving_info=None, erp=None):
    serving_identity = {key: serving_info[key] for key in
                        ("adapter_profile", "adapter_path", "adapter_model_sha256", "client_mode", "candidate_status")
                        if serving_info is not None and key in serving_info}
    if audiences is None:
        audiences = json.loads((ROOT / "data/erp/live-audiences.json").read_text(encoding="utf-8"))
        if erp is None:
            from support.live_erp import DemoERP
            erp = DemoERP.from_path(ROOT / "data/erp/live-objects.json")
    service = LiveService(runtime, policy, audiences, store_path=store_path, archive_path=archive_path,
                          serving_info=serving_identity, erp=erp)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await asyncio.to_thread(service.close)

    app = FastAPI(title="Capstone local support", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.service = service
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.middleware("http")
    async def local_boundary(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                    valid = (parsed.scheme == request.url.scheme and parsed.netloc == request.headers.get("host")
                             and not parsed.path and not parsed.query and not parsed.fragment)
                except ValueError:
                    valid = False
                if not valid:
                    return JSONResponse({"detail": "Разрешены только запросы с локальной страницы сервиса."}, status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Межсайтовый запрос запрещён."}, status_code=403)
            if "application/json" not in request.headers.get("content-type", "").lower():
                return JSONResponse({"detail": "Требуется application/json."}, status_code=415)
        # Unknown cookie values never grant access to an existing owner's records.
        if request.url.path.startswith("/api/"):
            try:
                request.state.owner = service.owner(request.cookies.get(COOKIE))
            except HTTPException as exc:
                return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        response = await call_next(request)
        if hasattr(request.state, "owner") and request.cookies.get(COOKIE) != request.state.owner:
            response.set_cookie(COOKIE, request.state.owner, httponly=True, samesite="strict", max_age=365 * 24 * 60 * 60)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers.setdefault("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'")
        return response

    @app.get("/health")
    def health():
        status = runtime.status()
        return {"status": "ok", "loaded": bool(status.get("loaded")), "busy": bool(status.get("busy")),
                **serving_identity}

    @app.get("/api/client/bootstrap")
    def bootstrap(request: Request):
        return service.bootstrap(request.state.owner)

    @app.post("/api/client/conversations")
    def create_conversation(body: ConversationBody, request: Request):
        return service.create_conversation(request.state.owner, body.audience)

    @app.get("/api/client/input-status")
    def input_status(request: Request):
        return service.input_status(request.state.owner)

    @app.get("/api/client/conversations/{conversation_id}/messages/{key}/status")
    def message_status(conversation_id: str, key: Key, request: Request):
        return service.submission_status(request.state.owner, conversation_id, key)

    @app.get("/api/client/conversations/{conversation_id}")
    def conversation(conversation_id: str, request: Request):
        with service.state_lock:
            return service.public_conversation(service.conversation(request.state.owner, conversation_id))

    @app.post("/api/client/conversations/{conversation_id}/messages")
    def message(conversation_id: str, body: MessageBody, request: Request):
        return service.submit(request.state.owner, conversation_id, body.message, body.idempotency_key)

    @app.get("/api/admin/state")
    def state(include_queue: bool = True):
        snapshot = service.snapshot(include_queue=include_queue)
        snapshot["runtime"].update(serving_identity)
        return snapshot

    @app.get("/api/admin/export")
    def export():
        return service.snapshot(full=True)

    @app.get("/api/admin/requests")
    def requests(q: str = Query(default="", max_length=4000), audience: Audience | None = None,
                 category: Category | None = None, route: ServerRoute | Literal["server_clarification"] | None = None,
                 page: int = Query(default=1, ge=1), page_size: int = Query(default=20, ge=1, le=100)):
        return service.list_requests(q=q, audience=audience, category=category, route=route, page=page, page_size=page_size)

    @app.get("/api/admin/requests/{request_id}")
    def request_detail(request_id: str):
        return service.request(request_id)

    @app.post("/api/admin/comparisons", status_code=202)
    def compare(body: CompareBody, request: Request):
        return service.enqueue_comparison(request.state.owner, body.message, body.audience, body.idempotency_key, body.parent_comparison_id)

    @app.get("/api/admin/comparisons")
    def comparisons(q: str = Query(default="", max_length=4000), audience: Audience | None = None,
                    page: int = Query(default=1, ge=1), page_size: int = Query(default=20, ge=1, le=100)):
        return service.list_comparisons(q=q, audience=audience, page=page, page_size=page_size)

    @app.get("/api/admin/comparisons/{comparison_id}")
    def comparison(comparison_id: str):
        return service.comparison(comparison_id)

    @app.get("/api/admin/queue")
    def queue(q: str = Query(default="", max_length=4000), status: Literal["draft", "approved"] | None = None,
              priority: Priority | None = None, page: int = Query(default=1, ge=1),
              page_size: int = Query(default=20, ge=1, le=100)):
        return service.list_queue(q=q, status=status, priority=priority, page=page, page_size=page_size)

    @app.get("/api/admin/queue/{ticket_id}")
    def queue_detail(ticket_id: str):
        return service.queue_ticket(ticket_id)

    @app.post("/api/admin/queue/{ticket_id}/review")
    def review(ticket_id: str, body: ReviewBody):
        return service.review(ticket_id, body.draft, body.status)

    @app.get("/api/admin/evaluation")
    def evaluation():
        if serving_identity.get("adapter_profile") == "quality-f":
            return live_evaluation.candidate_f_evaluation(serving_identity)
        note = "Validation проверяет метки. Качество текста и доля выдуманных утверждений ещё не оценены человеком; кандидат не выбран."
        empty = dict(available=False, split="validation", n=None, candidate_status="candidate_not_selected",
                     metrics=[], quality_note=note, report_url=None)
        if serving_identity.get("adapter_profile") == "dialogue-250":
            return {**empty, "split": "development", "quality_note":
                    "Активен диалоговый кандидат, обученный на 250 диалогах. Для него есть отдельный development-отчёт. "
                    "Прежняя validation-оценка на 150 примерах относится к full-v8, а не к активному адаптеру. "
                    "Оценка человеком остаётся незавершённой; эти метрики здесь не подменяются."}
        path = REPORT / "comparison.json"
        if not path.exists():
            return empty
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            base, tuned = data["base"]["accuracy"], data["fine_tuned"]["accuracy"]
            metrics = [dict(label=key, base=value["correct"], fine_tuned=tuned[key]["correct"], denominator=value["total"])
                       for key, value in base.items()]
            return {**empty, "available": True, "n": data["base"]["counts"]["cases_loaded"], "metrics": metrics,
                    "report_url": "/admin/validation-report" if (REPORT / "comparison.html").exists() else None}
        except (ValueError, KeyError, TypeError):
            return empty

    @app.get("/admin/candidate-f-report")
    def candidate_f_report():
        path = live_evaluation.CANDIDATE_REPORT
        if not path.is_file():
            raise HTTPException(404, "Отчёт F пока отсутствует.")
        return FileResponse(path, media_type="text/plain; charset=utf-8")

    @app.get("/admin/validation-report")
    def report():
        path = REPORT / "comparison.html"
        if not path.exists():
            raise HTTPException(404, "Отчёт пока отсутствует.")
        html = path.read_text(encoding="utf-8")
        hashes = ["'sha256-" + base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii") + "'"
                  for script in re.findall(r"<script>(.*?)</script>", html, re.S)]
        return FileResponse(path, headers={"Content-Security-Policy":
            "default-src 'none'; style-src 'unsafe-inline'; script-src " + " ".join(hashes)
            + "; frame-ancestors 'none'; base-uri 'none'"})

    static = Path(__file__).parent / "live_static"
    app.mount("/assets", StaticFiles(directory=static, check_dir=False), name="assets")

    @app.get("/")
    def home():
        return RedirectResponse("/client/")

    @app.get("/client/")
    def client_page():
        return FileResponse(static / "client.html")

    @app.get("/admin/")
    def admin_page():
        return FileResponse(static / "admin.html")

    return app
