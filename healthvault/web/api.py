"""Local-only FastAPI adapter for synthetic HealthVault report workflows."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import uuid
from dataclasses import replace
from datetime import date
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import quote, urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from healthvault.application.confirm_extracted_report import confirm_extracted_report
from healthvault.application.knowledge_support_gate import assess_health_knowledge_support
from healthvault.domain.records import ConfirmedSourceAsset
from healthvault.domain.reports import INDICATOR_CATALOG
from healthvault.ocr.pipeline import OCRProcessingError
from healthvault.ui.controller import HealthVaultUIController, configured_database_path
from healthvault.web.answers import (
    INSUFFICIENT_EVIDENCE_MARKER,
    ModelServiceError,
    answer_with_model,
    knowledge_evidence,
    official_web_evidence,
    retrieve_question_knowledge,
    visit_evidence,
)
from healthvault.web.model_runtime import _model_setting, configured_web_model, run_model_task
from healthvault.web.extraction import build_natural_preview, extract_with_model
from healthvault.web.documents import PDFProcessingError, extract_pdf_document
from healthvault.web.agent_tasks import (
    AgentTaskConflict,
    WebAgentTaskStore,
    public_task,
)
from healthvault.web.auth import AuthSession, AuthStore, DuplicateUsername, SESSION_SECONDS
from healthvault.web.answer_history import WebAnswerHistory
from healthvault.web.drafts import Draft, DraftConflict, DraftStore
from healthvault.rag.releases import KnowledgeReleaseStore
from healthvault.rag.sources import OfficialSourceWatcher


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_SKILL_NAME = "HealthVault-Task-Execution"
PROFILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
IMAGE_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class StrictModel(BaseModel):
    class Config:
        extra = "forbid"


class TextDraftInput(StrictModel):
    profile_id: str
    source_id: str = Field(min_length=1, max_length=200)
    report_text: str = Field(min_length=1, max_length=100_000)
    auto_extract: bool = False
    model_consent: bool = False


class DraftUpdateInput(StrictModel):
    profile_id: str
    report_text: str = Field(min_length=1, max_length=100_000)
    reviewed: bool
    model_consent: bool = False


class ConfirmInput(StrictModel):
    profile_id: str
    report_text: str
    preview_sha256: str
    acknowledged: bool


class AnswerInput(StrictModel):
    question: str = Field(min_length=3, max_length=1000)
    model_consent: bool = False
    allow_web_search: bool = False
    context_question: str = Field(default="", max_length=1000)
    allow_query_rewrite: bool = False


class AgentMessageInput(StrictModel):
    prompt: str = Field(min_length=3, max_length=2000)
    expected_version: int = Field(ge=0)
    model_consent: bool = False
    allow_web_search: bool = False


class AgentVersionInput(StrictModel):
    expected_version: int = Field(ge=0)
    model_consent: bool = False


class AgentWorkflowFeedbackInput(StrictModel):
    correction: str = Field(min_length=3, max_length=500)
    confirmed: bool = False
    model_consent: bool = False


class SkillDecisionInput(StrictModel):
    decision: Literal["approve", "reject", "watch"]


class PersonalSkillSettingsInput(StrictModel):
    manual_review: bool


class CalendarCreateInput(StrictModel):
    idempotency_key: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    scheduled_date: str
    remind_before_days: int = Field(default=1, ge=0, le=365)


class CalendarScheduleInput(StrictModel):
    scheduled_date: str
    remind_before_days: int = Field(default=1, ge=0, le=365)
    acknowledged: bool = False


class CalendarEditInput(StrictModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    scheduled_date: str | None = None
    remind_before_days: int | None = Field(default=None, ge=0, le=365)
    acknowledged: bool = False


class AccountInput(StrictModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=1, max_length=128)


def _valid_profile(profile_id: str) -> str:
    if not PROFILE_PATTERN.fullmatch(profile_id):
        raise HTTPException(422, "profile_id 只允许 1–80 位字母、数字、点、下划线或连字符。")
    return profile_id


def _skill_signal_context(prompt: str) -> dict[str, object]:
    """Classify a task without persisting its text, report content, or values."""
    from healthvault.application.workflow_contract import workflow_intent
    intent = workflow_intent(prompt)
    return {
        "intent": intent,
        "preconditions": ["authenticated task", "read-only tool access"],
        "expected_outcome": "validated read-only workflow response",
        "validator_status": "failed",
        "contract_checks": {"status": "failed"},
    }


def _trace_task_id(profile_id: str, task_id: str) -> str:
    return "web-" + hashlib.sha256(
        f"{profile_id}\0{task_id}".encode("utf-8")
    ).hexdigest()[:20]


def _public_skill_review(record: dict[str, Any]) -> dict[str, Any]:
    from healthvault.web.skill_descriptions import candidate_explanation
    return {**{
        key: value for key, value in record.items()
        if key not in {"candidate_content", "evidence_fingerprint"}
    }, 'explanation_zh': candidate_explanation(record)}


def _skill_review_receipt(record: dict[str, Any]) -> dict[str, Any]:
    """A contributor receives no other users' global evidence or workflow diff."""
    return {key: record[key] for key in ("candidate_id", "status")}


async def _model_skill_patch_side_query(
    prompt: str,
    schema: dict[str, Any],
) -> dict[str, Any] | str:
    system_prompt = (
        "You generate a bounded workflow-only Skill patch. Return one JSON object and "
        "nothing else. It must match the supplied JSON Schema exactly. Never add medical "
        "facts, patient data, permissions, safety-policy changes, or evaluation answers."
    )
    raw = await asyncio.to_thread(
        run_model_task,
        system_prompt,
        {"request": json.loads(prompt), "response_schema": schema},
        profile_id="shared-skill-review",
        failure_message="Skill 候选生成暂不可用。",
    )
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.I)
    return stripped


def _preview_hash(draft: Draft) -> str:
    assert draft.preview_state is not None
    payload = {
        "token": draft.token,
        "profile_id": draft.profile_id,
        "source_id": draft.source_id,
        "report_text": draft.report_text,
        "image_sha256": draft.image_sha256,
        "ocr_reviewed": draft.ocr_reviewed,
        "file_media_type": draft.image_media_type,
        "preview": draft.preview_state["preview"],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _public_draft(draft: Draft) -> dict[str, Any]:
    return {
        "token": draft.token,
        "source_id": draft.source_id,
        "report_text": draft.report_text,
        "input_kind": draft.input_kind,
        "image_sha256": draft.image_sha256,
        "ocr_text_sha256": draft.ocr_text_sha256,
        "ocr_blocks": draft.ocr_blocks or [],
        "ocr_confidence": draft.ocr_confidence,
        "ocr_reviewed": draft.ocr_reviewed,
        "image_url": (
            f"/api/drafts/{draft.token}/image?profile_id={quote(draft.profile_id)}"
            if draft.image_bytes else None
        ),
        "preview": draft.preview_state["preview"] if draft.preview_state else None,
        "preview_sha256": draft.preview_sha256,
        "original_text": draft.original_text,
        "processing_status": draft.processing_status,
        "processing_error": draft.processing_error,
        "document_artifact": draft.document_artifact,
        "archived_at": draft.archived_at,
        "duplicate_of": draft.duplicate_of,
        "saved": True,
    }


def create_app(
    controller: HealthVaultUIController | None = None,
    model_answerer: Callable[[str, str, dict[str, Any]], str] | None = None,
    model_extractor: Callable[[str], dict[str, Any]] | None = None,
    *,
    skill_root: str | Path | None = None,
    skill_evolution_root: str | Path | None = None,
    skill_patch_side_query: Callable[..., Any] | None = None,
    skill_workflow_evaluator: Callable[..., Any] | None = None,
    knowledge_release_root: str | Path | None = None,
    official_source_fetcher: Callable[..., Any] | None = None,
    model_answer_verifier: Callable[..., Any] | None = None,
    model_query_rewriter: Callable[..., Any] | None = None,
    model_question_planner: Callable[..., Any] | None = None,
    auto_plan_suggestions: bool = True,
    answer_repair_enabled: bool = True,
) -> FastAPI:
    """Profile-owned durable report intake; local development server."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.source_watcher.start()
        try:
            yield
        finally:
            application.state.source_watcher.stop()
    app = FastAPI(title="HealthVault local demo API", version="1", lifespan=lifespan)
    app.state.controller = controller or HealthVaultUIController(configured_database_path(), knowledge_release_root=knowledge_release_root)
    if controller is not None and knowledge_release_root is not None:
        controller.knowledge_release_root = Path(knowledge_release_root)
        controller.reload_knowledge()
    app.state.knowledge_releases = KnowledgeReleaseStore(getattr(app.state.controller, "knowledge_release_root", knowledge_release_root))
    app.state.source_watcher = OfficialSourceWatcher(app.state.knowledge_releases, **({"fetcher": official_source_fetcher} if official_source_fetcher else {}))
    app.state.knowledge_release_lock = threading.RLock()
    app.state.model_answerer = model_answerer
    app.state.model_answer_verifier = model_answer_verifier
    app.state.model_query_rewriter = model_query_rewriter
    app.state.model_question_planner = model_question_planner
    app.state.auto_plan_suggestions = auto_plan_suggestions
    app.state.answer_repair_enabled = answer_repair_enabled
    app.state.model_extractor = model_extractor
    if hasattr(app.state.controller, "knowledge_repository"):
        from healthvault.tools.catalog import build_healthvault_tool_registry
        from healthvault.tools.directory import HARNESS_READ_ONLY_TOOL_NAMES
        app.state.read_only_tools = build_healthvault_tool_registry(
            app.state.controller.record_repository,
            app.state.controller.knowledge_repository,
            allowed_names=HARNESS_READ_ONLY_TOOL_NAMES,
        )
    else:
        app.state.read_only_tools = None
    app.state.drafts_lock = threading.RLock()
    task_database = app.state.controller.record_repository.database_path
    app.state.draft_store = DraftStore(task_database.with_name(task_database.stem + "-uploads.sqlite3"))
    app.state.agent_task_path = task_database.with_name(task_database.stem + "-agent-tasks.sqlite3")
    app.state.agent_tasks = WebAgentTaskStore(app.state.agent_task_path)
    app.state.auth_store = AuthStore(task_database.with_name(task_database.stem + "-auth.sqlite3"))
    app.state.answer_history = WebAnswerHistory(task_database.with_name(task_database.stem + "-answers.sqlite3"))
    app.state.skill_root = Path(skill_root) if skill_root is not None else PROJECT_ROOT / ".bear" / "skills"
    app.state.skill_evolution_root = (
        Path(skill_evolution_root)
        if skill_evolution_root is not None
        else PROJECT_ROOT / ".bear" / "skill-evolution" / "healthvault"
    )
    from healthvault.evolution.review import SkillEvolutionService
    app.state.skill_evolution = SkillEvolutionService(
        app.state.skill_root,
        app.state.skill_evolution_root,
        skill_patch_side_query or _model_skill_patch_side_query,
    )
    from healthvault.evolution.personal import PersonalSkillManager
    from healthvault.evolution.workflow_eval import model_workflow_evaluator
    app.state.personal_skills = PersonalSkillManager(app.state.skill_root, app.state.skill_evolution_root,
                                                    skill_patch_side_query or _model_skill_patch_side_query,
                                                    evaluator=skill_workflow_evaluator or model_workflow_evaluator)
    app.state.agent_task_locks: dict[tuple[str, str], asyncio.Lock] = {}
    app.state.agent_task_locks_lock = threading.RLock()

    @app.middleware("http")
    async def guard_api(request: Request, call_next: Any) -> Response:
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            allowed = {
                str(request.base_url).rstrip("/"),
                "http://127.0.0.1:5173", "http://localhost:5173",
                os.environ.get("HEALTHVAULT_WEB_ORIGIN", ""),
            }
            if (origin and origin not in allowed) or request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "请求来源不被允许。"}, status_code=403)
        if path not in {"/api/auth/register", "/api/auth/login"}:
            session = app.state.auth_store.session(request.cookies.get("hv_session"))
            if session is None:
                return JSONResponse({"detail": "请先登录。"}, status_code=401)
            request.state.auth = session
            parts = path.split("/")
            if len(parts) > 3 and parts[2] == "profiles" and parts[3] != session.user.profile_id:
                return JSONResponse({"detail": "记录不存在。"}, status_code=404)
            if request.method not in {"GET", "HEAD", "OPTIONS"} and not secrets.compare_digest(
                request.headers.get("x-csrf-token", ""), session.csrf_token
            ):
                return JSONResponse({"detail": "页面会话已失效，请刷新后重试。"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def require_profile(request: Request, profile_id: str) -> str:
        profile_id = _valid_profile(profile_id)
        if profile_id != request.state.auth.user.profile_id:
            raise HTTPException(404, "记录不存在。")
        return profile_id

    def require_admin(request: Request) -> AuthSession:
        session: AuthSession = request.state.auth
        if not session.user.is_admin:
            raise HTTPException(403, "仅管理员可以审核或发布全局 Skill。")
        return session

    def account_response(request: Request, response: Response, user: Any) -> dict[str, Any]:
        token, csrf_token = app.state.auth_store.create_session(user)
        response.set_cookie(
            "hv_session", token, max_age=SESSION_SECONDS, httponly=True,
            secure=request.url.scheme == "https", samesite="strict", path="/api",
        )
        return {
            "username": user.username,
            "profile_id": user.profile_id,
            "csrf_token": csrf_token,
            "is_admin": user.is_admin,
        }

    @app.post("/api/auth/register")
    def register_account(body: AccountInput, request: Request, response: Response) -> dict[str, Any]:
        try:
            user = app.state.auth_store.register(body.username, body.password)
        except DuplicateUsername as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return account_response(request, response, user)

    @app.post("/api/auth/login")
    def login_account(body: AccountInput, request: Request, response: Response) -> dict[str, Any]:
        user = app.state.auth_store.authenticate(body.username, body.password)
        if user is None:
            raise HTTPException(401, "用户名或密码错误。")
        return account_response(request, response, user)

    @app.get("/api/auth/me")
    def current_account(request: Request) -> dict[str, Any]:
        session: AuthSession = request.state.auth
        return {
            "username": session.user.username,
            "profile_id": session.user.profile_id,
            "csrf_token": session.csrf_token,
            "is_admin": session.user.is_admin,
        }

    @app.post("/api/auth/logout")
    def logout_account(request: Request, response: Response) -> dict[str, bool]:
        app.state.auth_store.logout(request.cookies.get("hv_session"))
        response.delete_cookie("hv_session", path="/api")
        return {"logged_out": True}

    def task_lock(profile_id: str, task_id: str) -> asyncio.Lock:
        key = (profile_id, task_id)
        with app.state.agent_task_locks_lock:
            return app.state.agent_task_locks.setdefault(key, asyncio.Lock())

    def task_checkpoint(profile_id: str, task_id: str) -> Any:
        from healthvault.runtime_context import RuntimeContext
        from healthvault.task_memory import TaskMemoryStore

        store = TaskMemoryStore(
            app.state.agent_task_path,
            app.state.controller.record_repository,
        )
        return store.get(RuntimeContext(profile_id, "web-task-read"), task_id)

    def build_web_agent(
        profile_id: str,
        task_id: str,
        *,
        allow_web_search: bool = False,
        prompt: str | None = None,
        question_plan: dict | None = None,
    ) -> Any:
        from healthvault.agent_runtime import (
            HealthVaultAgentRuntimeConfig,
            build_healthvault_agent_runtime,
        )
        from healthvault.runtime_context import RuntimeContext

        try:
            model_config = configured_web_model()
        except ModelServiceError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        active_skill = app.state.skill_root / ACTIVE_SKILL_NAME / "SKILL.md"
        skill_names = (ACTIVE_SKILL_NAME,) if active_skill.is_file() else ()
        signal_context = _skill_signal_context(prompt) if prompt is not None else None
        trace_task_id = _trace_task_id(profile_id, task_id)
        personal_service = app.state.personal_skills.service(profile_id)

        async def verify_agent_answer(text: str, sources: list[dict]) -> dict:
            from healthvault.web.answer_validation import verify_answer
            cited = re.findall(r'\[((?:HVK|HVW|REC):[^\]]+)\]', text)
            available = {source['citation_id'] for source in sources}
            if not cited or not set(cited).issubset(available):
                return {'passed': False, 'kind': 'missing_current_tool_evidence'}
            return await asyncio.to_thread(verify_answer, question_plan['original_question'], text, sources,
                                           query=app.state.model_answer_verifier, profile_id=profile_id,
                                           resolved_question=question_plan['resolved_question'], audit_coverage=True,
                                           required_points=question_plan['points'])

        async def repair_agent_answer(_text: str, sources: list[dict], feedback: dict) -> str:
            evidence = {'sources': sources, 'question_plan': question_plan, 'answer_repair': feedback}
            mode = 'visit' if any(source['citation_id'].startswith('REC:') for source in sources) else 'knowledge'
            if app.state.model_answerer is not None:
                return await asyncio.to_thread(app.state.model_answerer, question_plan['original_question'], mode, evidence)
            return await asyncio.to_thread(answer_with_model, question_plan['original_question'], mode, evidence,
                                           profile_id=profile_id)

        return build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
            context=RuntimeContext(profile_id, uuid.uuid4().hex),
            database_path=app.state.controller.record_repository.database_path,
            knowledge_path=getattr(app.state.controller, "knowledge_path", None),
            knowledge_mode=getattr(app.state.controller, "knowledge_mode", "lexical"),
            hybrid_index_path=getattr(app.state.controller, "hybrid_index_path", None),
            hybrid_model_path=getattr(app.state.controller, "hybrid_model_path", None),
            model_config=model_config,
            task_id=task_id,
            task_memory_path=app.state.agent_task_path,
            allow_web_search=allow_web_search,
            skill_root=app.state.skill_root,
            skill_names=skill_names,
            evolution_root=(personal_service.root if skill_names and prompt is not None else None),
            personal_skill_root=app.state.personal_skills.injectable_root(profile_id) if skill_names else None,
            signal_context=signal_context,
            trace_task_id=trace_task_id,
            answer_verifier=verify_agent_answer if question_plan is not None else None,
            answer_repairer=repair_agent_answer if question_plan is not None and app.state.answer_repair_enabled else None,
            question_plan=question_plan,
            max_turns=6,
            max_cost_usd=1.0,
        ))

    def restore_web_agent(runtime: Any, saved: dict[str, Any] | None) -> None:
        if not saved:
            return
        runtime.agent.restore_session({
            "openaiMessages": saved["messages"],
            "foldedSessionMemories": saved["folded"],
        })
        if runtime.agent._openai_messages:
            runtime.agent._openai_messages[0]["content"] = runtime.agent._system_prompt

    def snapshot_web_agent(runtime: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            json.loads(json.dumps(runtime.agent._openai_messages, ensure_ascii=False)),
            json.loads(json.dumps(runtime.agent._folded_session_memories, ensure_ascii=False)),
        )

    def owned_draft(token: str, profile_id: str) -> Draft:
        _valid_profile(profile_id)
        draft = app.state.draft_store.get(token)
        if draft is None:
            raise HTTPException(404, "草稿不存在或已更新，请重新打开。")
        if draft.profile_id != profile_id:
            raise HTTPException(404, "草稿不存在或已更新，请重新打开。")
        return draft

    def save_draft(draft: Draft, *, previous_token: str | None = None) -> None:
        try:
            app.state.draft_store.save(draft, previous_token=previous_token)
        except DraftConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    def preview_and_save(draft: Draft, *, previous_token: str | None = None) -> None:
        try:
            set_preview(draft)
            draft.processing_status = "needs_review"
            draft.processing_error = (
                ("部分页面未完整识别，请逐页对照原件核对。" if draft.input_kind=='pdf' else "图像质量需要核对，请对照原件检查识别结果。")
                if draft.document_artifact and draft.document_artifact.get("warnings") else None
            )
        except HTTPException as exc:
            draft.processing_status = "extraction_failed"
            draft.processing_error = str(exc.detail)
        save_draft(draft, previous_token=previous_token)

    def set_preview(draft: Draft) -> None:
        if draft.input_kind in {"image", "pdf"} and not draft.ocr_reviewed:
            draft.preview_state = None
            draft.preview_sha256 = None
            return
        canonical_format = "result|" in draft.report_text and "report_id:" in draft.report_text
        if not canonical_format and (draft.auto_extract or "report_id:" not in draft.report_text):
            try:
                draft.extraction_payload = (
                    (app.state.model_extractor(draft.report_text)
                     if app.state.model_extractor is not None
                     else extract_with_model(draft.report_text, profile_id=draft.profile_id))
                    if draft.auto_extract else {}
                )
            except ModelServiceError as exc:
                raise HTTPException(exc.status_code, str(exc)) from exc
            preview = build_natural_preview(
                draft.source_id, draft.report_text, draft.extraction_payload,
                blocks=draft.ocr_blocks, original_text=draft.original_text,
            )
            draft.preview_state = {
                "input_kind": "natural",
                "source_id": draft.source_id,
                "report_text": draft.report_text,
                "sha256": hashlib.sha256(draft.report_text.encode("utf-8")).hexdigest(),
                "preview": preview,
            }
        elif draft.input_kind == "image":
            assert draft.image_sha256 is not None
            assert draft.ocr_text_sha256 is not None
            draft.preview_state = app.state.controller.preview_ocr_report(
                draft.source_id,
                draft.report_text,
                draft.image_sha256,
                draft.ocr_text_sha256,
                ocr_blocks=draft.ocr_blocks,
                reviewed=True,
            )
        else:
            draft.preview_state = app.state.controller.preview_report(
                draft.source_id, draft.report_text
            )
        if draft.preview_state.get("input_kind") != "natural":
            from healthvault.web.evidence import bind_preview_evidence
            bind_preview_evidence(
                draft.preview_state["preview"], draft.report_text,
                draft.ocr_blocks or [], draft.original_text,
            )
        draft.preview_sha256 = _preview_hash(draft)

    @app.post("/api/drafts/text")
    def create_text_draft(body: TextDraftInput, request: Request) -> dict[str, Any]:
        require_profile(request, body.profile_id)
        if body.auto_extract and not body.model_consent:
            raise HTTPException(409, "模型抽取需要先同意发送本次报告文字。")
        draft = Draft(
            token=secrets.token_urlsafe(32),
            profile_id=body.profile_id,
            source_id=body.source_id,
            report_text=body.report_text,
            input_kind="text",
            auto_extract=body.auto_extract,
            original_text=body.report_text,
        )
        save_draft(draft)
        preview_and_save(draft)
        return _public_draft(draft)

    @app.get("/api/profiles/{profile_id}/uploads")
    def report_uploads(profile_id: str, request: Request, status: Literal['active', 'pending', 'failed', 'archived', 'all'] = 'active') -> dict[str, Any]:
        require_profile(request, profile_id)
        return {"uploads": app.state.draft_store.list(profile_id, status=status)}

    @app.get("/api/drafts/{token}")
    def read_draft(token: str, profile_id: str, request: Request) -> dict[str, Any]:
        require_profile(request, profile_id)
        return _public_draft(owned_draft(token, profile_id))

    def organize_draft(token: str, action: Literal['archive', 'restore'], request: Request) -> dict[str, Any]:
        with app.state.drafts_lock:
            draft = owned_draft(token, request.state.auth.user.profile_id)
            if draft.processing_status == 'confirmed':
                raise HTTPException(409, '已确认的报告请从档案查看。')
            if action == 'archive':
                from datetime import datetime, timezone
                draft.archived_at = draft.archived_at or datetime.now(timezone.utc).isoformat(timespec='seconds')
            else:
                draft.archived_at = None
            save_draft(draft)
            return _public_draft(draft)

    @app.post('/api/drafts/{token}/archive')
    def archive_draft(token: str, request: Request) -> dict[str, Any]:
        return organize_draft(token, 'archive', request)

    @app.post('/api/drafts/{token}/restore')
    def restore_draft(token: str, request: Request) -> dict[str, Any]:
        return organize_draft(token, 'restore', request)

    @app.post("/api/drafts/image")
    async def create_image_draft(
        request: Request,
        profile_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        require_profile(request, profile_id)
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "图片为空或超过 8 MB。")
        extension = Path(file.filename or "").suffix.lower()
        if extension not in IMAGE_MEDIA_TYPES:
            raise HTTPException(415, "仅支持 PNG、JPG、JPEG。")
        if extension == ".png" and not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(422, "PNG 文件内容无效。")
        if extension in {".jpg", ".jpeg"} and not raw.startswith(b"\xff\xd8\xff"):
            raise HTTPException(422, "JPEG 文件内容无效。")
        draft = Draft(
            token=secrets.token_urlsafe(32),
            profile_id=profile_id,
            source_id=f"image:{hashlib.sha256(raw).hexdigest()}",
            report_text="",
            input_kind="image",
            image_bytes=raw,
            image_filename=Path(file.filename or "report.png").name,
            image_media_type=IMAGE_MEDIA_TYPES[extension],
            image_sha256=hashlib.sha256(raw).hexdigest(),
            auto_extract=True,
        )
        save_draft(draft)
        try:
            ocr = await run_in_threadpool(app.state.controller.process_image, raw, file.filename or "report.png")
            draft.report_text = draft.original_text = ocr["text"]
            draft.ocr_text_sha256 = ocr["text_sha256"]
            draft.ocr_blocks = ocr["blocks"]
            draft.ocr_confidence = ocr["confidence"]
            preparation = ocr.get('preprocessing',{})
            flags = preparation.get('quality',{}).get('flags',[])
            draft.document_artifact = {'kind':'image','preprocessing':preparation,'warnings':[{'code':'image_quality','flags':flags}] if flags else []}
            if flags:
                draft.processing_error = '图像质量较差，请对照原件检查识别结果。'
            draft.processing_status = "needs_review"
        except OCRProcessingError as exc:
            draft.processing_status = "extraction_failed"
            draft.processing_error = str(exc)
        save_draft(draft)
        return _public_draft(draft)

    @app.get("/api/drafts/{token}/image")
    def draft_image(token: str, profile_id: str, request: Request) -> Response:
        require_profile(request, profile_id)
        with app.state.drafts_lock:
            draft = owned_draft(token, profile_id)
            if draft.image_bytes is None:
                raise HTTPException(404, "该草稿没有原图。")
            return Response(
                draft.image_bytes,
                media_type=draft.image_media_type,
                headers={"Cache-Control": "no-store"},
            )

    @app.get("/api/drafts/{token}/pages/{page_number}")
    def draft_page(token: str, page_number: int, request: Request) -> Response:
        draft = owned_draft(token, request.state.auth.user.profile_id)
        if draft.input_kind != "pdf" or draft.image_bytes is None:
            raise HTTPException(404, "PDF 页面不存在。")
        import pymupdf
        with pymupdf.open(stream=draft.image_bytes, filetype="pdf") as document:
            if not 1 <= page_number <= len(document):
                raise HTTPException(404, "PDF 页面不存在。")
            page = document[page_number-1]; page.set_rotation(0)
            image = page.get_pixmap(matrix=pymupdf.Matrix(2,2), alpha=False)
            return Response(image.tobytes("png"), media_type="image/png", headers={"Cache-Control":"no-store"})

    @app.post("/api/drafts/pdf")
    async def create_pdf_draft(
        request: Request,
        profile_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        require_profile(request, profile_id)
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "PDF 为空或超过 8 MB。")
        if Path(file.filename or "").suffix.lower() != ".pdf" or not raw.startswith(b"%PDF-"):
            raise HTTPException(415, "请选择有效 PDF 报告。")
        file_hash = hashlib.sha256(raw).hexdigest()
        draft = Draft(
            token=secrets.token_urlsafe(32),
            profile_id=profile_id,
            source_id=f"pdf:{file_hash}",
            report_text="",
            input_kind="pdf",
            image_bytes=raw,
            image_filename=Path(file.filename or "report.pdf").name,
            image_media_type="application/pdf",
            image_sha256=file_hash,
            auto_extract=True,
        )
        save_draft(draft)
        try:
            artifact = await run_in_threadpool(extract_pdf_document, raw, app.state.controller.process_image)
            report_text = artifact["text"]
            draft.report_text = draft.original_text = report_text
            draft.ocr_text_sha256 = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
            draft.ocr_blocks = artifact["blocks"]
            draft.document_artifact = artifact
            draft.processing_status = "needs_review" if report_text else "extraction_failed"
            if artifact["warnings"]:
                draft.processing_error = "部分页面未完整识别，请逐页对照原件核对。"
        except (PDFProcessingError, OCRProcessingError) as exc:
            draft.processing_status = "extraction_failed"
            draft.processing_error = str(exc)
        save_draft(draft)
        return _public_draft(draft)

    @app.put("/api/drafts/{token}")
    def update_draft(token: str, body: DraftUpdateInput, request: Request) -> dict[str, Any]:
        require_profile(request, body.profile_id)
        with app.state.drafts_lock:
            draft = owned_draft(token, body.profile_id)
            if draft.archived_at:
                raise HTTPException(409, '草稿已归档，请先恢复再编辑。')
            if draft.processing_status == "confirmed":
                raise HTTPException(409, "这份报告已确认，请从报告详情查看原文。")
            updated = replace(
                draft,
                token=secrets.token_urlsafe(32),
                report_text=body.report_text,
                ocr_reviewed=body.reviewed if draft.input_kind in {"image", "pdf"} else False,
                preview_state=None,
                preview_sha256=None,
                extraction_payload=None,
                auto_extract=body.model_consent,
            )
        with app.state.drafts_lock:
            preview_and_save(updated, previous_token=token)
            return _public_draft(updated)

    @app.post("/api/drafts/{token}/confirm")
    def confirm_draft(token: str, body: ConfirmInput, request: Request) -> dict[str, Any]:
        require_profile(request, body.profile_id)
        with app.state.drafts_lock:
            draft = owned_draft(token, body.profile_id)
            if draft.archived_at:
                raise HTTPException(409, '草稿已归档，请先恢复再确认。')
            if not body.acknowledged:
                raise HTTPException(409, "请明确确认已核对报告。")
            if (
                draft.preview_state is None
                or draft.preview_sha256 is None
                or draft.preview_sha256 != body.preview_sha256
                or draft.report_text != body.report_text
                or (draft.input_kind in {"image", "pdf"} and not draft.ocr_reviewed)
            ):
                raise HTTPException(409, "报告与预览不一致，请重新预览。")
            original_asset = (
                ConfirmedSourceAsset(
                    filename=draft.image_filename or "report",
                    media_type=draft.image_media_type or "application/octet-stream",
                    sha256=draft.image_sha256 or "",
                    content=draft.image_bytes,
                )
                if draft.image_bytes is not None else None
            )
            if draft.preview_state.get("input_kind") == "natural":
                if draft.extraction_payload is None or build_natural_preview(
                    draft.source_id, draft.report_text, draft.extraction_payload,
                    blocks=draft.ocr_blocks, original_text=draft.original_text,
                ) != draft.preview_state["preview"]:
                    raise HTTPException(409, "抽取结果已变化，请重新预览。")
                result = confirm_extracted_report(
                    draft.profile_id,
                    draft.source_id,
                    draft.report_text,
                    draft.preview_state["preview"],
                    app.state.controller.record_repository,
                    original_asset=original_asset,
                )
            else:
                # confirm_preview reparses the documented structured format.
                result = app.state.controller.confirm_preview(
                    profile_id=draft.profile_id,
                    preview_state=draft.preview_state,
                    current_source_id=draft.source_id,
                    current_report_text=draft.report_text,
                    current_image_sha256=draft.image_sha256,
                    ocr_reviewed=draft.ocr_reviewed,
                    original_asset=original_asset,
                )
            if not result.get("confirmed"):
                raise HTTPException(409, result)
            draft.processing_status = "confirmed"
            save_draft(draft)
            result["calendar_sync"] = app.state.controller.sync_calendar(draft.profile_id)
            return result

    def calendar_result(result: dict[str, Any]) -> dict[str, Any]:
        if result["status"] == "not_found":
            raise HTTPException(404, "日历项目不存在。")
        if result["status"] in {"invalid_request", "invalid_transition", "idempotency_conflict"}:
            raise HTTPException(409, "日期、事件状态或重复请求内容不一致，请刷新后核对。")
        return result

    @app.get("/api/profiles/{profile_id}/calendar")
    def calendar_view(profile_id: str, request: Request, as_of: str | None = None) -> dict[str, Any]:
        require_profile(request, profile_id)
        app.state.controller.sync_calendar(profile_id)
        return calendar_result(app.state.controller.calendar(profile_id, as_of or date.today().isoformat()))

    @app.post("/api/profiles/{profile_id}/calendar/events")
    def calendar_create(profile_id: str, body: CalendarCreateInput, request: Request) -> dict[str, Any]:
        require_profile(request, profile_id)
        return calendar_result(app.state.controller.create_calendar_event(profile_id, **body.model_dump()))

    @app.post("/api/profiles/{profile_id}/calendar/plans/{clinical_item_id}/schedule")
    def calendar_schedule(profile_id: str, clinical_item_id: str, body: CalendarScheduleInput, request: Request) -> dict[str, Any]:
        require_profile(request, profile_id)
        if not body.acknowledged:
            raise HTTPException(409, "请核对原文并确认所选复诊日期。")
        return calendar_result(app.state.controller.calendar_service.schedule_plan(profile_id, clinical_item_id, body.scheduled_date, body.remind_before_days))

    @app.post("/api/profiles/{profile_id}/calendar/events/{event_id}/activate")
    def calendar_activate(profile_id: str, event_id: str, body: CalendarEditInput, request: Request) -> dict[str, Any]:
        require_profile(request, profile_id)
        if not body.acknowledged:
            raise HTTPException(409, "请核对原文并确认复诊日历安排。")
        return calendar_result(app.state.controller.activate_calendar_event(profile_id, event_id, **body.model_dump(exclude_none=True, exclude={"acknowledged"})))

    @app.patch("/api/profiles/{profile_id}/calendar/events/{event_id}")
    def calendar_update(profile_id: str, event_id: str, body: CalendarEditInput, request: Request) -> dict[str, Any]:
        require_profile(request, profile_id)
        return calendar_result(app.state.controller.update_calendar_event(profile_id, event_id, **body.model_dump(exclude_none=True, exclude={"acknowledged"})))

    @app.post("/api/profiles/{profile_id}/calendar/events/{event_id}/{action}")
    def calendar_transition(profile_id: str, event_id: str, action: Literal["complete", "cancel", "acknowledge"], request: Request) -> dict[str, Any]:
        require_profile(request, profile_id)
        if action == "acknowledge":
            result = app.state.controller.acknowledge_calendar_reminder(profile_id, event_id, date.today().isoformat())
        elif action == "complete":
            result = app.state.controller.complete_calendar_event(profile_id, event_id)
        else:
            result = app.state.controller.cancel_calendar_event(profile_id, event_id)
        return calendar_result(result)

    @app.get("/api/profiles/{profile_id}/agent/capabilities")
    def agent_capabilities(profile_id: str) -> dict[str, Any]:
        _valid_profile(profile_id)
        base = (_model_setting("HEALTHVAULT_OPENAI_BASE_URL") or _model_setting("OPENAI_BASE_URL")).strip()
        parsed = urlparse(base)
        configured = bool(
            (_model_setting("HEALTHVAULT_OPENAI_API_KEY") or _model_setting("OPENAI_API_KEY")).strip()
            and (_model_setting("HEALTHVAULT_OPENAI_MODEL") or _model_setting("OPENAI_MODEL")).strip()
            and parsed.scheme == "https" and parsed.hostname
        )
        return {
            "status": "ready" if configured else "model_unconfigured",
            "memory": True,
            "session_compaction": True,
            "task_resume": True,
            "mcp": True,
            "mcp_scope": "仅供外部 Agent/Codex/第三方客户端可选接入",
            "runtime_tool_transport": "native_registry",
            "global_skills": False,
            "project_skill": ACTIVE_SKILL_NAME,
            "skill_candidate_review": app.state.personal_skills.settings(profile_id)["manual_review"],
            "personal_skills": True,
        }

    @app.get("/api/profiles/{profile_id}/agent/tasks")
    def agent_task_list(profile_id: str) -> dict[str, Any]:
        return {
            "tasks": app.state.agent_tasks.list_tasks(_valid_profile(profile_id)),
        }

    @app.get("/api/profiles/{profile_id}/agent/tasks/{task_id}")
    def agent_task_get(profile_id: str, task_id: str) -> dict[str, Any]:
        _valid_profile(profile_id)
        try:
            saved = app.state.agent_tasks.get(profile_id, task_id)
            return public_task(saved, task_id, task_checkpoint(profile_id, task_id))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/profiles/{profile_id}/agent/tasks/{task_id}/messages")
    async def agent_task_message(
        profile_id: str, task_id: str, body: AgentMessageInput,
    ) -> dict[str, Any]:
        _valid_profile(profile_id)
        if not body.model_consent:
            raise HTTPException(409, "请先同意将本次问题及必要证据发送至模型服务。")
        try:
            app.state.agent_tasks._check_task_id(task_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        async with task_lock(profile_id, task_id):
            saved = app.state.agent_tasks.get(profile_id, task_id)
            if (saved["version"] if saved else 0) != body.expected_version:
                raise HTTPException(409, "任务已更新，请刷新后继续。")
            from healthvault.application.query_rewrite import prepare_query
            from healthvault.web.question_plan import create_question_plan
            previous = next((item['text'] for item in reversed(saved['dialogue']) if item['role'] == 'user'), '') if saved else ''
            prepared = prepare_query(body.prompt.strip(), context_question=previous, knowledge_only=False)
            if prepared.status == 'needs_clarification':
                raise HTTPException(422, '请明确你说的是哪个指标或检查，再继续提问。')
            try:
                plan = await asyncio.to_thread(create_question_plan, body.prompt.strip(), resolved_question=prepared.resolved_question,
                                               query=app.state.model_question_planner if app.state.auto_plan_suggestions else None,
                                               use_model=app.state.auto_plan_suggestions and app.state.model_answerer is None,
                                               profile_id=profile_id)
            except ModelServiceError as exc:
                raise HTTPException(exc.status_code, str(exc)) from exc
            runtime = build_web_agent(
                profile_id,
                task_id,
                allow_web_search=body.allow_web_search,
                prompt=body.prompt.strip(),
                question_plan=plan,
            )
            from agents.ui import quiet_agent_ui
            try:
                with quiet_agent_ui():
                    restore_web_agent(runtime, saved)
                    agent_prompt = body.prompt.strip()
                    if prepared.resolved_question != agent_prompt:
                        agent_prompt += '\n指代解析：' + prepared.resolved_question + '\n这是上一用户问题的指标名称，不是新的患者事实。原问题和约束保持不变。'
                    reply = await runtime.run(agent_prompt)
                    tool_names = list(runtime.manifest["tool_names"])
            except Exception as exc:
                logging.getLogger(__name__).exception("HealthVault web Agent failed")
                if runtime.trace_recorder is not None and runtime.trace_recorder.trace_path is not None:
                    app.state.personal_skills.record_usage(profile_id, runtime.trace_recorder.trace_path, injected=bool(runtime.manifest.get("personal_skill_injected")))
                raise HTTPException(
                    502, f"Agent 调用未完成：{type(exc).__name__}。请稍后重试。"
                ) from exc
            messages, folded = snapshot_web_agent(runtime)
            dialogue = list(saved["dialogue"] if saved else [])
            dialogue.extend([
                {"role": "user", "text": body.prompt.strip()},
                {"role": "assistant", "text": reply.text},
            ])
            try:
                updated = app.state.agent_tasks.save(
                    profile_id, task_id,
                    expected_version=body.expected_version,
                    messages=messages,
                    folded=folded,
                    dialogue=dialogue[-60:],
                )
            except AgentTaskConflict as exc:
                raise HTTPException(409, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(413, str(exc)) from exc
            skill_candidate = None
            if runtime.trace_recorder is not None and runtime.trace_recorder.trace_path is not None:
                try:
                    app.state.personal_skills.record_usage(profile_id, runtime.trace_recorder.trace_path, injected=bool(runtime.manifest.get("personal_skill_injected")))
                    candidate = await app.state.personal_skills.consider(profile_id)
                    skill_candidate = _skill_review_receipt(candidate) if candidate else None
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Skill candidate extraction failed after task %s: %s",
                        task_id,
                        type(exc).__name__,
                    )
            return {
                **public_task(updated, task_id, task_checkpoint(profile_id, task_id)),
                "answer": reply.text,
                "memory_status": reply.task_memory_status,
                "tool_names": tool_names,
                "project_skills": reply.project_skills,
                "skill_candidate": skill_candidate,
                **({'measurement': reply.measurement} if reply.measurement is not None else {}),
                **({'verification': reply.verification} if reply.verification is not None else {}),
            }

    @app.post("/api/profiles/{profile_id}/agent/tasks/{task_id}/workflow-feedback")
    async def agent_workflow_feedback(
        profile_id: str,
        task_id: str,
        body: AgentWorkflowFeedbackInput,
        request: Request,
    ) -> dict[str, Any]:
        require_profile(request, profile_id)
        if not body.confirmed:
            raise HTTPException(409, "请明确确认这是可复用的工作流纠正。")
        if not body.model_consent:
            raise HTTPException(409, "生成候选可能使用模型服务，请先确认。")
        trace_id = app.state.personal_skills.service(profile_id).latest_trace_id_for_task(
            _trace_task_id(profile_id, task_id)
        )
        if trace_id is None:
            raise HTTPException(404, "当前任务没有可用于纠正的脱敏 Trace。")
        try:
            candidate = await app.state.personal_skills.consider(profile_id,
                feedback_by_trace_id={trace_id: body.correction.strip()},
                user_confirmed_trace_ids={trace_id},
                only_trace_ids={trace_id},
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("Workflow feedback processing failed")
            raise HTTPException(422, f"纠正未形成安全候选：{type(exc).__name__}。") from exc
        if candidate is None:
            return {
                "status": "not_eligible",
                "reason": "纠正包含个人/医学事实、没有工具轨迹，或不满足候选证据门槛。",
            }
        return {"status": "candidate_created", "candidate": _skill_review_receipt(candidate)}

    @app.get("/api/skills")
    def personal_skills(request: Request) -> dict[str, Any]:
        profile = request.state.auth.user.profile_id
        service = app.state.personal_skills.service(profile)
        return {"settings": app.state.personal_skills.settings(profile),
                "usage": app.state.personal_skills.usage(profile),
                "candidates": [_public_skill_review(item) for item in service.store.list()]}

    @app.post("/api/skills/disable")
    def personal_skill_disable(request: Request) -> dict[str, Any]:
        return app.state.personal_skills.disable(request.state.auth.user.profile_id)

    @app.put("/api/skills/settings")
    def personal_skill_settings(body: PersonalSkillSettingsInput, request: Request) -> dict[str, Any]:
        return app.state.personal_skills.set_manual_review(request.state.auth.user.profile_id, body.manual_review)

    @app.post("/api/skills/candidates/{candidate_id}/decision")
    def personal_skill_decision(candidate_id: str, body: SkillDecisionInput, request: Request) -> dict[str, Any]:
        try:
            record = app.state.personal_skills.service(request.state.auth.user.profile_id).store.decide(
                candidate_id, body.decision, request.state.auth.user.username)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return _public_skill_review(record)

    @app.post("/api/skills/rollback")
    def personal_skill_rollback(request: Request) -> dict[str, Any]:
        result = app.state.personal_skills.rollback(request.state.auth.user.profile_id, request.state.auth.user.username)
        if result["status"] not in {"rolled_back", "already_rolled_back"}:
            raise HTTPException(409, f"Skill 回滚不可用：{result['status']}")
        return result

    @app.post("/api/skills/candidates/{candidate_id}/share")
    def share_skill(candidate_id: str, request: Request) -> dict[str, Any]:
        try:
            return app.state.personal_skills.share(request.state.auth.user.profile_id, candidate_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/admin/skills/proposals/{proposal_id}/decision")
    def shared_skill_decision(proposal_id: str, body: SkillDecisionInput, request: Request) -> dict[str, Any]:
        session = require_admin(request)
        try:
            return app.state.personal_skills.decide_proposal(proposal_id, body.decision, session.user.username)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/admin/skills/personal/{owner_ref}/traces/{trace_id}")
    def personal_skill_trace(owner_ref: str, trace_id: str, request: Request) -> dict[str, Any]:
        require_admin(request)
        if not re.fullmatch(r"[0-9a-f]{24}", owner_ref) or not re.fullmatch(r"[0-9a-f]{20}", trace_id):
            raise HTTPException(404, "Trace 不存在。")
        path = app.state.personal_skills.root / owner_ref / "evolution" / "traces" / f"{trace_id}.json"
        if not path.is_file():
            raise HTTPException(404, "Trace 不存在。")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/admin/skills/candidates")
    def skill_candidates(request: Request) -> dict[str, Any]:
        require_admin(request)
        return {
            "active_skill": ACTIVE_SKILL_NAME,
            "candidates": [
                _public_skill_review(item) for item in app.state.skill_evolution.store.list()
            ],
            "personal_candidates": [_public_skill_review(item) for item in app.state.personal_skills.candidates()],
            "shared_proposals": app.state.personal_skills.list_proposals(),
        }

    @app.post("/api/admin/skills/candidates/{candidate_id}/decision")
    def skill_candidate_decision(
        candidate_id: str,
        body: SkillDecisionInput,
        request: Request,
    ) -> dict[str, Any]:
        session = require_admin(request)
        try:
            record = app.state.skill_evolution.store.decide(
                candidate_id,
                body.decision,
                session.user.username,
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return _public_skill_review(record)

    @app.get("/api/admin/skills/traces/{trace_id}")
    def skill_source_trace(trace_id: str, request: Request) -> dict[str, Any]:
        require_admin(request)
        if not re.fullmatch(r"[0-9a-f]{20}", trace_id):
            raise HTTPException(404, "Trace 不存在。")
        path = app.state.skill_evolution_root / "traces" / f"{trace_id}.json"
        if not path.is_file():
            raise HTTPException(404, "Trace 不存在。")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/api/admin/skills/rollback")
    def skill_rollback(request: Request) -> dict[str, Any]:
        session = require_admin(request)
        result = app.state.skill_evolution.store.rollback(session.user.username)
        if result["status"] not in {"rolled_back", "already_rolled_back"}:
            raise HTTPException(409, f"Skill 回滚不可用：{result['status']}")
        return result

    def reload_web_knowledge() -> None:
        from healthvault.tools.catalog import build_healthvault_tool_registry
        from healthvault.tools.directory import HARNESS_READ_ONLY_TOOL_NAMES

        app.state.controller.reload_knowledge()
        app.state.read_only_tools = build_healthvault_tool_registry(
            app.state.controller.record_repository,
            app.state.controller.knowledge_repository,
            allowed_names=HARNESS_READ_ONLY_TOOL_NAMES,
        )

    @app.get("/api/admin/knowledge/candidates")
    def knowledge_candidates(request: Request) -> dict[str, Any]:
        require_admin(request)
        from healthvault.rag.releases import _snapshots
        return {"active": app.state.knowledge_releases.active(), "candidates": app.state.knowledge_releases.list(), "source_observations": app.state.source_watcher.list(),
                'sources':[{'source_id':source['source_id'],'title':source['title'],'publisher':source['publisher']} for source in _snapshots(app.state.knowledge_releases.active_root).values()]}

    @app.get('/api/admin/knowledge/update-template')
    def knowledge_update_template(source_id: str, request: Request) -> dict[str, Any]:
        require_admin(request)
        try: return app.state.knowledge_releases.update_template(source_id)
        except ValueError as exc: raise HTTPException(404,str(exc)) from exc

    @app.post("/api/admin/knowledge/check-sources")
    def check_knowledge_sources(request: Request) -> dict[str, Any]:
        require_admin(request)
        return app.state.source_watcher.check()

    @app.get("/api/admin/knowledge/sources/{identifier}/pending")
    def pending_knowledge_source(identifier: str, request: Request) -> dict[str, Any]:
        require_admin(request)
        try:
            return app.state.source_watcher.pending(identifier)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise HTTPException(404, "来源没有可查看的更新快照。") from exc

    @app.post("/api/admin/knowledge/candidates")
    def stage_knowledge_candidate(body: dict[str, Any], request: Request) -> dict[str, Any]:
        session = require_admin(request)
        try:
            return app.state.knowledge_releases.stage(body, session.user.username)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise HTTPException(422, f"知识更新未通过检查：{exc}") from exc

    @app.post("/api/admin/knowledge/candidates/{candidate_id}/decision")
    def knowledge_candidate_decision(candidate_id: str, body: SkillDecisionInput, request: Request) -> dict[str, Any]:
        session = require_admin(request)
        try:
            with app.state.knowledge_release_lock:
                record = app.state.knowledge_releases.decide(candidate_id, body.decision, session.user.username)
                if record["status"] == "published":
                    reload_web_knowledge()
                return record
        except FileNotFoundError as exc:
            raise HTTPException(404, "知识候选不存在。") from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/admin/knowledge/candidates/{candidate_id}/vectors")
    def build_candidate_vectors(candidate_id: str, request: Request) -> dict[str, Any]:
        session = require_admin(request)
        try:
            return app.state.knowledge_releases.build_vectors(candidate_id, session.user.username)
        except FileNotFoundError as exc:
            raise HTTPException(404, "知识候选不存在。") from exc
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(409, f"本地向量索引构建失败：{exc}") from exc

    @app.post("/api/admin/knowledge/rollback")
    def rollback_knowledge(request: Request) -> dict[str, Any]:
        session = require_admin(request)
        try:
            with app.state.knowledge_release_lock:
                result = app.state.knowledge_releases.rollback(session.user.username)
                reload_web_knowledge()
                return result
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/profiles/{profile_id}/agent/tasks/{task_id}/compact")
    async def agent_task_compact(
        profile_id: str, task_id: str, body: AgentVersionInput,
    ) -> dict[str, Any]:
        _valid_profile(profile_id)
        if not body.model_consent:
            raise HTTPException(409, "整理上下文会使用模型服务，请先确认。")
        async with task_lock(profile_id, task_id):
            try:
                saved = app.state.agent_tasks.get(profile_id, task_id)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if saved is None:
                raise HTTPException(404, "任务尚未创建。")
            if saved["version"] != body.expected_version:
                raise HTTPException(409, "任务已更新，请刷新后继续。")
            runtime = build_web_agent(profile_id, task_id)
            from agents.ui import quiet_agent_ui
            try:
                with quiet_agent_ui():
                    restore_web_agent(runtime, saved)
                    await runtime.agent.compact()
            except Exception as exc:
                raise HTTPException(502, "会话折叠暂不可用，请稍后重试。") from exc
            messages, folded = snapshot_web_agent(runtime)
            updated = app.state.agent_tasks.save(
                profile_id, task_id,
                expected_version=body.expected_version,
                messages=messages,
                folded=folded,
                dialogue=saved["dialogue"],
            )
            return public_task(updated, task_id, task_checkpoint(profile_id, task_id))

    @app.get("/api/profiles/{profile_id}/timeline")
    def timeline(profile_id: str) -> dict[str, Any]:
        return app.state.controller.timeline(_valid_profile(profile_id))

    @app.get("/api/profiles/{profile_id}/reports/{report_id}")
    def report_detail(profile_id: str, report_id: str) -> dict[str, object]:
        detail = app.state.controller.record_repository.get_report_detail(_valid_profile(profile_id), report_id)
        if detail is None:
            raise HTTPException(404, "报告不存在。")
        detail["original_text"] = app.state.draft_store.confirmed_original(
            profile_id, str(detail["source_id"]), str(detail["report_text"]),
        )
        return detail

    @app.get("/api/profiles/{profile_id}/reports/{report_id}/original")
    def report_original(profile_id: str, report_id: str) -> Response:
        asset = app.state.controller.record_repository.get_report_asset(_valid_profile(profile_id), report_id)
        if asset is None:
            raise HTTPException(404, "原始文件不存在。")
        content, media_type = asset
        return Response(content, media_type=media_type, headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
        })

    @app.get('/api/profiles/{profile_id}/reports/{report_id}/pages/{page_number}')
    def confirmed_report_page(profile_id: str, report_id: str, page_number: int) -> Response:
        asset = app.state.controller.record_repository.get_report_asset(_valid_profile(profile_id), report_id)
        if asset is None or asset[1] != 'application/pdf':
            raise HTTPException(404, 'PDF 页面不存在。')
        import pymupdf
        with pymupdf.open(stream=asset[0], filetype='pdf') as document:
            if not 1 <= page_number <= len(document):
                raise HTTPException(404, 'PDF 页面不存在。')
            page = document[page_number - 1]
            page.set_rotation(0)
            image = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            return Response(image.tobytes('png'), media_type='image/png', headers={'Cache-Control': 'no-store'})

    @app.get("/api/profiles/{profile_id}/comparisons")
    def comparisons(profile_id: str) -> dict[str, Any]:
        return app.state.controller.comparisons(_valid_profile(profile_id))

    @app.get("/api/knowledge")
    def knowledge(query: str) -> dict[str, Any]:
        return app.state.controller.search_knowledge(query)

    @app.get("/api/indicators")
    def indicators() -> dict[str, Any]:
        return {
            "count": len(INDICATOR_CATALOG),
            "indicators": [
                {"code": code, "display_name": spec.display_name, "unit": spec.unit}
                for code, spec in INDICATOR_CATALOG.items()
            ],
        }

    @app.get("/api/profiles/{profile_id}/visit-brief")
    def visit_brief(profile_id: str) -> dict[str, Any]:
        return app.state.controller.visit_brief(_valid_profile(profile_id))

    def search_knowledge_tool(query: str, limit: int = 5) -> dict[str, Any]:
        registry = app.state.read_only_tools
        if registry is None:
            return app.state.controller.search_knowledge(query, limit=limit)
        return json.loads(asyncio.run(registry.execute("search_health_knowledge", {"query": query, "limit": limit})))

    def search_official_tool(question: str, *, local_answer_insufficient: bool = False) -> dict[str, Any]:
        registry = app.state.read_only_tools
        if registry is None:
            return {"status": "unavailable", "results": []}
        return json.loads(asyncio.run(registry.execute("search_official_health_web", {
            "question": question, "limit": 3, "local_answer_insufficient": local_answer_insufficient,
        })))

    def visit_brief_tool(profile_id: str) -> dict[str, Any]:
        registry = app.state.read_only_tools
        if registry is None:
            return app.state.controller.visit_brief(profile_id)
        return json.loads(asyncio.run(registry.execute("generate_visit_brief", {"profile_id": profile_id})))

    def answer(profile_id: str, question: str, mode: str, evidence: dict[str, Any], sources: list[dict[str, Any]],
               *, answer_state: dict[str, Any] | None = None) -> dict[str, Any]:
        from healthvault.web.question_plan import create_question_plan
        from healthvault.web.answers import recover_missing_knowledge, answer_repair_feedback, AnswerGroundingError
        state = answer_state if answer_state is not None else {}
        state.setdefault('model_attempts', 0)
        state.setdefault('recovery_used', False)
        if state['model_attempts'] >= 2:
            return {'status': 'insufficient_evidence', 'question': question,
                    'answer': '当前证据不足以回答这个具体问题。', 'sources': []}
        plan = state.get('plan')
        if plan is None:
            try:
                plan = create_question_plan(question, resolved_question=evidence.get('question_context', {}).get('resolved_question'),
                                            query=app.state.model_question_planner if app.state.auto_plan_suggestions else None,
                                            use_model=app.state.auto_plan_suggestions and app.state.model_answerer is None,
                                            profile_id=profile_id)
            except ModelServiceError as exc:
                raise HTTPException(exc.status_code, str(exc)) from exc
            state['plan'] = plan
        evidence = {**evidence, 'question_plan': plan}
        measurement = {'entrypoint': 'knowledge_qa' if mode == 'knowledge' else 'visit_qa',
                       'personal_skill_injected': False, 'skill_versions': [],
                       'features': {'native_tools': False, 'model_plan_suggestions': app.state.auto_plan_suggestions,
                                    'answer_repair': app.state.answer_repair_enabled,
                                    'answer_verification': app.state.model_answerer is None or app.state.model_answer_verifier is not None},
                       'usage_scope': 'outer_ledger_required_for_planning_generation_verification'}
        recovery = state.setdefault('retrieval_recovery', [])
        repairs = state.setdefault('answer_repairs', [])
        for _attempt in range(2 - state['model_attempts']):
            state['model_attempts'] += 1
            measurement['generation_attempts'] = state['model_attempts']
            try:
                generated = (app.state.model_answerer(question, mode, evidence) if app.state.model_answerer is not None
                             else answer_with_model(question, mode, evidence, profile_id=profile_id))
            except AnswerGroundingError as exc:
                if app.state.answer_repair_enabled and state['model_attempts'] < 2:
                    evidence = {**evidence, 'answer_repair': exc.feedback}
                    repairs.append({'attempt': state['model_attempts'], 'reason': exc.feedback['reason']})
                    continue
                raise HTTPException(exc.status_code, str(exc)) from exc
            except ModelServiceError as exc:
                raise HTTPException(exc.status_code,str(exc)) from exc
            if generated.strip() == INSUFFICIENT_EVIDENCE_MARKER:
                return {"status": "insufficient_evidence", "question": question,
                        "answer": "当前证据不足以回答这个具体问题。", "sources": [], 'question_plan': plan, 'measurement': measurement}
            allowed = {source["citation_id"] for source in sources}
            cited = re.findall(r"\[((?:HVK|HVW|REC):[^\]]+)\]", generated)
            if not cited or any(citation_id not in allowed for citation_id in cited):
                if app.state.answer_repair_enabled and state['model_attempts'] < 2:
                    feedback = answer_repair_feedback(generated, {'kind': 'invalid_citations',
                        'invalid_citation_ids': sorted(set(cited) - allowed)})
                    evidence = {**evidence, 'answer_repair': feedback}
                    repairs.append({'attempt': state['model_attempts'], 'reason': feedback['reason']})
                    continue
                raise HTTPException(502, "模型回答缺少可核对的来源，请重试。")
            used_sources = [source for source in sources if source["citation_id"] in cited]
            verification = None
            if app.state.model_answerer is None or app.state.model_answer_verifier is not None:
                from healthvault.web.answer_validation import verify_answer
                try:
                    verification = verify_answer(question, generated, sources, query=app.state.model_answer_verifier,
                                                 profile_id=profile_id, resolved_question=plan['resolved_question'],
                                                 audit_coverage=True, required_points=plan['points'])
                    if (not verification['passed'] and mode == 'knowledge' and state['model_attempts'] < 2
                            and not state['recovery_used']
                            and verification.get('missing_evidence_points')):
                        state['recovery_used'] = True
                        expanded, recovery = recover_missing_knowledge(search_knowledge_tool, plan['resolved_question'],
                                                                       plan, verification, sources)
                        state['retrieval_recovery'] = recovery
                        if len(expanded) > len(sources):
                            sources = expanded
                            evidence = {**evidence, 'sources': sources, 'retrieval_recovery': recovery,
                                        'answer_repair': {'reason': 'new_evidence', 'previous_answer': generated,
                                                         'missing_evidence_points': verification['missing_evidence_points']}}
                            repairs.append({'attempt': state['model_attempts'], 'reason': 'new_evidence'})
                            continue
                    if not verification['passed'] and app.state.answer_repair_enabled and state['model_attempts'] < 2:
                        feedback = answer_repair_feedback(generated, verification)
                        if feedback is not None:
                            evidence = {**evidence, 'answer_repair': feedback}
                            repairs.append({'attempt': state['model_attempts'], 'reason': feedback['reason']})
                            continue
                except ModelServiceError as exc:
                    raise HTTPException(exc.status_code, str(exc)) from exc
                verification['retrieval_recovery'] = recovery
                verification['answer_repairs'] = list(repairs)
                if not verification['passed']:
                    if verification.get('kind') == 'answer_budget_exceeded':
                        raise HTTPException(502, verification['message'])
                    if verification.get('support_passed') and verification.get('omitted_points'):
                        raise HTTPException(502, '已找到相关资料，但回答遗漏了关键要点，请重试。')
                    if verification.get('kind') == 'invalid_verifier_output':
                        raise HTTPException(502, '回答复核结构无法核验，请重试。')
                    if not verification.get('support_passed') or not verification.get('coverage_contract_passed'):
                        raise HTTPException(502, '回答包含未能核验的结论，请重试。')
                    return {'status':'insufficient_evidence','question':question,'answer':'现有证据无法支持对这个具体问题的回答，请补充相关报告或咨询医生。','sources':[], 'verification': verification, 'question_plan': plan, 'measurement': measurement}
                generated = verification['answer']
            break
        app.state.answer_history.save(profile_id, mode, question, generated, used_sources,
                                      resolved_question=evidence.get('question_context', {}).get('resolved_question', question))
        return {"status": "answered", "question": question,
                "resolved_question": evidence.get('question_context', {}).get('resolved_question', question),
                "answer": generated, "sources": used_sources, 'question_plan': plan,
                'point_evidence': evidence.get('draft_grounding', []),
                'answer_repairs': list(repairs),
                'measurement': measurement,
                **({'verification':verification} if verification else {})}

    @app.get("/api/profiles/{profile_id}/answers/history")
    def answer_history(profile_id: str) -> dict[str, Any]:
        return {"answers": app.state.answer_history.list(_valid_profile(profile_id))}

    @app.post("/api/answers/knowledge")
    def answer_knowledge(body: AnswerInput, response: Response, request: Request) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        if not body.model_consent:
            raise HTTPException(409, "请先同意将本次问题及检索证据发送至模型服务。")
        question = body.question.strip()
        if len(question) < 3:
            raise HTTPException(422, "请输入至少 3 个字的问题。")
        from healthvault.application.query_rewrite import prepare_query
        from healthvault.web.model_runtime import run_model_task
        prepared = prepare_query(question, context_question=body.context_question,
                                 model_query=(app.state.model_query_rewriter or run_model_task) if body.allow_query_rewrite else None,
                                 profile_id=request.state.auth.user.profile_id)
        if prepared.status == 'needs_clarification':
            return {'status':'insufficient_evidence','question':question,'answer':'请明确你说的是哪个指标或检查；上一问题包含多个指标时，我不会替你选择。','sources':[]}
        evidence, sources = knowledge_evidence(retrieve_question_knowledge(search_knowledge_tool, prepared.resolved_question, search_query=prepared.search_query))
        if prepared.resolved_question != question:
            evidence['question_context'] = {'original_question':question,'resolved_question':prepared.resolved_question}
        had_local_sources = bool(sources)
        answer_state: dict[str, Any] = {}
        if had_local_sources:
            local_answer = answer(request.state.auth.user.profile_id, question, "knowledge", evidence, sources,
                                  answer_state=answer_state)
            if local_answer["status"] == "answered":
                return local_answer
            if not body.allow_web_search or answer_state['model_attempts'] >= 2:
                return local_answer
            evidence, sources = official_web_evidence(search_official_tool(prepared.resolved_question, local_answer_insufficient=True))
        if not had_local_sources and body.allow_web_search:
            evidence, sources = official_web_evidence(search_official_tool(prepared.resolved_question))
        if prepared.resolved_question != question:
            evidence['question_context'] = {'original_question':question,'resolved_question':prepared.resolved_question}
        if not sources:
            return {"status": "insufficient_evidence", "question": question, "answer": "当前可信资料中没有足够依据回答这个问题，请换一种问法或咨询医生。", "sources": []}
        return answer(request.state.auth.user.profile_id, question, "knowledge", evidence, sources,
                      answer_state=answer_state)

    @app.post("/api/profiles/{profile_id}/answers/visit")
    def answer_visit(profile_id: str, body: AnswerInput, response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        if not body.model_consent:
            raise HTTPException(409, "请先同意将本次问题及必要记录证据发送至模型服务。")
        profile_id = _valid_profile(profile_id)
        question = body.question.strip()
        if len(question) < 3:
            raise HTTPException(422, "请输入至少 3 个字的问题。")
        intent = assess_health_knowledge_support(question)
        if intent.reason_code in {
            "unsupported_diagnosis", "unsupported_treatment", "unsupported_future_prediction",
        }:
            return {
                "status": "insufficient_evidence", "question": question,
                "answer": "无法根据现有报告给出诊断、用药方案或精确预测。请把这个问题带给医生讨论。",
                "sources": [],
            }
        from healthvault.application.query_rewrite import prepare_query
        from healthvault.web.model_runtime import run_model_task
        prepared = prepare_query(question, context_question=body.context_question,
                                 model_query=(app.state.model_query_rewriter or run_model_task) if body.allow_query_rewrite else None,
                                 profile_id=profile_id, knowledge_only=False)
        if prepared.status == 'needs_clarification':
            return {'status':'insufficient_evidence','question':question,'answer':'请明确你说的是哪个指标或检查，再根据本人的已确认记录回答。','sources':[]}
        evidence, sources = visit_evidence(visit_brief_tool(profile_id))
        if prepared.resolved_question != question:
            evidence['question_context'] = {'original_question':question,'resolved_question':prepared.resolved_question}
        if not any(source["kind"] == "record" for source in sources):
            return {"status": "insufficient_evidence", "question": question, "answer": "当前档案没有可用的已确认记录，请先核对并保存报告。", "sources": []}
        return answer(profile_id, question, "visit", evidence, sources)

    return app


app = create_app()
