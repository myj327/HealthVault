#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Callable, Awaitable, Any

import anthropic
import openai

from agents.mcp_client import McpManager
from agents.memory import MemoryPrefetch, start_memory_prefetch, format_memories_for_injection
from agents.prompt import build_system_prompt
from agents.runtime_policy import AgentRuntimePolicy
from agents.session_memory import (
    FOLD_SESSION_MEMORY_SYSTEM,
    build_anthropic_transcript,
    build_folding_user_prompt,
    build_openai_transcript,
    fallback_folded_memory,
    format_folded_memory,
    parse_folded_memory,
)
from agents.session import save_folded_session_memory, save_session
from agents.tools import ToolDef, tool_definitions, execute_tool, CONCURRENCY_SAFE_TOOLS, check_permission, \
    get_active_tool_definitions
from agents.ui import print_info, print_divider, print_assistant_text, \
    start_spinner, stop_spinner, print_cost, print_tool_call, print_tool_result, print_confirmation, print_retry, \
    print_error


ToolExecutor = Callable[[str, dict, dict[str, float]], Awaitable[str]]


# 指数退避重试


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (429, 503, 529):
        return True
    msg = str(error)
    if "overloaded" in msg or "ECONNRESET" in msg or "ETIMEDOUT" in msg:
        return True
    return False


def _safe_utf8_text(value: object) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_for_utf8(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_utf8_text(value)
    if isinstance(value, list):
        return [_sanitize_for_utf8(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_utf8(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_for_utf8(key): _sanitize_for_utf8(item)
            for key, item in value.items()
        }
    return value


async def _with_retry(fn, max_retries: int = 3):
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            delay = min(1000 * (2 ** attempt), 30000) / 1000 + (hash(str(time.time())) % 1000) / 1000
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            reason = f"HTTP {status}" if status else (getattr(error, "code", None) or "network error")
            print_retry(attempt + 1, max_retries, reason)
            await asyncio.sleep(delay)

MODEL_CONTEXT = {
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-opus-4-20250514": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "deepseek-chat":200000
}

def _get_context_windows(model:str)->int:
    return MODEL_CONTEXT.get(model, 200000)


#多层级压缩常数
SNIP_THRESHOLD = 0.60
AUTO_COMPACT_THRESHOLD = 0.70
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
MICROCOMPACT_IDLE_S = 5 * 60  # 5 minutes

KEEP_RECENT_RESULTS = 3


def _get_max_output_tokens(model: str) -> int:
    m = model.lower()
    if "opus-4-6" in m:
        return 64000
    if "sonnet-4-6" in m:
        return 32000
    if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
        return 32000
    return 16384

#转换tool的形式到openai
def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


class Agent:
    def __init__(self,
                 *,
                 permission_mode:str="default",
                 model:str="deepseek-chat",
                 api_base: str | None=None,
                 anthropic_base_url: str | None=None,
                 api_key: str | None=None,
                 thinking: bool=False,
                 reasoning_effort: str | None=None,
                 max_cost_usd: float | None=None,
                 max_turns: int | None=None,
                 confirm_fn:Callable[[str], Awaitable[bool]] | None=None,
                 custom_system_prompt: str | None=None,
                 custom_tools: list[ToolDef] | None=None,
                 tool_executor: ToolExecutor | None=None,
                 completion_check: Callable[[str], str | None] | None=None,
                 runtime_policy: AgentRuntimePolicy | None=None,
                 is_sub_agent: bool=False,):
        self.permission_mode = permission_mode
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.model = model
        self.use_openai = bool(api_base)
        self.is_sub_agent = is_sub_agent
        self.tools = custom_tools if custom_tools is not None else tool_definitions
        self._tool_executor = tool_executor if tool_executor is not None else execute_tool
        self.completion_check = completion_check
        self._completion_rejections = 0
        self.runtime_policy = runtime_policy or AgentRuntimePolicy()
        self.max_cost_usd = max_cost_usd
        self.max_turns = max_turns
        self.confirm_fn = confirm_fn
        self._custom_system_prompt = custom_system_prompt
        self.effective_window=_get_context_windows(model) -20000
        self.session_id = uuid.uuid4().hex[:8]
        self.session_start_time= time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_token_count = 0
        self.current_turns = 0
        self.last_api_call_time = 0


        self._aborted = False
        #存储异步任务
        self._current_task:asyncio.Task | None = None
        #权限白名单
        self._confirmed_paths: set[str] = set()


        self._context_cleared : bool=False

        #思考模式
        self._thinking_mode = self._resolve_thinking_mode()

        #单次执行的输出缓存
        self._output_buffer: list[str] | None=None
        self._turn_output_buffer: list[str] | None = None

        # Optional state shared with application-injected tool executors.
        self._read_file_state: dict[str, float] ={}

        #MCP集成
        self._mcp_manager = McpManager()
        self._mcp_initialized = False

        #记忆回溯
        #记忆agent已经回答过的信息
        self._already_surfaced_memories: set[str] = set()
        #当前会话占用的字节数
        self._session_memory_bytes = 0

        #区分message的历史消息
        self._anthropic_messages: list[str] = []
        self._openai_messages: list[str] = []
        self._last_retrieved_skill_reference: dict[str, Any] | None = None
        self._last_retrieved_skill_hits: list[dict[str, Any]] = []
        self._pending_skill_extraction_window: dict[str, Any] | None = None
        self._background_skill_tasks: set[asyncio.Task] = set()
        self._folded_session_memories: list[dict[str, Any]] = []
        self._fold_last_time: float = 0.0
        self._fold_count: int = 0
        self._tool_error_streak: int = 0
        self._same_tool_repeat_count: int = 0
        self._last_tool_name: str = ""

        #构建系统提示词
        self._base_system_prompt = custom_system_prompt or build_system_prompt()

        self._system_prompt = self._base_system_prompt

        #初始化大模型客户端
        if self.use_openai:
            self._openai_client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
            self._anthropic_client = None
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        else:
            kwargs : dict[str,Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if anthropic_base_url:
                kwargs["base_url"] = anthropic_base_url
            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)
            self._openai_client = None

        self._refresh_runtime_system_prompt()

    #判断返回模型的思考模式
    def _resolve_thinking_mode(self) -> str:
        if not self.thinking:
            return "disabled"
        if not self._model_supports_thinking():
            return "disabled"

        if self._mode_supports_adaptive_thinking(self.model):
            return "adaptive"
        return "enabled"

    def _model_supports_thinking(self) -> bool:
        m = self.model.lower()
        if "claude-3-" in m or "3-5-" in m or "3-7-" in m:
            return False
        if "claude" in m and any(x in m for x in ("opus", "sonnet", "haiku")):
            return True
        return False
    def _model_supports_adaptive_thinking(self) -> bool:
        m = self.model.lower()
        return "opus-4-6" in m or "sonnet-4-6" in m


    #判断当前的任务所有的任务是否完成
    @property
    def is_processing(self)->bool:
        return self._current_task is not None and not self._current_task.done()

    #大模型调用的工厂方法,构建一个用于记忆召回（memory recall）的 sideQuery 可调用对象，兼容anthropic, openai。
    def _build_side_query(self, *, max_tokens: int = 256):
        if self._anthropic_client:
            client = self._anthropic_client
            model = self.model
            async def _sq(system:str, user_message:str)->str:

                resp = await client.messages.create(
                    model=model, max_tokens=max(1, int(max_tokens)), system=system,
                messages=[{"role": "user", "content": user_message}],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                if not text.strip():
                    block_types = [str(getattr(b, "type", "")) for b in getattr(resp, "content", [])]
                    logging.warning(
                        "side_query returned empty Anthropic-compatible response: model=%s stop_reason=%s content_block_types=%s",
                        getattr(resp, "model", model),
                        getattr(resp, "stop_reason", ""),
                        block_types,
                    )
                return text
            return _sq
        if self._openai_client:
            client = self._openai_client
            model = self.model
            async def _sq_openai(system:str, user_message:str)->str:
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=max(1, int(max_tokens)),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],

                )
                if not resp.choices:
                    logging.warning("side_query returned no OpenAI-compatible choices: model=%s", model)
                    return ""
                choice = resp.choices[0]
                content = choice.message.content or ""
                if not content.strip():
                    logging.warning(
                        "side_query returned empty OpenAI-compatible response: model=%s finish_reason=%s message=%s",
                        model,
                        getattr(choice, "finish_reason", ""),
                        choice.message,
                    )
                return content
            return _sq_openai
        return None
    #异步任务取消（Abort）
    def abort(self) -> None:
        self._aborted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def set_confirm_fn(self, fn:Callable[[str], Awaitable[bool]]) -> None:
        self.confirm_fn = fn


    def get_token_usage(self) -> dict:
        return {"input":self.total_input_tokens, "output":self.total_output_tokens}

    #主入口

    async def  chat(self, user_message:str)->None:
        #懒加载MCP服务在第一次chat的时候
        if (
            self.runtime_policy.mcp_enabled
            and not self._mcp_initialized
            and not self.is_sub_agent
        ):
            self._mcp_initialized = True
            try:
                await self._mcp_manager.load_and_connect()
                mcp_defs = self._mcp_manager.get_tool_definitions()
                if mcp_defs:
                    self.tools = self.tools + mcp_defs
            except Exception as e:
                print_error(f"MCP init failed: {e}")

        original_user_message = _safe_utf8_text(user_message)
        ready_skill_extraction_window: dict[str, Any] | None = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        if not self.is_sub_agent and self.runtime_policy.global_skills_enabled:
            ready_skill_extraction_window = self._pop_pending_skill_extraction_window(original_user_message)
            user_message, self._last_retrieved_skill_reference = self._augment_user_message_with_skill_context(
                original_user_message
            )

        self._aborted = False
        self._turn_output_buffer = []
        coro = self._chat_openai(user_message) if self.use_openai else self._chat_anthropic(user_message)
        self._current_task = asyncio.create_task(coro)
        try:
            await self._current_task
        except asyncio.CancelledError:
            self._aborted = True

        finally:
            self._current_task = None
        assistant_text = "".join(self._turn_output_buffer or []).strip()
        self._turn_output_buffer = None
        if (
            not self.is_sub_agent
            and not self._aborted
            and self.runtime_policy.global_skills_enabled
        ):
            self._schedule_background_skill_task(self._run_skill_usage_tracking(original_user_message, assistant_text))
            if ready_skill_extraction_window:
                self._schedule_background_skill_task(self._run_online_skill_evolution(ready_skill_extraction_window))
            self._set_pending_skill_extraction_window(
                original_user_message=original_user_message,
                assistant_text=assistant_text,
                retrieved_reference=self._last_retrieved_skill_reference,
            )
        if not self.is_sub_agent:
            print_divider()
            if self.runtime_policy.session_persistence_enabled:
                self._auto_save()


   #执行一次对话，收集本轮模型输出文本，并返回本轮消耗的 token 数
    async def run_once(self, prompt:str)->None:
        self._output_buffer = []
        prev_in = self.total_input_tokens
        prev_out = self.total_output_tokens
        try:
            await self.chat(prompt)
            text = "".join(self._output_buffer)
        finally:
            self._output_buffer = None
        return {
            "text": text,
            "tokens":{
                "input":self.total_input_tokens-prev_in,
                "output":self.total_output_tokens-prev_out
            },
        }

    #输出工具：统一处理模型输出文本。根据当前是否处于“收集输出”的模式
    # 决定是把文本存进缓冲区，还是直接打印到终端。
    def _emit_text(self, text:str)->None:
        text = _safe_utf8_text(text)
        if self._turn_output_buffer is not None:
            self._turn_output_buffer.append(text)
        if self._output_buffer is not None:
            self._output_buffer.append(text)
        else:
            print_assistant_text(text)

    def _build_fold_guidance_section(self) -> str:
        if self._custom_system_prompt is not None:
            return ""
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0.0
        last_fold = "never" if not self._fold_last_time else f"{int((time.time() - self._fold_last_time) / 60)}m ago"
        return (
            "\n\n# Runtime Fold Guidance\n"
            f"- Current context utilization: {utilization:.0%}\n"
            f"- Recent tool error streak: {self._tool_error_streak}\n"
            f"- Same tool repeat count: {self._same_tool_repeat_count}\n"
            f"- Last fold: {last_fold}\n"
            "- If the context is getting long, the same tool is being retried without progress, or tool failures are accumulating, call `compact_context` before trying more tools.\n"
            "- If you folded very recently and the next step is clear, prefer continuing rather than folding again.\n"
        )

    def _refresh_runtime_system_prompt(self) -> None:
        if self._custom_system_prompt is not None:
            return
        self._base_system_prompt = build_system_prompt()
        self._system_prompt = self._base_system_prompt
        self._system_prompt += self._build_fold_guidance_section()
        if self.use_openai and self._openai_messages:
            self._openai_messages[0]["content"] = self._system_prompt

    def _record_tool_outcome(self, tool_name: str, success: bool) -> None:
        if tool_name == self._last_tool_name:
            self._same_tool_repeat_count += 1
        else:
            self._same_tool_repeat_count = 1
        self._last_tool_name = tool_name
        if success:
            self._tool_error_streak = 0
        else:
            self._tool_error_streak += 1

    def _record_fold_event(self) -> None:
        self._fold_last_time = time.time()
        self._fold_count += 1
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""

    def _looks_like_tool_failure(self, tool_name: str, raw: str, result: str) -> bool:
        text = f"{raw}\n{result}".lower()
        if any(marker in text for marker in ("error", "denied", "timed out", "timeout")):
            return True
        if tool_name == "compact_context" and "no context compaction" in text:
            return True
        return False

    def _augment_user_message_with_skill_context(self, user_message: str) -> tuple[str, dict[str, Any] | None]:
        try:
            from .skills import format_retrieved_skill_context

            context, top_ref = format_retrieved_skill_context(user_message, limit=3)
        except Exception:
            return user_message, None
        if top_ref and isinstance(top_ref.get("all_hits"), list):
            self._last_retrieved_skill_hits = list(top_ref.get("all_hits") or [])
        if not context.strip():
            return user_message, top_ref
        return f"{user_message}\n\n{context}", top_ref

    def _strip_runtime_injections(self, text: str) -> str:
        return re.sub(r"\n*<retrieved_skills>.*?</retrieved_skills>\s*", "", str(text or ""), flags=re.DOTALL).strip()

    def _message_text(self, msg: dict[str, Any]) -> str:
        content = msg.get("content")
        if isinstance(content, str):
            return self._strip_runtime_injections(content)
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif "content" in block and block.get("type") not in {"tool_result", "tool_use"}:
                        parts.append(str(block.get("content") or ""))
            return self._strip_runtime_injections("\n".join(parts))
        return ""

    def _recent_dialog_messages(self, *, max_messages: int = 8) -> list[dict[str, str]]:
        raw_messages = self._openai_messages if self.use_openai else self._anthropic_messages
        out: list[dict[str, str]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = self._message_text(msg)
            if text:
                out.append({"role": role, "content": text})
        return out[-max(2, int(max_messages)) :]

    async def _confirm_online_skill_write(self, summary: str) -> bool:
        if self.permission_mode in {"bypassPermissions", "acceptEdits"}:
            return True
        if self.permission_mode in {"plan", "dontAsk"}:
            return False
        if self.confirm_fn is None:
            return False
        print_confirmation(summary)
        try:
            return bool(await self.confirm_fn(summary))
        except Exception:
            return False

    async def _confirm_background_online_skill_write(self, summary: str) -> bool:
        return self.permission_mode in {"bypassPermissions", "acceptEdits"}

    def _online_evolution_enabled(self) -> bool:
        if not self.runtime_policy.global_skills_enabled:
            return False
        raw = os.environ.get("BEAR_AUTO_SKILL_EVOLUTION", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _schedule_background_skill_task(self, coro) -> None:
        if self.permission_mode == "plan":
            try:
                coro.close()
            except Exception:
                pass
            return
        task = asyncio.create_task(coro)
        self._background_skill_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._background_skill_tasks.discard(done_task)
            try:
                done_task.result()
            except Exception:
                pass

        task.add_done_callback(_done)

    async def drain_background_skill_tasks(self) -> None:
        tasks = [task for task in self._background_skill_tasks if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _pop_pending_skill_extraction_window(self, next_user_feedback: str) -> dict[str, Any] | None:
        pending = self._pending_skill_extraction_window
        self._pending_skill_extraction_window = None
        if not pending:
            return None
        messages = list(pending.get("messages") or [])
        feedback = _safe_utf8_text(next_user_feedback).strip()
        if feedback:
            messages.append({"role": "user", "content": feedback})
        pending["messages"] = messages[-10:]
        pending["next_user_feedback"] = feedback
        return pending

    def _set_pending_skill_extraction_window(
        self,
        *,
        original_user_message: str,
        assistant_text: str,
        retrieved_reference: dict[str, Any] | None,
    ) -> None:
        if not original_user_message.strip() or not assistant_text.strip():
            return
        self._pending_skill_extraction_window = {
            "messages": self._recent_dialog_messages(max_messages=8),
            "latest_user": original_user_message,
            "latest_assistant": assistant_text,
            "retrieved_reference": self._compact_retrieved_reference(retrieved_reference),
            "session_id": self.session_id,
        }

    def _compact_retrieved_reference(self, ref: dict[str, Any] | None) -> dict[str, Any] | None:
        if not ref:
            return None
        return {k: v for k, v in ref.items() if k != "all_hits"}

    async def _run_online_skill_evolution(self, window: dict[str, Any], *, interactive_confirm: bool = False) -> None:
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return
        messages = list(window.get("messages") or [])
        if not messages:
            return

        side_query = self._build_side_query(max_tokens=2200)
        if side_query is None:
            return

        try:
            from .online_skill_evolution import online_ingest
        except Exception:
            return

        result = await online_ingest(
            messages=messages,
            side_query=side_query,
            retrieved_reference=window.get("retrieved_reference") or None,
            hint=str(window.get("hint") or ""),
            confirm_write=self._confirm_online_skill_write if interactive_confirm else self._confirm_background_online_skill_write,
            target=os.environ.get("BEAR_AUTO_SKILL_TARGET", "project"),
        )
        if result.get("ok"):
            if result.get("action") in {"add", "merge"}:
                self._refresh_runtime_system_prompt()
                print_info(f"Online skill {result.get('action')}: {result.get('skill')}")
        elif result.get("action") not in {"add_denied", "merge_denied"}:
            print_error(f"Online skill evolution failed: {result.get('error') or result}")

    async def _run_skill_usage_tracking(self, original_user_message: str, assistant_text: str) -> None:
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return
        hits = list(self._last_retrieved_skill_hits or [])
        if not hits or not assistant_text.strip():
            return
        side_query = self._build_side_query(max_tokens=700)
        try:
            from .online_skill_evolution import judge_retrieved_skill_usage
            from .skills import record_usage_judgments

            judgments = await judge_retrieved_skill_usage(
                hits=hits,
                user_message=original_user_message,
                assistant_text=assistant_text,
                side_query=side_query,
            )
            result = record_usage_judgments(judgments)
            if result.get("pruned"):
                self._refresh_runtime_system_prompt()
        except Exception:
            return

    async def extract_now(self, hint: str = "") -> dict[str, Any]:
        pending = self._pending_skill_extraction_window
        if not pending:
            return {"ok": False, "error": "no pending online skill extraction window"}
        window = dict(pending)
        window["hint"] = hint
        await self._run_online_skill_evolution(window, interactive_confirm=True)
        self._pending_skill_extraction_window = None
        return {"ok": True}


    def clear_history(self)->None:
        self._anthropic_messages = []
        self._openai_messages = []
        self._pending_skill_extraction_window = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        self._fold_last_time = 0.0
        self._fold_count = 0
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content":self._system_prompt})
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_token_count = 0
        print_info("Conversation cleared.")

    def show_cost(self):
        total = self._get_current_cost_usd()
        budget_info = f" / ${self.max_cost_usd} budget" if self.max_cost_usd else ""
        turn_info = f" | Turns: {self.current_turns}/{self.max_turns}" if self.max_turns else ""
        print_info(
            f"Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out\n  Estimated cost: ${total:.4f}{budget_info}{turn_info}")

    #获取当前的花费，
    def _get_current_cost_usd(self) -> float:
        return (self.total_input_tokens / 1_000_000) * 3 + (self.total_output_tokens / 1_000_000) * 15

    #检查预算
    def _check_budget(self) -> dict:
        if self.max_cost_usd is not None and self._get_current_cost_usd() >= self.max_cost_usd:
            return {"exceeded": True, "reason": f"Cost limit reached (${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})"}
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {"exceeded": True, "reason": f"Turn limit reached ({self.current_turns} >= {self.max_turns})"}
        return {"exceeded": False}

    #压缩会话
    async def compact(self)->None:
        compacted = await self._compact_conversation(trigger="manual")
        if not compacted:
            print_info("Nothing to compact yet.")


    #恢复会话信息
    def restore_session(self, data:dict)->None:
        if data.get("anthropicMessages"):
            self._anthropic_messages = self._normalize_anthropic_messages(_sanitize_for_utf8(data["anthropicMessages"]))
        if data.get("openaiMessages"):
            self._openai_messages = _sanitize_for_utf8(data["openaiMessages"])
        if isinstance(data.get("foldedSessionMemories"), list):
            self._folded_session_memories = _sanitize_for_utf8(data["foldedSessionMemories"])
        print_info(f"Session restored ({self._get_message_count()} messages).")


#整理 Anthropic 的历史消息，修正部分角色错误，并丢弃不合法的工具调用消息。
    def _normalize_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        role_normalized = []
        for msg in messages:
            copied = dict(msg)
            content = copied.get("content")
            if copied.get("role") == "user" and isinstance(content, list):
                if any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content):
                    copied["role"] = "assistant"
            role_normalized.append(copied)

        normalized = []
        i = 0
        while i < len(role_normalized):
            msg = role_normalized[i]
            tool_use_ids = self._anthropic_tool_use_ids(msg)
            if not tool_use_ids:
                normalized.append(msg)
                i += 1
                continue

            next_msg = role_normalized[i + 1] if i + 1 < len(role_normalized) else None
            result_ids = self._anthropic_tool_result_ids(next_msg) if next_msg else set()
            if tool_use_ids.issubset(result_ids):
                normalized.append(msg)
                normalized.append(next_msg)
                i += 2
                continue

            i += 1
        return normalized

    @staticmethod
    def _anthropic_tool_use_ids(msg: dict | None) -> set[str]:
        if not msg or msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
        }

    @staticmethod
    def _anthropic_tool_result_ids(msg: dict | None) -> set[str]:
        if not msg or msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("tool_use_id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id")
        }

    def _get_message_count(self) -> int:
        return len(self._openai_messages) if self.use_openai else len(self._anthropic_messages)

    def _auto_save(self) -> None:
        try:
            save_session(self.session_id, {
                "metadata": {
                    "id": self.session_id,
                    "model": self.model,
                    "cwd": str(Path.cwd()),
                    "startTime": self.session_start_time,
                    "messageCount": self._get_message_count(),
                },
                "anthropicMessages": _sanitize_for_utf8(self._anthropic_messages) if not self.use_openai else None,
                "openaiMessages": _sanitize_for_utf8(self._openai_messages) if self.use_openai else None,
                "foldedSessionMemories": _sanitize_for_utf8(self._folded_session_memories),
            })
        except Exception:
            pass

    #自动压缩
    async def _check_and_compact(self)->None:
        if self.last_input_token_count > self.effective_window * AUTO_COMPACT_THRESHOLD:
            print_info("Context window filling up, compacting conversation...")
            await self._compact_conversation(trigger="auto")

    async def _compact_conversation(self, *, trigger: str = "manual")->bool:
        if self.use_openai:
            compacted = await self._compact_openai(trigger=trigger)
        else:
            compacted = await self._compact_anthropic(trigger=trigger)
        if compacted:
            print_info("Conversation compacted.")
        return compacted

    async def _compact_anthropic(self, *, trigger: str)->bool:
        if len (self._anthropic_messages)<4:
            return False

        transcript = build_anthropic_transcript(_sanitize_for_utf8(self._anthropic_messages))
        if not transcript.strip():
            return False
        memory = await self._generate_folded_session_memory(transcript)
        self._record_folded_session_memory(trigger, memory)
        self._record_fold_event()
        self._anthropic_messages = [{"role": "user", "content": format_folded_memory(memory)}]
        self.last_input_token_count = 0
        self._refresh_runtime_system_prompt()
        return True

    async def _compact_openai(self, *, trigger: str)->bool:
        if len (self._openai_messages)<4:
            return False
        system_msg = self._openai_messages[0]
        transcript = build_openai_transcript(_sanitize_for_utf8(self._openai_messages))
        if not transcript.strip():
            return False
        memory = await self._generate_folded_session_memory(transcript)
        self._record_folded_session_memory(trigger, memory)
        self._record_fold_event()
        self._openai_messages=[
            system_msg,
            {"role": "user", "content": format_folded_memory(memory)},
        ]
        self.last_input_token_count=0
        self._refresh_runtime_system_prompt()
        return True

    async def _generate_folded_session_memory(self, transcript: str) -> dict[str, Any]:
        side_query = self._build_side_query(max_tokens=6000)
        if side_query is None:
            return fallback_folded_memory(transcript)
        try:
            raw = await side_query(FOLD_SESSION_MEMORY_SYSTEM, build_folding_user_prompt(transcript))
            return parse_folded_memory(raw)
        except Exception:
            return fallback_folded_memory(transcript)

    def _record_folded_session_memory(self, trigger: str, memory: dict[str, Any]) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trigger": trigger,
            "session_id": self.session_id,
            **memory,
        }
        self._folded_session_memories.append(record)
        if self.runtime_policy.session_persistence_enabled:
            try:
                save_folded_session_memory(
                    self.session_id,
                    _sanitize_for_utf8(record),
                )
            except Exception:
                pass

    #多层级压缩流水线
    def _run_compression_pipeline(self)->None:
        if self.use_openai:
            self._budget_tool_results_openai()
            self._snip_stale_results_openai()
            self._microcompact_openai()
        else:
            self._budget_tool_results_anthropic()
            self._snip_stale_results_anthropic()
            self._microcompact_anthropic()

    #第一层级压缩，预算压缩
    def _budget_tool_results_anthropic(self)->None:
        #计算利用率：utilization = 已用Token / 有效窗口大小。
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        #如果利用率低于 50%，说明空间还很充裕，直接返回，不做任何处理。
        if utilization < 0.5:
            return
        #动态预算（Budget）：危急状态（>70%）：如果利用率很高，允许单个工具结果保留 15,000 个字符。
        # 警戒状态（50%-70%）：如果利用率中等，只允许保留 30000 个字符。
        budget = 15000 if utilization > 0.7 else 30000

        for msg in self._anthropic_messages:

            #只处理 role 为 "user" 的消息。在工具调用流程中，工具的执行结果通常是以“用户”的身份反馈给模型的。

            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and len(block["content"]) > budget:
                    #计算保留长度 (keep)：keep = (budget - 80) // 2 这里预留了约 80 个字符的空间给中间的提示语，剩下的长度平分给开头和结尾。
                    keep = (budget - 80) // 2
                    #重组新内容 = 开头部分 + 提示语 + 结尾部分
                    block["content"] = block["content"][:keep] + f"\n\n[... budgeted: {len(block['content']) - keep * 2} chars truncated ...]\n\n" + block["content"][-keep:]

    def _budget_tool_results_openai(self)->None:
        #计算利用率：utilization = 已用Token / 有效窗口大小。
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        #如果利用率低于 50%，说明空间还很充裕，直接返回，不做任何处理。
        if utilization < 0.5:
            return
        #动态预算（Budget）：危急状态（>70%）：如果利用率很高，允许单个工具结果保留 15,000 个字符。
        # 警戒状态（50%-70%）：如果利用率中等，只允许保留 30000 个字符。
        budget = 15000 if utilization > 0.7 else 30000

        for msg in self._openai_messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > budget:
                keep = (budget - 80) // 2
                msg["content"] = msg["content"][:keep] + f"\n\n[... budgeted: {len(msg['content']) - keep * 2} chars truncated ...]\n\n" + msg["content"][-keep:]


    #第二级策略：修剪过期的工具执行结果
    def _snip_stale_results_anthropic(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < SNIP_THRESHOLD:
            return
        results = [block for message in self._anthropic_messages
                   if message.get("role") == "user" and isinstance(message.get("content"), list)
                   for block in message["content"]
                   if isinstance(block, dict) and block.get("type") == "tool_result"
                   and isinstance(block.get("content"), str) and block["content"] != SNIP_PLACEHOLDER]
        for block in results[:-KEEP_RECENT_RESULTS]:
            block["content"] = SNIP_PLACEHOLDER

    def _snip_stale_results_openai(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < SNIP_THRESHOLD:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] != SNIP_PLACEHOLDER:
                tool_msgs.append(i)
        if len(tool_msgs) <= KEEP_RECENT_RESULTS:
            return
        snip_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(snip_count):
            self._openai_messages[tool_msgs[i]]["content"] = SNIP_PLACEHOLDER

    #微压缩

    #基于“时间”的上下文瘦身策略，
    #如果已经很久没说话了，说明之前的工具执行结果你已经看完了，那就把它们清理掉，腾出空间

    def _microcompact_anthropic(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return

        all_results = []
        for mindex, msg in enumerate(self._anthropic_messages):
            if msg.get("role")!="user" or not isinstance(msg.get("content"), list):
                continue
            for bindex, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                    all_results.append((mindex, bindex))

        clear_count = len(all_results) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            mi, bi = all_results[i]
            self._anthropic_messages[mi]["content"][bi]["content"] = "[Old result cleared]"

    def _microcompact_openai(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                tool_msgs.append(i)
        clear_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            self._openai_messages[tool_msgs[i]]["content"] = "[Old result cleared]"




    #执行工具入口

    async def _execute_tool_call(self, name: str, inp: dict) -> str:
        allowed_tool_names = self.runtime_policy.allowed_tool_names
        if allowed_tool_names is not None and name not in allowed_tool_names:
            return f"Tool unavailable under Agent runtime policy: {name}"
        if name == "compact_context":
            return await self._execute_compact_context_tool(inp)
        if (
            self.runtime_policy.mcp_enabled
            and self._mcp_manager.is_mcp_tool(name)
        ):
            return await self._mcp_manager.call_tool(name, inp)
        if name not in {tool["name"] for tool in self.tools}:
            return f"Tool unavailable: {name}"
        return await self._tool_executor(name, inp, self._read_file_state)


    async def _execute_compact_context_tool(self, inp: dict) -> str:
        reason = str(inp.get("reason") or "").strip()
        compacted = await self._compact_conversation(trigger="tool")
        if not compacted:
            self._record_tool_outcome("compact_context", False)
            return "No context compaction was performed because there is not enough conversation history yet."
        self._record_tool_outcome("compact_context", True)
        self._context_cleared = True
        suffix = f"\nReason: {reason}" if reason else ""
        return (
            "Context compacted into structured session memory. "
            "Continue from the folded memory now present in the conversation context."
            f"{suffix}"
        )


#--------------Anthropic 后端---------------
    async def _buffer_checked_response(self, call):
        """Keep rejected final text out of output; retain the provider message."""
        if self.completion_check is None:
            return await call(), None
        emitted = []
        original_emit = self._emit_text
        self._emit_text = emitted.append
        try:
            return await call(), emitted
        finally:
            self._emit_text = original_emit

    def _accept_checked_completion(self, text, messages, emitted):
        if self.completion_check is None:
            return True
        feedback = self.completion_check(text)
        if feedback is None:
            self._emit_text("".join(emitted or []) or text)
            return True
        self.current_turns += 1
        self._completion_rejections += 1
        if self._completion_rejections > 1 or self._check_budget()["exceeded"]:
            raise RuntimeError("Agent completion conditions remain unsatisfied.")
        messages.append({"role": "user", "content": "Runtime completion check: " + feedback})
        return False

    async def  _chat_anthropic(self, user_message: str) -> None:
        self._completion_rejections = 0
        self._anthropic_messages = self._normalize_anthropic_messages(_sanitize_for_utf8(self._anthropic_messages))
        user_message = _safe_utf8_text(user_message)
        # 先把本轮用户输入放入 Anthropic 消息历史，后续每轮模型调用都会带上这段上下文。
        self._anthropic_messages.append({"role": "user", "content": user_message})

        # 异步内存预取：主 agent 才需要查 memory，sub agent 不额外注入记忆。
        # 这里只启动后台任务，不阻塞当前模型调用流程。
        memory_prefetch:MemoryPrefetch | None = None
        if not self.is_sub_agent and self.runtime_policy.memory_enabled:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )
        while True:
            # 外部请求中止时，结束整个 agent loop。
            if self._aborted:
                break

            # 每轮调用模型前尝试压缩上下文，避免消息历史过长。
            self._run_compression_pipeline()

            # 如果记忆预取任务已经完成，就把取回来的 memory 内容追加到最后一条用户消息里。
            # consumed 用来保证同一批 memory 只注入一次。
            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        injection_text = format_memories_for_injection(memories)
                        injection_text = _safe_utf8_text(injection_text)
                        last = self._anthropic_messages[-1] if self._anthropic_messages else None
                        if last and last.get("role") == "user":
                            content = last.get("content", "")
                            if isinstance(content, str):
                                # 字符串不可变，需要重新赋值回 message。
                                last["content"] = content + "\n\n" + injection_text
                            elif isinstance(content, list):
                                # list 是可变对象，append 会直接修改 last["content"] 指向的列表。
                                content.append({"type": "text", "text": injection_text})
                        else:
                            # 如果最后一条不是 user message，就单独追加一条用户消息承载 memory。
                            self._anthropic_messages.append({"role": "user", "content": injection_text})

                        for m in memories:
                            # 记录本 session 已经注入过的 memory，后续检索时可避免重复 surfaced。
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += m.size
                except:
                    # memory 注入失败不应该中断主对话流程。
                    pass

            if not self.is_sub_agent:
                start_spinner()


            # 保存“提前执行”的工具任务。key 是 Anthropic 返回的 tool_use block id。
            early_executions: dict[str, asyncio.Task] = {}


            def _on_tool_block(block:dict):
                # 流式响应中一旦完整收到 tool_use block，如果工具是并发安全且权限允许，
                # 就可以提前开始执行，减少等待完整模型响应后的空档时间。
                if block["name"] in CONCURRENCY_SAFE_TOOLS:
                    perm = check_permission(block["name"], block["input"], self.permission_mode)
                    if perm["action"]=="allow":
                        task =asyncio.create_task(self._execute_tool_call(block["name"], block["input"]))
                        early_executions[block["id"]] = task


            # 调用 Anthropic 流式接口；流式过程中完成 tool block 时会触发 _on_tool_block。
            response, emitted = await self._buffer_checked_response(
                lambda: self._call_anthropic_stream(on_tool_block_complete=_on_tool_block))
            if not self.is_sub_agent:
                stop_spinner()

            # 记录本次模型调用的耗时点和 token 消耗，用于成本展示与预算控制。
            self.last_api_call_time = time.time()
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.last_input_token_count = response.usage.input_tokens

            # Anthropic 的响应内容里可能混有 text block 和 tool_use block，这里只挑出工具调用。
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            # 把模型返回的所有 content block 写入消息历史，后续 tool_result 要与这些 tool_use 对应。
            self._anthropic_messages.append({
                "role": "assistant",
                "content": [self._block_to_dict(b) for b in response.content],
            })

            # 没有工具调用，说明模型已经给出最终回复，本轮对话结束。
            if not tool_uses:
                text = "".join(b.text for b in response.content if b.type == "text")
                if not self._accept_checked_completion(text, self._anthropic_messages, emitted):
                    continue
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens, self.total_output_tokens)
                break

            # 有工具调用时，进入下一轮工具执行。这里同时检查 turn/budget 限制。
            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                if self.completion_check is not None:
                    raise RuntimeError("Agent completion budget exhausted before tool execution.")
                print_info(f"Budget exceeded: {budget['reason']}")
                self._anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": f"Tool execution skipped: {budget['reason']}",
                        }
                        for tu in tool_uses
                    ],
                })
                break


            # 收集本轮所有工具结果，之后作为 tool_result 消息回传给模型。
            tool_results: list[dict] = []
            context_break = False

            for tu in tool_uses:
                # context_break 表示某个工具执行期间清理了上下文，需要停止继续处理本轮剩余工具。
                if context_break or self._aborted:
                    break

                # 将工具入参转为普通 dict，便于权限检查、打印和实际执行。
                inp = dict(tu.input) if hasattr(tu, "items") else tu.input
                print_tool_call(tu.name, inp)

                # 如果这个工具已经在流式阶段提前开始执行，这里只需要等待它完成并收集结果。
                early_task = early_executions.get(tu.id)
                if early_task:
                    try:
                        raw = await early_task
                    except Exception as e:
                        raw = f"Error executing tool: {e}"
                    raw = _safe_utf8_text(raw)
                    res = raw
                    print_tool_result(tu.name, res)
                    self._record_tool_outcome(tu.name, not self._looks_like_tool_failure(tu.name, raw, res))
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})
                    continue

                # 如果不是提前执行的工具，就在真正执行前做权限检查。

                perm = check_permission(tu.name, inp, self.permission_mode)
                if perm["action"] == "deny":
                    # 权限拒绝时，也要返回一个 tool_result，让模型知道该工具调用失败的原因。
                    print_info(f"Denied: {perm.get('message', '')}")
                    self._record_tool_outcome(tu.name, False)
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id,
                                         "content": f"Action denied: {perm.get('message', '')}"})
                    continue

                if perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                    # 高风险操作需要用户确认；同一个 message 确认过后会缓存，避免重复询问。
                    confirmed = await self._confirm_dangerous(perm["message"])
                    if not confirmed:
                        self._record_tool_outcome(tu.name, False)
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": tu.id, "content": "User denied this action."})
                        continue
                    self._confirmed_paths.add(perm["message"])

                # 权限通过后执行工具；结果留在会话中，由现有压缩流程控制长度。
                try:
                    raw = await self._execute_tool_call(tu.name, inp)
                except Exception as e:
                    raw = f"Error executing tool: {e}"
                raw = _safe_utf8_text(raw)
                res = raw
                print_tool_result(tu.name, res)
                self._record_tool_outcome(tu.name, not self._looks_like_tool_failure(tu.name, raw, res))

                if self._context_cleared:
                    # 工具执行过程中如果清理了上下文，就把结果作为新的用户消息写入，
                    # 并停止继续处理本轮剩余工具，避免旧上下文和新上下文混在一起。
                    self._context_cleared = False
                    self._anthropic_messages.append({"role": "user", "content": res})
                    context_break = True
                    break

                # Anthropic 要求 tool_result 使用 tool_use_id 对应到前面的 tool_use block。
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})

            if not context_break and tool_results:
                # Anthropic 要求 assistant/tool_use 后面紧跟一条 user/tool_result 消息，
                # 且这条消息必须包含本轮所有 tool_use 的对应结果。
                self._anthropic_messages.append({"role": "user", "content": tool_results})

            self._context_cleared = False

            # 工具结果可能很长，每轮工具执行后检查是否需要压缩上下文。
            self._refresh_runtime_system_prompt()
            await self._check_and_compact()

    @staticmethod
    def _block_to_dict(block) -> dict:
        if block.type == "text":
            return {"type": "text", "text": _safe_utf8_text(block.text)}
        if block.type == "tool_use":
            raw_input = dict(block.input) if hasattr(block.input, 'items') else block.input
            return {"type": "tool_use", "id": _safe_utf8_text(block.id), "name": _safe_utf8_text(block.name), "input": _sanitize_for_utf8(raw_input)}
        # Fallback
        return {"type": _safe_utf8_text(block.type)}

    async def _call_anthropic_stream(self, on_tool_block_complete=None):

        async def _do():
            max_output =  _get_max_output_tokens(self.model)

            create_params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_output if self._thinking_mode != "disabled" else 16384,
                "system": _safe_utf8_text(self._system_prompt),
                "tools": _sanitize_for_utf8(get_active_tool_definitions(self.tools)),
                "messages": _sanitize_for_utf8(self._anthropic_messages),
            }
            #如果开启了思考模式，就给 Anthropic 请求加上 thinking 参数。
            if self._thinking_mode  in ("adaptive", "enabled"):
                create_params["thinking"]={"type": "enabled", "budget_tokens": max_output - 1}

            first_text = True

            tool_blocks_by_index: dict[int, dict] = {}

            async with self._anthropic_client.messages.stream(**create_params)as stream:
                async for event in stream:
                    if not hasattr(event, 'type'):
                        continue
                    # 当事件是工具调用开始：
                    if event.type == "content_block_start":
                        cb = getattr(event, 'content_block', None)
                        #如果 block 类型是 tool_use，就记录这个工具调用：
                        if cb and getattr(cb, 'type', None) == "tool_use":
                            #因为工具参数 JSON 是流式分片返回的，所以先准备一个空字符串 input_json。
                            tool_blocks_by_index[event.index]= {
                                "id": cb.id, "name": cb.name, "input_json": "",
                            }
                    #当事件是内容增量，分三种情况。
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        # 第一种，普通文本：模型输出正文时，
                        # 调用 _emit_text()。如果是普通交互，就打印；
                        # 如果是 run_once()，就写入 _output_buffer。
                        if hasattr(delta, "text"):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n")
                                first_text = False
                            self._emit_text(delta.text)
                        #第二种，thinking 内容：
                        #如果模型返回思考内容，也输出出来，并在开头加：[thinking]
                        elif hasattr(delta, 'thinking'):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n  [thinking] ")
                                first_text = False
                            self._emit_text(delta.thinking)
                        #第三种，工具参数 JSON 片段：工具调用的参数不是一次性返回，
                        # 而是一段一段返回，所以这里不断拼接到 input_json。
                        elif hasattr(delta, 'partial_json'):
                            tb = tool_blocks_by_index.get(event.index)
                            if tb:
                                tb["input_json"] += _safe_utf8_text(delta.partial_json)
                    #当一个 content block 结束：
                    #如果结束的是之前记录的工具调用，就把拼好的 JSON 解析出来：
                    elif event.type == "content_block_stop":
                        tb = tool_blocks_by_index.pop(event.index, None)
                        if tb and on_tool_block_complete:
                            import json as _json
                            try:
                                parsed = _json.loads(tb["input_json"] or "{}")
                            except Exception:
                                parsed = {}
                            #然后调用回调：
                            #这个回调的作用通常是：工具调用一完整，
                            # 就可以提前开始执行工具，不必等整条 assistant 消息全部结束。
                            on_tool_block_complete({
                                "type": "tool_use", "id": _safe_utf8_text(tb["id"]),
                                "name": _safe_utf8_text(tb["name"]), "input": _sanitize_for_utf8(parsed),
                            })
                final_message = await stream.get_final_message()

            #过滤思考的message（因为 thinking 内容一般不应该进入历史消息，否则后续上下文会变大，也可能不符合 API 消息格式要求。）
            final_message.content = [b for b in final_message.content if b.type != "thinking"]
            return final_message
#调用 _do()，如果遇到可重试错误，就由 _with_retry() 负责重试。
        return await _with_retry(_do)

    #openAI后端

    async def _chat_openai(self, user_message:str) -> None:
        self._completion_rejections = 0
        user_message = _safe_utf8_text(user_message)
        self._openai_messages.append({"role": "user", "content": user_message})

        #预取句柄 MemoryPrefetch
        memory_prefetch: MemoryPrefetch | None = None
        if not self.is_sub_agent and self.runtime_policy.memory_enabled:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )

        while True:
            if self._aborted:
                break

            self._run_compression_pipeline()

            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        injection_text = format_memories_for_injection(memories)
                        injection_text = _safe_utf8_text(injection_text)
                        last = self._openai_messages[-1] if self._openai_messages else None

                        if last and last.get("role") == "user":
                            last["content"] = (last.get("content") or "") + "\n\n" + injection_text
                        else:
                            self._openai_messages.append({"role": "user", "content": injection_text})

                        for m in memories:
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += len(m.content.encode())
                except Exception:
                    pass

            if not self.is_sub_agent:
                start_spinner()

            response, emitted = await self._buffer_checked_response(self._call_openai_stream)

            if not self.is_sub_agent:
                stop_spinner()

            self.last_api_call_time = time.time()

            if response.get("usage"):
                self.total_input_tokens += response["usage"]["prompt_tokens"]
                self.total_output_tokens += response["usage"]["completion_tokens"]
                self.last_input_token_count = response["usage"]["prompt_tokens"]

            choice = response.get("choices", [{}])[0] if response.get("choices") else {}
            message = choice.get("message", {})

            self._openai_messages.append(message)

            tool_calls = message.get("tool_calls")

            if not tool_calls:
                if not self._accept_checked_completion(message.get("content") or "", self._openai_messages, emitted):
                    continue
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens, self.total_output_tokens)
                break

            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                for tc in tool_calls:
                    if tc.get("type") == "function":
                        self._openai_messages.append({"role": "tool", "tool_call_id": tc["id"],
                                                     "content": f"Tool call skipped: agent budget exhausted ({budget['reason']})."})
                if self.completion_check is not None:
                    raise RuntimeError("Agent completion budget exhausted before tool execution.")
                print_info(f"Budget exceeded: {budget['reason']}")
                break

            await self._execute_openai_tool_batch(message)
            self._refresh_runtime_system_prompt()
            await self._check_and_compact()

    async def _execute_openai_tool_batch(self, message: dict) -> None:
        tool_calls = [tc for tc in message["tool_calls"] if tc.get("type") == "function"]
        checked: list[dict] = []
        results: dict[str, str] = {}
        replied: set[str] = set()

        def append_results(calls: list[dict]) -> None:
            # Context-clearing tools remove the initiating assistant message as well.
            if not any(item is message for item in self._openai_messages):
                return
            for tc in calls:
                if tc["id"] in results and tc["id"] not in replied:
                    self._openai_messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": results[tc["id"]]})
                    replied.add(tc["id"])

        async def run_tool(ct: dict) -> None:
            if not ct["allowed"] or self._aborted or self._context_cleared:
                return
            if ct["permission_state"] != self.permission_mode:
                perm = check_permission(ct["fn"], ct["inp"], self.permission_mode)
                if perm["action"] == "deny" or (
                    perm["action"] == "confirm" and perm.get("message") not in self._confirmed_paths
                ):
                    results[ct["tc"]["id"]] = "Action denied: permissions changed; request this tool again for authorization."
                    self._record_tool_outcome(ct["fn"], False)
                    return
            try:
                raw = _safe_utf8_text(await self._execute_tool_call(ct["fn"], ct["inp"]))
                result = raw
            except asyncio.CancelledError:
                results[ct["tc"]["id"]] = "Tool execution interrupted before a result was available."
                self._record_tool_outcome(ct["fn"], False)
                raise
            except Exception as error:
                raw = result = _safe_utf8_text(f"Error executing tool {ct['fn']}: {error}")
            results[ct["tc"]["id"]] = result
            print_tool_result(ct["fn"], result)
            self._record_tool_outcome(ct["fn"], not self._looks_like_tool_failure(ct["fn"], raw, result))

        try:
            # Finish parsing and authorization before starting any tool in this batch.
            for tc in tool_calls:
                if self._aborted:
                    break
                fn_name = tc["function"]["name"]
                try:
                    inp = json.loads(tc["function"]["arguments"])
                except Exception:
                    inp = {}
                print_tool_call(fn_name, inp)
                perm = check_permission(fn_name, inp, self.permission_mode)
                denied_result = None
                if perm["action"] == "deny":
                    print_info(f"Denied: {perm.get('message', '')}")
                    denied_result = f"Action denied: {perm.get('message', '')}"
                elif perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                    if await self._confirm_dangerous(perm["message"]):
                        self._confirmed_paths.add(perm["message"])
                    else:
                        denied_result = "User denied this action."
                if denied_result is not None:
                    results[tc["id"]] = denied_result
                    self._record_tool_outcome(fn_name, False)
                checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": denied_result is None,
                                "permission_state": self.permission_mode})

            batches: list[dict] = []
            for ct in checked:
                safe = ct["allowed"] and ct["fn"] in CONCURRENCY_SAFE_TOOLS
                if safe and batches and batches[-1]["concurrent"]:
                    batches[-1]["items"].append(ct)
                else:
                    batches.append({"concurrent": safe, "items": [ct]})

            for batch in batches:
                if self._aborted or self._context_cleared:
                    break
                if batch["concurrent"]:
                    tasks = [asyncio.create_task(run_tool(ct)) for ct in batch["items"]]
                    try:
                        await asyncio.gather(*tasks)
                    except BaseException:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                else:
                    await run_tool(batch["items"][0])
                append_results([ct["tc"] for ct in batch["items"]])
                if self._context_cleared:
                    if not any(item is message for item in self._openai_messages):
                        content = "\n".join(results[ct["tc"]["id"]] for ct in batch["items"]
                                            if ct["tc"]["id"] in results)
                        self._openai_messages.append({"role": "user", "content": content})
                    break
        finally:
            # Cancellation can occur during approval or execution. Retained calls still need replies.
            if any(item is message for item in self._openai_messages):
                for tc in tool_calls:
                    if tc["id"] not in results:
                        reason = "context was cleared" if self._context_cleared else "batch was interrupted"
                        results[tc["id"]] = f"Tool call skipped because the {reason}."
                        self._record_tool_outcome(tc["function"]["name"], False)
                append_results(tool_calls)
            self._context_cleared = False

    async def _call_openai_stream(self) -> dict:
        async def _do():
            active_tools = _sanitize_for_utf8(_to_openai_tools(get_active_tool_definitions(self.tools)))
            request = {
                "model": self.model,
                "messages": _sanitize_for_utf8(self._openai_messages),
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if active_tools:
                request["tools"] = active_tools
            if self.reasoning_effort is not None:
                request["reasoning_effort"] = self.reasoning_effort
            stream = await self._openai_client.chat.completions.create(**request)

            content = ""
            reasoning_content = ""
            reasoning_received = False
            first_text = True
            tool_calls: dict[int, dict] = {}
            finish_reason = ""
            usage = None

            async for chunk in stream:
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                reasoning_delta = getattr(delta, "reasoning_content", None)
                if reasoning_delta is not None:
                    reasoning_received = True
                    reasoning_content += _safe_utf8_text(reasoning_delta)

                if delta and delta.content:
                    if first_text:
                        stop_spinner()
                        self._emit_text("\n")
                        first_text = False
                    self._emit_text(delta.content)
                    content += _safe_utf8_text(delta.content)

                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        existing = tool_calls.get(tc.index)
                        if existing:
                            if tc.function and tc.function.arguments:
                                existing["arguments"] += _safe_utf8_text(tc.function.arguments)
                        else:
                            tool_calls[tc.index] = {
                                "id": _safe_utf8_text(tc.id or ""),
                                "name": _safe_utf8_text((tc.function.name if tc.function else "") or ""),
                                "arguments": _safe_utf8_text((tc.function.arguments if tc.function else "") or ""),
                            }

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            assembled = None
            if tool_calls:
                assembled = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for _, tc in sorted(tool_calls.items())
                ]

            message = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": assembled,
            }
            if reasoning_received:
                message["reasoning_content"] = reasoning_content

            return {
                "choices": [{
                    "message": message,
                    "finish_reason": finish_reason or "stop",
                }],
                "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
            }

        return await _with_retry(_do)

    async def _confirm_dangerous(self, command: str) -> bool:
        print_confirmation(command)
        if self.confirm_fn:
            return await self.confirm_fn(command)
        # Fallback: blocking input
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False
