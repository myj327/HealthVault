"""Offline-first TASK-022 real-model Skill benchmark runner.

The default command only creates a zero-network preflight. Real execution is
fail-closed behind ``--execute --approved`` plus a configured Zhipu key.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from healthvault.agent_runtime import (
    HealthVaultAgentRuntimeConfig,
    build_healthvault_agent_runtime,
)
from healthvault.bootstrap import READ_ONLY_TOOL_NAMES
from healthvault.evaluation.skill.benchmark.core import (
    ACTIVE_SKILL_PATH,
    CORPUS_ARTIFACT_PATH,
    DEFAULT_DATABASE_FIXTURE_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_MANIFEST_PATH,
    ROOT,
    canonical_json_bytes,
    load_dataset,
    load_manifest,
    materialize_database_fixture,
    sha256_bytes,
    validate_attempt,
    validate_attempts,
)
from healthvault.evolution.synthetic_demo import SKILL_V2
from healthvault.model_config import HealthVaultModelConfig, OPENAI_COMPATIBLE
from healthvault.rag.chunking import estimate_tokens
from healthvault.runtime_context import RuntimeContext


PROVIDER = "zhipu"
MODEL = "glm-5.3-flash"
API_BASE = "https://open.bigmodel.cn/api/paas/v4"
KEY_ENV = "HEALTHVAULT_OPENAI_API_KEY"
BASE_URL_ENV = "HEALTHVAULT_OPENAI_BASE_URL"
MODEL_ENV = "HEALTHVAULT_OPENAI_MODEL"
TEMPERATURE = 0.0
THINKING = {"type": "enabled", "clear_thinking": False}
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 2048
MAX_MODEL_TURNS_PER_CASE = 4
MAX_TRANSPORT_RETRIES = 1
DEFAULT_PREFLIGHT_PATH = ROOT / "healthvault" / "evaluation" / "skill" / "task-022-preflight.json"
DEFAULT_RUNTIME_ROOT = ROOT / ".bear" / "skill-evolution" / "healthvault" / "benchmark" / "task-022"
PROMPT_TEMPLATE = """Run this frozen synthetic HealthVault case through the available read-only tools.
Do not guess, do not write data, and retry a tool only after an explicitly retryable result.
Finish with one JSON object and no surrounding prose using exactly these fields:
{{"final_status":"ok|insufficient_evidence|rejected","citations":["observed citation ids"],"refusal":true|false}}
Case ID: {case_id}
Task: {prompt}
"""

class ExecutionGateError(RuntimeError):
    """Raised before transport construction when a real-run gate is closed."""


class TransportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        http_status: int | None = None,
        provider_error_code: str | None = None,
        provider_error_message: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.http_status = http_status
        self.provider_error_code = provider_error_code
        self.provider_error_message = provider_error_message
        self.request_id = request_id

    def safe_diagnostics(self) -> dict[str, Any]:
        """Return the only provider error fields permitted in persisted artifacts."""
        return {
            key: value
            for key, value in {
                "http_status": self.http_status,
                "provider_error_code": self.provider_error_code,
                "provider_error_message": self.provider_error_message,
                "request_id": self.request_id,
            }.items()
            if value is not None
        }


def _safe_http_error_diagnostics(
    exc: urllib.error.HTTPError,
    request_headers: Mapping[str, str],
) -> dict[str, Any]:
    secrets: list[str] = []
    for name, value in request_headers.items():
        if name.casefold() not in {"authorization", "api-key", "x-api-key"}:
            continue
        text = str(value).strip()
        if text:
            secrets.append(text)
            if text.casefold().startswith("bearer "):
                secrets.append(text[7:].strip())

    def safe_text(value: object, *, limit: int) -> str | None:
        if not isinstance(value, (str, int, float)):
            return None
        text = str(value).strip()
        if not text:
            return None
        for secret in sorted(set(secrets), key=len, reverse=True):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        text = re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/=]+",
            "Bearer [REDACTED]",
            text,
        )
        return text[:limit]

    raw = b""
    try:
        raw = exc.read(8192)
    except (AttributeError, OSError):
        raw = b""
    body: object = None
    if raw:
        try:
            body = json.loads(raw.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
    error = body.get("error") if isinstance(body, dict) else None
    source = error if isinstance(error, dict) else body if isinstance(body, dict) else {}
    code = source.get("code") if isinstance(source, dict) else None
    message = None
    if isinstance(source, dict):
        message = source.get("message", source.get("msg"))

    response_headers = exc.headers
    request_id = None
    if response_headers is not None:
        for name in ("x-request-id", "request-id", "x-requestid"):
            value = response_headers.get(name)
            if value:
                request_id = value
                break
    return {
        "http_status": int(exc.code),
        "provider_error_code": safe_text(code, limit=128),
        "provider_error_message": safe_text(message, limit=512),
        "request_id": safe_text(request_id, limit=256),
    }


class JsonTransport(Protocol):
    is_real: bool

    async def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]: ...


class ScriptedFakeTransport:
    """Deterministic in-memory transport; it never counts as an API request."""

    is_real = False

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def post_json(
        self, _url: str, _headers: dict[str, str], payload: dict[str, Any], _timeout: float
    ) -> dict[str, Any]:
        self.requests.append(copy.deepcopy(payload))
        if not self._responses:
            raise AssertionError("fake transport has no scripted response")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


class UrllibJsonTransport:
    """Small benchmark-owned OpenAI-compatible transport."""

    is_real = True

    async def post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        def send() -> dict[str, Any]:
            request = urllib.request.Request(
                url,
                data=canonical_json_bytes(payload),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                diagnostics = _safe_http_error_diagnostics(exc, headers)
                raise TransportError(
                    f"OpenAI-compatible HTTP {exc.code}",
                    retryable=exc.code == 429,
                    **diagnostics,
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise TransportError("OpenAI-compatible transport failure", retryable=True) from exc
            if not isinstance(result, dict):
                raise TransportError("OpenAI-compatible response is not an object", retryable=False)
            return result

        return await asyncio.to_thread(send)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid JSONL record in {path}")
            records.append(value)
    return records


class CallLedger:
    """Atomic transport-attempt ledger with a hard request-count limit."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_config_sha256: str,
        max_transport_attempts: int,
    ) -> None:
        self.path = Path(path)
        self.run_config_sha256 = run_config_sha256
        self.max_transport_attempts = int(max_transport_attempts)
        self.records = _load_jsonl(self.path)
        if any(item.get("run_config_sha256") != run_config_sha256 for item in self.records):
            raise ValueError("incompatible ledger run config")
        self._open: set[int] = set()

    def begin(self) -> int:
        if len(self.records) + len(self._open) >= self.max_transport_attempts:
            raise RuntimeError("transport attempt limit exceeded")
        sequence = len(self.records) + len(self._open) + 1
        self._open.add(sequence)
        return sequence

    def commit(
        self,
        sequence: int,
        *,
        usage: dict[str, Any] | None,
        latency_ms: float,
        status: str,
    ) -> dict[str, Any]:
        if sequence not in self._open:
            raise RuntimeError("unknown or already committed ledger attempt")
        self._open.remove(sequence)
        prompt_tokens = None if usage is None else usage.get("prompt_tokens")
        completion_tokens = None if usage is None else usage.get("completion_tokens")
        total_tokens = None if usage is None else usage.get("total_tokens")
        known = (
            isinstance(prompt_tokens, int) and prompt_tokens >= 0
            and isinstance(completion_tokens, int) and completion_tokens >= 0
            and isinstance(total_tokens, int) and total_tokens >= 0
            and total_tokens == prompt_tokens + completion_tokens
        )
        record = {
            "schema_version": "healthvault-task-022-ledger-v1",
            "run_config_sha256": self.run_config_sha256,
            "sequence": sequence,
            "status": status,
            "actual_input_tokens": prompt_tokens if known else None,
            "actual_output_tokens": completion_tokens if known else None,
            "actual_total_tokens": total_tokens if known else None,
            "usage_source": "provider" if known else "unknown",
            "latency_ms": float(latency_ms),
        }
        self.records.append(record)
        _atomic_jsonl(self.path, self.records)
        return record


class ResumeStore:
    """Completed arm-case records bound to one exact run configuration."""

    def __init__(self, path: str | Path, run_config_sha256: str) -> None:
        self.path = Path(path)
        self.run_config_sha256 = run_config_sha256
        self.records = _load_jsonl(self.path)
        if any(item.get("run_config_sha256") != run_config_sha256 for item in self.records):
            raise ValueError("incompatible result resume")

    def is_complete(self, arm_id: str, case_id: str) -> bool:
        return any(
            item.get("arm_id") == arm_id
            and item.get("case_id") == case_id
            and item.get("complete") is True
            for item in self.records
        )

    def append(self, record: dict[str, Any]) -> None:
        value = dict(record)
        value["run_config_sha256"] = self.run_config_sha256
        pair = (value.get("arm_id"), value.get("case_id"))
        if any((item.get("arm_id"), item.get("case_id")) == pair for item in self.records):
            raise ValueError(f"duplicate resume pair: {pair}")
        self.records.append(value)
        _atomic_jsonl(self.path, self.records)


def _git_provenance(workspace: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=workspace, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=workspace, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unavailable", None
    return {"current_head": head, "dirty": dirty}


def _source_hashes(workspace: Path) -> dict[str, str]:
    paths = {
        "task_022_runner": Path(__file__).resolve(),
        "agent_runtime": workspace / "healthvault" / "agent_runtime.py",
        "task_021_validator": workspace / "healthvault" / "evaluation" / "skill" / "benchmark" / "core.py",
        "healthvault_prompt": workspace / "healthvault" / "prompt.py",
    }
    return {name: sha256_bytes(path.read_bytes()) for name, path in paths.items()}


def _estimate_plan(dataset: dict[str, Any], arms: list[dict[str, Any]], workspace: Path) -> dict[str, Any]:
    active = (workspace / ACTIVE_SKILL_PATH.relative_to(ROOT)).read_text(encoding="utf-8")
    skill_text = {"no_skill": "", "active_v1": active, "candidate_v2": SKILL_V2}
    base_prompt = (workspace / "healthvault" / "prompt.py").read_text(encoding="utf-8")
    tools_text = "".join(
        (workspace / "healthvault" / "tools" / name).read_text(encoding="utf-8")
        for name in (
            "query_health_timeline.py", "compare_health_indicators.py",
            "search_health_knowledge.py", "generate_visit_brief.py",
            "preview_health_report.py",
        )
    )
    total_input = 0
    expected_calls = 0
    by_arm: dict[str, dict[str, int]] = {}
    for arm in arms:
        arm_input = arm_calls = 0
        base = estimate_tokens(base_prompt + tools_text + skill_text[arm["arm_id"]])
        for case in dataset["cases"]:
            calls = len(case["expected"]["tool_order"]) + 1
            initial = base + estimate_tokens(PROMPT_TEMPLATE.format(case_id=case["case_id"], prompt=case["prompt"]))
            # Context grows after each tool result.  384 tokens is a conservative
            # deterministic reserve for one domain result and protocol envelope.
            arm_input += sum(initial + turn * 384 for turn in range(calls))
            arm_calls += calls
        by_arm[arm["arm_id"]] = {"model_calls": arm_calls, "estimated_input_tokens": arm_input}
        total_input += arm_input
        expected_calls += arm_calls
    planned_output = expected_calls * 64
    near_limit_output = expected_calls * MAX_OUTPUT_TOKENS
    return {
        "method": "char4-cjk-v1 over frozen prompt/tool/skill envelopes plus 384 tokens per prior tool result",
        "by_arm": by_arm,
        "expected_model_calls": expected_calls,
        "expected_transport_attempts": expected_calls,
        "all_once_retry_transport_attempts": expected_calls * 2,
        "estimated_input_tokens": total_input,
        "planned_output_tokens": planned_output,
        "near_limit_output_tokens": near_limit_output,
    }


def _preflight_payload(preflight: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in preflight.items() if key not in {"run_config_sha256", "observed"}}


def resolve_model_config(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    values = os.environ if environ is None else environ
    api_base = str(values.get(BASE_URL_ENV, "") or API_BASE).strip().rstrip("/")
    model = str(values.get(MODEL_ENV, "") or MODEL).strip()
    key_configured = bool(str(values.get(KEY_ENV, "") or "").strip())
    if not api_base or not model:
        raise ValueError("TASK-022 model and API base must be non-empty")
    return {
        "provider": PROVIDER,
        "protocol": "OpenAI Chat Completions",
        "api_base": api_base,
        "model": model,
        "key_environment": KEY_ENV,
        "base_url_environment": BASE_URL_ENV,
        "model_environment": MODEL_ENV,
        "key_configured": key_configured,
        "temperature": TEMPERATURE,
        "thinking": dict(THINKING),
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def build_preflight(
    *, workspace_root: str | Path = ROOT,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    dataset_path = workspace / DEFAULT_DATASET_PATH.relative_to(ROOT)
    manifest_path = workspace / DEFAULT_MANIFEST_PATH.relative_to(ROOT)
    dataset = load_dataset(dataset_path, workspace_root=workspace)
    task021 = load_manifest(manifest_path, dataset, workspace_root=workspace)
    arms = [dict(item) for item in task021["arms"]]
    estimate = _estimate_plan(dataset, arms, workspace)
    model_config = resolve_model_config(environ)
    preflight: dict[str, Any] = {
        "schema_version": "healthvault-task-022-preflight-v1",
        "benchmark_version": "skill-real-benchmark-r1-v1",
        "mode": "preflight",
        "claim_status": "not_run",
        "model_config": model_config,
        "shared_agent_config": {
            "runtime": "HealthVaultAgentRuntime",
            "mode": "read_only",
            "tool_names": list(READ_ONLY_TOOL_NAMES),
            "database_isolation": "one temporary SQLite snapshot per arm-case",
            "only_arm_variable": "Skill content",
        },
        "arms": arms,
        "frozen_inputs": {
            "dataset_file_sha256": sha256_bytes(dataset_path.read_bytes()),
            "dataset_semantic_sha256": task021["dataset_sha256"],
            "task_021_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
            "database_fixture_sha256": sha256_bytes((workspace / DEFAULT_DATABASE_FIXTURE_PATH.relative_to(ROOT)).read_bytes()),
            "corpus_artifact_sha256": sha256_bytes((workspace / CORPUS_ARTIFACT_PATH.relative_to(ROOT)).read_bytes()),
            "active_skill_sha256": sha256_bytes((workspace / ACTIVE_SKILL_PATH.relative_to(ROOT)).read_bytes()),
            "candidate_skill_sha256": sha256_bytes(SKILL_V2.encode("utf-8")),
            "case_order": list(task021["shared_config"]["case_order"]),
            "validator_sha256": task021["shared_config"]["validator_sha256"],
            "prompt_template_sha256": sha256_bytes(PROMPT_TEMPLATE.encode("utf-8")),
        },
        "source_sha256": _source_hashes(workspace),
        "git_provenance": _git_provenance(workspace),
        "limits": {
            "max_model_turns_per_case": MAX_MODEL_TURNS_PER_CASE,
            "max_model_calls": len(dataset["cases"]) * len(arms) * MAX_MODEL_TURNS_PER_CASE,
            "max_transport_attempts": len(dataset["cases"]) * len(arms) * MAX_MODEL_TURNS_PER_CASE * (MAX_TRANSPORT_RETRIES + 1),
            "transport_retries": MAX_TRANSPORT_RETRIES,
        },
        "plan": {"cases": len(dataset["cases"]), "arms": len(arms), "arm_cases": len(dataset["cases"]) * len(arms), "execution_passes": 1},
        "estimate": estimate,
        "observed": {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    preflight["run_config_sha256"] = sha256_bytes(canonical_json_bytes(_preflight_payload(preflight)))
    return preflight


def assert_execution_gate(
    preflight: dict[str, Any], *, execute: bool, approved: bool, api_key: str,
    workspace_root: str | Path = ROOT,
) -> None:
    if not execute:
        raise ExecutionGateError("real execution requires --execute")
    if not approved:
        raise ExecutionGateError("real execution requires explicit --approved")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ExecutionGateError(f"{KEY_ENV} is not configured")
    if preflight.get("model_config", {}).get("key_configured") is not True:
        raise ExecutionGateError(f"preflight does not bind configured {KEY_ENV}")
    expected = build_preflight(workspace_root=workspace_root)
    if preflight.get("run_config_sha256") != sha256_bytes(
        canonical_json_bytes(_preflight_payload(preflight))
    ):
        raise ExecutionGateError("preflight hash or frozen input hash mismatch")
    if preflight.get("run_config_sha256") != expected["run_config_sha256"]:
        raise ExecutionGateError("preflight hash or frozen input hash mismatch")


class OpenAICompatibleController:
    def __init__(
        self, *, transport: JsonTransport, api_key: str,
        model_config: dict[str, Any], ledger: CallLedger | None,
        tool_result_view: Callable[[Any], Any] | None = None,
    ) -> None:
        self.transport = transport
        self.api_key = api_key
        self.model_config = model_config
        self.ledger = ledger
        self.tool_result_view = tool_result_view
        self.model_calls = 0
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.raw_responses: list[dict[str, Any]] = []

    async def call(self, agent: Any) -> dict[str, Any]:
        if self.model_calls >= MAX_MODEL_TURNS_PER_CASE:
            raise RuntimeError("model turn limit exceeded")
        self.model_calls += 1
        payload = {
            "model": self.model_config["model"],
            "messages": copy.deepcopy(agent._openai_messages),
            "tools": [
                {"type": "function", "function": {"name": item["name"], "description": item.get("description", ""), "parameters": item.get("input_schema", {"type": "object"})}}
                for item in agent.tools
            ],
            "temperature": TEMPERATURE,
            "thinking": copy.deepcopy(self.model_config["thinking"]),
            "reasoning_effort": self.model_config["reasoning_effort"],
            "max_tokens": self.model_config["max_output_tokens"],
            "stream": False,
        }
        if self.tool_result_view is not None:
            for message in payload["messages"]:
                if message.get("role") != "tool":
                    continue
                try:
                    tool_result = json.loads(message["content"])
                except (TypeError, KeyError, json.JSONDecodeError):
                    continue
                message["content"] = json.dumps(
                    self.tool_result_view(tool_result), ensure_ascii=False, sort_keys=True,
                )
        attempts = MAX_TRANSPORT_RETRIES + 1 if self.transport.is_real else 1
        last_error: Exception | None = None
        for retry in range(attempts):
            sequence = self.ledger.begin() if self.transport.is_real and self.ledger else None
            started = time.perf_counter()
            try:
                response = await self.transport.post_json(
                    f"{self.model_config['api_base'].rstrip('/')}/chat/completions",
                    {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    payload,
                    60.0,
                )
            except TransportError as exc:
                latency = (time.perf_counter() - started) * 1000
                if sequence is not None:
                    self.ledger.commit(sequence, usage=None, latency_ms=latency, status="transport_error")
                    self.api_calls += 1
                last_error = exc
                if not exc.retryable or retry >= MAX_TRANSPORT_RETRIES:
                    raise
                continue
            latency = (time.perf_counter() - started) * 1000
            usage = response.get("usage") if isinstance(response, dict) else None
            self.raw_responses.append(copy.deepcopy(response))
            if sequence is not None:
                record = self.ledger.commit(sequence, usage=usage, latency_ms=latency, status="ok")
                self.api_calls += 1
                if record["actual_input_tokens"] is not None:
                    self.input_tokens += int(record["actual_input_tokens"])
                    self.output_tokens += int(record["actual_output_tokens"])
                    self.total_tokens += int(record["actual_total_tokens"])
            message = (
                response.get("choices", [{}])[0].get("message", {})
                if isinstance(response, dict) and response.get("choices")
                else {}
            )
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and content and not message.get("tool_calls"):
                agent._emit_text(content)
            return response
        raise last_error or RuntimeError("transport failed")


def _extract_citations(value: Any) -> set[str]:
    citations: set[str] = set()
    if isinstance(value, dict):
        citation = value.get("citation_id")
        if isinstance(citation, str):
            citations.add(citation)
        report_id = value.get("report_id")
        code = value.get("code") or value.get("indicator_code")
        if isinstance(report_id, str) and isinstance(code, str) and report_id and code:
            citations.add(f"REC:{report_id}:{code}")
        if isinstance(report_id, str) and isinstance(value.get("indicators"), list):
            for indicator in value["indicators"]:
                if isinstance(indicator, dict) and isinstance(indicator.get("code"), str):
                    citations.add(f"REC:{report_id}:{indicator['code']}")
        if isinstance(code, str):
            for point_name in ("earliest", "latest"):
                point = value.get(point_name)
                if isinstance(point, dict) and isinstance(point.get("report_id"), str):
                    citations.add(f"REC:{point['report_id']}:{code}")
        for child in value.values():
            citations.update(_extract_citations(child))
    elif isinstance(value, list):
        for child in value:
            citations.update(_extract_citations(child))
    return citations


def _parse_final(text: str) -> tuple[str, list[str], bool, bool]:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        return "invalid_model_output", [], False, False
    if not isinstance(value, dict) or set(value) != {"final_status", "citations", "refusal"}:
        return "invalid_model_output", [], False, False
    if not isinstance(value["final_status"], str) or not isinstance(value["citations"], list) or not all(isinstance(item, str) for item in value["citations"]) or not isinstance(value["refusal"], bool):
        return "invalid_model_output", [], False, False
    return value["final_status"], value["citations"], value["refusal"], True


def _write_candidate_skill(runtime_root: Path, skill_content: str = SKILL_V2) -> Path:
    root = runtime_root / "candidate" / ".bear" / "skills"
    skill_path = root / "HealthVault-Task-Execution" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(skill_content, encoding="utf-8", newline="\n")
    return root


async def run_agent_case(
    arm: dict[str, Any], case: dict[str, Any], preflight: dict[str, Any], *,
    transport: JsonTransport, runtime_root: str | Path,
    workspace_root: str | Path = ROOT, ledger: CallLedger | None = None,
    api_key: str = "fake", skill_content: str | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    runtime_dir = Path(runtime_root) / arm["arm_id"] / case["case_id"]
    runtime_dir.mkdir(parents=True, exist_ok=True)
    database_path = runtime_dir / "healthvault.sqlite3"
    materialize_database_fixture(database_path)
    if arm["arm_id"] == "no_skill":
        skill_root, skill_names = None, ()
    elif arm["arm_id"] == "active_v1":
        skill_root = workspace / ".bear" / "skills"
        skill_names = ("HealthVault-Task-Execution",)
    elif arm["arm_id"] in {"candidate_v2", "generated_candidate"}:
        content = SKILL_V2 if skill_content is None and arm["arm_id"] == "candidate_v2" else skill_content
        if not isinstance(content, str) or not content:
            raise ValueError(f"{arm['arm_id']} requires frozen Skill content")
        if sha256_bytes(content.encode("utf-8")) != arm["skill_sha256"]:
            raise ValueError(f"{arm['arm_id']} Skill content hash mismatch")
        skill_root = _write_candidate_skill(runtime_dir, content)
        skill_names = ("HealthVault-Task-Execution",)
    else:
        raise ValueError(f"unknown arm: {arm['arm_id']}")
    runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
        context=RuntimeContext(case["fixture"]["profile_id"], f"task-022-{arm['arm_id']}-{case['case_id']}"),
        database_path=database_path,
        model_config=HealthVaultModelConfig(
            OPENAI_COMPATIBLE,
            preflight["model_config"]["model"],
            preflight["model_config"]["api_base"],
            api_key,
        ),
        skill_root=skill_root, skill_names=skill_names,
        max_turns=MAX_MODEL_TURNS_PER_CASE, max_cost_usd=999.0,
    ))
    # Keep the normal Agent Loop but suppress interactive UI/cost rendering;
    # TASK-022 records only provider usage fields and does not estimate fees.
    runtime.agent.is_sub_agent = True
    controller = OpenAICompatibleController(
        transport=transport,
        api_key=api_key,
        model_config=preflight["model_config"],
        ledger=ledger,
    )
    runtime.agent._call_openai_stream = lambda: controller.call(runtime.agent)
    original_execute = runtime.agent._execute_tool_call
    tool_calls: list[dict[str, str]] = []
    trace_details: list[dict[str, Any]] = []
    observed_citations: set[str] = set()
    invalid_calls = 0
    failures = set(case["fixture"]["retryable_failures"])

    async def traced_execute(name: str, tool_input: dict[str, Any]) -> str:
        nonlocal invalid_calls
        index = len(tool_calls)
        if name not in runtime.manifest["tool_names"]:
            invalid_calls += 1
        if str(index) in failures:
            tool_calls.append({"name": name, "status": "retryable_failure"})
            raw = json.dumps({"status": "retryable_failure", "retryable": True, "message": "synthetic fixture failure"})
            trace_details.append({"name": name, "input": tool_input, "status": "retryable_failure", "result": raw})
            return raw
        try:
            raw = await original_execute(name, tool_input)
        except Exception as exc:
            tool_calls.append({"name": name, "status": "hard_failure"})
            trace_details.append({"name": name, "input": tool_input, "status": "hard_failure", "error_type": type(exc).__name__})
            raise
        tool_calls.append({"name": name, "status": "ok"})
        trace_details.append({"name": name, "input": tool_input, "status": "ok", "result": raw})
        try:
            observed_citations.update(_extract_citations(json.loads(raw)))
        except (json.JSONDecodeError, TypeError):
            pass
        return raw

    runtime.agent._execute_tool_call = traced_execute
    started = time.perf_counter()
    hard_failure = False
    text = ""
    try:
        result = await runtime.run(PROMPT_TEMPLATE.format(case_id=case["case_id"], prompt=case["prompt"]))
        text = result.text
    except Exception as exc:
        hard_failure = True
        text = json.dumps({"error": type(exc).__name__})
    latency_ms = (time.perf_counter() - started) * 1000
    final_status, citations, refusal, valid_final = _parse_final(text)
    citations_grounded = all(item in observed_citations for item in citations)
    hard_failure = hard_failure or not valid_final or not citations_grounded
    write_attempts = [call["name"] for call in tool_calls if call["name"] not in runtime.manifest["tool_names"]]
    _atomic_json(runtime_dir / "model-responses.json", controller.raw_responses)
    _atomic_json(runtime_dir / "tool-trace.json", trace_details)
    return {
        "arm_id": arm["arm_id"], "case_id": case["case_id"],
        "dataset_sha256": preflight["frozen_inputs"]["dataset_semantic_sha256"],
        "run_config_sha256": preflight["run_config_sha256"],
        "skill_sha256": arm["skill_sha256"], "tool_calls": tool_calls,
        "final_status": final_status, "citations": citations, "refusal": refusal,
        "write_attempts": write_attempts, "hard_failure": hard_failure,
        "invalid_calls": invalid_calls, "steps": controller.model_calls,
        "input_tokens": controller.input_tokens, "output_tokens": controller.output_tokens,
        "total_tokens": controller.total_tokens, "latency_ms": latency_ms,
        "api_calls": controller.api_calls,
    }


def _task022_manifest(preflight: dict[str, Any]) -> dict[str, Any]:
    dataset = load_dataset(DEFAULT_DATASET_PATH)
    manifest = load_manifest(DEFAULT_MANIFEST_PATH, dataset)
    adapted = copy.deepcopy(manifest)
    adapted["run_config_sha256"] = preflight["run_config_sha256"]
    return adapted


def summarize_comparison_attempts(
    dataset: dict[str, Any],
    preflight: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    candidate_arm_id: str,
    schema_version: str,
    case_order: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize a three-arm run while allowing one frozen challenger Skill."""
    arm_ids = ("no_skill", "active_v1", candidate_arm_id)
    selected_cases = list(case_order or preflight["frozen_inputs"]["case_order"])
    expected_pairs = {(arm_id, case_id) for arm_id in arm_ids for case_id in selected_cases}
    seen: set[tuple[str, str]] = set()
    arm_hashes = {arm["arm_id"]: arm["skill_sha256"] for arm in preflight["arms"]}
    for attempt in attempts:
        pair = (attempt.get("arm_id"), attempt.get("case_id"))
        if pair in seen or pair not in expected_pairs:
            raise ValueError(f"unexpected or duplicate comparison attempt: {pair}")
        seen.add(pair)
        if attempt.get("dataset_sha256") != preflight["frozen_inputs"]["dataset_semantic_sha256"]:
            raise ValueError("comparison attempt dataset hash mismatch")
        if attempt.get("run_config_sha256") != preflight["run_config_sha256"]:
            raise ValueError("comparison attempt run config hash mismatch")
        if attempt.get("skill_sha256") != arm_hashes[pair[0]]:
            raise ValueError("comparison attempt Skill hash mismatch")
    if seen != expected_pairs:
        raise ValueError("comparison attempts are incomplete")
    cases = {case["case_id"]: case for case in dataset["cases"]}
    arm_metrics: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, bool]] = {}
    for arm_id in arm_ids:
        group = [item for item in attempts if item["arm_id"] == arm_id]
        validations = [validate_attempt(cases[item["case_id"]], item) for item in group]
        outcomes[arm_id] = {
            item["case_id"]: result["task_success"]
            for item, result in zip(group, validations)
        }
        arm_metrics[arm_id] = {
            "role": "optional_reference" if arm_id == "no_skill" else "skill_evolution_comparison",
            "attempts": len(group),
            "task_success_rate": sum(item["task_success"] for item in validations) / len(group),
            "correct_tool_call_rate": sum(item["tool_call_correct"] for item in validations) / len(group),
            "citation_correctness_rate": sum(item["citation_correct"] for item in validations) / len(group),
            "hard_failure_rate": sum(item["hard_failure"] for item in group) / len(group),
            "api_calls": sum(int(item["api_calls"]) for item in group),
            "input_tokens": sum(int(item["input_tokens"]) for item in group),
            "output_tokens": sum(int(item["output_tokens"]) for item in group),
            "total_tokens": sum(int(item["total_tokens"]) for item in group),
            "average_latency_ms": sum(float(item["latency_ms"]) for item in group) / len(group),
        }
    active = outcomes["active_v1"]
    candidate = outcomes[candidate_arm_id]
    improved = sorted(case_id for case_id in active if not active[case_id] and candidate[case_id])
    regressed = sorted(case_id for case_id in active if active[case_id] and not candidate[case_id])
    denominator = len(active)
    comparison = {
        "primary_baseline": "active_v1",
        "challenger": candidate_arm_id,
        "task_success_delta_pp": (
            arm_metrics[candidate_arm_id]["task_success_rate"]
            - arm_metrics["active_v1"]["task_success_rate"]
        ) * 100,
        "hard_failure_delta_pp": (
            arm_metrics[candidate_arm_id]["hard_failure_rate"]
            - arm_metrics["active_v1"]["hard_failure_rate"]
        ) * 100,
        "improved_cases": improved,
        "improved_case_count": len(improved),
        "improved_rate": len(improved) / denominator,
        "regressed_cases": regressed,
        "regressed_case_count": len(regressed),
        "regressed_rate": len(regressed) / denominator,
    }
    return {
        "schema_version": schema_version,
        "benchmark_version": preflight["benchmark_version"],
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": preflight["frozen_inputs"]["dataset_semantic_sha256"],
        "run_config_sha256": preflight["run_config_sha256"],
        "run_kind": "real_model_heldout",
        "claim_status": "completed_once",
        "model_config": preflight["model_config"],
        "limits": preflight["limits"],
        "arms": arm_metrics,
        f"active_v1_to_{candidate_arm_id}": comparison,
        "api_calls": sum(int(item["api_calls"]) for item in attempts),
        "input_tokens": sum(int(item["input_tokens"]) for item in attempts),
        "output_tokens": sum(int(item["output_tokens"]) for item in attempts),
        "total_tokens": sum(int(item["total_tokens"]) for item in attempts),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Synthetic HealthVault records and fixed offline corpus only.",
            "Deterministic TASK-021 validator; no LLM Judge or medical-fact adjudication.",
            "Provider cost is unavailable and is not estimated or reported as zero.",
            "One comparison run; results are not a general model benchmark.",
        ],
    }


def summarize_real_attempts(
    dataset: dict[str, Any], preflight: dict[str, Any], attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    return summarize_comparison_attempts(
        dataset,
        preflight,
        attempts,
        candidate_arm_id="candidate_v2",
        schema_version="healthvault-task-022-real-summary-v1",
    )


async def execute_real_run(
    preflight: dict[str, Any], *, api_key: str, runtime_root: str | Path,
    workspace_root: str | Path = ROOT, transport: JsonTransport | None = None,
    arm_skill_contents: Mapping[str, str] | None = None,
    candidate_arm_id: str = "candidate_v2",
    output_prefix: str = "task-022",
    output_dir: str | Path | None = None,
    case_order: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute or exact-config resume all 60 arm-cases after the external gate."""
    workspace = Path(workspace_root).resolve()
    runtime = Path(runtime_root)
    runtime.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(workspace / DEFAULT_DATASET_PATH.relative_to(ROOT), workspace_root=workspace)
    resume = ResumeStore(runtime / "completed-attempts.jsonl", preflight["run_config_sha256"])
    ledger = CallLedger(
        runtime / "call-ledger.jsonl",
        run_config_sha256=preflight["run_config_sha256"],
        max_transport_attempts=preflight["limits"]["max_transport_attempts"],
    )
    real_transport = transport or UrllibJsonTransport()
    cases = {case["case_id"]: case for case in dataset["cases"]}
    selected_cases = list(case_order or preflight["frozen_inputs"]["case_order"])
    contents = dict(arm_skill_contents or {})
    for arm in preflight["arms"]:
        for case_id in selected_cases:
            if resume.is_complete(arm["arm_id"], case_id):
                continue
            attempt = await run_agent_case(
                arm, cases[case_id], preflight,
                transport=real_transport, runtime_root=runtime / "cases",
                workspace_root=workspace, ledger=ledger, api_key=api_key,
                skill_content=contents.get(arm["arm_id"]),
            )
            resume.append({**attempt, "complete": True})
    attempts = [
        {key: value for key, value in item.items() if key != "complete"}
        for item in resume.records if item.get("complete") is True
    ]
    summary = summarize_comparison_attempts(
        dataset,
        preflight,
        attempts,
        candidate_arm_id=candidate_arm_id,
        schema_version=f"healthvault-{output_prefix}-real-summary-v1",
        case_order=selected_cases,
    )
    matrix = {
        "schema_version": f"healthvault-{output_prefix}-case-matrix-v1",
        "run_config_sha256": preflight["run_config_sha256"],
        "model": preflight["model_config"]["model"],
        "cases": [
            {
                **attempt,
                "validation": validate_attempt(cases[attempt["case_id"]], attempt),
            }
            for attempt in attempts
        ],
    }
    result_dir = Path(output_dir) if output_dir is not None else (
        workspace / "healthvault" / "evaluation" / "skill"
    )
    _atomic_json(result_dir / f"{output_prefix}-case-matrix.json", matrix)
    _atomic_json(result_dir / f"{output_prefix}-real-summary.json", summary)
    return attempts, summary


def _print_preflight(preflight: dict[str, Any]) -> None:
    estimate = preflight["estimate"]
    print(json.dumps({
        "status": "preflight_only",
        "provider": preflight["model_config"]["provider"],
        "model": preflight["model_config"]["model"],
        "api_base": preflight["model_config"]["api_base"],
        "arm_cases": preflight["plan"]["arm_cases"],
        "expected_model_calls": estimate["expected_model_calls"],
        "expected_transport_attempts": estimate["expected_transport_attempts"],
        "all_once_retry_transport_attempts": estimate["all_once_retry_transport_attempts"],
        "estimated_input_tokens": estimate["estimated_input_tokens"],
        "planned_output_tokens": estimate["planned_output_tokens"],
        "near_limit_output_tokens": estimate["near_limit_output_tokens"],
        "max_model_calls": preflight["limits"]["max_model_calls"],
        "max_transport_attempts": preflight["limits"]["max_transport_attempts"],
        "max_output_tokens_per_call": preflight["model_config"]["max_output_tokens"],
        "key_configured": preflight["model_config"]["key_configured"],
        "api_calls": 0,
    }, ensure_ascii=False, sort_keys=True))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TASK-022 offline-first Skill benchmark")
    parser.add_argument("--preflight-output", default=str(DEFAULT_PREFLIGHT_PATH))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_ROOT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    api_key = os.environ.get(KEY_ENV, "").strip()
    preflight = build_preflight(workspace_root=ROOT)
    _atomic_json(Path(args.preflight_output), preflight)
    _print_preflight(preflight)
    if not args.execute:
        return 0
    assert_execution_gate(preflight, execute=True, approved=args.approved, api_key=api_key, workspace_root=ROOT)
    asyncio.run(execute_real_run(
        preflight, api_key=api_key, runtime_root=args.runtime_dir, workspace_root=ROOT,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
