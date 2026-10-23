"""Minimal redacted task traces around the HealthVault tool executor."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping
from healthvault.tools.directory import TOOL_SPECS
from healthvault.evolution.failures import classify_trace_failure


ToolExecutor = Callable[[str, dict[str, Any], dict[str, float] | None], Awaitable[str]]

_SIGNAL_CONTEXT_KEYS = frozenset({
    "intent", "preconditions", "expected_outcome", "validator_status", "contract_checks",
})
_KNOWN_TOOL_NAMES = frozenset(spec.name for spec in TOOL_SPECS)
_SENSITIVE_CONTEXT = re.compile(
    r"(?i)(患者|病人|诊断|用药|血压|血糖|glucose|\bBP\b|\bdiagnosis\b|"
    r"\bmedication\b|\b\d{2,3}/\d{2,3}\b)"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _input_summary(name: str, tool_input: dict[str, Any]) -> dict[str, object]:
    allowed = next((set(spec.input_schema.get("properties", {})) for spec in TOOL_SPECS if spec.name == name), set())
    known_input = {key: value for key, value in tool_input.items() if key in allowed}
    return {
        "keys": sorted(known_input),
        "unknown_key_count": len(tool_input) - len(known_input),
        "types": {
            str(key): type(value).__name__
            for key, value in sorted(known_input.items())
        },
    }


def _result_metadata(result: str) -> tuple[str, str | None, str | None]:
    status = "ok"
    validator_code = None
    failure_code = None
    try:
        parsed = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        status = str(parsed.get("status", "ok"))
        validator = parsed.get("validator_code")
        failure = parsed.get("failure_code") or parsed.get("error_code")
        validator_code = str(validator) if validator is not None else None
        failure_code = str(failure) if failure is not None else None
    elif str(result).lower().startswith(("error", "tool unavailable", "tool input rejected")):
        status = "rejected"
        failure_code = "tool_rejected"
    return status, validator_code, failure_code


def _signal_context(context: Mapping[str, object] | None) -> dict[str, object] | None:
    """Persist only explicitly supplied, non-medical task metadata for manual batching."""
    if context is None:
        return None
    if set(context) != _SIGNAL_CONTEXT_KEYS:
        raise ValueError("signal context must use the exact task-end schema")
    intent = context["intent"]
    expected = context["expected_outcome"]
    validator = context["validator_status"]
    preconditions = context["preconditions"]
    contracts = context["contract_checks"]
    if not all(isinstance(value, str) and value.strip() for value in (intent, expected, validator)):
        raise ValueError("signal context strings must be non-empty")
    if (
        not isinstance(preconditions, (tuple, list))
        or not all(isinstance(item, str) and item.strip() for item in preconditions)
        or not isinstance(contracts, Mapping)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in contracts.items())
    ):
        raise ValueError("signal context preconditions and contract checks are invalid")
    flattened = "\n".join([intent, expected, *preconditions])
    if _SENSITIVE_CONTEXT.search(flattened):
        raise ValueError("signal context must not persist patient or medical facts")
    return {
        "intent": intent.strip(),
        "preconditions": list(preconditions),
        "expected_outcome": expected.strip(),
        "validator_status": validator.strip(),
        "contract_checks": {str(key): str(value) for key, value in contracts.items()},
    }


class TaskTraceRecorder:
    """Collect one task's redacted tool metadata and persist it at task end."""

    def __init__(
        self,
        evolution_root: str | Path,
        *,
        request_id: str,
        task_id: str,
        skill: dict[str, str],
        executor: ToolExecutor,
        signal_context: Mapping[str, object] | None = None,
    ) -> None:
        self._root = Path(evolution_root)
        self._request_id = request_id
        self._task_id = task_id
        self._skill = {"name": skill["name"], "sha256": skill["sha256"]}
        self._executor = executor
        self._signal_context = _signal_context(signal_context)
        self._started = time.perf_counter()
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._tool_calls: list[dict[str, object]] = []
        self._finished = False
        self.trace_path: Path | None = None

    @property
    def signal_context(self) -> Mapping[str, object] | None:
        return self._signal_context

    def permission_denied(self, name: str) -> None:
        """Record a rejected attempt without arguments, user values or denial text."""
        spec = next((item for item in TOOL_SPECS if item.name == name), None)
        self._tool_calls.append({
            "tool_name": name if spec is not None else "unapproved_tool",
            "input_summary": {"keys": [], "unknown_key_count": 0, "types": {}},
            "result_status": "rejected", "validator_code": "runtime_permission_denied",
            "failure_code": "safety_write_violation" if spec is not None and not spec.read_only else "safety_tool_policy_violation",
            "elapsed_ms": 0.0,
        })

    def workflow_rejected(self, name: str, tool_input: dict[str, Any], code: str) -> None:
        """Record a workflow refusal without search text or patient values."""
        self._tool_calls.append({
            "tool_name": name if name in _KNOWN_TOOL_NAMES else "unapproved_tool",
            "input_summary": _input_summary(name, tool_input),
            "result_status": "rejected", "validator_code": "workflow_protocol_violation",
            "failure_code": code, "elapsed_ms": 0.0,
        })

    async def execute(
        self,
        name: str,
        tool_input: dict[str, Any],
        read_file_state: dict[str, float] | None = None,
    ) -> str:
        started = time.perf_counter()
        recorded_name = name if name in _KNOWN_TOOL_NAMES else "unapproved_tool"
        try:
            result = await self._executor(name, tool_input, read_file_state)
        except Exception as exc:
            self._tool_calls.append({
                "tool_name": recorded_name,
                "input_summary": _input_summary(name, tool_input),
                "result_status": "failed",
                "validator_code": None,
                "failure_code": type(exc).__name__,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            })
            raise
        status, validator_code, failure_code = _result_metadata(result)
        self._tool_calls.append({
            "tool_name": recorded_name,
            "input_summary": _input_summary(name, tool_input),
            "result_status": status,
            "validator_code": validator_code,
            "failure_code": failure_code,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        })
        return result

    def finish(
        self,
        result_status: str,
        *,
        tokens: dict[str, int] | None = None,
        cost_usd: float = 0.0,
        failure_code: str | None = None,
        validator_status: str | None = None,
        contract_checks: Mapping[str, str] | None = None,
    ) -> Path:
        if self._finished and self.trace_path is not None:
            return self.trace_path
        safety = next((call["failure_code"] for call in self._tool_calls
                       if str(call.get("failure_code") or "").startswith("safety_")), None)
        if safety:
            result_status, failure_code, validator_status = "failed", str(safety), "failed"
            contract_checks = {**(contract_checks or (self._signal_context or {}).get("contract_checks", {})), "safety": "failed"}
        trace_id = hashlib.sha256(
            f"{self._request_id}\0{self._task_id}\0{self._started_at}".encode("utf-8")
        ).hexdigest()[:20]
        trace = {
            "schema_version": "healthvault-task-trace-v1",
            "trace_id": trace_id,
            "request_id": self._request_id,
            "task_id": self._task_id,
            "skill": dict(self._skill),
            "result_status": result_status,
            "failure_code": failure_code,
            "tokens": {
                "input": int((tokens or {}).get("input", 0)),
                "output": int((tokens or {}).get("output", 0)),
            },
            "cost_usd": float(cost_usd),
            "elapsed_ms": round((time.perf_counter() - self._started) * 1000, 3),
            "tool_calls": list(self._tool_calls),
            "candidate_generation_triggered": False,
            "started_at": self._started_at,
        }
        if self._signal_context is not None:
            context = dict(self._signal_context)
            if validator_status is not None:
                context["validator_status"] = validator_status
            if contract_checks is not None:
                context["contract_checks"] = dict(contract_checks)
            trace["signal_context"] = _signal_context(context)
        trace['failure_category'] = classify_trace_failure(trace)
        trace_dir = self._root / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        target = trace_dir / f"{trace_id}.json"
        temporary = trace_dir / f".{trace_id}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(_canonical_json(trace) + b"\n")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self._finished = True
        self.trace_path = target
        return target
