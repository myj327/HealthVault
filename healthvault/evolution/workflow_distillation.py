"""TASK-023 deterministic workflow distillation and zero-API preflight."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from healthvault.evaluation.skill.benchmark.task_022 import (
    KEY_ENV,
    MAX_OUTPUT_TOKENS,
    MAX_TRANSPORT_RETRIES,
    JsonTransport,
    TransportError,
    UrllibJsonTransport,
    build_preflight as build_task022_preflight,
    execute_real_run,
    resolve_model_config,
)
from healthvault.evolution.candidates import (
    CandidateGenerator,
    EvolutionSignal,
    SkillCandidate,
    TaskEndProcessor,
)
from healthvault.evolution.synthetic_demo import SKILL_V2
from healthvault.evolution.verifier import (
    REQUIRED_SECTIONS,
    CandidateVerifier,
    VerificationResult,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_PATH = ROOT / "healthvault" / "evaluation" / "skill" / "task-023-dev-traces.json"
DEFAULT_PREFLIGHT_PATH = ROOT / "healthvault" / "evaluation" / "skill" / "task-023-preflight.json"
DEFAULT_RUNTIME_ROOT = ROOT / ".task_outputs" / "TASK-023"
TASK_022_PREFLIGHT_PATH = ROOT / "healthvault" / "evaluation" / "skill" / "task-022-preflight.json"
ACTIVE_SKILL_PATH = ROOT / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"
REPEAT_THRESHOLD = 3
CANDIDATE_MODEL = "glm-5.3"
CANDIDATE_MODEL_ENV = "HEALTHVAULT_CANDIDATE_MODEL"
CANDIDATE_REASONING_EFFORT = "high"
CANDIDATE_MAX_OUTPUT_TOKENS = 4096
CANDIDATE_TIMEOUT_SECONDS = 120.0

TRACE_KEYS = frozenset({
    "trace_id",
    "task_id",
    "task_type",
    "signal_type",
    "tool_calls",
    "task_status",
    "validator_status",
    "hard_failure",
    "reusable_lesson",
    "contains_patient_or_medical_fact",
})
TOOL_CALL_KEYS = frozenset({"name", "status"})
SIGNAL_TYPES = frozenset({
    "successful_workflow",
    "explicit_correction",
    "deterministic_failure",
    "patient_fact_noise",
})
TOOL_STATUSES = frozenset({"ok", "rejected", "retryable_failure"})

FAKE_CANDIDATE_CONTENT = SKILL_V2.replace(
    "2. Run only approved read-only domain tools and preserve returned citations.",
    "2. Reuse a repeated validator-passed read-only tool sequence only when every tool status is verified.\n"
    "3. Run only approved read-only domain tools and preserve returned citations.",
).replace(
    "3. Return the tool status accurately; never present a rejection as success.",
    "4. Return the tool status accurately; never present a rejection as success.",
)


class WorkflowTraceError(ValueError):
    """Raised when the frozen synthetic dev Trace artifact is malformed."""


class ApiGateError(RuntimeError):
    """Raised before any real candidate or benchmark transport can be used."""


def resolve_candidate_model_config(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve a reproducible candidate model independently of the tested Agent."""
    values = os.environ if environ is None else environ
    execution_config = resolve_model_config(values)
    model = str(values.get(CANDIDATE_MODEL_ENV, "") or CANDIDATE_MODEL).strip()
    if not model:
        raise ValueError("candidate model must be non-empty")
    return {
        **execution_config,
        "model": model,
        "model_environment": CANDIDATE_MODEL_ENV,
        "do_sample": False,
        "temperature": 0.0,
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": CANDIDATE_REASONING_EFFORT,
        "max_output_tokens": CANDIDATE_MAX_OUTPUT_TOKENS,
        "timeout_seconds": CANDIDATE_TIMEOUT_SECONDS,
    }


@dataclass(frozen=True)
class DistillationResult:
    dataset_version: str
    trace_count: int
    workflow_signals: tuple[EvolutionSignal, ...]
    recovery_signals: tuple[EvolutionSignal, ...]
    one_off_workflow_count: int
    duplicate_success_count: int
    excluded_patient_fact_count: int
    excluded_hard_failure_count: int
    excluded_validator_failure_count: int

    @property
    def candidate_signals(self) -> tuple[EvolutionSignal, ...]:
        return self.workflow_signals + self.recovery_signals


class CandidateModelSideQuery:
    """One logical candidate call using the shared TASK-022 transport contract."""

    def __init__(
        self,
        *,
        transport: JsonTransport,
        api_key: str,
        model_config: Mapping[str, Any],
        active_skill_content: str,
        candidate_context: Mapping[str, Any] | None = None,
        content_contract: Mapping[str, Any] | None = None,
    ) -> None:
        self.transport = transport
        self.api_key = api_key
        self.model_config = dict(model_config)
        self.active_skill_content = active_skill_content
        self.candidate_context = dict(candidate_context or {})
        self.content_contract = dict(content_contract or {})
        self.transport_requests = 0
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.latency_ms = 0.0

    async def __call__(self, prompt: str, schema: dict[str, Any]) -> str:
        request = json.loads(prompt)
        if not isinstance(request, dict):
            raise ValueError("candidate request must be a JSON object")
        request["active_skill_content"] = self.active_skill_content
        request["candidate_context"] = self.candidate_context
        contract = {
            "kind": "complete_replacement_skill_markdown",
            "required_frontmatter": {
                "name": "HealthVault-Task-Execution",
                "description": "non-empty string",
            },
            "required_h2_sections": list(REQUIRED_SECTIONS),
            "must_preserve_safe_existing_behavior": True,
            "must_add_a_material_reusable_increment": True,
            "summary_or_patch_only_is_invalid": True,
        }
        contract.update(self.content_contract)
        request["content_contract"] = contract
        payload = {
            "model": self.model_config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object matching this schema. "
                        "The content field must contain a complete replacement SKILL.md, "
                        "not a summary, prose description, or patch. Preserve the exact "
                        "target name, provide a non-empty description, and include every "
                        "required H2 section from content_contract. "
                        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False, sort_keys=True),
                },
            ],
            "response_format": {"type": "json_object"},
            "do_sample": self.model_config.get("do_sample", True),
            "thinking": copy.deepcopy(self.model_config["thinking"]),
            "reasoning_effort": self.model_config["reasoning_effort"],
            "max_tokens": self.model_config["max_output_tokens"],
            "stream": False,
        }
        if payload["do_sample"]:
            payload["temperature"] = self.model_config["temperature"]
        last_error: Exception | None = None
        for retry in range(MAX_TRANSPORT_RETRIES + 1):
            started = time.perf_counter()
            self.transport_requests += 1
            if self.transport.is_real:
                self.api_calls += 1
            try:
                response = await self.transport.post_json(
                    f"{self.model_config['api_base'].rstrip('/')}/chat/completions",
                    {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    payload,
                    float(self.model_config.get("timeout_seconds", 60.0)),
                )
            except TransportError as exc:
                self.latency_ms += (time.perf_counter() - started) * 1000
                last_error = exc
                if not exc.retryable or retry >= MAX_TRANSPORT_RETRIES:
                    raise
                continue
            self.latency_ms += (time.perf_counter() - started) * 1000
            usage = response.get("usage") if isinstance(response, dict) else None
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
                if (
                    isinstance(prompt_tokens, int)
                    and isinstance(completion_tokens, int)
                    and isinstance(total_tokens, int)
                    and min(prompt_tokens, completion_tokens, total_tokens) >= 0
                    and total_tokens == prompt_tokens + completion_tokens
                ):
                    self.input_tokens += prompt_tokens
                    self.output_tokens += completion_tokens
                    self.total_tokens += total_tokens
            choices = response.get("choices") if isinstance(response, dict) else None
            message = choices[0].get("message") if isinstance(choices, list) and choices else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("candidate model response has no JSON content")
            return content
        raise last_error or RuntimeError("candidate transport failed")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
            + b"\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_dev_traces(path: str | Path = DEFAULT_TRACE_PATH) -> dict[str, Any]:
    """Load the exact synthetic schema without consulting TASK-021 held-out data."""
    target = Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "dataset_version", "traces"
    }:
        raise WorkflowTraceError("dev Trace artifact does not match the exact schema")
    if value["schema_version"] != "healthvault-workflow-traces-v1":
        raise WorkflowTraceError("unsupported dev Trace schema")
    if not isinstance(value["dataset_version"], str) or not value["dataset_version"].strip():
        raise WorkflowTraceError("dev Trace dataset_version must be non-empty")
    traces = value["traces"]
    if not isinstance(traces, list) or len(traces) != 10:
        raise WorkflowTraceError("TASK-023 requires exactly 10 synthetic dev Traces")

    seen_trace_ids: set[str] = set()
    for trace in traces:
        if not isinstance(trace, dict) or set(trace) != TRACE_KEYS:
            raise WorkflowTraceError("dev Trace does not match the exact schema")
        for key in ("trace_id", "task_id", "task_type", "reusable_lesson"):
            if not isinstance(trace[key], str) or not trace[key].strip():
                raise WorkflowTraceError(f"dev Trace {key} must be non-empty")
        if trace["trace_id"] in seen_trace_ids:
            raise WorkflowTraceError("dev trace_id values must be unique")
        seen_trace_ids.add(trace["trace_id"])
        if trace["signal_type"] not in SIGNAL_TYPES:
            raise WorkflowTraceError("dev Trace signal_type is invalid")
        if trace["task_status"] not in {"succeeded", "failed", "rejected"}:
            raise WorkflowTraceError("dev Trace task_status is invalid")
        if trace["validator_status"] not in {"passed", "failed"}:
            raise WorkflowTraceError("dev Trace validator_status is invalid")
        for key in ("hard_failure", "contains_patient_or_medical_fact"):
            if not isinstance(trace[key], bool):
                raise WorkflowTraceError(f"dev Trace {key} must be boolean")
        calls = trace["tool_calls"]
        if not isinstance(calls, list) or not calls:
            raise WorkflowTraceError("dev Trace tool_calls must be non-empty")
        for call in calls:
            if not isinstance(call, dict) or set(call) != TOOL_CALL_KEYS:
                raise WorkflowTraceError("dev Trace tool call does not match exact schema")
            if not isinstance(call["name"], str) or not call["name"].strip():
                raise WorkflowTraceError("dev Trace tool name must be non-empty")
            if call["status"] not in TOOL_STATUSES:
                raise WorkflowTraceError("dev Trace tool status is invalid")
    return value


def normalize_tool_workflow(tool_calls: Iterable[Mapping[str, str]]) -> str:
    """Retain only normalized tool names and statuses."""
    return ">".join(
        f"{str(item['name']).strip().lower()}:{str(item['status']).strip().lower()}"
        for item in tool_calls
    )


def distill_workflows(dataset: Mapping[str, Any]) -> DistillationResult:
    traces = list(dataset["traces"])
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    duplicate_success_count = 0
    excluded_patient = 0
    excluded_hard = 0
    excluded_validator = 0

    for trace in traces:
        if trace["contains_patient_or_medical_fact"]:
            excluded_patient += 1
            continue
        if trace["hard_failure"]:
            excluded_hard += 1
            continue
        if trace["validator_status"] != "passed":
            excluded_validator += 1
            continue
        if trace["signal_type"] != "successful_workflow" or trace["task_status"] != "succeeded":
            continue
        workflow = normalize_tool_workflow(trace["tool_calls"])
        tasks = grouped.setdefault(workflow, {})
        if trace["task_id"] in tasks:
            duplicate_success_count += 1
            continue
        tasks[trace["task_id"]] = trace

    workflow_signals: list[EvolutionSignal] = []
    one_off_workflow_count = 0
    for workflow, task_map in sorted(grouped.items()):
        supporting = sorted(task_map.values(), key=lambda item: item["trace_id"])
        if len(supporting) < REPEAT_THRESHOLD:
            one_off_workflow_count += len(supporting)
            continue
        trace_ids = tuple(item["trace_id"] for item in supporting)
        aggregate_id = "WORKFLOW-" + _sha256_bytes(workflow.encode("utf-8"))[:12]
        steps = " -> ".join(
            f"{part.split(':', 1)[0]} (status={part.split(':', 1)[1]})"
            for part in workflow.split(">")
        )
        workflow_signals.append(EvolutionSignal(
            trace_id=aggregate_id,
            signal_type="verified_reusable_workflow",
            lesson=f"For repeated validator-passed tasks, use this read-only workflow: {steps}.",
            reusable=True,
            source_trace_ids=trace_ids,
        ))

    recovery_signals = tuple(
        EvolutionSignal(
            trace_id=trace["trace_id"],
            signal_type=trace["signal_type"],
            lesson=trace["reusable_lesson"],
            reusable=True,
        )
        for trace in traces
        if trace["signal_type"] in {"explicit_correction", "deterministic_failure"}
        and trace["task_status"] == "succeeded"
        and trace["validator_status"] == "passed"
        and not trace["hard_failure"]
        and not trace["contains_patient_or_medical_fact"]
    )
    return DistillationResult(
        dataset_version=str(dataset["dataset_version"]),
        trace_count=len(traces),
        workflow_signals=tuple(workflow_signals),
        recovery_signals=recovery_signals,
        one_off_workflow_count=one_off_workflow_count,
        duplicate_success_count=duplicate_success_count,
        excluded_patient_fact_count=excluded_patient,
        excluded_hard_failure_count=excluded_hard,
        excluded_validator_failure_count=excluded_validator,
    )


def build_candidate_artifact(
    candidate: SkillCandidate,
    verification: VerificationResult,
    *,
    model_config: Mapping[str, Any],
    trace_artifact_sha256: str,
    generation_mode: str,
    preflight_sha256: str | None = None,
    credential_binding_sha256: str | None = None,
    approved: bool = False,
    freeze: bool = False,
) -> dict[str, Any]:
    """Bind a candidate to its evidence, model configuration, and content hash."""
    payload: dict[str, Any] = {
        "schema_version": "healthvault-task-023-candidate-v1",
        "generation_mode": generation_mode,
        "candidate_id": candidate.candidate_id,
        "action": candidate.action,
        "skill_name": candidate.skill_name,
        "status": "frozen" if freeze and verification.status == "verified" else candidate.status,
        "content": candidate.content,
        "trace_ids": list(candidate.trace_ids),
        "rationale": candidate.rationale,
        "candidate_content_sha256": _sha256_bytes(candidate.content.encode("utf-8")),
        "verification": {
            "status": verification.status,
            "reasons": list(verification.reasons),
            "content_sha256": verification.content_sha256,
        },
        "model_config": dict(model_config),
        "model_config_sha256": _sha256_bytes(_canonical_json(dict(model_config))),
        "trace_artifact_sha256": trace_artifact_sha256,
        "preflight_sha256": preflight_sha256,
        "credential_binding_sha256": credential_binding_sha256,
        "approval": {"execute": approved, "approved": approved},
        "published": False,
    }
    payload["artifact_sha256"] = _sha256_bytes(_canonical_json(payload))
    return payload


async def generate_fake_candidate(
    result: DistillationResult,
    *,
    evolution_root: str | Path,
    active_content: str,
    model_config: Mapping[str, Any],
    trace_artifact_sha256: str,
) -> tuple[SkillCandidate, VerificationResult, dict[str, Any], int]:
    """Exercise the existing candidate and Verifier chain without a model call."""

    async def fake_side_query(prompt: str, _schema: dict[str, Any]) -> dict[str, Any]:
        request = json.loads(prompt)
        trace_ids = list(dict.fromkeys(
            trace_id
            for evidence in request["evidence"]
            for trace_id in evidence["trace_ids"]
        ))
        return {
            "action": "evolve",
            "skill_name": "HealthVault-Task-Execution",
            "content": FAKE_CANDIDATE_CONTENT,
            "rationale": "Reuse repeated verified tool statuses and retain deterministic recovery rules.",
            "trace_ids": trace_ids,
        }

    generator = CandidateGenerator(evolution_root, fake_side_query)
    processor = TaskEndProcessor(generator)
    for signal in result.candidate_signals:
        if processor.record(signal) != "eligible":
            raise RuntimeError("distilled signal was rejected by TaskEndProcessor")
    candidate = await processor.process("evolve")
    if candidate is None:
        raise RuntimeError("fake candidate batch was empty")
    verification = CandidateVerifier(candidate.skill_name).verify_content(
        candidate.content,
        action=candidate.action,
        active_content=active_content,
    )
    artifact = build_candidate_artifact(
        candidate,
        verification,
        model_config=model_config,
        trace_artifact_sha256=trace_artifact_sha256,
        generation_mode="fake_side_query",
    )
    return candidate, verification, artifact, generator.call_count


def _dependency_hashes(root: Path) -> dict[str, str]:
    paths = {
        "task_019_summary": root / "healthvault/evaluation/skill/task-019-release-summary.json",
        "task_020_summary": root / "healthvault/evaluation/skill/task-020-lifecycle-summary.json",
        "task_021_manifest": root / "healthvault/evaluation/skill/task-021-run-manifest.json",
        "task_022_preflight": root / "healthvault/evaluation/skill/task-022-preflight.json",
        "task_022_runner": root / "healthvault/evaluation/skill/benchmark/task_022.py",
        "candidate_generator": root / "healthvault/evolution/candidates.py",
        "usage_lifecycle": root / "healthvault/evolution/usage_lifecycle.py",
        "workflow_distillation": root / "healthvault/evolution/workflow_distillation.py",
        "candidate_verifier": root / "healthvault/evolution/verifier.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"TASK-023 dependency missing: {', '.join(missing)}")
    return {name: _sha256_bytes(path.read_bytes()) for name, path in paths.items()}


def _preflight_payload(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in preflight.items()
        if key not in {"run_config_sha256", "observed"}
    }


def _credential_binding(api_key: str) -> str | None:
    if not api_key.strip():
        return None
    return _sha256_bytes(("healthvault-task-023-key-v1\0" + api_key).encode("utf-8"))


def build_preflight(
    *,
    workspace_root: str | Path = ROOT,
    trace_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    traces_path = Path(trace_path) if trace_path is not None else (
        root / DEFAULT_TRACE_PATH.relative_to(ROOT)
    )
    dataset = load_dev_traces(traces_path)
    result = distill_workflows(dataset)
    trace_hash = _sha256_bytes(traces_path.read_bytes())
    values = os.environ if environ is None else environ
    api_key = str(values.get(KEY_ENV, "") or "").strip()
    benchmark_model_config = resolve_model_config(values)
    candidate_model_config = resolve_candidate_model_config(values)
    task022 = json.loads(
        (root / TASK_022_PREFLIGHT_PATH.relative_to(ROOT)).read_text(encoding="utf-8")
    )
    expected_model = {
        key: value
        for key, value in task022["model_config"].items()
        if key != "key_configured"
    }
    actual_model = {
        key: value for key, value in benchmark_model_config.items() if key != "key_configured"
    }
    if actual_model != expected_model:
        raise ValueError("TASK-023 model configuration no longer matches TASK-022")
    active_path = root / ACTIVE_SKILL_PATH.relative_to(ROOT)
    active_content = active_path.read_text(encoding="utf-8")
    active_before = _sha256_bytes(active_path.read_bytes())
    with tempfile.TemporaryDirectory(prefix="healthvault-task-023-fake-") as temporary:
        candidate, verification, artifact, side_query_calls = asyncio.run(
            generate_fake_candidate(
                result,
                evolution_root=Path(temporary),
                active_content=active_content,
                model_config=candidate_model_config,
                trace_artifact_sha256=trace_hash,
            )
        )
    active_after = _sha256_bytes(active_path.read_bytes())
    if active_before != active_after:
        raise RuntimeError("TASK-023 preflight changed the active Skill")

    workflow_support = [
        list(signal.source_trace_ids) for signal in result.workflow_signals
    ]
    preflight: dict[str, Any] = {
        "schema_version": "healthvault-task-023-preflight-v1",
        "mode": "offline_preflight",
        "claim_status": "not_run",
        "dev_trace": {
            "dataset_version": result.dataset_version,
            "artifact_sha256": trace_hash,
            "trace_count": result.trace_count,
            "heldout_content_read": False,
            "repeat_threshold": REPEAT_THRESHOLD,
            "verified_reusable_workflows": len(result.workflow_signals),
            "workflow_support_trace_ids": workflow_support,
            "recovery_signals": len(result.recovery_signals),
            "one_off_workflows": result.one_off_workflow_count,
            "duplicate_successes": result.duplicate_success_count,
            "excluded_patient_fact": result.excluded_patient_fact_count,
            "excluded_hard_failure": result.excluded_hard_failure_count,
            "excluded_validator_failure": result.excluded_validator_failure_count,
        },
        "fake_candidate": {
            "candidate_id": candidate.candidate_id,
            "candidate_content_sha256": artifact["candidate_content_sha256"],
            "artifact_sha256": artifact["artifact_sha256"],
            "trace_ids": list(candidate.trace_ids),
            "verification": verification.status,
            "side_query_calls": side_query_calls,
            "published": False,
        },
        "model_config": benchmark_model_config,
        "model_config_sha256": _sha256_bytes(_canonical_json(benchmark_model_config)),
        "candidate_model_config": candidate_model_config,
        "candidate_model_config_sha256": _sha256_bytes(
            _canonical_json(candidate_model_config)
        ),
        "credential_binding_sha256": _credential_binding(api_key),
        "frozen_inputs": {
            "active_skill_sha256": active_before,
            "task_021_binding_source": "task-022-preflight only; held-out content not loaded",
            "task_021_dataset_semantic_sha256": task022["frozen_inputs"]["dataset_semantic_sha256"],
            "task_021_validator_sha256": task022["frozen_inputs"]["validator_sha256"],
            "task_022_run_config_sha256": task022["run_config_sha256"],
        },
        "dependency_sha256": _dependency_hashes(root),
        "plan": {
            "candidate_generation_model_calls": 1,
            "candidate_generation_max_transport_attempts": 1 + MAX_TRANSPORT_RETRIES,
            "candidate_generation_max_output_tokens": candidate_model_config[
                "max_output_tokens"
            ],
            "benchmark_cases": 20,
            "benchmark_arms": ["no_skill", "active_v1", "generated_candidate"],
            "benchmark_arm_cases": 60,
            "primary_comparison": "active_v1_to_generated_candidate",
            "reference_comparison": "no_skill_to_active_v1",
        },
        "api_gate": {
            "status": "stopped_awaiting_user_approval",
            "candidate_generation_approved": False,
            "real_benchmark_approved": False,
            "real_candidate_generated": False,
        },
        "observed": {
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
    preflight["run_config_sha256"] = _sha256_bytes(
        _canonical_json(_preflight_payload(preflight))
    )
    return preflight


def assert_api_gate(
    preflight: Mapping[str, Any],
    *,
    execute: bool,
    approved: bool,
    api_key: str,
    current_preflight: Mapping[str, Any],
) -> None:
    """Fail before transport creation unless approval and exact config both match."""
    if not execute:
        raise ApiGateError("TASK-023 real execution requires --execute")
    if not approved:
        raise ApiGateError("TASK-023 real execution requires renewed user approval")
    if not api_key.strip():
        raise ApiGateError(f"TASK-023 real execution requires {KEY_ENV}")
    if preflight.get("model_config", {}).get("key_configured") is not True:
        raise ApiGateError("TASK-023 preflight is not bound to a configured Key")
    if preflight.get("credential_binding_sha256") != _credential_binding(api_key):
        raise ApiGateError("TASK-023 Key binding mismatch")
    supplied_hash = preflight.get("run_config_sha256")
    if supplied_hash != _sha256_bytes(_canonical_json(_preflight_payload(preflight))):
        raise ApiGateError("TASK-023 preflight hash mismatch")
    if supplied_hash != current_preflight.get("run_config_sha256"):
        raise ApiGateError("TASK-023 configuration or dependency hash mismatch")


async def generate_approved_candidate(
    preflight: Mapping[str, Any],
    result: DistillationResult,
    *,
    api_key: str,
    transport: JsonTransport,
    runtime_root: str | Path,
    active_content: str,
) -> tuple[SkillCandidate, dict[str, Any], CandidateModelSideQuery]:
    """Generate and freeze exactly one candidate after the external Gate."""
    runtime = Path(runtime_root)
    side_query = CandidateModelSideQuery(
        transport=transport,
        api_key=api_key,
        model_config=preflight["candidate_model_config"],
        active_skill_content=active_content,
    )
    generator = CandidateGenerator(runtime / "evolution", side_query)
    processor = TaskEndProcessor(generator)
    for signal in result.candidate_signals:
        if processor.record(signal) != "eligible":
            raise RuntimeError("distilled signal was rejected before approved generation")
    candidate = await processor.process("evolve")
    if candidate is None or generator.call_count != 1:
        raise RuntimeError("TASK-023 must generate exactly one candidate")
    candidate_traces = set(candidate.trace_ids)
    if not any(
        set(signal.source_trace_ids).issubset(candidate_traces)
        for signal in result.workflow_signals
    ):
        raise ValueError(
            "candidate must cite every supporting Trace of a verified reusable workflow"
        )
    verification = CandidateVerifier(candidate.skill_name).verify_content(
        candidate.content,
        action=candidate.action,
        active_content=active_content,
    )
    if verification.status != "verified":
        raise ValueError(f"candidate verification failed: {verification.reasons}")
    artifact = build_candidate_artifact(
        candidate,
        verification,
        model_config=preflight["candidate_model_config"],
        trace_artifact_sha256=preflight["dev_trace"]["artifact_sha256"],
        generation_mode="approved_model_path",
        preflight_sha256=str(preflight["run_config_sha256"]),
        credential_binding_sha256=str(preflight["credential_binding_sha256"]),
        approved=True,
        freeze=True,
    )
    _atomic_json(runtime / "task-023-frozen-candidate.json", artifact)
    return candidate, artifact, side_query


def _benchmark_preflight(
    source_preflight: Mapping[str, Any],
    candidate_artifact: Mapping[str, Any],
    *,
    workspace_root: Path,
    environ: Mapping[str, str],
    case_order: list[str] | None,
) -> dict[str, Any]:
    benchmark = build_task022_preflight(
        workspace_root=workspace_root,
        environ=environ,
    )
    if benchmark["model_config"] != source_preflight["model_config"]:
        raise ApiGateError("benchmark model configuration changed after preflight")
    arms = [dict(item) for item in benchmark["arms"][:2]]
    arms.append({
        "arm_id": "generated_candidate",
        "skill_source": "task-023-frozen-candidate",
        "skill_sha256": candidate_artifact["candidate_content_sha256"],
        "content_sha256": candidate_artifact["candidate_content_sha256"],
    })
    selected_cases = list(case_order or benchmark["frozen_inputs"]["case_order"])
    benchmark["schema_version"] = "healthvault-task-023-benchmark-preflight-v1"
    benchmark["benchmark_version"] = "skill-workflow-distillation-r1-v1"
    benchmark["arms"] = arms
    benchmark["frozen_inputs"]["candidate_skill_sha256"] = candidate_artifact[
        "candidate_content_sha256"
    ]
    benchmark["frozen_inputs"]["case_order"] = selected_cases
    benchmark["frozen_inputs"]["task_023_candidate_artifact_sha256"] = candidate_artifact[
        "artifact_sha256"
    ]
    benchmark["frozen_inputs"]["task_023_preflight_sha256"] = source_preflight[
        "run_config_sha256"
    ]
    benchmark["plan"] = {
        "cases": len(selected_cases),
        "arms": 3,
        "arm_cases": len(selected_cases) * 3,
        "execution_passes": 1,
    }
    benchmark["limits"]["max_model_calls"] = (
        len(selected_cases) * 3 * benchmark["limits"]["max_model_turns_per_case"]
    )
    benchmark["limits"]["max_transport_attempts"] = (
        benchmark["limits"]["max_model_calls"]
        * (benchmark["limits"]["transport_retries"] + 1)
    )
    benchmark["run_config_sha256"] = _sha256_bytes(
        _canonical_json(_preflight_payload(benchmark))
    )
    return benchmark


async def execute_approved_run(
    preflight: Mapping[str, Any],
    *,
    execute: bool,
    approved: bool,
    api_key: str,
    current_preflight: Mapping[str, Any],
    runtime_root: str | Path,
    workspace_root: str | Path = ROOT,
    candidate_transport: JsonTransport | None = None,
    benchmark_transport: JsonTransport | None = None,
    output_dir: str | Path | None = None,
    case_order: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run approved candidate generation then inject it into the shared runner."""
    assert_api_gate(
        preflight,
        execute=execute,
        approved=approved,
        api_key=api_key,
        current_preflight=current_preflight,
    )
    root = Path(workspace_root).resolve()
    runtime = Path(runtime_root)
    runtime.mkdir(parents=True, exist_ok=True)
    if (runtime / "task-023-frozen-candidate.json").exists():
        raise RuntimeError("TASK-023 candidate is already frozen; refusing repeated generation")
    values = dict(os.environ if environ is None else environ)
    values[KEY_ENV] = api_key
    active_path = root / ACTIVE_SKILL_PATH.relative_to(ROOT)
    active_before = _sha256_bytes(active_path.read_bytes())
    if active_before != preflight["frozen_inputs"]["active_skill_sha256"]:
        raise ApiGateError("active Skill changed after preflight")
    dataset = load_dev_traces(root / DEFAULT_TRACE_PATH.relative_to(ROOT))
    result = distill_workflows(dataset)
    candidate, artifact, candidate_query = await generate_approved_candidate(
        preflight,
        result,
        api_key=api_key,
        transport=candidate_transport or UrllibJsonTransport(),
        runtime_root=runtime,
        active_content=active_path.read_text(encoding="utf-8"),
    )
    benchmark_preflight = _benchmark_preflight(
        preflight,
        artifact,
        workspace_root=root,
        environ=values,
        case_order=case_order,
    )
    _atomic_json(runtime / "task-023-benchmark-preflight.json", benchmark_preflight)
    _, summary = await execute_real_run(
        benchmark_preflight,
        api_key=api_key,
        runtime_root=runtime / "benchmark",
        workspace_root=root,
        transport=benchmark_transport or UrllibJsonTransport(),
        arm_skill_contents={"generated_candidate": candidate.content},
        candidate_arm_id="generated_candidate",
        output_prefix="task-023",
        output_dir=output_dir,
        case_order=list(case_order) if case_order is not None else None,
    )
    comparison = summary["active_v1_to_generated_candidate"]
    publish_eligible = (
        comparison["task_success_delta_pp"] > 0
        and comparison["hard_failure_delta_pp"] <= 0
        and comparison["regressed_case_count"] == 0
    )
    summary["candidate_generation"] = {
        "candidate_id": candidate.candidate_id,
        "candidate_content_sha256": artifact["candidate_content_sha256"],
        "candidate_artifact_sha256": artifact["artifact_sha256"],
        "trace_ids": list(candidate.trace_ids),
        "logical_calls": 1,
        "transport_attempts": candidate_query.transport_requests,
        "api_calls": candidate_query.api_calls,
        "input_tokens": candidate_query.input_tokens,
        "output_tokens": candidate_query.output_tokens,
        "total_tokens": candidate_query.total_tokens,
        "latency_ms": round(candidate_query.latency_ms, 3),
        "published": False,
    }
    summary["candidate_decision"] = (
        "publish_eligible" if publish_eligible else "reject/archive_candidate"
    )
    summary["api_calls"] += candidate_query.api_calls
    summary["input_tokens"] += candidate_query.input_tokens
    summary["output_tokens"] += candidate_query.output_tokens
    summary["total_tokens"] += candidate_query.total_tokens
    summary["active_skill_sha256_before"] = active_before
    summary["active_skill_sha256_after"] = _sha256_bytes(active_path.read_bytes())
    if summary["active_skill_sha256_after"] != active_before:
        raise RuntimeError("TASK-023 execution changed the active Skill")
    result_dir = Path(output_dir) if output_dir is not None else (
        root / "healthvault" / "evaluation" / "skill"
    )
    _atomic_json(result_dir / "task-023-real-summary.json", summary)
    _atomic_json(runtime / "task-023-candidate-assessment.json", {
        "candidate_artifact_sha256": artifact["artifact_sha256"],
        "decision": summary["candidate_decision"],
        "active_skill_unchanged": True,
        "published": False,
    })
    return artifact, summary


def write_preflight(
    path: str | Path = DEFAULT_PREFLIGHT_PATH,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    preflight = build_preflight(environ=environ)
    _atomic_json(Path(path), preflight)
    return preflight


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TASK-023 workflow distillation offline preflight"
    )
    parser.add_argument("--preflight-output", default=str(DEFAULT_PREFLIGHT_PATH))
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.execute:
        preflight_path = Path(arguments.preflight_output)
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        current_preflight = build_preflight()
        api_key = os.environ.get(KEY_ENV, "").strip()
        _, summary = asyncio.run(execute_approved_run(
            preflight,
            execute=True,
            approved=arguments.approved,
            api_key=api_key,
            current_preflight=current_preflight,
            runtime_root=arguments.runtime_dir,
        ))
        print(json.dumps({
            "status": "completed_once",
            "candidate_decision": summary["candidate_decision"],
            "api_calls": summary["api_calls"],
            "input_tokens": summary["input_tokens"],
            "output_tokens": summary["output_tokens"],
            "total_tokens": summary["total_tokens"],
        }, ensure_ascii=False, sort_keys=True))
        return 0
    preflight = write_preflight(arguments.preflight_output)
    print(json.dumps({
        "status": "preflight_only",
        "dev_traces": preflight["dev_trace"]["trace_count"],
        "verified_reusable_workflows": preflight["dev_trace"]["verified_reusable_workflows"],
        "fake_candidate_verification": preflight["fake_candidate"]["verification"],
        "candidate_generation_model_calls_planned": preflight["plan"]["candidate_generation_model_calls"],
        "benchmark_arm_cases_planned": preflight["plan"]["benchmark_arm_cases"],
        "api_gate": preflight["api_gate"]["status"],
        "api_calls": 0,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
