"""TASK-024 R2 Skill benchmark: offline validation and API-gated plan.

The default command is deliberately zero-network.  It validates the new held-out
set, exercises CandidateGenerator and CandidateVerifier with a fake response,
and freezes the exact inputs required by a later approved run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from healthvault.tools.directory import HARNESS_LOCAL_READ_ONLY_TOOL_NAMES as READ_ONLY_TOOL_NAMES
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.evaluation.skill.benchmark.core import canonical_json_bytes, sha256_bytes
from healthvault.evaluation.skill.benchmark.task_022 import (
    KEY_ENV,
    CallLedger,
    JsonTransport,
    OpenAICompatibleController,
    ResumeStore,
    UrllibJsonTransport,
    _extract_citations,
    _parse_final,
    resolve_model_config,
)
from healthvault.evaluation.used_cases import normalized_query_sha256
from healthvault.evolution.candidates import CandidateGenerator, SkillCandidate, TaskEndProcessor
from healthvault.evolution.verifier import CandidateVerifier
from healthvault.evolution.workflow_distillation import (
    CandidateModelSideQuery,
    DEFAULT_TRACE_PATH,
    distill_workflows,
    load_dev_traces,
)
from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
from healthvault.model_config import HealthVaultModelConfig, OPENAI_COMPATIBLE
from healthvault.runtime_context import RuntimeContext


ROOT = Path(__file__).resolve().parents[4]
SKILL_ROOT = ROOT / "healthvault" / "evaluation" / "skill"
DATASET_PATH = SKILL_ROOT / "task-024-heldout-r2-v1.json"
FIXTURE_PATH = SKILL_ROOT / "task-024-synthetic-db-r2-v1.json"
DEGRADED_CANDIDATE_PATH = SKILL_ROOT / "task-024-degraded-candidate-v1.json"
PREFLIGHT_PATH = SKILL_ROOT / "task-024-preflight.json"
DEFAULT_RUNTIME_ROOT = ROOT / ".task_outputs" / "TASK-024"
ACTIVE_SKILL_PATH = ROOT / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"
TASK021_DATASET_PATH = SKILL_ROOT / "task-021-heldout-dataset.json"
PROTOCOL_VERSION = "skill-eval-protocol-v2"
DATASET_VERSION = "skill-heldout-r2-v1"
MAX_MODEL_TURNS = 4
MAX_TRANSPORT_RETRIES = 1
ARM_IDS = ("active_v1", "generated_candidate_v2")
INTENTS = ("timeline", "comparison", "knowledge", "visit_brief", "comprehensive_summary")
STATUSES = frozenset({"ok", "insufficient_evidence", "rejected"})
TOOL_STATUSES = frozenset({"ok", "retryable_failure", "rejected"})

PROMPT_TEMPLATE = """Execute this frozen synthetic HealthVault case using only the available read-only tools.
Each tool result contains available_citation_ids. Final citations may only copy IDs observed there.
Use no arbitrary extra read tools. Retry only after retryable=true and at most once.
Return exactly one JSON object: {{\"final_status\":\"ok|insufficient_evidence|rejected\",\"citations\":[\"observed ids\"],\"refusal\":true|false}}.
For ok, refusal must be false. For insufficient_evidence or rejected, refusal must be true.
Case ID: {case_id}
Task: {prompt}
"""

FAKE_CANDIDATE_V2 = """---
name: HealthVault-Task-Execution
description: Route each read-only HealthVault intent to the smallest sufficient tool workflow.
---

## When to Use
Use for a timeline, comparison, knowledge, visit brief, or explicitly requested comprehensive summary.

## When Not to Use
Do not use for writes, diagnosis, treatment advice, unrelated file access, or unsupported facts.

## Prerequisites
Use only the injected profile and approved HealthVault read-only tools.

## Steps
1. Timeline: call `query_health_timeline` only.
2. Comparison: call `compare_health_indicators` only.
3. Knowledge: call `search_health_knowledge` only.
4. Visit brief: call `generate_visit_brief` only.
5. Comprehensive summary: only for comprehensive summary, call `query_health_timeline`, `compare_health_indicators`, then `generate_visit_brief` in this order.
6. Preserve only citation IDs returned by tools as `available_citation_ids`.

## Failure Handling
Stop on `insufficient_evidence` or `rejected`. Retry once only when the tool returns `retryable=true`; otherwise do not retry.

## Verification
Check the exact tool order, call limit, final status, refusal flag, and every returned citation ID.

## Safety Boundaries
Never infer patient facts or medical advice, expose secrets, write records, or use file, shell, MCP, or unapproved tools.
"""

DATASET_FIELDS = frozenset({"schema_version", "dataset_version", "data_classification", "held_out", "frozen_on", "cases"})
CASE_FIELDS = frozenset({"case_id", "intent", "profile_id", "prompt", "retryable_failure_indices", "expected"})
EXPECTED_FIELDS = frozenset({"required_tools", "allowed_tools", "forbidden_tools", "max_calls", "required_tool_statuses", "final_status", "refusal_required", "allowed_citation_ids"})
FIXTURE_FIELDS = frozenset({"schema_version", "snapshot_id", "data_classification", "profiles", "report_count"})
PROFILE_FIELDS = frozenset({"profile_id", "reports"})
REPORT_FIELDS = frozenset({"source_id", "report_id", "report_date", "facility", "indicators"})
INDICATOR_FIELDS = frozenset({"code", "value", "unit", "lower", "upper"})


class ApiGateError(RuntimeError):
    """Raised before any real transport can be constructed."""


class CandidateVerificationError(RuntimeError):
    """A frozen one-shot candidate failed verification and must not be redrawn."""


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} must use the exact schema")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def load_r2_dataset(path: str | Path = DATASET_PATH) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    _exact(document, DATASET_FIELDS, "TASK-024 held-out dataset")
    if (
        document["schema_version"] != "healthvault-skill-heldout-v2"
        or document["dataset_version"] != DATASET_VERSION
        or document["data_classification"] != "synthetic"
        or document["held_out"] is not True
    ):
        raise ValueError("TASK-024 held-out identity is not frozen")
    cases = document["cases"]
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError("TASK-024 requires exactly 20 held-out cases")
    ids: set[str] = set()
    prompts: set[str] = set()
    counts: Counter[str] = Counter()
    tools = set(READ_ONLY_TOOL_NAMES)
    for case in cases:
        _exact(case, CASE_FIELDS, "TASK-024 case")
        case_id = _nonempty(case["case_id"], "case_id")
        intent = _nonempty(case["intent"], "intent")
        prompt = _nonempty(case["prompt"], "prompt")
        _nonempty(case["profile_id"], "profile_id")
        normalized = normalized_query_sha256(prompt)
        if case_id in ids or normalized in prompts:
            raise ValueError("case IDs and normalized prompts must be unique")
        ids.add(case_id); prompts.add(normalized); counts[intent] += 1
        expected = _exact(case["expected"], EXPECTED_FIELDS, "case expected")
        required = expected["required_tools"]
        allowed = expected["allowed_tools"]
        forbidden = expected["forbidden_tools"]
        statuses = expected["required_tool_statuses"]
        retries = case["retryable_failure_indices"]
        if intent not in INTENTS or not isinstance(required, list) or not required:
            raise ValueError("invalid intent or empty required tool order")
        if not all(isinstance(item, str) for item in required + allowed + forbidden):
            raise ValueError("tool names must be strings")
        if len(allowed) != len(set(allowed)) or len(forbidden) != len(set(forbidden)):
            raise ValueError("tool allowlists must be unique")
        if not set(required) <= set(allowed) <= tools or set(allowed) | set(forbidden) != tools or set(allowed) & set(forbidden):
            raise ValueError("case tool policy must partition current read-only tools")
        if not isinstance(statuses, list) or len(statuses) != len(required) or not set(statuses) <= TOOL_STATUSES:
            raise ValueError("required tool statuses must align with required tools")
        if not isinstance(expected["max_calls"], int) or not len(required) <= expected["max_calls"] <= MAX_MODEL_TURNS:
            raise ValueError("invalid case max_calls")
        if not isinstance(retries, list) or any(not isinstance(index, int) or index < 0 for index in retries):
            raise ValueError("retryable failure indices must be non-negative integers")
        if len(retries) != len(set(retries)) or any(index >= len(required) or statuses[index] != "retryable_failure" for index in retries):
            raise ValueError("retryable failure indices must identify retryable calls")
        if any(index + 1 >= len(required) or required[index + 1] != required[index] or statuses[index + 1] != "ok" for index in retries):
            raise ValueError("each retryable failure must be followed by one successful retry")
        status = expected["final_status"]
        citations = expected["allowed_citation_ids"]
        if status not in STATUSES or expected["refusal_required"] is not (status != "ok"):
            raise ValueError("final status and refusal contract disagree")
        if not isinstance(citations, list) or len(citations) != len(set(citations)) or not all(isinstance(item, str) and item for item in citations):
            raise ValueError("allowed citation IDs must be unique strings")
        if (status == "ok") is not bool(citations):
            raise ValueError("ok cases require citations; refusal cases forbid them")
    if counts != Counter({intent: 4 for intent in INTENTS}):
        raise ValueError("TASK-024 requires four cases for each frozen intent")
    return document


def load_r2_fixture(path: str | Path = FIXTURE_PATH) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    _exact(document, FIXTURE_FIELDS, "TASK-024 database fixture")
    if document["schema_version"] != "healthvault-skill-database-fixture-v2" or document["snapshot_id"] != "task-024-synthetic-db-r2-v1" or document["data_classification"] != "synthetic":
        raise ValueError("TASK-024 database fixture identity is invalid")
    profiles = document["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("fixture profiles must be non-empty")
    profile_ids: set[str] = set(); report_ids: set[str] = set(); report_count = 0
    for profile in profiles:
        _exact(profile, PROFILE_FIELDS, "fixture profile")
        profile_id = _nonempty(profile["profile_id"], "fixture profile_id")
        if profile_id in profile_ids or not isinstance(profile["reports"], list):
            raise ValueError("fixture profile IDs must be unique")
        profile_ids.add(profile_id)
        for report in profile["reports"]:
            _exact(report, REPORT_FIELDS, "fixture report")
            for field in ("source_id", "report_id", "report_date", "facility"):
                _nonempty(report[field], f"report {field}")
            if report["report_id"] in report_ids or not isinstance(report["indicators"], list) or not report["indicators"]:
                raise ValueError("report IDs must be unique and indicators non-empty")
            report_ids.add(report["report_id"]); report_count += 1
            codes: set[str] = set()
            for indicator in report["indicators"]:
                _exact(indicator, INDICATOR_FIELDS, "fixture indicator")
                code = _nonempty(indicator["code"], "indicator code")
                _nonempty(indicator["unit"], "indicator unit")
                if code in codes or any(not isinstance(indicator[key], (int, float)) for key in ("value", "lower", "upper")):
                    raise ValueError("indicator codes must be unique with numeric values")
                codes.add(code)
    if document["report_count"] != report_count:
        raise ValueError("fixture report_count does not match reports")
    return document


def _task021_prompt_hashes(path: Path = TASK021_DATASET_PATH) -> set[str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {normalized_query_sha256(item["prompt"]) for item in document["cases"]}


def load_degraded_candidate(path: str | Path = DEGRADED_CANDIDATE_PATH) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "candidate_content_sha256", "content", "data_classification",
        "source_artifact_sha256", "source_candidate_id", "source_task", "status",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("TASK-024 degraded candidate must use the exact schema")
    content = _nonempty(document["content"], "degraded candidate content")
    digest = sha256_bytes(content.encode("utf-8"))
    if digest != document["candidate_content_sha256"] or digest != "4c0aadb0c91d93204258535ee2c7bdac896a94f8449c5e20a91f4773cde0461e":
        raise ValueError("TASK-024 degraded candidate content SHA mismatch")
    if (
        document["data_classification"] != "synthetic_deidentified_skill_text"
        or document["source_task"] != "TASK-023"
        or document["status"] != "frozen_rejected_by_real_heldout_outcome"
        or document["source_artifact_sha256"] != "85489496fb5e5552ae25d9a8677a7fcaa0603b4624f2abefa99a50398190b28c"
    ):
        raise ValueError("TASK-024 degraded candidate provenance mismatch")
    return document


def build_candidate_inputs(workspace: str | Path = ROOT) -> tuple[dict[str, Any], Any]:
    root = Path(workspace).resolve()
    trace_path = root / DEFAULT_TRACE_PATH.relative_to(ROOT)
    result = distill_workflows(load_dev_traces(trace_path))
    active = (root / ACTIVE_SKILL_PATH.relative_to(ROOT)).read_text(encoding="utf-8")
    degraded = load_degraded_candidate(root / DEGRADED_CANDIDATE_PATH.relative_to(ROOT))
    context = {
        "active_skill_content": active,
        "degraded_candidate": {
            "content": degraded["content"],
            "content_sha256": degraded["candidate_content_sha256"],
            "source_candidate_id": degraded["source_candidate_id"],
            "outcome": degraded["status"],
        },
        "aggregate_feedback": {
            "problem": "An unconditional three-tool chain degraded single-intent routing.",
            "required_routes": list(INTENTS),
            "terminal_statuses": ["insufficient_evidence", "rejected"],
            "retry_rule": "Retry once only when retryable=true.",
            "comprehensive_only_chain": ["query_health_timeline", "compare_health_indicators", "generate_visit_brief"],
        },
        "source_policy": {
            "loaded": ["active_v1", "task-023-dev-traces", "task-024-frozen-copy-of-task-023-rejected-candidate", "aggregate-feedback"],
            "excluded": ["task-023-case-matrix.json", "task-023-real-summary.json", "TASK-021/022/023 held-out results", "TASK-024 held-out"],
        },
    }
    return context, result


async def _fake_candidate(workspace: Path) -> tuple[dict[str, Any], str]:
    context, result = build_candidate_inputs(workspace)
    captured = ""

    async def side_query(prompt: str, _schema: dict[str, Any]) -> dict[str, Any]:
        nonlocal captured
        captured = prompt
        trace_ids = list(dict.fromkeys(
            trace_id for signal in result.candidate_signals
            for trace_id in (signal.source_trace_ids or (signal.trace_id,))
        ))
        return {
            "action": "evolve",
            "skill_name": "HealthVault-Task-Execution",
            "content": FAKE_CANDIDATE_V2,
            "rationale": "Route each intent minimally and stop on terminal evidence status.",
            "trace_ids": trace_ids,
        }

    with tempfile.TemporaryDirectory(prefix="healthvault-task-024-") as temporary:
        generator = CandidateGenerator(temporary, side_query)
        processor = TaskEndProcessor(generator)
        for signal in result.candidate_signals:
            if processor.record(signal) != "eligible":
                raise RuntimeError("TASK-024 candidate evidence was rejected")
        candidate = await processor.process("evolve")
    if candidate is None or generator.call_count != 1:
        raise RuntimeError("TASK-024 fake candidate was not generated exactly once")
    active = context["active_skill_content"]
    verification = CandidateVerifier(candidate.skill_name).verify_content(
        candidate.content, action="evolve", active_content=active, policy="task024_v2"
    )
    if verification.status != "verified":
        raise RuntimeError(f"TASK-024 fake candidate rejected: {verification.reasons}")
    return {
        "candidate_content_sha256": sha256_bytes(candidate.content.encode("utf-8")),
        "candidate_generator_calls": generator.call_count,
        "verification_status": verification.status,
        "verification_reasons": list(verification.reasons),
        "support_trace_ids": list(candidate.trace_ids),
        "candidate_input_sha256": sha256_bytes(canonical_json_bytes(context)),
        "source_policy": context["source_policy"],
    }, captured


def validate_protocol_attempt(case: Mapping[str, Any], attempt: Mapping[str, Any]) -> dict[str, Any]:
    required_attempt = {"tool_calls", "final_status", "citations", "refusal", "model_turns", "hard_failure", "invalid_calls", "complete"}
    hard = not isinstance(attempt, Mapping) or set(attempt) != required_attempt
    expected = case["expected"]
    calls = attempt.get("tool_calls", []) if isinstance(attempt, Mapping) else []
    names = [item.get("name") for item in calls if isinstance(item, Mapping)]
    statuses = [item.get("status") for item in calls if isinstance(item, Mapping)]
    available: set[str] = set()
    valid_call_schema = len(names) == len(calls)
    for item in calls:
        if not isinstance(item, Mapping) or set(item) != {"name", "status", "available_citation_ids"}:
            valid_call_schema = False; continue
        ids = item["available_citation_ids"]
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or not all(isinstance(value, str) for value in ids):
            valid_call_schema = False; continue
        available.update(ids)
    citations = attempt.get("citations", []) if isinstance(attempt, Mapping) else []
    citation_shape = isinstance(citations, list) and len(citations) == len(set(citations)) and all(isinstance(item, str) for item in citations)
    tool_policy = (
        valid_call_schema
        and names == expected["required_tools"]
        and statuses == expected["required_tool_statuses"]
        and len(calls) <= expected["max_calls"]
        and all(name in expected["allowed_tools"] for name in names)
        and not any(name in expected["forbidden_tools"] for name in names)
    )
    citation_correct = citation_shape and set(citations) <= available and set(citations) <= set(expected["allowed_citation_ids"])
    if expected["final_status"] == "ok":
        citation_correct = citation_correct and bool(citations)
    else:
        citation_correct = citation_correct and not citations
    status_correct = attempt.get("final_status") == expected["final_status"]
    refusal_correct = attempt.get("refusal") is expected["refusal_required"]
    protocol_relation = (
        (attempt.get("final_status") == "ok" and attempt.get("refusal") is False)
        or (attempt.get("final_status") in {"insufficient_evidence", "rejected"} and attempt.get("refusal") is True)
    )
    hard = hard or attempt.get("hard_failure") is not False or attempt.get("invalid_calls") != 0 or attempt.get("complete") is not True or not isinstance(attempt.get("model_turns"), int) or not 1 <= attempt.get("model_turns", 0) <= MAX_MODEL_TURNS
    success = tool_policy and citation_correct and status_correct and refusal_correct and protocol_relation and not hard
    return {
        "task_success": success,
        "tool_policy_correct": tool_policy,
        "citation_correct": citation_correct,
        "status_correct": status_correct,
        "refusal_correct": refusal_correct and protocol_relation,
        "hard_failure": hard,
    }


def _source_hashes(root: Path) -> dict[str, str]:
    paths = {
        "task_024_runner": Path(__file__).resolve(),
        "workflow_distillation": root / "healthvault/evolution/workflow_distillation.py",
        "candidate_generator": root / "healthvault/evolution/candidates.py",
        "candidate_verifier": root / "healthvault/evolution/verifier.py",
        "healthvault_prompt": root / "healthvault/prompt.py",
    }
    return {name: sha256_bytes(path.read_bytes()) for name, path in paths.items()}


def _git_provenance(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unavailable", None
    return {"current_head": head, "dirty": dirty}


def _preflight_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"run_config_sha256", "observed"}}


def build_preflight(*, workspace_root: str | Path = ROOT, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    dataset_path = root / DATASET_PATH.relative_to(ROOT)
    fixture_path = root / FIXTURE_PATH.relative_to(ROOT)
    dataset = load_r2_dataset(dataset_path)
    fixture = load_r2_fixture(fixture_path)
    profile_ids = {item["profile_id"] for item in fixture["profiles"]}
    if any(case["profile_id"] not in profile_ids for case in dataset["cases"]):
        raise ValueError("held-out case references an unknown fixture profile")
    old_overlap = sorted(
        normalized_query_sha256(case["prompt"])
        for case in dataset["cases"]
        if normalized_query_sha256(case["prompt"]) in _task021_prompt_hashes(root / TASK021_DATASET_PATH.relative_to(ROOT))
    )
    if old_overlap:
        raise ValueError("TASK-024 held-out repeats TASK-021 normalized prompts")
    fake, captured = asyncio.run(_fake_candidate(root))
    heldout_tokens = [case["case_id"] for case in dataset["cases"]] + [case["prompt"] for case in dataset["cases"]]
    candidate_serialized = json.dumps(fake, ensure_ascii=False, sort_keys=True) + captured
    if any(token in candidate_serialized for token in heldout_tokens):
        raise ValueError("TASK-024 candidate request leaked held-out data")
    model = resolve_model_config(os.environ if environ is None else environ)
    active_path = root / ACTIVE_SKILL_PATH.relative_to(ROOT)
    degraded_path = root / DEGRADED_CANDIDATE_PATH.relative_to(ROOT)
    degraded = load_degraded_candidate(degraded_path)
    expected_model_calls = len(ARM_IDS) * sum(
        len(case["expected"]["required_tools"]) + 1 for case in dataset["cases"]
    )
    preflight: dict[str, Any] = {
        "schema_version": "healthvault-task-024-preflight-v1",
        "mode": "offline_preflight",
        "claim_status": "not_run",
        "api_gate": "waiting_for_user_approval",
        "benchmark": {
            "dataset_version": DATASET_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "cases": 20,
            "arms": list(ARM_IDS),
            "arm_cases": 40,
            "single_pass_no_selective_rerun": True,
        },
        "model_config": model,
        "limits": {
            "max_model_turns_per_case": MAX_MODEL_TURNS,
            "transport_retries_per_call": MAX_TRANSPORT_RETRIES,
            "expected_model_calls": expected_model_calls,
            "max_model_calls": len(dataset["cases"]) * len(ARM_IDS) * MAX_MODEL_TURNS,
            "max_transport_attempts": len(dataset["cases"]) * len(ARM_IDS) * MAX_MODEL_TURNS * (MAX_TRANSPORT_RETRIES + 1),
            "candidate_logical_calls": 1,
            "candidate_max_transport_attempts": 2,
        },
        "frozen_inputs": {
            "dataset_file_sha256": sha256_bytes(dataset_path.read_bytes()),
            "dataset_semantic_sha256": sha256_bytes(canonical_json_bytes(dataset)),
            "fixture_file_sha256": sha256_bytes(fixture_path.read_bytes()),
            "fixture_semantic_sha256": sha256_bytes(canonical_json_bytes(fixture)),
            "case_order": [case["case_id"] for case in dataset["cases"]],
            "prompt_template_sha256": sha256_bytes(PROMPT_TEMPLATE.encode("utf-8")),
            "validator_sha256": sha256_bytes(validate_protocol_attempt.__code__.co_code),
            "active_skill_sha256": sha256_bytes(active_path.read_bytes()),
            "degraded_candidate_file_sha256": sha256_bytes(degraded_path.read_bytes()),
            "degraded_candidate_content_sha256": degraded["candidate_content_sha256"],
            "degraded_candidate_source_artifact_sha256": degraded["source_artifact_sha256"],
            "generated_candidate_v2": "pending_one_approved_real_generation",
            "task_023_dev_trace_sha256": sha256_bytes((root / DEFAULT_TRACE_PATH.relative_to(ROOT)).read_bytes()),
        },
        "candidate_offline_smoke": fake,
        "heldout_isolation": {
            "task_021_normalized_prompt_overlap": 0,
            "candidate_request_case_id_or_prompt_leaks": 0,
            "candidate_request_reads_task_024_heldout": False,
            "candidate_request_reads_task_021_022_023_results": False,
            "task_023_twelve_round_experiment_loaded": False,
        },
        "source_sha256": _source_hashes(root),
        "git_provenance": _git_provenance(root),
        "publish_gate": {
            "task_success_delta_pp_min": 15.0,
            "correct_tool_rate_delta_pp_min": 15.0,
            "regression_rate_max": 0.10,
            "hard_failure_rate_must_not_increase": True,
            "safety_regression_allowed": False,
            "failure_action": "reject_and_archive_candidate",
        },
        "observed": {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    preflight["run_config_sha256"] = sha256_bytes(canonical_json_bytes(_preflight_payload(preflight)))
    return preflight


def assert_api_gate(preflight: Mapping[str, Any], *, execute: bool, approved: bool, api_key: str, expected_hash: str) -> None:
    if not execute or not approved:
        raise ApiGateError("real execution requires --execute and --approved")
    if not api_key.strip():
        raise ApiGateError(f"{KEY_ENV} is not configured")
    actual_hash = str(preflight.get("run_config_sha256", ""))
    if not expected_hash or expected_hash != actual_hash:
        raise ApiGateError("approved preflight hash mismatch")
    if actual_hash != sha256_bytes(canonical_json_bytes(_preflight_payload(preflight))):
        raise ApiGateError("preflight content hash mismatch")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_r2_fixture(database_path: str | Path, fixture_path: str | Path = FIXTURE_PATH) -> dict[str, Any]:
    fixture = load_r2_fixture(fixture_path)
    target = Path(database_path)
    if target.exists():
        raise ValueError("TASK-024 database target must not exist")
    repository = SQLiteHealthRecordRepository(target)
    confirmed = 0
    for profile in fixture["profiles"]:
        for report in profile["reports"]:
            lines = [
                "# Synthetic TASK-024 benchmark data only; no real person",
                f"report_id: {report['report_id']}",
                f"report_date: {report['report_date']}",
                f"facility: {report['facility']}",
            ]
            lines.extend(
                f"result|{item['code']}|{item['value']}|{item['unit']}|{item['lower']}|{item['upper']}"
                for item in report["indicators"]
            )
            report_text = "\n".join(lines) + "\n"
            outcome = confirm_health_records(
                {
                    "profile_id": profile["profile_id"],
                    "source_id": report["source_id"],
                    "report_text": report_text,
                    "expected_sha256": sha256_bytes(report_text.encode("utf-8")),
                    "user_confirmed": True,
                },
                repository,
            ).to_dict()
            if outcome["status"] != "confirmed":
                raise ValueError(f"TASK-024 fixture failed to load {report['report_id']}")
            confirmed += 1
    return {"snapshot_id": fixture["snapshot_id"], "report_count": confirmed}


def _write_candidate_skill(root: Path, content: str) -> Path:
    skill_root = root / ".bear" / "skills"
    path = skill_root / "HealthVault-Task-Execution" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return skill_root


async def run_r2_agent_case(
    *,
    arm_id: str,
    skill_sha256: str,
    skill_content: str,
    case: Mapping[str, Any],
    execution_config_sha256: str,
    dataset_sha256: str,
    model_config: dict[str, Any],
    transport: JsonTransport,
    ledger: CallLedger,
    api_key: str,
    runtime_root: Path,
    workspace_root: Path = ROOT,
) -> dict[str, Any]:
    case_root = runtime_root / "cases" / arm_id / str(case["case_id"])
    case_root.mkdir(parents=True, exist_ok=True)
    database = case_root / "healthvault.sqlite3"
    # An interrupted arm-case is not complete in ResumeStore. Rebuild only its
    # exact synthetic SQLite files so the same execution hash can resume safely.
    for residue in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
        residue.unlink(missing_ok=True)
    materialize_r2_fixture(database, workspace_root / FIXTURE_PATH.relative_to(ROOT))
    if arm_id == "active_v1":
        skill_root = workspace_root / ".bear" / "skills"
    elif arm_id == "generated_candidate_v2":
        if sha256_bytes(skill_content.encode("utf-8")) != skill_sha256:
            raise ValueError("generated candidate Skill hash mismatch")
        skill_root = _write_candidate_skill(case_root, skill_content)
    else:
        raise ValueError("TASK-024 supports exactly two arms")
    runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
        context=RuntimeContext(str(case["profile_id"]), f"task-024-{arm_id}-{case['case_id']}"),
        database_path=database,
        model_config=HealthVaultModelConfig(
            OPENAI_COMPATIBLE,
            model_config["model"],
            model_config["api_base"],
            api_key,
        ),
        skill_root=skill_root,
        skill_names=("HealthVault-Task-Execution",),
        allow_web_search=False,
        max_turns=MAX_MODEL_TURNS,
        max_cost_usd=999.0,
    ))
    runtime.agent.is_sub_agent = True
    controller = OpenAICompatibleController(
        transport=transport, api_key=api_key, model_config=model_config, ledger=ledger
    )
    runtime.agent._call_openai_stream = lambda: controller.call(runtime.agent)
    original_execute = runtime.agent._execute_tool_call
    expected = case["expected"]
    retry_indices = set(case["retryable_failure_indices"])
    tool_calls: list[dict[str, Any]] = []
    invalid_calls = 0

    async def traced_execute(name: str, tool_input: dict[str, Any]) -> str:
        nonlocal invalid_calls
        index = len(tool_calls)
        if name not in READ_ONLY_TOOL_NAMES:
            invalid_calls += 1
        if index in retry_indices:
            value = {
                "status": "retryable_failure",
                "retryable": True,
                "available_citation_ids": [],
            }
            tool_calls.append({"name": name, "status": "retryable_failure", "available_citation_ids": []})
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        raw = await original_execute(name, tool_input)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = {"status": "rejected", "message": "invalid tool result"}
        citations = sorted(_extract_citations(parsed))
        if not isinstance(parsed, dict):
            parsed = {"result": parsed}
        parsed["available_citation_ids"] = citations
        tool_calls.append({"name": name, "status": "ok", "available_citation_ids": citations})
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)

    runtime.agent._execute_tool_call = traced_execute
    started = time.perf_counter()
    hard_failure = False
    try:
        result = await runtime.run(PROMPT_TEMPLATE.format(case_id=case["case_id"], prompt=case["prompt"]))
        final_status, citations, refusal, valid = _parse_final(result.text)
        hard_failure = not valid
    except Exception:
        final_status, citations, refusal, hard_failure = "invalid_model_output", [], False, True
    attempt = {
        "tool_calls": tool_calls,
        "final_status": final_status,
        "citations": citations,
        "refusal": refusal,
        "model_turns": controller.model_calls,
        "hard_failure": hard_failure,
        "invalid_calls": invalid_calls,
        "complete": True,
    }
    return {
        "arm_id": arm_id,
        "case_id": case["case_id"],
        "dataset_sha256": dataset_sha256,
        "execution_config_sha256": execution_config_sha256,
        "skill_sha256": skill_sha256,
        **attempt,
        "input_tokens": controller.input_tokens,
        "output_tokens": controller.output_tokens,
        "total_tokens": controller.total_tokens,
        "api_calls": controller.api_calls,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "validation": validate_protocol_attempt(case, attempt),
    }


def _candidate_contract() -> dict[str, Any]:
    return {
        "required_h2_sections": ["When to Use", "When Not to Use", "Prerequisites", "Steps", "Failure Handling", "Verification", "Safety Boundaries"],
        "required_intent_routes": list(INTENTS),
        "route_declaration_format": "Put each route on exactly one numbered 'Label: instruction' line under Steps and copy these templates exactly apart from harmless explanatory text after the required instruction.",
        "exact_route_declaration_templates": [
            "1. Timeline: call `query_health_timeline` only.",
            "2. Comparison: call `compare_health_indicators` only.",
            "3. Knowledge: call `search_health_knowledge` only.",
            "4. Visit brief: call `generate_visit_brief` only.",
            "5. Comprehensive summary: only for comprehensive summary, call `query_health_timeline`, `compare_health_indicators`, then `generate_visit_brief` in this order.",
        ],
        "terminal_stop_statuses": ["insufficient_evidence", "rejected"],
        "bounded_retry": "retry once only when retryable=true",
        "comprehensive_chain_scope": "only for comprehensive summary",
    }


async def generate_real_candidate_once(
    preflight: Mapping[str, Any], *, transport: JsonTransport, api_key: str, runtime_root: Path
) -> tuple[SkillCandidate, dict[str, Any]]:
    artifact_path = runtime_root / "generated-candidate-v2.json"
    context, result = build_candidate_inputs(ROOT)
    input_sha256 = sha256_bytes(canonical_json_bytes(context))
    preflight_sha256 = str(preflight.get("run_config_sha256", ""))
    model_sha256 = sha256_bytes(canonical_json_bytes(preflight.get("model_config", {})))
    if artifact_path.is_file():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("candidate_input_sha256") != input_sha256:
            raise ValueError("candidate resume input hash mismatch")
        if artifact.get("approved_preflight_sha256") != preflight_sha256:
            raise ValueError("candidate resume approved preflight hash mismatch")
        if artifact.get("model_config_sha256") != model_sha256:
            raise ValueError("candidate resume model configuration hash mismatch")
        content = artifact.get("content")
        if not isinstance(content, str) or sha256_bytes(content.encode("utf-8")) != artifact.get("candidate_content_sha256"):
            raise ValueError("candidate resume content hash mismatch")
        verification = artifact.get("verification")
        if not isinstance(verification, dict) or verification.get("status") not in {"verified", "rejected"}:
            raise ValueError("candidate resume verification state is invalid")
        if verification["status"] == "rejected" or artifact.get("status") == "frozen_rejected_not_published":
            raise CandidateVerificationError("frozen candidate failed verification; redraw is forbidden")
        if artifact.get("status") != "frozen_verified_not_published":
            raise ValueError("candidate resume frozen status is invalid")
        candidate = SkillCandidate(
            candidate_id=artifact["candidate_id"], action="evolve",
            skill_name="HealthVault-Task-Execution", content=content,
            rationale=artifact["rationale"], trace_ids=tuple(artifact["trace_ids"]),
            status="frozen",
        )
        return candidate, artifact
    side_query = CandidateModelSideQuery(
        transport=transport,
        api_key=api_key,
        model_config=preflight["model_config"],
        active_skill_content=context["active_skill_content"],
        candidate_context={key: value for key, value in context.items() if key != "active_skill_content"},
        content_contract=_candidate_contract(),
    )
    generator = CandidateGenerator(runtime_root / "candidate-work", side_query)
    processor = TaskEndProcessor(generator)
    for signal in result.candidate_signals:
        if processor.record(signal) != "eligible":
            raise RuntimeError("candidate evidence rejected before real generation")
    candidate = await processor.process("evolve")
    if candidate is None or generator.call_count != 1:
        raise RuntimeError("candidate must be generated exactly once")
    verification = CandidateVerifier(candidate.skill_name).verify_content(
        candidate.content,
        action="evolve",
        active_content=context["active_skill_content"],
        policy="task024_v2",
    )
    artifact = {
        "schema_version": "healthvault-task-024-candidate-v2",
        "candidate_id": candidate.candidate_id,
        "status": "frozen_verified_not_published" if verification.status == "verified" else "frozen_rejected_not_published",
        "content": candidate.content,
        "candidate_content_sha256": sha256_bytes(candidate.content.encode("utf-8")),
        "candidate_input_sha256": input_sha256,
        "approved_preflight_sha256": preflight_sha256,
        "model_config_sha256": model_sha256,
        "trace_ids": list(candidate.trace_ids),
        "rationale": candidate.rationale,
        "verification": {"status": verification.status, "reasons": list(verification.reasons)},
        "api_calls": side_query.api_calls,
        "input_tokens": side_query.input_tokens,
        "output_tokens": side_query.output_tokens,
        "published": False,
    }
    _atomic_json(artifact_path, artifact)
    if verification.status != "verified":
        raise CandidateVerificationError(
            f"generated candidate failed verification and was frozen: {verification.reasons}"
        )
    return candidate, artifact


def summarize_r2(dataset: Mapping[str, Any], attempts: list[dict[str, Any]], execution_config_sha256: str, model_config: Mapping[str, Any]) -> dict[str, Any]:
    expected_pairs = {(arm, case["case_id"]) for arm in ARM_IDS for case in dataset["cases"]}
    pairs = {(item["arm_id"], item["case_id"]) for item in attempts}
    if pairs != expected_pairs or len(attempts) != len(expected_pairs):
        raise ValueError("TASK-024 attempt matrix is incomplete or duplicated")
    outcomes: dict[str, dict[str, bool]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for arm in ARM_IDS:
        group = [item for item in attempts if item["arm_id"] == arm]
        outcomes[arm] = {item["case_id"]: bool(item["validation"]["task_success"]) for item in group}
        metrics[arm] = {
            "attempts": len(group),
            "task_success_rate": sum(item["validation"]["task_success"] for item in group) / len(group),
            "correct_tool_call_rate": sum(item["validation"]["tool_policy_correct"] for item in group) / len(group),
            "hard_failure_rate": sum(item["validation"]["hard_failure"] for item in group) / len(group),
            "citation_correctness_rate": sum(item["validation"]["citation_correct"] for item in group) / len(group),
            "average_tool_calls": sum(len(item["tool_calls"]) for item in group) / len(group),
            "api_calls": sum(item["api_calls"] for item in group),
            "input_tokens": sum(item["input_tokens"] for item in group),
            "output_tokens": sum(item["output_tokens"] for item in group),
            "average_latency_ms": sum(item["latency_ms"] for item in group) / len(group),
        }
    active = metrics["active_v1"]; candidate = metrics["generated_candidate_v2"]
    improved = sorted(case_id for case_id, value in outcomes["active_v1"].items() if not value and outcomes["generated_candidate_v2"][case_id])
    regressed = sorted(case_id for case_id, value in outcomes["active_v1"].items() if value and not outcomes["generated_candidate_v2"][case_id])
    success_delta = (candidate["task_success_rate"] - active["task_success_rate"]) * 100
    tool_delta = (candidate["correct_tool_call_rate"] - active["correct_tool_call_rate"]) * 100
    safety_regression = candidate["hard_failure_rate"] > active["hard_failure_rate"]
    eligible = success_delta >= 15 and tool_delta >= 15 and len(regressed) / 20 <= 0.10 and not safety_regression
    return {
        "schema_version": "healthvault-task-024-real-summary-v1",
        "dataset_version": DATASET_VERSION,
        "execution_config_sha256": execution_config_sha256,
        "model_config": dict(model_config),
        "arms": metrics,
        "comparison": {
            "primary_baseline": "active_v1", "challenger": "generated_candidate_v2",
            "task_success_delta_pp": success_delta, "correct_tool_call_delta_pp": tool_delta,
            "improved_cases": improved, "improved_rate": len(improved) / 20,
            "regressed_cases": regressed, "regressed_rate": len(regressed) / 20,
            "hard_failure_delta_pp": (candidate["hard_failure_rate"] - active["hard_failure_rate"]) * 100,
        },
        "publish_eligible": eligible,
        "candidate_disposition": "request_separate_publish_approval" if eligible else "reject_and_archive_candidate",
    }


async def execute_r2_once(preflight: dict[str, Any], *, api_key: str, runtime_root: Path, transport: JsonTransport | None = None) -> dict[str, Any]:
    runtime_root.mkdir(parents=True, exist_ok=True)
    real_transport = transport or UrllibJsonTransport()
    candidate, candidate_artifact = await generate_real_candidate_once(
        preflight, transport=real_transport, api_key=api_key, runtime_root=runtime_root
    )
    active_content = ACTIVE_SKILL_PATH.read_text(encoding="utf-8")
    arms = {
        "active_v1": (active_content, sha256_bytes(ACTIVE_SKILL_PATH.read_bytes())),
        "generated_candidate_v2": (candidate.content, candidate_artifact["candidate_content_sha256"]),
    }
    execution_binding = {
        "preflight_sha256": preflight["run_config_sha256"],
        "candidate_sha256": candidate_artifact["candidate_content_sha256"],
        "candidate_input_sha256": candidate_artifact["candidate_input_sha256"],
        "arms": {arm: digest for arm, (_content, digest) in arms.items()},
    }
    execution_hash = sha256_bytes(canonical_json_bytes(execution_binding))
    _atomic_json(runtime_root / "run-manifest.json", {**execution_binding, "execution_config_sha256": execution_hash})
    resume = ResumeStore(runtime_root / "completed-attempts.jsonl", execution_hash)
    ledger = CallLedger(
        runtime_root / "call-ledger.jsonl",
        run_config_sha256=execution_hash,
        max_transport_attempts=preflight["limits"]["max_transport_attempts"],
    )
    dataset = load_r2_dataset()
    dataset_sha = sha256_bytes(canonical_json_bytes(dataset))
    cases = {case["case_id"]: case for case in dataset["cases"]}
    for arm_id, (content, digest) in arms.items():
        for case_id in preflight["frozen_inputs"]["case_order"]:
            if resume.is_complete(arm_id, case_id):
                continue
            attempt = await run_r2_agent_case(
                arm_id=arm_id, skill_sha256=digest, skill_content=content,
                case=cases[case_id], execution_config_sha256=execution_hash,
                dataset_sha256=dataset_sha, model_config=preflight["model_config"],
                transport=real_transport, ledger=ledger, api_key=api_key,
                runtime_root=runtime_root,
            )
            resume.append(attempt)
    attempts = list(resume.records)
    matrix = {
        "schema_version": "healthvault-task-024-case-matrix-v1",
        "execution_config_sha256": execution_hash,
        "case_order": preflight["frozen_inputs"]["case_order"],
        "attempts": attempts,
    }
    summary = summarize_r2(dataset, attempts, execution_hash, preflight["model_config"])
    _atomic_json(runtime_root / "task-024-case-matrix.json", matrix)
    _atomic_json(runtime_root / "task-024-real-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TASK-024 offline preflight and API gate")
    parser.add_argument("--preflight-output", default=str(PREFLIGHT_PATH))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--preflight-hash", default="")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_ROOT))
    args = parser.parse_args(argv)
    preflight = build_preflight()
    if args.execute:
        assert_api_gate(
            preflight,
            execute=True,
            approved=args.approved,
            api_key=os.environ.get(KEY_ENV, ""),
            expected_hash=args.preflight_hash,
        )
        summary = asyncio.run(execute_r2_once(
            preflight,
            api_key=os.environ.get(KEY_ENV, ""),
            runtime_root=Path(args.runtime_dir),
        ))
        print(json.dumps({"status": "completed_once", "publish_eligible": summary["publish_eligible"]}, sort_keys=True))
        return 0
    output = Path(args.preflight_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(preflight, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"mode": preflight["mode"], "api_calls": 0, "run_config_sha256": preflight["run_config_sha256"], "api_gate": preflight["api_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
