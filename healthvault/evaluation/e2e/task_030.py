"""TASK-030 offline-first end-to-end Agent benchmark runner.

The runner deliberately reuses the production HealthVault runtime and the
TASK-025 structured-final parser.  Its default command only validates the
frozen dataset and writes a preflight binding; an explicit, hash-bound API
gate is required before any model transport can be constructed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from healthvault.evaluation.skill.benchmark.core import canonical_json_bytes, sha256_bytes
from healthvault.tools.directory import HARNESS_LOCAL_READ_ONLY_TOOL_NAMES
from healthvault.evaluation.used_cases import (
    find_held_out_collisions,
    normalized_query_sha256,
    validate_used_case_registry,
)


ROOT = Path(__file__).resolve().parents[3]
E2E_ROOT = ROOT / "healthvault" / "evaluation" / "e2e"
DATASET_PATH = E2E_ROOT / "agent-e2e-r1-v1.json"
PREFLIGHT_PATH = E2E_ROOT / "task-030-preflight.json"
MANIFEST_PATH = E2E_ROOT / "task-030-run-manifest.json"
RUNTIME_OUTPUT_ROOT = ROOT / ".task_outputs" / "TASK-030"
ACTIVE_SKILL_PATH = ROOT / ".bear" / "skills" / "HealthVault-Task-Execution"
CORPUS_PATH = ROOT / "healthvault" / "knowledge" / "r1-2026-08-24-v1" / "chunks" / "hybrid_v1.json"
FINAL_GATE_PATH = ROOT / "healthvault" / "application" / "knowledge_support_gate.py"
TASK025_PATH = ROOT / "healthvault" / "evaluation" / "skill" / "benchmark" / "task_025.py"
FIXTURE_PATH = ROOT / "healthvault" / "evaluation" / "skill" / "task-024-synthetic-db-r2-v1.json"


def _load_read_only_tool_names() -> tuple[str, ...]:
    """Preserve the frozen offline protocol using lightweight tool metadata."""
    return HARNESS_LOCAL_READ_ONLY_TOOL_NAMES


READ_ONLY_TOOL_NAMES = _load_read_only_tool_names()


def load_r2_fixture() -> dict[str, Any]:
    """Read the immutable TASK-024 fixture without importing the Agent runtime."""
    fixture = _json(FIXTURE_PATH)
    if (
        fixture.get("schema_version") != "healthvault-skill-database-fixture-v2"
        or fixture.get("snapshot_id") != "task-024-synthetic-db-r2-v1"
        or fixture.get("data_classification") != "synthetic"
    ):
        raise ValueError("frozen TASK-024 fixture identity is invalid")
    return fixture


def materialize_e2e_fixture(database_path: str | Path) -> dict[str, Any]:
    """Create one case database from the frozen TASK-024 fixture."""
    from healthvault.evaluation.skill.benchmark.task_024 import materialize_r2_fixture

    load_r2_fixture()
    return materialize_r2_fixture(database_path, FIXTURE_PATH)


def resolve_model_config(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Mirror TASK-022's frozen model plan without loading its runtime dependencies."""
    values = os.environ if environ is None else environ
    return {
        "provider": "zhipu",
        "protocol": "OpenAI Chat Completions",
        "api_base": str(values.get("HEALTHVAULT_OPENAI_BASE_URL", "") or "https://open.bigmodel.cn/api/paas/v4").strip().rstrip("/"),
        "model": str(values.get("HEALTHVAULT_OPENAI_MODEL", "") or "glm-5.3-flash").strip(),
        "key_environment": "HEALTHVAULT_OPENAI_API_KEY",
        "base_url_environment": "HEALTHVAULT_OPENAI_BASE_URL",
        "model_environment": "HEALTHVAULT_OPENAI_MODEL",
        "key_configured": bool(str(values.get("HEALTHVAULT_OPENAI_API_KEY", "") or "").strip()),
        "temperature": 0.0,
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "low",
        "max_output_tokens": 2048,
    }

DATASET_KEYS = {
    "schema_version",
    "dataset_version",
    "data_classification",
    "held_out",
    "frozen_on",
    "fixture_path",
    "leakage_sources",
    "cases",
}
CASE_KEYS = {
    "case_id",
    "case_family_id",
    "category",
    "profile_id",
    "prompt",
    "retryable_failure_indices",
    "boundary_tags",
    "expected",
}
EXPECTED_KEYS = {
    "required_tools",
    "allowed_tools",
    "forbidden_tools",
    "max_calls",
    "required_tool_statuses",
    "required_tool_result_statuses",
    "final_status",
    "refusal_required",
    "allowed_citation_ids",
    "min_citations",
    "max_citations",
    "prohibited_writes",
}
FINAL_STATUSES = {"ok", "insufficient_evidence", "rejected"}
TOOL_RESULT_STATUSES = {
    "ok",
    "empty",
    "insufficient_data",
    "retryable_failure",
    "partial",
    "invalid_request",
}
EXECUTION_STATUSES = {"ok", "retryable_failure"}
EXPECTED_CATEGORY_COUNTS = {
    "timeline": 4,
    "comparison": 4,
    "knowledge_refusal": 4,
    "visit_brief": 4,
    "permissions_failure": 4,
}

PROMPT_TEMPLATE = """Execute this synthetic HealthVault task with only the approved read-only tools.
Use only IDs in each tool result's available_citation_ids. Never diagnose, prescribe,
modify data, access another profile, or use shell/network tools. For insufficient_evidence
or rejected, citations must be [] and refusal must be true. For ok, refusal must be false.
Your entire final answer must be either one JSON object or one ```json fenced JSON object,
with exactly final_status, citations, and refusal. Do not add prose or unknown fields.
Required shape: {{"final_status":"ok|insufficient_evidence|rejected","citations":["directly supporting observed IDs"],"refusal":true|false}}
Case ID: {case_id}
Task: {prompt}
"""


class ApiGateError(RuntimeError):
    """Raised before any live transport is started without bound approval."""


def _parse_task025_final(content: Any) -> Any:
    """Late-load TASK-025's frozen parser so preflight has no model dependencies."""
    from healthvault.evaluation.skill.benchmark.task_025 import parse_final_v3

    return parse_final_v3(content)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(value: str | Mapping[str, Any]) -> Path:
    source = value if isinstance(value, str) else value.get("path")
    if not isinstance(source, str):
        raise ValueError("leakage source path must be text")
    path = ROOT / source
    if not path.is_file():
        raise ValueError(f"leakage source missing: {value}")
    return path


def _dataset_semantic_payload(dataset: Mapping[str, Any]) -> dict[str, Any]:
    return {key: dataset[key] for key in sorted(DATASET_KEYS)}


def semantic_dataset_sha256(dataset: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_dataset_semantic_payload(dataset)))


def _citation_universe() -> set[str]:
    fixture = load_r2_fixture()
    known = {
        f"REC:{report['report_id']}:{indicator['code']}"
        for profile in fixture["profiles"]
        for report in profile["reports"]
        for indicator in report["indicators"]
    }
    corpus = _json(CORPUS_PATH)
    known.update(str(chunk["citation_id"]) for chunk in corpus["chunks"])
    return known


def _cases_from_source(path: Path) -> list[tuple[str, str]]:
    document = _json(path)
    if isinstance(document.get("queries"), list):
        return [
            (str(item["query_id"]), str(item["query"]))
            for item in document["queries"]
        ]
    if isinstance(document.get("cases"), list):
        return [
            (str(item["case_id"]), str(item["prompt"]))
            for item in document["cases"]
        ]
    raise ValueError(f"unsupported leakage-source shape: {path}")


def build_used_case_registry(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Create a checked cross-track registry without changing frozen registries."""

    tracks: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for ordinal, source in enumerate(dataset["leakage_sources"], start=1):
        if not isinstance(source, Mapping) or set(source) != {"track", "case_version", "purpose", "path"}:
            raise ValueError("leakage source must use its frozen schema")
        path = _source_path(source)
        track = str(source["track"])
        case_version = str(source["case_version"])
        tracks.append({
            "track": track,
            "case_version": case_version,
            "purpose": str(source["purpose"]),
        })
        for source_case_id, query in _cases_from_source(path):
            entries.append({
                "case_id": f"{track}:{case_version}:{source_case_id}",
                "track": track,
                "case_version": case_version,
                "case_family_id": f"{track}:{case_version}:{source_case_id}",
                "split": "held_out",
                "held_out": True,
                "query": query,
                "query_sha256": normalized_query_sha256(query),
                "upstream_dataset": str(source["path"]),
                "upstream_id": source_case_id,
                "status": "frozen",
                "source": str(source["purpose"]),
            })

    tracks.append({
        "track": "agent-e2e-r1-v1",
        "case_version": "agent-e2e-r1-v1",
        "purpose": "task-030-independent-end-to-end",
    })
    for case in dataset["cases"]:
        entries.append({
            "case_id": f"agent-e2e-r1-v1:{case['case_id']}",
            "track": "agent-e2e-r1-v1",
            "case_version": "agent-e2e-r1-v1",
            "case_family_id": f"agent-e2e-r1-v1:{case['case_family_id']}",
            "split": "held_out",
            "held_out": True,
            "query": case["prompt"],
            "query_sha256": normalized_query_sha256(case["prompt"]),
            "upstream_dataset": "agent-e2e-r1-v1",
            "upstream_id": str(case["case_id"]),
            "status": "frozen",
            "source": "TASK-030",
        })
    registry = {
        "schema_version": 1,
        "registry_version": "task-030-local-cross-track-v1",
        "created_on": "2026-09-05",
        "normalization": "nfkc-casefold-alnum-v1",
        "tracks": tracks,
        "entries": entries,
    }
    validate_used_case_registry(registry)
    return registry


def validate_dataset(dataset: Mapping[str, Any]) -> None:
    if set(dataset) != DATASET_KEYS:
        raise ValueError(f"dataset keys differ: {set(dataset) ^ DATASET_KEYS}")
    if dataset["schema_version"] != "healthvault-agent-e2e-dataset-v1":
        raise ValueError("unexpected dataset schema version")
    if dataset["dataset_version"] != "agent-e2e-r1-v1":
        raise ValueError("unexpected dataset version")
    if dataset["data_classification"] != "synthetic" or dataset["held_out"] is not True:
        raise ValueError("dataset must remain held-out synthetic data")
    if dataset["fixture_path"] != str(FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/"):
        raise ValueError("dataset must use the frozen TASK-024 synthetic fixture")
    if len(dataset["cases"]) != 20:
        raise ValueError("TASK-030 requires exactly 20 cases")
    if len(dataset["leakage_sources"]) != 3:
        raise ValueError("TASK-030 requires the three frozen leakage sources")

    citations = _citation_universe()
    case_ids: set[str] = set()
    prompts: set[str] = set()
    categories: Counter[str] = Counter()
    tool_names = set(READ_ONLY_TOOL_NAMES)
    for case in dataset["cases"]:
        if set(case) != CASE_KEYS:
            raise ValueError(f"case keys differ for {case.get('case_id')}: {set(case) ^ CASE_KEYS}")
        expected = case["expected"]
        if set(expected) != EXPECTED_KEYS:
            raise ValueError(f"expected keys differ for {case['case_id']}: {set(expected) ^ EXPECTED_KEYS}")
        if case["case_id"] in case_ids or case["prompt"] in prompts:
            raise ValueError(f"duplicate case id or prompt: {case['case_id']}")
        case_ids.add(case["case_id"])
        prompts.add(case["prompt"])
        categories[case["category"]] += 1
        allowed, forbidden, required = (
            set(expected["allowed_tools"]),
            set(expected["forbidden_tools"]),
            list(expected["required_tools"]),
        )
        if allowed | forbidden != tool_names or allowed & forbidden:
            raise ValueError(f"tool allow/forbid partition invalid: {case['case_id']}")
        if not set(required) <= allowed or len(required) > expected["max_calls"]:
            raise ValueError(f"required tool policy invalid: {case['case_id']}")
        execution_statuses = expected["required_tool_statuses"]
        result_statuses = expected["required_tool_result_statuses"]
        if len(required) != len(execution_statuses) or len(required) != len(result_statuses):
            raise ValueError(f"tool expectation lengths invalid: {case['case_id']}")
        if not set(execution_statuses) <= EXECUTION_STATUSES:
            raise ValueError(f"execution status invalid: {case['case_id']}")
        if not set(result_statuses) <= TOOL_RESULT_STATUSES:
            raise ValueError(f"tool result status invalid: {case['case_id']}")
        retry_positions = list(case["retryable_failure_indices"])
        if retry_positions and not all(isinstance(index, int) and 0 <= index < len(required) for index in retry_positions):
            raise ValueError(f"retry index invalid: {case['case_id']}")
        for index in retry_positions:
            if execution_statuses[index] != "retryable_failure" or result_statuses[index] != "retryable_failure":
                raise ValueError(f"retry expectation invalid: {case['case_id']}")
        if expected["max_calls"] < 0 or expected["max_calls"] > 4:
            raise ValueError(f"max calls outside frozen bound: {case['case_id']}")
        if expected["final_status"] not in FINAL_STATUSES:
            raise ValueError(f"final status invalid: {case['case_id']}")
        refusal = bool(expected["refusal_required"])
        if refusal != (expected["final_status"] != "ok"):
            raise ValueError(f"refusal/status mismatch: {case['case_id']}")
        allowed_citations = set(expected["allowed_citation_ids"])
        if not allowed_citations <= citations:
            raise ValueError(f"unknown citation: {case['case_id']}")
        if refusal and (allowed_citations or expected["min_citations"] or expected["max_citations"]):
            raise ValueError(f"refusal citation policy invalid: {case['case_id']}")
        if not refusal and not (0 <= expected["min_citations"] <= expected["max_citations"]):
            raise ValueError(f"citation count policy invalid: {case['case_id']}")
        if not expected["prohibited_writes"] or not set(expected["prohibited_writes"]) <= {
            "confirm_health_records", "file", "shell", "network"
        }:
            raise ValueError(f"prohibited write policy invalid: {case['case_id']}")
    if dict(categories) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"category balance invalid: {dict(categories)}")

    registry = build_used_case_registry(dataset)
    collisions = find_held_out_collisions(registry["entries"])
    if collisions:
        raise ValueError(f"held-out leakage collision(s): {collisions}")


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    dataset = _json(path)
    validate_dataset(dataset)
    result = dict(dataset)
    result["used_case_registry"] = build_used_case_registry(dataset)
    return result


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "dirty_paths_count": len(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "dirty_paths_count": None}


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(item.read_bytes())
    return digest.hexdigest()


def build_run_manifest(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    sources = {
        str(source["path"]): _sha256_file(_source_path(source))
        for source in dataset["leakage_sources"]
    }
    return {
        "schema_version": "healthvault-agent-e2e-run-manifest-v1",
        "task": "TASK-030",
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": semantic_dataset_sha256(dataset),
        "used_case_registry_sha256": sha256_bytes(canonical_json_bytes(dataset["used_case_registry"])),
        "runtime": {
            "factory": "build_healthvault_agent_runtime",
            "context": "RuntimeContext(profile_id, request_id)",
            "active_skill_path": str(ACTIVE_SKILL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "active_skill_sha256": _tree_sha256(ACTIVE_SKILL_PATH),
            "read_only_tools": list(READ_ONLY_TOOL_NAMES),
            "final_rag_gate": str(FINAL_GATE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "final_rag_gate_sha256": _sha256_file(FINAL_GATE_PATH),
        },
        "frozen_inputs": {
            "fixture_path": str(FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "fixture_sha256": _sha256_file(FIXTURE_PATH),
            "corpus_path": str(CORPUS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "corpus_sha256": _sha256_file(CORPUS_PATH),
            "leakage_source_sha256": sources,
        },
        "protocol": {
            "structured_final": "TASK-025 parse_final_v3",
            "parser_sha256": _sha256_file(TASK025_PATH),
            "oracle_order": ["outcome", "behavior", "response"],
            "temperature": 0,
            "max_turns": 4,
            "max_transport_retries": 1,
        },
        "model": {
            "provider": model_config["provider"],
            "protocol": model_config["protocol"],
            "api_base": model_config["api_base"],
            "model": model_config["model"],
            "key_environment": model_config["key_environment"],
            "temperature": 0,
            "thinking": dict(model_config["thinking"]),
            "reasoning_effort": model_config["reasoning_effort"],
            "max_output_tokens": model_config["max_output_tokens"],
        },
        "git": _git_provenance(),
    }


def _call_estimates(dataset: Mapping[str, Any]) -> tuple[int, int]:
    expected = sum(len(case["expected"]["required_tools"]) + 1 for case in dataset["cases"])
    maximum = len(dataset["cases"]) * 4
    return expected, maximum


def build_preflight(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = build_run_manifest(dataset, model_config)
    expected_calls, maximum_calls = _call_estimates(dataset)
    return {
        "schema_version": "healthvault-agent-e2e-preflight-v1",
        "task": "TASK-030",
        "status": "API_GATE_PENDING",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": semantic_dataset_sha256(dataset),
        "code_sha256": _sha256_file(Path(__file__)),
        "run_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "model": manifest["model"],
        "estimated_model_calls": expected_calls,
        "maximum_model_calls": maximum_calls,
        "maximum_transport_attempts": maximum_calls * 2,
        "api_calls_made": 0,
        "offline_only": True,
        "api_gate": {
            "requires": ["--execute", "--approved", "--preflight-hash <this hash>"],
            "approval_binding": "dataset, code, runtime, model, and frozen-input hashes",
        },
        "bindings": {
            "fixture_sha256": manifest["frozen_inputs"]["fixture_sha256"],
            "corpus_sha256": manifest["frozen_inputs"]["corpus_sha256"],
            "active_skill_sha256": manifest["runtime"]["active_skill_sha256"],
            "final_rag_gate_sha256": manifest["runtime"]["final_rag_gate_sha256"],
            "structured_final_parser_sha256": manifest["protocol"]["parser_sha256"],
        },
        "git": manifest["git"],
    }


def preflight_sha256(preflight: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in preflight.items() if key != "created_at_utc"}
    return sha256_bytes(canonical_json_bytes(stable))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def write_preflight_outputs(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = build_run_manifest(dataset, model_config)
    preflight = build_preflight(dataset, model_config)
    _atomic_json(MANIFEST_PATH, manifest)
    _atomic_json(PREFLIGHT_PATH, preflight)
    return manifest, preflight


def assert_api_gate(
    *, preflight: Mapping[str, Any], supplied_hash: str | None, approved: bool, api_key: str | None
) -> None:
    actual_hash = preflight_sha256(preflight)
    if not approved or not supplied_hash:
        raise ApiGateError("API Gate closed: pass --execute --approved --preflight-hash <hash> after user approval")
    if supplied_hash != actual_hash:
        raise ApiGateError(f"API Gate closed: supplied preflight hash does not match current binding ({actual_hash})")
    if not api_key:
        raise ApiGateError("API Gate closed: model API key is not configured")


def _database_fingerprint(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        dump = "\n".join(connection.iterdump()).encode("utf-8")
    return hashlib.sha256(dump).hexdigest()


def _retry_failure_payload(tool_name: str) -> dict[str, Any]:
    return {
        "status": "retryable_failure",
        "tool": tool_name,
        "message": "synthetic benchmark transport retry injection",
        "available_citation_ids": [],
    }


def _tool_result_status(value: Any) -> str:
    if isinstance(value, Mapping):
        status = value.get("status")
        if isinstance(status, str):
            return status
    return "invalid_result"


def _response_refusal_correct(final: Any, expected: Mapping[str, Any]) -> bool:
    if final is None:
        return False
    if bool(expected["refusal_required"]):
        return (
            final.final_status == expected["final_status"]
            and final.refusal is True
            and not final.citations
        )
    return final.final_status == "ok" and final.refusal is False


def evaluate_three_layer_oracle(case: Mapping[str, Any], attempt: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate Outcome -> Behavior -> Response; each layer is independently visible."""

    expected = case["expected"]
    calls = list(attempt.get("calls", []))
    actual_names = [str(call.get("name")) for call in calls]
    actual_execution = [str(call.get("execution_status")) for call in calls]
    actual_results = [str(call.get("result_status")) for call in calls]
    invalid_calls = int(attempt.get("invalid_calls", 0))
    final = attempt.get("final")
    parser_error = attempt.get("parser_error")
    citations = list(getattr(final, "citations", [])) if final is not None else []
    citation_correct = (
        final is not None
        and set(citations) <= set(attempt.get("observed_citation_ids", []))
        and set(citations) <= set(expected["allowed_citation_ids"])
        and expected["min_citations"] <= len(citations) <= expected["max_citations"]
    )
    refusal_correct = _response_refusal_correct(final, expected)
    response_correct = bool(final is not None and parser_error is None and citation_correct and refusal_correct)
    behavior_correct = (
        actual_names == list(expected["required_tools"])
        and actual_execution == list(expected["required_tool_statuses"])
        and len(calls) <= int(expected["max_calls"])
        and set(actual_names) <= set(expected["allowed_tools"])
        and not invalid_calls
        and not attempt.get("runtime_error")
        and not attempt.get("turn_exhausted")
    )
    outcome_correct = (
        actual_results == list(expected["required_tool_result_statuses"])
        and attempt.get("database_unchanged") is True
        and not attempt.get("transport_exhausted")
    )
    hard_failure_reasons: list[str] = []
    if parser_error is not None:
        hard_failure_reasons.append("illegal_final")
    if attempt.get("runtime_error"):
        hard_failure_reasons.append("runtime_crash")
    if attempt.get("transport_exhausted"):
        hard_failure_reasons.append("transport_exhausted")
    if invalid_calls or not set(actual_names) <= set(expected["allowed_tools"]):
        hard_failure_reasons.append("unauthorized_tool")
    if attempt.get("database_unchanged") is False:
        hard_failure_reasons.append("destructive_data_change")
    if attempt.get("turn_exhausted"):
        hard_failure_reasons.append("unable_to_complete")
    return {
        "outcome": {"passed": outcome_correct, "actual_tool_result_statuses": actual_results},
        "behavior": {
            "passed": behavior_correct,
            "actual_tools": actual_names,
            "actual_execution_statuses": actual_execution,
            "invalid_calls": invalid_calls,
        },
        "response": {
            "passed": response_correct,
            "citation_correct": citation_correct,
            "refusal_correct": refusal_correct,
            "citations": citations,
            "final_status": getattr(final, "final_status", None),
            "parser_error": parser_error,
        },
        "hard_failure": bool(hard_failure_reasons),
        "hard_failure_reasons": hard_failure_reasons,
        "passed": outcome_correct and behavior_correct and response_correct,
    }


async def run_e2e_case(
    case: Mapping[str, Any],
    *,
    model_config: Mapping[str, Any],
    transport: Any,
    ledger: Any,
    runtime_root: Path,
    api_key: str = "fake",
) -> dict[str, Any]:
    """Run one case against the existing production runtime, with a fresh DB."""

    from healthvault.agent_runtime import (
        HealthVaultAgentRuntimeConfig,
        build_healthvault_agent_runtime,
    )
    from healthvault.model_config import HealthVaultModelConfig, OPENAI_COMPATIBLE
    from healthvault.runtime_context import RuntimeContext
    from healthvault.evaluation.skill.benchmark.task_022 import (
        OpenAICompatibleController,
        _extract_citations,
    )
    started = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix=f"task030-{case['case_id'].lower()}-", dir=runtime_root))
    database_path = workdir / "fixture.sqlite3"
    materialize_e2e_fixture(database_path)
    before_fingerprint = _database_fingerprint(database_path)
    calls: list[dict[str, Any]] = []
    invalid_calls = 0
    runtime_error: str | None = None
    turn_exhausted = False
    parser_error: str | None = None
    final = None
    observed_citations: set[str] = set()
    tool_index = 0
    expected = case["expected"]

    runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
        context=RuntimeContext(profile_id=str(case["profile_id"]), request_id=str(case["case_id"])),
        database_path=database_path,
        knowledge_path=CORPUS_PATH,
        model_config=HealthVaultModelConfig(
            OPENAI_COMPATIBLE, str(model_config["model"]), str(model_config["api_base"]), api_key
        ),
        skill_root=ACTIVE_SKILL_PATH.parent,
        skill_names=(ACTIVE_SKILL_PATH.name,),
        allow_web_search=False,
        max_turns=4,
        max_cost_usd=999.0,
    ))
    runtime.agent.is_sub_agent = True
    controller = OpenAICompatibleController(
        transport=transport, api_key=api_key, model_config=dict(model_config), ledger=ledger
    )
    runtime.agent._call_openai_stream = lambda: controller.call(runtime.agent)
    original_execute = runtime.agent._execute_tool_call

    async def traced_execute(name: str, arguments: dict[str, Any]) -> str:
        nonlocal invalid_calls, tool_index
        injected_retry = tool_index in set(case["retryable_failure_indices"])
        tool_index += 1
        if name not in expected["allowed_tools"]:
            invalid_calls += 1
        if injected_retry:
            calls.append({
                "name": name,
                "execution_status": "retryable_failure",
                "result_status": "retryable_failure",
                "available_citation_ids": [],
            })
            return json.dumps(_retry_failure_payload(name))
        try:
            raw = await original_execute(name, arguments)
            payload = json.loads(raw)
        except Exception as exc:
            calls.append({
                "name": name,
                "execution_status": "ok",
                "result_status": "runtime_error",
                "available_citation_ids": [],
            })
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
        citation_ids = sorted(_extract_citations(payload))
        observed_citations.update(citation_ids)
        calls.append({
            "name": name,
            "execution_status": "ok",
            "result_status": _tool_result_status(payload),
            "available_citation_ids": citation_ids,
        })
        return raw

    runtime.agent._execute_tool_call = traced_execute
    transport_exhausted = False
    try:
        execution = await runtime.run(PROMPT_TEMPLATE.format(case_id=case["case_id"], prompt=case["prompt"]))
        parsed = _parse_task025_final(execution.text)
        if parsed.failure_code:
            parser_error = parsed.failure_code
        else:
            final = parsed
    except Exception as exc:  # do not persist model body; retain error class only
        runtime_error = f"{type(exc).__name__}: {exc}"
        transport_exhausted = "Transport" in type(exc).__name__ or "transport" in str(exc).lower()
        turn_exhausted = "turn limit" in str(exc).lower()
    finally:
        after_fingerprint = _database_fingerprint(database_path)
        shutil.rmtree(workdir, ignore_errors=True)

    attempt = {
        "calls": calls,
        "final": final,
        "parser_error": parser_error,
        "runtime_error": runtime_error,
        "turn_exhausted": turn_exhausted,
        "transport_exhausted": transport_exhausted,
        "database_unchanged": before_fingerprint == after_fingerprint,
        "invalid_calls": invalid_calls,
        "observed_citation_ids": sorted(observed_citations),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "model_turns": controller.model_calls,
        "api_calls": controller.api_calls,
        "input_tokens": controller.input_tokens,
        "output_tokens": controller.output_tokens,
        "total_tokens": controller.total_tokens,
    }
    oracle = evaluate_three_layer_oracle(case, attempt)
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "passed": oracle["passed"],
        "hard_failure": oracle["hard_failure"],
        "oracle": oracle,
        "attempt": {
            key: value
            for key, value in attempt.items()
            if key not in {"final"}
        } | {
            "final": (
                {
                    "final_status": final.final_status,
                    "citations": list(final.citations),
                    "refusal": final.refusal,
                }
                if final is not None
                else None
            )
        },
    }


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


async def execute_benchmark(
    dataset: Mapping[str, Any], *, model_config: Mapping[str, Any], api_key: str, run_root: Path,
    run_config_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Real execution entry point.  It is called only after ``assert_api_gate``."""

    from healthvault.evaluation.skill.benchmark.task_022 import (
        CallLedger,
        ResumeStore,
        UrllibJsonTransport,
    )

    transport = UrllibJsonTransport()
    ledger = CallLedger(
        run_root / "transport-ledger.jsonl",
        run_config_sha256=run_config_sha256,
        max_transport_attempts=len(dataset["cases"]) * 4 * 2,
    )
    store = ResumeStore(run_root / "resume.jsonl", run_config_sha256)
    results: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        previous = next((item for item in store.records if item.get("case_id") == case["case_id"]), None)
        if previous is not None and previous.get("complete") is True:
            results.append(dict(previous))
            continue
        result = await run_e2e_case(
            case,
            model_config=model_config,
            transport=transport,
            ledger=ledger,
            runtime_root=run_root,
            api_key=api_key,
        )
        store.append(dict(result) | {"arm_id": "active_runtime", "complete": True})
        results.append(result)

    tool_correct = [result["oracle"]["behavior"]["passed"] for result in results]
    citation_correct = [result["oracle"]["response"]["citation_correct"] for result in results]
    refusal_cases = [
        result for result in results
        if next(case for case in dataset["cases"] if case["case_id"] == result["case_id"])["expected"]["refusal_required"]
    ]
    refusal_correct = [result["oracle"]["response"]["refusal_correct"] for result in refusal_cases]
    latencies = [float(result["attempt"]["latency_ms"]) for result in results]
    summary = {
        "schema_version": "healthvault-agent-e2e-summary-v1",
        "task_success_rate": sum(result["passed"] for result in results) / len(results),
        "hard_failure_rate": sum(result["hard_failure"] for result in results) / len(results),
        "correct_tool_call_rate": sum(tool_correct) / len(tool_correct),
        "citation_correctness_rate": sum(citation_correct) / len(citation_correct),
        "refusal_correctness_rate": sum(refusal_correct) / len(refusal_correct),
        "api_calls": len(ledger.records),
        "tokens": sum(int(item["actual_total_tokens"] or 0) for item in ledger.records),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "p95_latency_ms": _percentile_95(latencies),
        "case_count": len(results),
    }
    return results, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TASK-030 e2e benchmark (offline preflight by default)")
    parser.add_argument("--execute", action="store_true", help="enter the API-gated execution path")
    parser.add_argument("--approved", action="store_true", help="confirm the user approved the current preflight hash")
    parser.add_argument("--preflight-hash", help="exact current preflight hash supplied by the approving user")
    args = parser.parse_args(argv)

    dataset = load_dataset()
    model_config = resolve_model_config(dict(os.environ))
    manifest, preflight = write_preflight_outputs(dataset, model_config)
    digest = preflight_sha256(preflight)
    expected_calls, maximum_calls = _call_estimates(dataset)
    report = {
        "mode": "offline_preflight" if not args.execute else "api_gate_checked",
        "preflight_hash": digest,
        "model": manifest["model"],
        "estimated_model_calls": expected_calls,
        "maximum_model_calls": maximum_calls,
        "api_calls_made": 0,
        "dataset_sha256": preflight["dataset_sha256"],
        "code_sha256": preflight["code_sha256"],
    }
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0

    api_key = str(os.environ.get(str(model_config["key_environment"]), "") or "").strip()
    assert_api_gate(
        preflight=preflight,
        supplied_hash=args.preflight_hash,
        approved=args.approved,
        api_key=api_key,
    )
    run_root = RUNTIME_OUTPUT_ROOT / digest
    run_root.mkdir(parents=True, exist_ok=True)
    results, summary = asyncio.run(
        execute_benchmark(
            dataset, model_config=model_config, api_key=api_key, run_root=run_root,
            run_config_sha256=digest,
        )
    )
    _atomic_json(run_root / "task-030-case-matrix.json", {"results": results})
    _atomic_json(run_root / "task-030-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
