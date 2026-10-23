"""TASK-045 independent Skill A/B, offline until a candidate and hash are approved."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from healthvault.adapters.bearcode.final_response_normalizer import FinalResponseNormalizer
from healthvault.application.citation_policy import actual_available_citation_ids
from healthvault.evaluation.e2e import task_030 as base
from healthvault.evaluation.e2e import task_043 as oracle
from healthvault.evaluation.skill.benchmark.core import canonical_json_bytes, sha256_bytes
from healthvault.evaluation.split_integrity import find_patient_profile_split_conflicts
from healthvault.evaluation.used_cases import normalize_case_text, normalized_query_sha256
from healthvault.evolution import skill_patch as patching
from healthvault.evolution.signal_extractor import ExperienceCard, SignalExtractor, TaskObservation
from healthvault.evolution.verifier import CandidateVerifier


ROOT = base.ROOT
SKILL_ROOT = ROOT / "healthvault" / "evaluation" / "skill"
DATASET_PATH = SKILL_ROOT / "skill-heldout-r5-v1.json"
FIXTURE_PATH = SKILL_ROOT / "skill-heldout-r5-v1-fixture.json"
SPLIT_PATH = SKILL_ROOT / "task-045-patient-profile-splits.json"
LEAKAGE_AUDIT_PATH = SKILL_ROOT / "task-045-leakage-audit.json"
CASE_MATRIX_PATH = SKILL_ROOT / "task-045-case-matrix.json"
READINESS_PATH = SKILL_ROOT / "task-045-offline-readiness.json"
PREFLIGHT_PATH = SKILL_ROOT / "task-045-preflight.json"
CANDIDATE_GATE_A_PREFLIGHT_PATH = SKILL_ROOT / "task-045-candidate-gate-a-preflight.json"
CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH = (
    SKILL_ROOT / "task-045-candidate-gate-a-auth-probe-preflight.json"
)
CANDIDATE_GATE_A_CONSUMPTION_ROOT = (
    ROOT / ".task_outputs" / "TASK-045" / "candidate-gate-a-consumptions"
)
CANDIDATE_GATE_A_AUTH_PROBE_ROOT = (
    ROOT / ".task_outputs" / "TASK-045" / "candidate-gate-a-auth-probes"
)
SIGNOFF_TEMPLATE_PATH = SKILL_ROOT / "task-045-human-signoff-template.json"
ARTIFACT_TEMPLATE_PATH = SKILL_ROOT / "task-045-run-artifact-template.json"
SOURCE_TRACE_MANIFEST_PATH = SKILL_ROOT / "task-045-authorized-source-traces.json"
DEV_TRACE_PATH = SKILL_ROOT / "task-037-dev-traces.json"
ACTIVE_SKILL_PATH = ROOT / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"
CORPUS_PATH = oracle.CORPUS_PATH
PATCH_PIPELINE_PATH = ROOT / "healthvault" / "evolution" / "skill_patch.py"
PATCH_SCHEMA_PATH = ROOT / "healthvault" / "evolution" / "skill-patch-schema.json"
TRANSPORT_SOURCE_PATH = (
    ROOT / "healthvault" / "evaluation" / "skill" / "benchmark" / "task_022.py"
)
CANDIDATE_GATE_A_ANCHOR_TABLE_PATH = (
    SKILL_ROOT / "task-045-candidate-gate-a-anchors-v1.json"
)
CANDIDATE_GATE_A_PATCH_SCHEMA_PATH = (
    SKILL_ROOT / "task-045-candidate-gate-a-patch-schema.json"
)
ARM_IDS = ("active", "candidate")
DATASET_KEYS = {
    "schema_version", "dataset_version", "data_classification", "held_out",
    "frozen_on", "fixture_path", "cases",
}
CASE_KEYS = {
    "case_id", "case_family_id", "category", "patient_id", "profile_id",
    "prompt", "scenario", "boundary_tags",
}
CATEGORY_COUNTS = {
    "confirmation_boundary": 3,
    "timeline": 3,
    "comparison": 3,
    "knowledge": 3,
    "visit_brief": 3,
    "safety_isolation": 5,
}
CANDIDATE_KEYS = {
    "schema_version", "candidate_id", "status", "active_skill_sha256",
    "candidate_content", "candidate_sha256", "patch_artifact_sha256",
    "patch_artifact_path", "source_trace_ids", "source_splits",
    "verifier_status", "gate_a_preflight_sha256", "frozen_at_utc",
}
PATCH_ARTIFACT_KEYS = {
    "schema_version", "status", "reasons", "patches", "active_skill_sha256",
    "candidate_skill_sha256", "unified_diff", "generator_calls", "published",
    "api_calls",
}
SOURCE_TRACE_MANIFEST_KEYS = {"schema_version", "status", "entries"}
SOURCE_TRACE_ENTRY_KEYS = {
    "trace_id", "split", "source_path", "source_file_sha256", "trace_sha256",
}
CANDIDATE_SOURCE_TRACE_IDS = ("TASK037-DEV-1", "TASK037-DEV-2", "TASK037-DEV-3")
CANDIDATE_GATE_A_MAX_PATCHES = 1
CANDIDATE_GATE_A_CONTENT_MAX_LENGTH = 240
CANDIDATE_GATE_A_CONTENT_PATTERN = r"^[^\r\n.!?。！？]+[.!?。！？]?$"
CANDIDATE_GATE_A_AUTH_PROBE_MAX_TOKENS = 1
_EVAL_TARGETING_PATTERN = re.compile(
    r"(?i)\b(?:oracle|gold|held[-_\s]?out|expected\s+(?:answer|tool(?:\s+sequence)?|"
    r"citation(?:\s+id)?|status|refusal))\b"
)
PUBLISH_THRESHOLDS = {
    "minimum_task_success_delta": 0.10,
    "maximum_regression_rate": 0.05,
    "hard_failure_must_not_increase": True,
    "tool_correctness_must_not_decrease": True,
}
MODEL_CONFIG = {
    "provider": "zhipu",
    "protocol": "OpenAI Chat Completions",
    "api_base": "https://open.bigmodel.cn/api/paas/v4/",
    "model": "glm-5.3-flash",
    "key_environment": "HEALTHVAULT_OPENAI_API_KEY",
    "temperature": 0.0,
    "thinking": {"type": "enabled", "clear_thinking": False},
    "reasoning_effort": "low",
    "max_output_tokens": 4096,
}
CANDIDATE_GATE_A_MODEL = {
    "provider": "zhipu",
    "protocol": "OpenAI Chat Completions",
    "api_base": "https://open.bigmodel.cn/api/paas/v4/",
    "model": "glm-5.3",
    "key_environment": "HEALTHVAULT_OPENAI_API_KEY",
    "do_sample": False,
    "temperature": 0.0,
    "thinking": {"type": "enabled", "clear_thinking": False},
    "reasoning_effort": "high",
    "max_output_tokens": 4096,
    "timeout_seconds": 120.0,
}


class CandidateGateError(RuntimeError):
    """Raised before a formal preflight can exist without a frozen candidate."""


class ApiGateError(RuntimeError):
    """Raised before any real transport is used without exact-hash approval."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _exclusive_json(path: Path, value: Mapping[str, Any], *, exists_message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ApiGateError(exists_message) from exc


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    fixture = _json(path)
    if set(fixture) != oracle.FIXTURE_KEYS:
        raise ValueError("TASK-045 fixture schema is invalid")
    if (
        fixture.get("schema_version") != "healthvault-agent-e2e-fixture-v2"
        or fixture.get("snapshot_id") != "skill-heldout-r5-fixture-v1"
        or fixture.get("data_classification") != "synthetic"
    ):
        raise ValueError("TASK-045 fixture identity is invalid")
    profiles = fixture.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 5:
        raise ValueError("TASK-045 requires five isolated profiles")
    patients: set[str] = set()
    profile_ids: set[str] = set()
    report_ids: set[str] = set()
    source_ids: set[str] = set()
    reports = 0
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) != {"patient_id", "profile_id", "reports"}:
            raise ValueError("TASK-045 fixture profile schema is invalid")
        patient_id, profile_id = profile["patient_id"], profile["profile_id"]
        if patient_id in patients or profile_id in profile_ids:
            raise ValueError("TASK-045 fixture identity is duplicated")
        patients.add(patient_id); profile_ids.add(profile_id)
        for report in profile["reports"]:
            if not isinstance(report, dict) or set(report) != {"source_id", "report_id", "report_date", "facility", "indicators"}:
                raise ValueError("TASK-045 fixture report schema is invalid")
            if report["report_id"] in report_ids or report["source_id"] in source_ids:
                raise ValueError("TASK-045 fixture report/source identity is duplicated")
            report_ids.add(report["report_id"]); source_ids.add(report["source_id"]); reports += 1
            if not report["indicators"]:
                raise ValueError("TASK-045 fixture report must contain indicators")
    if fixture.get("report_count") != reports:
        raise ValueError("TASK-045 fixture report count is invalid")
    return fixture


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    dataset = _json(path)
    fixture = load_fixture(ROOT / str(dataset.get("fixture_path", "")))
    if set(dataset) != DATASET_KEYS:
        raise ValueError("TASK-045 dataset schema is invalid")
    if (
        dataset.get("schema_version") != "healthvault-skill-heldout-dataset-v1"
        or dataset.get("dataset_version") != "skill-heldout-r5-v1"
        or dataset.get("data_classification") != "synthetic"
        or dataset.get("held_out") is not True
        or dataset.get("fixture_path") != _relative(FIXTURE_PATH)
    ):
        raise ValueError("TASK-045 dataset identity or fixture binding is invalid")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError("TASK-045 requires exactly twenty held-out cases")
    if Counter(case.get("category") for case in cases) != CATEGORY_COUNTS:
        raise ValueError("TASK-045 category matrix is invalid")
    pairs = {(profile["patient_id"], profile["profile_id"]) for profile in fixture["profiles"]}
    result: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != CASE_KEYS:
            raise ValueError("TASK-045 case schema is invalid")
        if (case["patient_id"], case["profile_id"]) not in pairs:
            raise ValueError("TASK-045 case identity is outside its fixture")
        if not isinstance(case["prompt"], str) or not case["prompt"].strip():
            raise ValueError("TASK-045 prompt is invalid")
        result.append(dict(case) | {"expected": oracle.derive_expected(case, fixture)})
    if len({case["case_id"] for case in result}) != 20:
        raise ValueError("TASK-045 case IDs must be unique")
    if len({normalized_query_sha256(case["prompt"]) for case in result}) != 20:
        raise ValueError("TASK-045 prompts must be unique")
    return dict(dataset) | {"cases": result}


def _split_document(path: Path) -> dict[str, Any]:
    value = _json(path)
    if (
        set(value) != {"schema_version", "purpose", "entries"}
        or value["schema_version"] != "healthvault-patient-profile-splits-v1"
    ):
        raise ValueError(f"invalid patient/profile split registry: {path}")
    return value


def _registry_entry(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": f"skill-r5:{case['case_id']}",
        "case_family_id": f"skill-r5:{case['case_family_id']}",
        "track": "skill-r5",
        "case_version": "skill-heldout-r5-v1",
        "split": "held_out",
        "held_out": True,
        "query": str(case["prompt"]),
        "query_sha256": normalized_query_sha256(str(case["prompt"])),
        "upstream_dataset": _relative(DATASET_PATH),
        "upstream_id": str(case["case_id"]),
        "status": "frozen",
        "source": "TASK-045",
    }


def _prior_source_signature() -> tuple[tuple[str, str], ...]:
    paths = [path for _, _, _, path in oracle.LEAKAGE_SOURCES]
    paths.extend((oracle.DATASET_PATH, oracle.DEV_CASES_PATH))
    return tuple(sorted((_relative(path), _sha(path)) for path in paths))


@lru_cache(maxsize=4)
def _cached_prior_entries(
    _signature: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    registry = oracle.build_used_case_registry(oracle.load_dataset())
    return tuple(
        json.dumps(entry, ensure_ascii=False, sort_keys=True)
        for entry in registry["entries"]
    )


def _prior_entries() -> list[dict[str, Any]]:
    return [json.loads(item) for item in _cached_prior_entries(_prior_source_signature())]


def _text_collisions(current: list[dict[str, Any]], prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collisions: list[dict[str, Any]] = []
    for left in current:
        left_text = normalize_case_text(left["query"])
        left_tokens = set(left_text.split())
        for right in prior:
            right_text = normalize_case_text(right["query"])
            right_tokens = set(right_text.split())
            union = left_tokens | right_tokens
            jaccard = len(left_tokens & right_tokens) / len(union) if union else 1.0
            ratio = SequenceMatcher(None, left_text, right_text).ratio()
            if left_text == right_text:
                reason = "exact_normalized_query"
            elif jaccard >= 0.60 or ratio >= 0.72:
                reason = "strict_near_synonym"
            else:
                continue
            collisions.append({
                "left_case_id": left["case_id"],
                "right_case_id": right["case_id"],
                "right_split": right["split"],
                "reason": reason,
                "token_jaccard": round(jaccard, 6),
                "sequence_ratio": round(ratio, 6),
            })
    return collisions


def _internal_text_collisions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collisions: list[dict[str, Any]] = []
    for index, left in enumerate(entries):
        collisions.extend(_text_collisions([left], entries[index + 1:]))
    return collisions


def _legacy_split_entries() -> list[dict[str, str]]:
    entries = list(_split_document(oracle.DEV_SPLIT_PATH)["entries"])
    entries.extend(_split_document(oracle.FORMAL_SPLIT_PATH)["entries"])
    for path in sorted((ROOT / "healthvault" / "evaluation" / "e2e").glob("agent-e2e-r*-v1-fixture.json")):
        fixture = _json(path)
        for profile in fixture.get("profiles", []):
            profile_id = str(profile.get("profile_id", ""))
            if not profile_id or any(item["profile_id"] == profile_id for item in entries):
                continue
            patient_id = str(profile.get("patient_id") or f"legacy-{profile_id}")
            entries.append({"patient_id": patient_id, "profile_id": profile_id, "split": "prior_held_out"})
    return entries


def load_authorized_source_traces(
    path: str | Path = SOURCE_TRACE_MANIFEST_PATH,
) -> dict[str, dict[str, str]]:
    """Resolve the frozen discovery/dev Trace registry back to its source bytes."""
    manifest = _json(Path(path))
    if set(manifest) != SOURCE_TRACE_MANIFEST_KEYS:
        raise CandidateGateError("authorized source Trace manifest schema is invalid")
    if (
        manifest.get("schema_version") != "healthvault-task-045-source-trace-manifest-v1"
        or manifest.get("status") != "frozen"
        or not isinstance(manifest.get("entries"), list)
        or not manifest["entries"]
    ):
        raise CandidateGateError("authorized source Trace manifest is not frozen")
    authorized: dict[str, dict[str, str]] = {}
    source_documents: dict[Path, dict[str, Any]] = {}
    for entry in manifest["entries"]:
        if not isinstance(entry, dict) or set(entry) != SOURCE_TRACE_ENTRY_KEYS:
            raise CandidateGateError("authorized source Trace entry schema is invalid")
        if entry.get("split") not in {"discovery", "dev"}:
            raise CandidateGateError("authorized source Trace split is invalid")
        trace_id = entry.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id or trace_id in authorized:
            raise CandidateGateError("authorized source Trace ID is invalid or duplicated")
        source_path = ROOT / str(entry.get("source_path", ""))
        try:
            source_path.resolve(strict=True).relative_to(ROOT.resolve(strict=True))
        except (FileNotFoundError, ValueError):
            raise CandidateGateError("authorized source Trace file is missing or outside the workspace")
        if _sha(source_path) != entry.get("source_file_sha256"):
            raise CandidateGateError("authorized source Trace file hash mismatch")
        document = source_documents.setdefault(source_path, _json(source_path))
        traces = document.get("traces")
        if not isinstance(traces, list):
            raise CandidateGateError("authorized source Trace document is invalid")
        matches = [item for item in traces if isinstance(item, dict) and item.get("trace_id") == trace_id]
        if len(matches) != 1:
            raise CandidateGateError("authorized source Trace is missing or duplicated")
        if sha256_bytes(canonical_json_bytes(matches[0])) != entry.get("trace_sha256"):
            raise CandidateGateError("authorized source Trace content hash mismatch")
        authorized[trace_id] = entry
    return authorized


def _candidate_gate_a_trace_document() -> dict[str, Any]:
    authorized = load_authorized_source_traces()
    if set(authorized) != set(CANDIDATE_SOURCE_TRACE_IDS):
        raise CandidateGateError("Candidate Gate A source manifest must contain only TASK037-DEV-1/2/3")
    document = _json(DEV_TRACE_PATH)
    if (
        document.get("schema_version") != 1
        or document.get("dataset_version") != "task-037-dev-traces-v1"
        or document.get("role") != "candidate_input_only"
        or not isinstance(document.get("feedback"), str)
        or not document["feedback"].strip()
        or not isinstance(document.get("traces"), list)
    ):
        raise CandidateGateError("Candidate Gate A dev Trace document is invalid")
    traces = document["traces"]
    if [item.get("trace_id") for item in traces if isinstance(item, dict)] != list(CANDIDATE_SOURCE_TRACE_IDS):
        raise CandidateGateError("Candidate Gate A requires the three frozen TASK037 dev Traces in order")
    return document


def _candidate_gate_a_observations() -> tuple[TaskObservation, ...]:
    document = _candidate_gate_a_trace_document()
    skill_sha = _sha(ACTIVE_SKILL_PATH)
    observations: list[TaskObservation] = []
    required = {
        "trace_id", "task_id", "intent", "preconditions", "expected_outcome",
        "validator_status", "contract_checks", "tool_calls",
    }
    for item in document["traces"]:
        if not isinstance(item, dict) or set(item) != required:
            raise CandidateGateError("Candidate Gate A dev Trace schema is invalid")
        observations.append(TaskObservation(
            trace={
                "trace_id": item["trace_id"],
                "task_id": item["task_id"],
                "skill": {"sha256": skill_sha},
                "result_status": "succeeded",
                "failure_code": None,
                "tool_calls": item["tool_calls"],
                "feedback": document["feedback"],
            },
            intent=str(item["intent"]),
            preconditions=tuple(item["preconditions"]),
            expected_outcome=str(item["expected_outcome"]),
            validator_status=str(item["validator_status"]),
            contract_checks={str(key): str(value) for key, value in item["contract_checks"].items()},
        ))
    return tuple(observations)


async def _candidate_gate_a_cards() -> tuple[ExperienceCard, ...]:
    report = await SignalExtractor().extract(_candidate_gate_a_observations())
    cards = tuple(card for card in report.cards if card.reusable)
    if (
        len(cards) != 1
        or cards[0].verification_strength != "strong"
        or set(cards[0].trace_ids) != set(CANDIDATE_SOURCE_TRACE_IDS)
        or cards[0].skill_sha256 != _sha(ACTIVE_SKILL_PATH)
    ):
        raise CandidateGateError("Candidate Gate A requires one verified card from exactly three dev Traces")
    return cards


def _candidate_gate_a_anchor_table(
    *,
    active_content: str | None = None,
) -> dict[str, Any]:
    table = _json(CANDIDATE_GATE_A_ANCHOR_TABLE_PATH)
    if set(table) != {
        "schema_version", "anchor_table_version", "active_skill_sha256", "anchors",
    }:
        raise CandidateGateError("Candidate Gate A anchor table schema is invalid")
    if (
        table.get("schema_version")
        != "healthvault-task-045-candidate-gate-a-anchor-table-v1"
        or table.get("anchor_table_version") != "task-045-candidate-gate-a-anchors-v1"
        or not isinstance(table.get("anchors"), list)
        or not table["anchors"]
    ):
        raise CandidateGateError("Candidate Gate A anchor table version is invalid")
    anchors: list[dict[str, Any]] = []
    anchor_ids: set[str] = set()
    covered_sections: set[str] = set()
    for entry in table["anchors"]:
        if not isinstance(entry, dict) or set(entry) != {
            "target_anchor_id", "target_section", "target_paragraph", "allowed_operations",
        }:
            raise CandidateGateError("Candidate Gate A anchor entry schema is invalid")
        anchor_id = entry.get("target_anchor_id")
        section = entry.get("target_section")
        paragraph = entry.get("target_paragraph")
        operations = entry.get("allowed_operations")
        if (
            not isinstance(anchor_id, str)
            or not anchor_id.strip()
            or anchor_id in anchor_ids
            or not isinstance(section, str)
            or section not in {"Steps", "Failure Handling", "Verification"}
            or not isinstance(paragraph, str)
            or not paragraph.strip()
            or not isinstance(operations, list)
            or not operations
            or len(operations) != len(set(operations))
            or not set(operations) <= {"add", "replace", "delete"}
        ):
            raise CandidateGateError("Candidate Gate A anchor entry is invalid")
        anchor_ids.add(anchor_id)
        covered_sections.add(section)
        anchors.append(dict(entry))
    if covered_sections != {"Steps", "Failure Handling", "Verification"}:
        raise CandidateGateError("Candidate Gate A anchor table does not cover required sections")
    content = active_content if active_content is not None else ACTIVE_SKILL_PATH.read_text(encoding="utf-8")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != table.get("active_skill_sha256"):
        raise CandidateGateError("Candidate Gate A anchor table active Skill hash mismatch")
    normalized = dict(table)
    normalized["anchors"] = anchors
    return normalized


def resolve_candidate_gate_a_anchor(
    target_anchor_id: str,
    operation: str,
    *,
    active_content: str,
    anchor_table: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Resolve one exact, versioned anchor without fuzzy or semantic matching."""
    table = dict(anchor_table) if anchor_table is not None else _candidate_gate_a_anchor_table(
        active_content=active_content,
    )
    if hashlib.sha256(active_content.encode("utf-8")).hexdigest() != table.get(
        "active_skill_sha256"
    ):
        raise CandidateGateError("Candidate Gate A anchor table active Skill hash mismatch")
    anchors = table.get("anchors")
    matches = [
        entry for entry in anchors if isinstance(entry, dict)
        and entry.get("target_anchor_id") == target_anchor_id
    ] if isinstance(anchors, list) else []
    if len(matches) != 1:
        raise CandidateGateError("Candidate Gate A unknown target_anchor_id")
    anchor = matches[0]
    allowed_operations = anchor.get("allowed_operations")
    if not isinstance(allowed_operations, list) or operation not in allowed_operations:
        raise CandidateGateError(
            f"Candidate Gate A operation is not allowed for anchor:{target_anchor_id}:{operation}"
        )
    section = str(anchor.get("target_section", ""))
    paragraph = str(anchor.get("target_paragraph", ""))
    spans = patching._section_spans(active_content)
    if section not in spans:
        raise CandidateGateError(
            f"Candidate Gate A anchor section is missing:{target_anchor_id}:{section}"
        )
    start, end = spans[section]
    if active_content[start:end].count(paragraph) != 1:
        raise CandidateGateError(
            f"Candidate Gate A anchor paragraph is missing or ambiguous:{target_anchor_id}"
        )
    return {"target_section": section, "target_paragraph": paragraph}


def _candidate_gate_a_patch_schema(
    *,
    active_content: str | None = None,
) -> dict[str, Any]:
    table = _candidate_gate_a_anchor_table(active_content=active_content)
    schema = _json(CANDIDATE_GATE_A_PATCH_SCHEMA_PATH)
    try:
        patches_contract = schema["properties"]["patches"]
        item = patches_contract["items"]
        required = set(item["required"])
        properties = item["properties"]
        anchor_enum = properties["target_anchor_id"]["enum"]
        operation_enum = properties["operation"]["enum"]
        tool_enum = properties["tool_path"]["items"]["enum"]
        content_contract = properties["content"]
    except (KeyError, TypeError) as exc:
        raise CandidateGateError("Candidate Gate A Patch schema is invalid") from exc
    expected_required = set(patching.PATCH_SCHEMA["properties"]["patches"]["items"]["required"])
    expected_required -= {"target_section", "target_paragraph"}
    expected_required.add("target_anchor_id")
    anchor_ids = [entry["target_anchor_id"] for entry in table["anchors"]]
    anchor_operation_sets = [entry["allowed_operations"] for entry in table["anchors"]]
    if (
        schema.get("$id") != "healthvault-task-045-candidate-gate-a-patch-v1"
        or patches_contract.get("minItems") != CANDIDATE_GATE_A_MAX_PATCHES
        or patches_contract.get("maxItems") != CANDIDATE_GATE_A_MAX_PATCHES
        or required != expected_required
        or "target_section" in properties
        or "target_paragraph" in properties
        or anchor_enum != anchor_ids
        or not isinstance(operation_enum, list)
        or not operation_enum
        or tool_enum != list(patching.ALLOWED_TOOL_PATH_VALUES)
        or any(allowed_operations != operation_enum for allowed_operations in anchor_operation_sets)
        or content_contract.get("minLength") != 1
        or content_contract.get("maxLength") != CANDIDATE_GATE_A_CONTENT_MAX_LENGTH
        or content_contract.get("pattern") != CANDIDATE_GATE_A_CONTENT_PATTERN
    ):
        raise CandidateGateError("Candidate Gate A Patch schema and anchor table disagree")
    return schema


def _safe_candidate_gate_a_patch_controls(
    response: object,
    *,
    active_content: str,
) -> list[dict[str, Any]]:
    if not isinstance(response, dict) or not isinstance(response.get("patches"), list):
        return []
    table = _candidate_gate_a_anchor_table(active_content=active_content)
    known_anchors = {entry["target_anchor_id"] for entry in table["anchors"]}
    controls: list[dict[str, Any]] = []
    for raw in response["patches"]:
        if not isinstance(raw, dict):
            continue
        patch_id = raw.get("patch_id")
        safe_patch_id = (
            patch_id if isinstance(patch_id, str)
            and re.fullmatch(r"(?:PATCH|TASK037)[A-Za-z0-9._-]{0,120}", patch_id)
            else "[REDACTED]"
        )
        operation = raw.get("operation")
        safe_operation = operation if operation in {"add", "replace", "delete"} else "[REDACTED]"
        anchor_id = raw.get("target_anchor_id")
        safe_anchor_id = anchor_id if anchor_id in known_anchors else "[REDACTED]"
        resolved_section: str | None = None
        if isinstance(anchor_id, str) and isinstance(operation, str):
            try:
                resolved_section = resolve_candidate_gate_a_anchor(
                    anchor_id,
                    operation,
                    active_content=active_content,
                    anchor_table=table,
                )["target_section"]
            except CandidateGateError:
                resolved_section = None
        tool_values = raw.get("tool_path")
        safe_tools = [
            value if isinstance(value, str) and value in patching.ALLOWED_TOOL_PATH else "[REDACTED]"
            for value in tool_values
        ] if isinstance(tool_values, list) else []
        controls.append({
            "patch_id": safe_patch_id,
            "operation": safe_operation,
            "target_anchor_id": safe_anchor_id,
            "target_anchor_id_sha256": (
                sha256_bytes(anchor_id.encode("utf-8")) if isinstance(anchor_id, str) else None
            ),
            "resolved_target_section": resolved_section,
            "tool_path": safe_tools,
        })
    return controls


def _resolve_candidate_gate_a_model_response(
    response: object,
    *,
    active_content: str,
) -> dict[str, Any]:
    schema = _candidate_gate_a_patch_schema(active_content=active_content)
    if not isinstance(response, dict) or set(response) != {"patches"}:
        raise CandidateGateError("Candidate Gate A Patch response schema is invalid")
    patches = response.get("patches")
    if not isinstance(patches, list) or len(patches) != CANDIDATE_GATE_A_MAX_PATCHES:
        raise CandidateGateError("Candidate Gate A Patch count is invalid")
    required = set(schema["properties"]["patches"]["items"]["required"])
    resolved: list[dict[str, Any]] = []
    for raw in patches:
        if not isinstance(raw, dict):
            raise CandidateGateError("Candidate Gate A Patch entry is invalid")
        if "target_paragraph" in raw or "target_section" in raw:
            raise CandidateGateError("Candidate Gate A free-text target fields are forbidden")
        if set(raw) != required:
            raise CandidateGateError("Candidate Gate A Patch entry schema is invalid")
        content = raw.get("content")
        if (
            not isinstance(content, str)
            or not 1 <= len(content) <= CANDIDATE_GATE_A_CONTENT_MAX_LENGTH
            or re.fullmatch(CANDIDATE_GATE_A_CONTENT_PATTERN, content) is None
        ):
            raise CandidateGateError(
                "Candidate Gate A Patch content must be one sentence of at most 240 characters"
            )
        anchor_id = raw.get("target_anchor_id")
        operation = raw.get("operation")
        if not isinstance(anchor_id, str) or not isinstance(operation, str):
            raise CandidateGateError("Candidate Gate A anchor selection is invalid")
        target = resolve_candidate_gate_a_anchor(
            anchor_id,
            operation,
            active_content=active_content,
        )
        converted = dict(raw)
        converted.pop("target_anchor_id")
        converted.update(target)
        resolved.append(converted)
    return {"patches": resolved}


class PatchModelSideQuery:
    """One bounded Patch-generation request containing no held-out evaluation data."""

    def __init__(
        self,
        *,
        transport: Any,
        api_key: str,
        model_config: Mapping[str, Any],
        active_skill_content: str,
    ) -> None:
        self.transport = transport
        self.api_key = api_key
        self.model_config = dict(model_config)
        self.active_skill_content = active_skill_content
        self.transport_requests = 0
        self.api_calls = 0
        self.total_tokens = 0
        self.last_payload: dict[str, Any] | None = None
        self.attempts: list[dict[str, Any]] = []
        self.last_safe_patch_controls: list[dict[str, Any]] = []

    async def __call__(self, prompt: str, schema: dict[str, Any]) -> str:
        if schema != patching.PATCH_SCHEMA:
            raise ValueError("Candidate Gate A expected the TASK-044 compatibility schema")
        request = json.loads(prompt)
        expected_keys = {
            "target_skill", "active_skill_sha256", "active_skill",
            "eligible_experience_cards", "rules",
        }
        if not isinstance(request, dict) or set(request) != expected_keys:
            raise ValueError("Candidate Gate A Patch request schema is invalid")
        if (
            request.get("target_skill") != "HealthVault-Task-Execution"
            or request.get("active_skill") != self.active_skill_content
            or request.get("active_skill_sha256")
            != hashlib.sha256(self.active_skill_content.encode("utf-8")).hexdigest()
        ):
            raise ValueError("Candidate Gate A active Skill binding mismatch")
        cards = request.get("eligible_experience_cards")
        trace_ids = {
            trace_id
            for card in cards if isinstance(cards, list) and isinstance(card, dict)
            for trace_id in card.get("trace_ids", [])
        } if isinstance(cards, list) else set()
        if trace_ids != set(CANDIDATE_SOURCE_TRACE_IDS):
            raise ValueError("Candidate Gate A generator input contains an unauthorized Trace")
        anchor_table = _candidate_gate_a_anchor_table(active_content=self.active_skill_content)
        gate_schema = _candidate_gate_a_patch_schema(active_content=self.active_skill_content)
        request["authorized_dev_traces"] = _candidate_gate_a_trace_document()["traces"]
        request["candidate_gate_a_anchor_catalog"] = [
            {
                "target_anchor_id": entry["target_anchor_id"],
                "target_section": entry["target_section"],
                "allowed_operations": list(entry["allowed_operations"]),
            }
            for entry in anchor_table["anchors"]
        ]
        request["rules"] += (
            " Choose exactly one target_anchor_id from candidate_gate_a_anchor_catalog for each "
            "Patch. Do not return target_section or target_paragraph; the local resolver supplies "
            "both exact values."
        )
        payload: dict[str, Any] = {
            "model": self.model_config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object matching the supplied local Patch schema. "
                        "Generate exactly one sourced, narrowly scoped workflow Patch. Do not rewrite "
                        "the complete Skill and do not add medical facts, knowledge content, "
                        "diagnosis or treatment rules, doses, reference values, safety boundaries, "
                        "permission policy, patient facts, case IDs, Gold, or expected answers. "
                        "Every Patch tool_path item must be exactly one of: "
                        + ", ".join(patching.ALLOWED_TOOL_PATH_VALUES)
                        + ". Never put a result-status suffix, a natural-language "
                        "step, or a nonexistent tool name in tool_path. Preserve observed status "
                        "transitions only through applicability_conditions or stop_conditions. "
                        "Choose only target_anchor_id from the supplied enum. Never return "
                        "target_section or target_paragraph. Patch content must be one short "
                        "workflow sentence of at most 240 characters, with no list, heading, or "
                        "second sentence. "
                        + json.dumps(gate_schema, ensure_ascii=False, sort_keys=True)
                    ),
                },
                {"role": "user", "content": json.dumps(request, ensure_ascii=False, sort_keys=True)},
            ],
            "response_format": {"type": "json_object"},
            "do_sample": False,
            "thinking": dict(self.model_config["thinking"]),
            "reasoning_effort": self.model_config["reasoning_effort"],
            "max_tokens": self.model_config["max_output_tokens"],
            "stream": False,
        }
        self.last_payload = json.loads(json.dumps(payload, ensure_ascii=False))
        last_error: Exception | None = None
        for retry in range(2):
            self.transport_requests += 1
            if self.transport.is_real:
                self.api_calls += 1
            try:
                response = await self.transport.post_json(
                    f"{str(self.model_config['api_base']).rstrip('/')}/chat/completions",
                    {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    payload,
                    float(self.model_config["timeout_seconds"]),
                )
            except Exception as exc:
                if not isinstance(getattr(exc, "retryable", None), bool):
                    raise
                self.attempts.append({
                    "sequence": self.transport_requests,
                    "status": "transport_error",
                    "retryable": bool(getattr(exc, "retryable")),
                    "response_sha256": None,
                    "usage": None,
                    "provider_error": _safe_transport_diagnostics(exc),
                })
                last_error = exc
                if not bool(getattr(exc, "retryable")) or retry == 1:
                    raise
                continue
            usage = response.get("usage") if isinstance(response, dict) else None
            if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
                self.total_tokens += int(usage["total_tokens"])
            self.attempts.append({
                "sequence": self.transport_requests,
                "status": "model_response",
                "retryable": False,
                "response_sha256": sha256_bytes(canonical_json_bytes(response)),
                "usage": dict(usage) if isinstance(usage, dict) else None,
            })
            choices = response.get("choices") if isinstance(response, dict) else None
            message = choices[0].get("message") if isinstance(choices, list) and choices else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Candidate Gate A model response has no Patch JSON")
            try:
                raw_response = json.loads(content)
            except json.JSONDecodeError as exc:
                raise CandidateGateError("Candidate Gate A model response is invalid JSON") from exc
            self.last_safe_patch_controls = _safe_candidate_gate_a_patch_controls(
                raw_response,
                active_content=self.active_skill_content,
            )
            resolved = _resolve_candidate_gate_a_model_response(
                raw_response,
                active_content=self.active_skill_content,
            )
            return json.dumps(resolved, ensure_ascii=False, sort_keys=True)
        raise last_error or RuntimeError("Candidate Gate A transport failed")


def _candidate_gate_a_regression(active: str, candidate: str) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    for start, end, reason in (
        ("## When to Use", "## Prerequisites", "when_to_use_changed"),
        ("## Prerequisites", "## Steps", "prerequisites_changed"),
        ("## Safety Boundaries", None, "safety_boundaries_changed"),
    ):
        active_part = active.split(start, 1)[1].split(end, 1)[0] if end else active.split(start, 1)[1]
        candidate_part = candidate.split(start, 1)[1].split(end, 1)[0] if end else candidate.split(start, 1)[1]
        if active_part != candidate_part:
            reasons.append(reason)
    for text in (
        "Run only approved read-only domain tools and preserve returned citations.",
        "Return the tool status accurately; never present a rejection as success.",
    ):
        if text not in candidate:
            reasons.append("existing_safe_workflow_removed")
    lowered = candidate.lower()
    if "retryable" not in lowered or "validator" not in lowered:
        reasons.append("verified_retry_and_validation_workflow_missing")
    return not reasons, tuple(dict.fromkeys(reasons))


def _candidate_gate_a_model(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    values = dict(environ or {})
    return dict(CANDIDATE_GATE_A_MODEL) | {
        "api_base": str(values.get("HEALTHVAULT_OPENAI_BASE_URL", CANDIDATE_GATE_A_MODEL["api_base"])),
    }


def _safe_transport_diagnostics(error: BaseException) -> dict[str, Any] | None:
    reader = getattr(error, "safe_diagnostics", None)
    if not callable(reader):
        return None
    value = reader()
    if not isinstance(value, dict):
        return None
    allowed = {
        "http_status": int,
        "provider_error_code": str,
        "provider_error_message": str,
        "request_id": str,
    }
    safe: dict[str, Any] = {}
    for key, expected_type in allowed.items():
        item = value.get(key)
        if isinstance(item, expected_type):
            safe[key] = item
    return safe or None


def _candidate_gate_a_auth_probe_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": model["model"],
        "messages": [{"role": "user", "content": "Reply OK."}],
        "do_sample": False,
        "thinking": {"type": "disabled"},
        "max_tokens": CANDIDATE_GATE_A_AUTH_PROBE_MAX_TOKENS,
        "stream": False,
    }


def build_candidate_gate_a_auth_probe_preflight(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    model = _candidate_gate_a_model(environ)
    request_payload = _candidate_gate_a_auth_probe_payload(model)
    payload: dict[str, Any] = {
        "schema_version": "healthvault-task-045-candidate-gate-a-auth-probe-v1",
        "task": "TASK-045",
        "gate": "Candidate-A-Auth-Probe",
        "status": "API_GATE_PENDING",
        "api_calls_made": 0,
        "model": model,
        "limits": {
            "maximum_api_calls": 1,
            "maximum_transport_attempts": 1,
            "maximum_output_tokens": CANDIDATE_GATE_A_AUTH_PROBE_MAX_TOKENS,
        },
        "request": {
            "method": "POST",
            "endpoint_suffix": "/chat/completions",
            "payload_sha256": sha256_bytes(canonical_json_bytes(request_payload)),
            "candidate_generation_forbidden": True,
            "response_content_persistence_forbidden": True,
        },
        "bindings": {
            "transport_source_sha256": _sha(TRANSPORT_SOURCE_PATH),
            "probe_runner_sha256": _sha(Path(__file__)),
        },
    }
    payload["preflight_sha256"] = preflight_sha256(payload)
    return payload


def _candidate_gate_a_auth_probe_path(preflight_hash: str) -> Path:
    return CANDIDATE_GATE_A_AUTH_PROBE_ROOT / f"{preflight_hash}.json"


def _candidate_gate_a_auth_probe_seal_path(preflight_hash: str) -> Path:
    return CANDIDATE_GATE_A_AUTH_PROBE_ROOT / f"{preflight_hash}-seal.json"


def _current_candidate_gate_a_auth_probe_preflight(preflight_hash: str) -> dict[str, Any]:
    if not CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH.is_file():
        raise CandidateGateError("Candidate Gate A auth probe preflight is missing")
    stored = _json(CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH)
    if (
        stored.get("preflight_sha256") != preflight_hash
        or preflight_sha256(stored) != preflight_hash
    ):
        raise CandidateGateError("Candidate Gate A auth probe preflight self-hash is invalid")
    model = stored.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("api_base"), str):
        raise CandidateGateError("Candidate Gate A auth probe preflight bindings are stale")
    current = build_candidate_gate_a_auth_probe_preflight({
        "HEALTHVAULT_OPENAI_BASE_URL": model["api_base"],
    })
    if current["preflight_sha256"] != preflight_hash:
        raise CandidateGateError("Candidate Gate A auth probe preflight bindings are stale")
    return stored


def _verified_candidate_gate_a_auth_probe(
    preflight_hash: str,
) -> dict[str, Any]:
    if not isinstance(preflight_hash, str) or re.fullmatch(r"[0-9a-f]{64}", preflight_hash) is None:
        raise CandidateGateError("Candidate Gate A auth probe hash is invalid")
    _current_candidate_gate_a_auth_probe_preflight(preflight_hash)
    preflight_artifact_sha256 = _sha(CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH)
    path = _candidate_gate_a_auth_probe_path(preflight_hash)
    if not path.is_file():
        raise CandidateGateError("Candidate Gate A auth probe result is missing")
    record = _json(path)
    expected_record_keys = {
        "schema_version", "preflight_sha256", "preflight_artifact_sha256",
        "status", "disposition", "started_at_utc", "finished_at_utc",
        "api_calls", "transport_attempts", "total_tokens", "response_sha256",
        "request_id", "provider_error", "failure_class", "failure_reason",
        "candidate_generated", "response_content_persisted",
    }
    if (
        set(record) != expected_record_keys
        or record.get("schema_version") != "healthvault-task-045-candidate-gate-a-auth-probe-result-v1"
        or record.get("preflight_sha256") != preflight_hash
        or record.get("preflight_artifact_sha256") != preflight_artifact_sha256
        or record.get("status") != "passed"
        or record.get("disposition") != "authentication_verified"
        or record.get("candidate_generated") is not False
        or record.get("response_content_persisted") is not False
        or record.get("provider_error") is not None
        or record.get("failure_class") is not None
        or record.get("failure_reason") is not None
        or record.get("transport_attempts") != 1
        or not isinstance(record.get("api_calls"), int)
        or record.get("api_calls") not in {0, 1}
        or not isinstance(record.get("total_tokens"), int)
        or record.get("total_tokens", -1) < 0
        or not isinstance(record.get("response_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", record["response_sha256"]) is None
    ):
        raise CandidateGateError("Candidate Gate A auth probe did not pass")
    seal_path = _candidate_gate_a_auth_probe_seal_path(preflight_hash)
    if not seal_path.is_file():
        raise CandidateGateError("Candidate Gate A auth probe result seal is missing")
    seal = _json(seal_path)
    if (
        set(seal) != {"schema_version", "preflight_sha256", "result_sha256"}
        or seal.get("schema_version") != "healthvault-task-045-candidate-gate-a-auth-probe-seal-v1"
        or seal.get("preflight_sha256") != preflight_hash
        or seal.get("result_sha256") != _sha(path)
    ):
        raise CandidateGateError("Candidate Gate A auth probe result hash is invalid")
    return {
        "preflight_sha256": preflight_hash,
        "preflight_artifact_sha256": preflight_artifact_sha256,
        "result_artifact_sha256": _sha(path),
        "result_seal_sha256": _sha(seal_path),
        "request_id": record.get("request_id"),
    }


def _candidate_gate_a_auth_probe_hash(preflight: Mapping[str, Any]) -> str | None:
    binding = preflight.get("auth_probe")
    if not isinstance(binding, dict):
        return None
    value = binding.get("preflight_sha256")
    return value if isinstance(value, str) else None


def build_candidate_gate_a_preflight(
    environ: Mapping[str, str] | None = None,
    *,
    auth_probe_hash: str | None = None,
) -> dict[str, Any]:
    """Bind candidate generation without loading TASK-045 held-out data."""
    authorized = load_authorized_source_traces()
    _candidate_gate_a_trace_document()
    active_content = ACTIVE_SKILL_PATH.read_text(encoding="utf-8")
    anchor_table = _candidate_gate_a_anchor_table(active_content=active_content)
    _candidate_gate_a_patch_schema(active_content=active_content)
    for anchor in anchor_table["anchors"]:
        for operation in anchor["allowed_operations"]:
            resolve_candidate_gate_a_anchor(
                anchor["target_anchor_id"],
                operation,
                active_content=active_content,
                anchor_table=anchor_table,
            )
    payload: dict[str, Any] = {
        "schema_version": "healthvault-task-045-candidate-gate-a-v1",
        "task": "TASK-045",
        "gate": "Candidate-A",
        "status": "API_GATE_PENDING",
        "api_calls_made": 0,
        "candidate_model": _candidate_gate_a_model(environ),
        "limits": {
            "candidate_generation_calls": 1,
            "maximum_model_calls": 1,
            "maximum_transport_attempts": 2,
        },
        "generator_inputs": {
            "active_skill_sha256": _sha(ACTIVE_SKILL_PATH),
            "source_trace_ids": list(CANDIDATE_SOURCE_TRACE_IDS),
            "source_splits": ["dev"],
            "anchor_table_version": anchor_table["anchor_table_version"],
            "source_trace_sha256": {
                trace_id: authorized[trace_id]["trace_sha256"]
                for trace_id in CANDIDATE_SOURCE_TRACE_IDS
            },
            "heldout_or_oracle_exposed": False,
        },
        "bindings": {
            "active_skill_sha256": _sha(ACTIVE_SKILL_PATH),
            "dev_trace_sha256": _sha(DEV_TRACE_PATH),
            "authorized_source_trace_manifest_sha256": _sha(SOURCE_TRACE_MANIFEST_PATH),
            "patch_pipeline_sha256": _sha(PATCH_PIPELINE_PATH),
            "patch_schema_sha256": _sha(PATCH_SCHEMA_PATH),
            "candidate_gate_a_patch_schema_sha256": _sha(
                CANDIDATE_GATE_A_PATCH_SCHEMA_PATH
            ),
            "candidate_gate_a_anchor_table_sha256": _sha(
                CANDIDATE_GATE_A_ANCHOR_TABLE_PATH
            ),
            "candidate_gate_a_anchor_resolver_sha256": sha256_bytes(
                inspect.getsource(resolve_candidate_gate_a_anchor).encode("utf-8")
            ),
            "signal_extractor_sha256": _sha(ROOT / "healthvault/evolution/signal_extractor.py"),
            "candidate_verifier_sha256": sha256_bytes(
                inspect.getsource(CandidateVerifier).encode("utf-8")
            ),
            "gate_runner_sha256": _sha(Path(__file__)),
        },
        "protocol": {
            "patch_only": True,
            "target_anchor_id_required": True,
            "free_text_target_forbidden": True,
            "task_044_verifier_and_regression_required": True,
            "post_generation_task_045_leakage_check_required": True,
            "candidate_status": "frozen_unpublished",
            "active_skill_write_forbidden": True,
            "formal_ab_forbidden": True,
        },
    }
    if auth_probe_hash is not None:
        payload["auth_probe"] = _verified_candidate_gate_a_auth_probe(auth_probe_hash)
    payload["preflight_sha256"] = preflight_sha256(payload)
    return payload


def assert_candidate_gate_a(
    preflight: Mapping[str, Any],
    *,
    approved: bool,
    supplied_hash: str,
    api_key: str,
) -> None:
    expected = preflight_sha256(preflight)
    if (
        not approved
        or not api_key
        or supplied_hash != expected
        or preflight.get("preflight_sha256") != expected
    ):
        raise ApiGateError("TASK-045 Candidate Gate A requires approval of the exact current hash")


def assert_candidate_gate_a_auth_probe(
    preflight: Mapping[str, Any],
    *,
    approved: bool,
    supplied_hash: str,
    api_key: str,
) -> None:
    expected = preflight_sha256(preflight)
    if (
        not approved
        or not api_key
        or supplied_hash != expected
        or preflight.get("preflight_sha256") != expected
    ):
        raise ApiGateError(
            "TASK-045 Candidate Gate A auth probe requires approval of the exact current hash"
        )


async def execute_candidate_gate_a_auth_probe_once(
    preflight: Mapping[str, Any],
    *,
    approved: bool,
    supplied_hash: str,
    api_key: str,
    transport: Any,
) -> dict[str, Any]:
    """Run one minimal model-access probe without invoking Candidate generation."""
    assert_candidate_gate_a_auth_probe(
        preflight,
        approved=approved,
        supplied_hash=supplied_hash,
        api_key=api_key,
    )
    preflight_hash = str(preflight["preflight_sha256"])
    try:
        _current_candidate_gate_a_auth_probe_preflight(preflight_hash)
    except CandidateGateError as exc:
        raise ApiGateError(str(exc)) from exc
    preflight_artifact_sha256 = _sha(CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH)
    result_path = _candidate_gate_a_auth_probe_path(preflight_hash)
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record: dict[str, Any] = {
        "schema_version": "healthvault-task-045-candidate-gate-a-auth-probe-result-v1",
        "preflight_sha256": preflight_hash,
        "preflight_artifact_sha256": preflight_artifact_sha256,
        "status": "started",
        "disposition": "probe_consumed",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "api_calls": 0,
        "transport_attempts": 0,
        "total_tokens": 0,
        "response_sha256": None,
        "request_id": None,
        "provider_error": None,
        "failure_class": None,
        "failure_reason": None,
        "candidate_generated": False,
        "response_content_persisted": False,
    }
    _exclusive_json(
        result_path,
        record,
        exists_message="TASK-045 Candidate Gate A auth probe hash is already consumed",
    )
    api_calls = 1 if bool(getattr(transport, "is_real", False)) else 0
    try:
        response = await transport.post_json(
            f"{str(preflight['model']['api_base']).rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            _candidate_gate_a_auth_probe_payload(preflight["model"]),
            float(preflight["model"]["timeout_seconds"]),
        )
        if not isinstance(response, dict):
            raise CandidateGateError("Candidate Gate A auth probe response is not an object")
        usage = response.get("usage")
        total_tokens = (
            int(usage["total_tokens"])
            if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int)
            else 0
        )
        record.update({
            "status": "passed",
            "disposition": "authentication_verified",
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "api_calls": api_calls,
            "transport_attempts": 1,
            "total_tokens": total_tokens,
            "response_sha256": sha256_bytes(canonical_json_bytes(response)),
            "request_id": None,
        })
        _atomic_json(result_path, record)
        seal_path = _candidate_gate_a_auth_probe_seal_path(preflight_hash)
        _atomic_json(seal_path, {
            "schema_version": "healthvault-task-045-candidate-gate-a-auth-probe-seal-v1",
            "preflight_sha256": preflight_hash,
            "result_sha256": _sha(result_path),
        })
        return {
            "status": "passed",
            "preflight_sha256": preflight_hash,
            "api_calls": api_calls,
            "transport_attempts": 1,
            "total_tokens": total_tokens,
            "response_sha256": record["response_sha256"],
            "request_id": record["request_id"],
            "candidate_generated": False,
            "result_path": str(result_path.resolve()),
            "result_sha256": _sha(result_path),
            "result_seal_path": str(seal_path.resolve()),
            "result_seal_sha256": _sha(seal_path),
        }
    except BaseException as exc:
        record.update({
            "status": "failed",
            "disposition": "rejected_consumed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "api_calls": api_calls,
            "transport_attempts": 1,
            "provider_error": _safe_transport_diagnostics(exc),
            "failure_class": type(exc).__name__,
            "failure_reason": str(exc),
        })
        _atomic_json(result_path, record)
        raise


def _candidate_gate_a_consumption_path(preflight_hash: str) -> Path:
    return CANDIDATE_GATE_A_CONSUMPTION_ROOT / f"{preflight_hash}.json"


def _candidate_gate_a_validation_failure_path(preflight_hash: str) -> Path:
    return CANDIDATE_GATE_A_CONSUMPTION_ROOT / f"{preflight_hash}-validation-failure.json"


def _redacted_rejected_tool_path(value: str) -> dict[str, Any]:
    """Retain safe tool/status evidence without persisting arbitrary model text."""
    tool_name, separator, result_status = value.partition(":")
    safe_status_transition = (
        tool_name in patching.ALLOWED_TOOL_PATH
        and bool(separator)
        and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", result_status) is not None
    )
    return {
        "value": value if safe_status_transition else "[REDACTED]",
        "value_sha256": sha256_bytes(value.encode("utf-8")),
        "redacted": not safe_status_transition,
        "classification": (
            "status_qualified_tool" if safe_status_transition else "untrusted_out_of_scope_value"
        ),
    }


def _write_candidate_gate_a_validation_failure(
    *,
    preflight_hash: str,
    side_query: Any,
    preview: Any,
    failure: BaseException,
) -> tuple[Path, str] | None:
    attempts = list(getattr(side_query, "attempts", []))
    response_hashes = [
        item.get("response_sha256")
        for item in attempts
        if isinstance(item, dict)
        and item.get("status") == "model_response"
        and isinstance(item.get("response_sha256"), str)
    ]
    if not response_hashes:
        return None
    patches = tuple(getattr(preview, "patches", ()) or ())
    rejected_fields: list[dict[str, Any]] = []
    for patch_index, patch in enumerate(patches):
        out_of_scope = [
            str(value)
            for value in getattr(patch, "tool_path", ())
            if value not in patching.ALLOWED_TOOL_PATH
        ]
        if out_of_scope:
            rejected_fields.append({
                "patch_index": patch_index,
                "field": "tool_path",
                "values": [_redacted_rejected_tool_path(value) for value in out_of_scope],
            })
    reasons = [
        str(reason).split(":", 1)[0]
        for reason in (getattr(preview, "reasons", ()) or ())
    ]
    if not reasons:
        reasons = [str(failure).split(":", 1)[0] or type(failure).__name__]
    patch_controls = list(getattr(side_query, "last_safe_patch_controls", []) or [])
    artifact = {
        "schema_version": "healthvault-task-045-candidate-gate-a-validation-failure-v2",
        "preflight_sha256": preflight_hash,
        "disposition": "rejected_consumed",
        "response_sha256": response_hashes[-1],
        "validation_reasons": reasons,
        "parsed_patch_count": len(patch_controls) if patch_controls else len(patches),
        "patch_controls": patch_controls,
        "rejected_fields": rejected_fields,
        "redaction_policy": (
            "Only bounded workflow patch IDs, registered anchor IDs, allowlisted tool IDs, and "
            "safe status-qualified tool values are retained; all other model-provided values "
            "are replaced by a hash-bound redaction marker."
        ),
        "contains_api_key": False,
        "contains_authorization_header": False,
        "contains_raw_model_response": False,
    }
    path = _candidate_gate_a_validation_failure_path(preflight_hash)
    _atomic_json(path, artifact)
    return path.resolve(), _sha(path)


def _create_candidate_gate_a_consumption(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    """Claim one approved hash exactly once, before any transport attempt."""
    _exclusive_json(
        path,
        value,
        exists_message="TASK-045 Candidate Gate A approval hash is already consumed",
    )


def _validate_candidate_gate_a_chain(
    candidate_path: Path,
    candidate: Mapping[str, Any],
    *,
    require_consumption: bool,
) -> None:
    if not CANDIDATE_GATE_A_PREFLIGHT_PATH.is_file():
        raise CandidateGateError("Candidate Gate A preflight artifact is missing")
    gate = _json(CANDIDATE_GATE_A_PREFLIGHT_PATH)
    gate_hash = gate.get("preflight_sha256")
    if not isinstance(gate_hash, str) or gate_hash != preflight_sha256(gate):
        raise CandidateGateError("Candidate Gate A preflight self-hash is invalid")
    current = build_candidate_gate_a_preflight(
        {
            "HEALTHVAULT_OPENAI_BASE_URL": gate.get("candidate_model", {}).get("api_base", ""),
        },
        auth_probe_hash=_candidate_gate_a_auth_probe_hash(gate),
    )
    if gate_hash != current["preflight_sha256"]:
        raise CandidateGateError("Candidate Gate A preflight bindings are stale")
    if candidate.get("gate_a_preflight_sha256") != gate_hash:
        raise CandidateGateError("candidate is not bound to the approved Candidate Gate A preflight")
    if not require_consumption:
        return
    consumption_path = _candidate_gate_a_consumption_path(gate_hash)
    if not consumption_path.is_file():
        raise CandidateGateError("Candidate Gate A consumption record is missing")
    consumption = _json(consumption_path)
    patch_path = Path(str(candidate.get("patch_artifact_path", "")))
    if not patch_path.is_absolute():
        patch_path = ROOT / patch_path
    if (
        consumption.get("schema_version") != "healthvault-task-045-candidate-gate-a-consumption-v1"
        or consumption.get("status") != "completed"
        or consumption.get("disposition") != "frozen_unpublished"
        or consumption.get("preflight_sha256") != gate_hash
        or consumption.get("candidate_sha256") != candidate.get("candidate_sha256")
        or consumption.get("patch_artifact_sha256") != candidate.get("patch_artifact_sha256")
        or consumption.get("candidate_artifact_path") != str(candidate_path.resolve())
        or consumption.get("patch_artifact_path") != str(patch_path.resolve())
        or consumption.get("candidate_artifact_sha256") != _sha(candidate_path)
    ):
        raise CandidateGateError("Candidate Gate A completed consumption record does not match candidate bytes")


def _load_bound_patch_artifact(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = Path(str(candidate.get("patch_artifact_path", "")))
    artifact_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
    if not artifact_path.is_file():
        raise CandidateGateError("candidate Patch artifact is missing")
    if _sha(artifact_path) != candidate.get("patch_artifact_sha256"):
        raise CandidateGateError("candidate Patch artifact hash mismatch")
    artifact = _json(artifact_path)
    if set(artifact) != PATCH_ARTIFACT_KEYS:
        raise CandidateGateError("candidate Patch artifact schema is invalid")
    if (
        artifact.get("schema_version") != "healthvault-skill-local-patch-artifact-v1"
        or artifact.get("status") != "pending_manual_review"
        or artifact.get("reasons") != []
        or artifact.get("published") is not False
        or artifact.get("generator_calls") != 1
        or not isinstance(artifact.get("api_calls"), int)
        or not 0 <= artifact.get("api_calls") <= 2
        or artifact.get("active_skill_sha256") != candidate.get("active_skill_sha256")
        or artifact.get("candidate_skill_sha256") != candidate.get("candidate_sha256")
    ):
        raise CandidateGateError("candidate is not bound to a valid TASK-044 Patch artifact")
    try:
        patches = patching._parse_patch_response({"patches": artifact.get("patches")})
        merged, unified_diff = patching.merge_patches(
            ACTIVE_SKILL_PATH.read_text(encoding="utf-8"), patches,
        )
    except ValueError as exc:
        raise CandidateGateError(f"candidate Patch artifact cannot be replayed: {exc}") from exc
    if merged != candidate.get("candidate_content") or unified_diff != artifact.get("unified_diff"):
        raise CandidateGateError("candidate does not match the replayed TASK-044 Patch artifact")
    patch_trace_ids = {trace_id for patch in patches for trace_id in patch.source_trace_ids}
    if patch_trace_ids != set(candidate.get("source_trace_ids", [])):
        raise CandidateGateError("candidate source Traces do not match the TASK-044 Patch artifact")
    return artifact


def load_frozen_candidate(
    path: str | Path,
    *,
    require_gate_consumption: bool = True,
) -> dict[str, Any]:
    candidate_path = Path(path)
    candidate = _json(candidate_path)
    if set(candidate) != CANDIDATE_KEYS:
        raise CandidateGateError("candidate artifact schema is invalid")
    content = candidate.get("candidate_content")
    digest = candidate.get("candidate_sha256")
    if (
        candidate.get("schema_version") != "healthvault-task-045-candidate-v1"
        or candidate.get("status") != "frozen_unpublished"
        or candidate.get("verifier_status") != "verified"
        or candidate.get("active_skill_sha256") != _sha(ACTIVE_SKILL_PATH)
        or not isinstance(content, str)
        or not content.strip()
        or not isinstance(digest, str)
        or hashlib.sha256(content.encode("utf-8")).hexdigest() != digest
        or content == ACTIVE_SKILL_PATH.read_text(encoding="utf-8")
    ):
        raise CandidateGateError("candidate artifact is not a verified frozen local Patch")
    trace_ids = candidate.get("source_trace_ids")
    source_splits = candidate.get("source_splits")
    if not isinstance(trace_ids, list) or not trace_ids or len(trace_ids) != len(set(trace_ids)):
        raise CandidateGateError("candidate source Trace list is invalid")
    if not isinstance(source_splits, list) or not source_splits or not set(source_splits) <= {"discovery", "dev"}:
        raise CandidateGateError("candidate sources must be discovery/dev only")
    for field in (
        "candidate_id", "patch_artifact_path", "patch_artifact_sha256",
        "gate_a_preflight_sha256", "frozen_at_utc",
    ):
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            raise CandidateGateError(f"candidate field is invalid: {field}")
    if len(candidate["gate_a_preflight_sha256"]) != 64 or any(
        character not in "0123456789abcdef"
        for character in candidate["gate_a_preflight_sha256"]
    ):
        raise CandidateGateError("candidate Gate A preflight hash is invalid")
    patch_digest = candidate["patch_artifact_sha256"]
    if len(patch_digest) != 64 or any(character not in "0123456789abcdef" for character in patch_digest):
        raise CandidateGateError("candidate Patch artifact hash is invalid")
    if any(not isinstance(trace_id, str) or not trace_id.strip() for trace_id in trace_ids):
        raise CandidateGateError("candidate source Trace IDs must be non-empty text")
    authorized = load_authorized_source_traces()
    unknown = sorted(set(trace_ids) - set(authorized))
    if unknown:
        raise CandidateGateError("candidate contains a Trace absent from the authorized discovery/dev manifest")
    authoritative_splits = {authorized[trace_id]["split"] for trace_id in trace_ids}
    if set(source_splits) != authoritative_splits:
        raise CandidateGateError("candidate source splits do not match the authorized Trace manifest")
    _load_bound_patch_artifact(candidate)
    verification = CandidateVerifier("HealthVault-Task-Execution").verify_content(
        content,
        action="evolve",
        active_content=ACTIVE_SKILL_PATH.read_text(encoding="utf-8"),
    )
    if verification.status != "verified" or verification.content_sha256 != digest:
        raise CandidateGateError("candidate does not pass the current deterministic verifier")
    _validate_candidate_gate_a_chain(
        candidate_path,
        candidate,
        require_consumption=require_gate_consumption,
    )
    return candidate


def build_leakage_audit(
    dataset: Mapping[str, Any],
    *,
    candidate_path: str | Path | None = None,
    require_gate_consumption: bool = True,
) -> dict[str, Any]:
    fixture = load_fixture()
    formal = _split_document(SPLIT_PATH)
    split_entries = _legacy_split_entries() + list(formal["entries"])
    split_conflicts = find_patient_profile_split_conflicts(split_entries)
    fixture_pairs = {(item["patient_id"], item["profile_id"]) for item in fixture["profiles"]}
    formal_pairs = {(item["patient_id"], item["profile_id"]) for item in formal["entries"]}
    case_pairs = {(item["patient_id"], item["profile_id"]) for item in dataset["cases"]}
    if fixture_pairs != formal_pairs or case_pairs - formal_pairs:
        raise ValueError("TASK-045 fixture, split registry, and cases disagree")

    prior_entries = _prior_entries()
    current_entries = [_registry_entry(case) for case in dataset["cases"]]
    text_collisions = (
        _text_collisions(current_entries, prior_entries)
        + _internal_text_collisions(current_entries)
    )
    candidate_status = "pending"
    candidate_reasons: list[str] = ["candidate_not_frozen"]
    candidate_sha256 = None
    if candidate_path is not None:
        candidate = load_frozen_candidate(
            candidate_path,
            require_gate_consumption=require_gate_consumption,
        )
        candidate_status = "passed"
        candidate_reasons = []
        candidate_sha256 = candidate["candidate_sha256"]
        heldout_trace_ids = {f"TASK045-HELDOUT-{case['case_id']}" for case in dataset["cases"]}
        overlap = sorted(set(candidate["source_trace_ids"]) & heldout_trace_ids)
        if overlap:
            candidate_reasons.append("candidate_source_trace_overlaps_heldout")
        normalized_candidate = normalize_case_text(candidate["candidate_content"])
        for case in dataset["cases"]:
            prompt = normalize_case_text(case["prompt"])
            if prompt in normalized_candidate:
                candidate_reasons.append(f"candidate_contains_heldout_prompt:{case['case_id']}")
        heldout_identifiers: dict[str, str] = {}
        for case in dataset["cases"]:
            for kind in ("case_id", "case_family_id", "patient_id", "profile_id"):
                heldout_identifiers[str(case[kind])] = kind
            heldout_identifiers[f"TASK045-HELDOUT-{case['case_id']}"] = "trace_id"
        for profile in fixture["profiles"]:
            heldout_identifiers[str(profile["patient_id"])] = "patient_id"
            heldout_identifiers[str(profile["profile_id"])] = "profile_id"
            for report in profile["reports"]:
                heldout_identifiers[str(report["report_id"])] = "report_id"
                heldout_identifiers[str(report["source_id"])] = "source_id"
        for identifier, kind in heldout_identifiers.items():
            if re.search(
                rf"(?i)(?<![a-z0-9]){re.escape(identifier)}(?![a-z0-9])",
                candidate["candidate_content"],
            ):
                candidate_reasons.append(
                    f"candidate_contains_heldout_identifier:{kind}:{identifier}"
                )
        if _EVAL_TARGETING_PATTERN.search(candidate["candidate_content"]):
            candidate_reasons.append("candidate_contains_oracle_or_expected_answer_targeting")
        if candidate_reasons:
            candidate_status = "failed"
    baseline_passed = not split_conflicts and not text_collisions
    return {
        "schema_version": "healthvault-task-045-leakage-audit-v1",
        "dataset_version": dataset["dataset_version"],
        "checked_case_count": len(dataset["cases"]),
        "checked_prior_case_count": len(prior_entries),
        "formal_patient_count": len({item["patient_id"] for item in formal["entries"]}),
        "formal_profile_count": len({item["profile_id"] for item in formal["entries"]}),
        "patient_profile_split_conflict_count": len(split_conflicts),
        "patient_profile_split_conflicts": split_conflicts,
        "text_collision_count": len(text_collisions),
        "text_collisions": text_collisions,
        "candidate_audit_status": candidate_status,
        "candidate_audit_reasons": candidate_reasons,
        "candidate_sha256": candidate_sha256,
        "baseline_isolation_passed": baseline_passed,
        "passed": baseline_passed and candidate_status == "passed",
        "source_sha256": {
            _relative(DATASET_PATH): _sha(DATASET_PATH),
            _relative(FIXTURE_PATH): _sha(FIXTURE_PATH),
            _relative(SPLIT_PATH): _sha(SPLIT_PATH),
            _relative(oracle.DATASET_PATH): _sha(oracle.DATASET_PATH),
            _relative(oracle.DEV_CASES_PATH): _sha(oracle.DEV_CASES_PATH),
            _relative(oracle.DEV_SPLIT_PATH): _sha(oracle.DEV_SPLIT_PATH),
            _relative(oracle.FORMAL_SPLIT_PATH): _sha(oracle.FORMAL_SPLIT_PATH),
            **{_relative(path): _sha(path) for _, _, _, path in oracle.LEAKAGE_SOURCES},
            _relative(SOURCE_TRACE_MANIFEST_PATH): _sha(SOURCE_TRACE_MANIFEST_PATH),
            **({"candidate_artifact": _sha(Path(candidate_path))} if candidate_path else {}),
        },
    }


def build_case_matrix(dataset: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for case in dataset["cases"]:
        expected = case["expected"]
        rows.append({
            "case_id": case["case_id"],
            "category": case["category"],
            "patient_id": case["patient_id"],
            "profile_id": case["profile_id"],
            "outcome": {
                "tool_result_statuses": expected["required_tool_result_statuses"],
                "database_change_allowed": expected["database_change_allowed"],
            },
            "behavior": {
                "required_tools": expected["required_tools"],
                "max_calls": expected["max_calls"],
                "prohibited_writes": expected["prohibited_writes"],
            },
            "response": {
                "final_status": expected["final_status"],
                "refusal_required": expected["refusal_required"],
                "min_citations": expected["min_citations"],
                "max_citations": expected["max_citations"],
            },
        })
    return {
        "schema_version": "healthvault-task-045-case-matrix-v1",
        "dataset_version": dataset["dataset_version"],
        "case_count": len(rows),
        "oracle_order": ["outcome", "behavior", "response"],
        "rows": rows,
    }


def _model_config(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    values = dict(environ or {})
    return dict(MODEL_CONFIG) | {
        "api_base": str(values.get("HEALTHVAULT_OPENAI_BASE_URL", MODEL_CONFIG["api_base"])),
    }


def _preflight_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in value if key != "preflight_sha256"}


def preflight_sha256(preflight: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_preflight_payload(preflight)))


def build_preflight(
    candidate_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a formal preflight only after the local Patch candidate is frozen."""
    dataset = load_dataset()
    candidate_path = Path(candidate_path)
    candidate = load_frozen_candidate(candidate_path)
    audit = build_leakage_audit(dataset, candidate_path=candidate_path)
    if not audit["passed"]:
        raise CandidateGateError("candidate or held-out leakage audit failed")
    estimated_model_calls = 2 * sum(
        len(case["expected"]["required_tools"]) + 1
        for case in dataset["cases"]
    )
    payload: dict[str, Any] = {
        "schema_version": "healthvault-task-045-preflight-v1",
        "task": "TASK-045",
        "status": "API_GATE_PENDING",
        "api_calls_made": 0,
        "dataset_version": dataset["dataset_version"],
        "model": _model_config(environ),
        "arms": {
            "order": list(ARM_IDS),
            "alternating_first_arm_by_case": True,
            "only_variable": "skill_content",
            "active_skill_sha256": _sha(ACTIVE_SKILL_PATH),
            "candidate_skill_sha256": candidate["candidate_sha256"],
        },
        "limits": {
            "case_count": 20,
            "attempt_count": 40,
            "estimated_model_calls": estimated_model_calls,
            "maximum_model_calls": 160,
            "maximum_transport_attempts": 320,
            "max_turns": 4,
            "max_transport_retries": 1,
        },
        "publish_thresholds": dict(PUBLISH_THRESHOLDS),
        "bindings": {
            "dataset_sha256": _sha(DATASET_PATH),
            "fixture_sha256": _sha(FIXTURE_PATH),
            "split_registry_sha256": _sha(SPLIT_PATH),
            "candidate_artifact_sha256": _sha(candidate_path),
            "candidate_sha256": candidate["candidate_sha256"],
            "candidate_patch_artifact_sha256": candidate["patch_artifact_sha256"],
            "authorized_source_trace_manifest_sha256": _sha(SOURCE_TRACE_MANIFEST_PATH),
            "active_skill_sha256": _sha(ACTIVE_SKILL_PATH),
            "corpus_sha256": _sha(CORPUS_PATH),
            "runner_sha256": _sha(Path(__file__)),
            "oracle_sha256": _sha(Path(oracle.__file__).resolve()),
            "structured_final_parser_sha256": _sha(ROOT / "healthvault/evaluation/skill/benchmark/task_025.py"),
            "normalizer_sha256": sha256_bytes(inspect.getsource(FinalResponseNormalizer).encode("utf-8")),
            "candidate_verifier_sha256": sha256_bytes(inspect.getsource(CandidateVerifier).encode("utf-8")),
            "patch_pipeline_sha256": _sha(PATCH_PIPELINE_PATH),
            "patch_schema_sha256": _sha(PATCH_SCHEMA_PATH),
            "runtime_sha256": _sha(ROOT / "healthvault/agent_runtime.py"),
            "prompt_sha256": _sha(ROOT / "healthvault/prompt.py"),
            "product_source_sha256": {
                name: _sha(path) for name, path in sorted(oracle.PRODUCT_SOURCES.items())
            },
            "signoff_template_sha256": _sha(SIGNOFF_TEMPLATE_PATH),
            "artifact_template_sha256": _sha(ARTIFACT_TEMPLATE_PATH),
            "leakage_audit_sha256": sha256_bytes(canonical_json_bytes(audit)),
        },
        "protocol": {
            "same_runtime_prompt_tools_fixture_model_and_budget": True,
            "oracle_order": ["outcome", "behavior", "response"],
            "one_attempt_per_arm_case": True,
            "no_selective_rerun": True,
            "environment_incidents_invalidate_formal_metrics": True,
            "candidate_remains_unpublished_until_human_signoff": True,
        },
    }
    payload["preflight_sha256"] = preflight_sha256(payload)
    return payload


def assert_api_gate(
    preflight: Mapping[str, Any],
    *,
    approved: bool,
    supplied_hash: str,
    api_key: str,
) -> None:
    expected = preflight_sha256(preflight)
    if (
        not approved
        or not api_key
        or supplied_hash != expected
        or preflight.get("preflight_sha256") != expected
    ):
        raise ApiGateError("TASK-045 requires explicit approval of the current complete preflight hash")


def materialize_fixture(database_path: str | Path) -> dict[str, Any]:
    from healthvault.application.confirm_health_records import confirm_health_records
    from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository

    target = Path(database_path)
    if target.exists():
        raise ValueError("TASK-045 fixture target already exists")
    fixture = load_fixture()
    repository = SQLiteHealthRecordRepository(target)
    count = 0
    for profile in fixture["profiles"]:
        for report in profile["reports"]:
            lines = [
                "# Synthetic TASK-045 benchmark data only; no real person",
                f"report_id: {report['report_id']}",
                f"report_date: {report['report_date']}",
                f"facility: {report['facility']}",
            ]
            lines.extend(
                f"result|{item['code']}|{item['value']}|{item['unit']}|{item['lower']}|{item['upper']}"
                for item in report["indicators"]
            )
            text = "\n".join(lines) + "\n"
            outcome = confirm_health_records(
                {
                    "profile_id": profile["profile_id"],
                    "source_id": report["source_id"],
                    "report_text": text,
                    "expected_sha256": sha256_bytes(text.encode("utf-8")),
                    "user_confirmed": True,
                },
                repository,
            ).to_dict()
            if outcome["status"] != "confirmed":
                raise ValueError(f"TASK-045 fixture report failed: {report['report_id']} ({outcome})")
            count += 1
    return {"snapshot_id": fixture["snapshot_id"], "report_count": count}


async def run_case(
    case: Mapping[str, Any],
    *,
    arm_id: str,
    skill_content: str,
    model_config: Mapping[str, Any],
    transport: Any,
    ledger: Any,
    runtime_root: Path,
    api_key: str,
) -> dict[str, Any]:
    """Run one arm/case through the production Runtime with an isolated Skill root."""
    from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
    from healthvault.evaluation.skill.benchmark.task_022 import OpenAICompatibleController, TransportError
    from healthvault.model_config import HealthVaultModelConfig, OPENAI_COMPATIBLE
    from healthvault.runtime_context import RuntimeContext

    if arm_id not in ARM_IDS:
        raise ValueError("unknown TASK-045 arm")
    started = time.perf_counter()
    runtime_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"task045-{arm_id}-{case['case_id'].lower()}-", dir=runtime_root))
    database_path = workdir / "fixture.sqlite3"
    materialize_fixture(database_path)
    before = oracle.world_state(database_path)
    skill_path = workdir / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(skill_content, encoding="utf-8", newline="\n")
    expected = case["expected"]
    calls: list[dict[str, Any]] = []
    observed: set[str] = set()
    invalid_calls = 0
    runtime_error: str | None = None
    environment_incident = False
    incident_class: str | None = None
    parser_error: str | None = None
    turn_exhausted = transport_exhausted = False
    final = None
    runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
        context=RuntimeContext(
            profile_id=str(case["profile_id"]),
            request_id=f"TASK-045-{case['case_id']}",
        ),
        database_path=database_path,
        knowledge_path=CORPUS_PATH,
        knowledge_mode="lexical",
        model_config=HealthVaultModelConfig(
            OPENAI_COMPATIBLE, str(model_config["model"]), str(model_config["api_base"]), api_key,
        ),
        skill_root=skill_path.parents[1],
        skill_names=("HealthVault-Task-Execution",),
        max_turns=4,
        max_cost_usd=999.0,
    ))
    runtime.agent.is_sub_agent = True
    controller = OpenAICompatibleController(
        transport=transport,
        api_key=api_key,
        model_config=dict(model_config),
        ledger=ledger,
    )
    runtime.agent._call_openai_stream = lambda: controller.call(runtime.agent)
    original_execute = runtime.agent._execute_tool_call

    async def traced_execute(name: str, arguments: dict[str, Any]) -> str:
        nonlocal invalid_calls
        if name not in expected["allowed_tools"]:
            invalid_calls += 1
        try:
            raw = await original_execute(name, arguments)
            payload = json.loads(raw)
        except Exception as exc:
            calls.append({
                "name": name,
                "arguments": dict(arguments),
                "execution_status": "runtime_error",
                "result_status": "runtime_error",
                "available_citation_ids": [],
            })
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
        citation_ids = actual_available_citation_ids(payload)
        observed.update(citation_ids)
        calls.append({
            "name": name,
            "arguments": dict(arguments),
            "execution_status": "ok",
            "result_status": base._tool_result_status(payload),
            "available_citation_ids": citation_ids,
        })
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    runtime.agent._execute_tool_call = traced_execute
    try:
        execution = await runtime.run(oracle.PROMPT_TEMPLATE.format(prompt=case["prompt"]))
        parsed = base._parse_task025_final(execution.text)
        if parsed.failure_code:
            parser_error = parsed.failure_code
        else:
            final = parsed
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, TransportError):
            environment_incident = True; incident_class = "provider_or_transport"
        elif isinstance(exc, UnicodeEncodeError) or "unicodeencodeerror" in runtime_error.casefold():
            environment_incident = True; incident_class = "console_encoding"
        transport_exhausted = isinstance(exc, TransportError) or "transport" in runtime_error.casefold()
        turn_exhausted = "turn limit" in runtime_error.casefold()
    finally:
        after = oracle.world_state(database_path)
        shutil.rmtree(workdir, ignore_errors=True)
    duplicate_write_detected = bool(expected["duplicate_write_scenario"] and before["row_count"] != after["row_count"])
    attempt = {
        "calls": calls,
        "final": final,
        "parser_error": parser_error,
        "runtime_error": runtime_error,
        "environment_incident": environment_incident,
        "incident_class": incident_class,
        "turn_exhausted": turn_exhausted,
        "transport_exhausted": transport_exhausted,
        "database_unchanged": before["sha256"] == after["sha256"],
        "duplicate_write_detected": duplicate_write_detected,
        "invalid_calls": invalid_calls,
        "observed_citation_ids": sorted(observed),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "model_turns": controller.model_calls,
        "api_calls": controller.api_calls,
        "input_tokens": controller.input_tokens,
        "output_tokens": controller.output_tokens,
        "total_tokens": controller.total_tokens,
    }
    evaluated = oracle.evaluate_three_layer_oracle(case, attempt)
    public_attempt = {key: value for key, value in attempt.items() if key != "final"} | {
        "final": None if final is None else {
            "final_status": final.final_status,
            "citations": list(final.citations),
            "refusal": final.refusal,
        },
    }
    return {
        "arm_id": arm_id,
        "case_id": case["case_id"],
        "category": case["category"],
        "passed": evaluated["passed"],
        "hard_failure": evaluated["hard_failure"],
        "oracle": evaluated,
        "attempt": public_attempt,
    }


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


def _arm_metrics(group: list[Mapping[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["attempt"]["latency_ms"]) for item in group]
    tokens = [int(item["attempt"]["total_tokens"]) for item in group]
    return {
        "attempts": len(group),
        "task_success_rate": sum(bool(item["passed"]) for item in group) / len(group),
        "correct_tool_call_rate": sum(bool(item["oracle"]["behavior"]["passed"]) for item in group) / len(group),
        "hard_failure_rate": sum(bool(item["hard_failure"]) for item in group) / len(group),
        "average_calls": sum(len(item["attempt"]["calls"]) for item in group) / len(group),
        "total_tokens": sum(tokens),
        "average_tokens": sum(tokens) / len(group),
        "mean_latency_ms": sum(latencies) / len(group),
        "p95_latency_ms": _percentile_95(latencies),
        "api_calls": sum(int(item["attempt"]["api_calls"]) for item in group),
    }


def _badcase_classification(result: Mapping[str, Any]) -> str:
    attempt = result["attempt"]
    if attempt.get("environment_incident") or attempt.get("transport_exhausted"):
        return "environment_incident"
    if attempt.get("runtime_error"):
        return "runtime_error"
    if attempt.get("parser_error"):
        return "illegal_final_json"
    evaluated = result["oracle"]
    failed = [layer for layer in ("outcome", "behavior", "response") if not evaluated[layer]["passed"]]
    return "+".join(failed) if failed else "none"


def summarize(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    expected_pairs = {(arm, f"S5-{index:02d}") for arm in ARM_IDS for index in range(1, 21)}
    actual_pairs = {(str(item.get("arm_id")), str(item.get("case_id"))) for item in results}
    if len(results) != 40 or len(actual_pairs) != 40 or actual_pairs != expected_pairs:
        raise ValueError("TASK-045 requires exactly one attempt per arm/case")
    incidents = [
        item for item in results
        if item["attempt"].get("environment_incident") or item["attempt"].get("transport_exhausted")
    ]
    base_summary: dict[str, Any] = {
        "schema_version": "healthvault-task-045-summary-v1",
        "task": "TASK-045",
        "dataset_version": "skill-heldout-r5-v1",
        "case_count": 20,
        "attempt_count": 40,
        "silent_exclusion_count": 0,
        "environment_incident_count": len(incidents),
        "formal_metrics_valid": not incidents,
        "run_status": "invalid_environment_run" if incidents else "complete",
        "publish_thresholds": dict(PUBLISH_THRESHOLDS),
        "active_skill_unchanged": True,
    }
    if incidents:
        return base_summary | {
            "incident_attempts": [f"{item['arm_id']}:{item['case_id']}" for item in incidents],
            "arms": None,
            "paired": None,
            "badcases": [
                {"arm_id": item["arm_id"], "case_id": item["case_id"], "classification": _badcase_classification(item)}
                for item in results if not item["passed"]
            ],
            "publish_eligible": False,
            "candidate_disposition": "candidate_remains_unpublished_invalid_environment",
        }

    groups = {arm: [item for item in results if item["arm_id"] == arm] for arm in ARM_IDS}
    arms = {arm: _arm_metrics(groups[arm]) for arm in ARM_IDS}
    active_by_id = {item["case_id"]: bool(item["passed"]) for item in groups["active"]}
    candidate_by_id = {item["case_id"]: bool(item["passed"]) for item in groups["candidate"]}
    per_case = []
    gained: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    for case_id in sorted(active_by_id):
        active_passed = active_by_id[case_id]
        candidate_passed = candidate_by_id[case_id]
        if not active_passed and candidate_passed:
            classification = "gained"; gained.append(case_id)
        elif active_passed and not candidate_passed:
            classification = "regressed"; regressed.append(case_id)
        else:
            classification = "unchanged_pass" if active_passed else "unchanged_fail"
            unchanged.append(case_id)
        per_case.append({
            "case_id": case_id,
            "active_passed": active_passed,
            "candidate_passed": candidate_passed,
            "classification": classification,
        })
    active_successes = sum(active_by_id.values())
    candidate_successes = sum(candidate_by_id.values())
    success_delta = (candidate_successes - active_successes) / 20
    regression_rate = len(regressed) / 20
    gate_checks = {
        "task_success_delta": success_delta >= PUBLISH_THRESHOLDS["minimum_task_success_delta"],
        "regression_rate": regression_rate <= PUBLISH_THRESHOLDS["maximum_regression_rate"],
        "hard_failure": arms["candidate"]["hard_failure_rate"] <= arms["active"]["hard_failure_rate"],
        "tool_correctness": arms["candidate"]["correct_tool_call_rate"] >= arms["active"]["correct_tool_call_rate"],
    }
    eligible = all(gate_checks.values())
    return base_summary | {
        "arms": arms,
        "paired": {
            "task_success_delta": success_delta,
            "regression_rate": regression_rate,
            "gained_cases": gained,
            "regressed_cases": regressed,
            "unchanged_cases": unchanged,
            "per_case": per_case,
        },
        "gate_checks": gate_checks,
        "badcases": [
            {"arm_id": item["arm_id"], "case_id": item["case_id"], "classification": _badcase_classification(item)}
            for item in results if not item["passed"]
        ],
        "publish_eligible": eligible,
        "candidate_disposition": (
            "publish_eligible_requires_human_signoff"
            if eligible else "candidate_remains_unpublished"
        ),
    }


async def execute_ab_once(
    preflight: Mapping[str, Any],
    *,
    candidate_path: str | Path,
    approved: bool,
    supplied_hash: str,
    api_key: str,
    transport: Any,
    run_root: Path,
) -> dict[str, Any]:
    """Run the fixed alternating 20x2 matrix once; quality failures are never retried."""
    from healthvault.evaluation.skill.benchmark.task_022 import CallLedger

    run_root = Path(run_root)
    summary_path = run_root / "summary.json"
    attempts_root = run_root / "attempts"
    if summary_path.exists() or (attempts_root.exists() and any(attempts_root.iterdir())):
        raise ApiGateError("TASK-045 run directory already contains attempts; selective rerun is forbidden")
    assert_api_gate(
        preflight,
        approved=approved,
        supplied_hash=supplied_hash,
        api_key=api_key,
    )
    current = build_preflight(candidate_path, environ={"HEALTHVAULT_OPENAI_BASE_URL": preflight["model"]["api_base"]})
    if current["preflight_sha256"] != preflight["preflight_sha256"]:
        raise ApiGateError("TASK-045 bound inputs changed after approval")
    dataset = load_dataset()
    candidate = load_frozen_candidate(candidate_path)
    active_content = ACTIVE_SKILL_PATH.read_text(encoding="utf-8")
    skill_contents = {"active": active_content, "candidate": candidate["candidate_content"]}
    run_root.mkdir(parents=True, exist_ok=True)
    attempts_root.mkdir(parents=True, exist_ok=True)
    ledger = CallLedger(
        run_root / "transport-ledger.jsonl",
        run_config_sha256=str(preflight["preflight_sha256"]),
        max_transport_attempts=int(preflight["limits"]["maximum_transport_attempts"]),
    )
    results: list[dict[str, Any]] = []
    for index, case in enumerate(dataset["cases"]):
        order = ARM_IDS if index % 2 == 0 else tuple(reversed(ARM_IDS))
        for arm_id in order:
            result = await run_case(
                case,
                arm_id=arm_id,
                skill_content=skill_contents[arm_id],
                model_config=preflight["model"],
                transport=transport,
                ledger=ledger,
                runtime_root=run_root,
                api_key=api_key,
            )
            _atomic_json(attempts_root / f"{case['case_id']}-{arm_id}.json", result)
            results.append(result)
    summary = summarize(results) | {
        "preflight_sha256": preflight["preflight_sha256"],
        "api_calls": len(ledger.records),
    }
    _atomic_json(summary_path, summary)
    _atomic_json(run_root / "candidate-disposition.json", {
        "schema_version": "healthvault-task-045-candidate-disposition-v1",
        "candidate_sha256": candidate["candidate_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "publish_eligible": summary["publish_eligible"],
        "disposition": summary["candidate_disposition"],
        "active_skill_unchanged": True,
        "human_signoff_required": True,
    })
    return summary


async def _execute_candidate_gate_a_generation(
    preflight: Mapping[str, Any],
    *,
    approved: bool,
    supplied_hash: str,
    api_key: str,
    transport: Any,
    runtime_root: Path,
    execution_state: dict[str, Any],
) -> dict[str, Any]:
    """Generate, verify, leakage-check, and freeze exactly one unpublished Patch candidate."""
    runtime_root = Path(runtime_root)
    patch_path = runtime_root / "task-045-patch-artifact.json"
    candidate_path = runtime_root / "task-045-frozen-candidate.json"
    if patch_path.exists() or candidate_path.exists():
        raise ApiGateError("TASK-045 Candidate Gate A output already exists; second generation is forbidden")
    assert_candidate_gate_a(
        preflight,
        approved=approved,
        supplied_hash=supplied_hash,
        api_key=api_key,
    )
    current = build_candidate_gate_a_preflight(
        {"HEALTHVAULT_OPENAI_BASE_URL": preflight["candidate_model"]["api_base"]},
        auth_probe_hash=_candidate_gate_a_auth_probe_hash(preflight),
    )
    if current["preflight_sha256"] != preflight["preflight_sha256"]:
        raise ApiGateError("TASK-045 Candidate Gate A inputs changed after approval")

    active_content = ACTIVE_SKILL_PATH.read_text(encoding="utf-8")
    active_sha = _sha(ACTIVE_SKILL_PATH)
    cards = await _candidate_gate_a_cards()
    side_query = PatchModelSideQuery(
        transport=transport,
        api_key=api_key,
        model_config=preflight["candidate_model"],
        active_skill_content=active_content,
    )
    execution_state["side_query"] = side_query
    preview = await patching.build_patch_preview(
        cards,
        active_content=active_content,
        generator=patching.PatchGenerator(side_query),
        verifier=CandidateVerifier("HealthVault-Task-Execution"),
        regression_check=lambda candidate: _candidate_gate_a_regression(active_content, candidate),
    )
    execution_state["preview"] = preview
    if (
        preview.status != "pending_manual_review"
        or preview.generator_calls != 1
        or preview.candidate_content is None
        or preview.candidate_skill_sha256 is None
        or not preview.patches
    ):
        raise CandidateGateError(
            "Candidate Gate A Patch failed TASK-044 verification: " + ",".join(preview.reasons)
        )
    if set(trace_id for item in preview.patches for trace_id in item.source_trace_ids) != set(
        CANDIDATE_SOURCE_TRACE_IDS
    ):
        raise CandidateGateError("Candidate Gate A Patch does not cite exactly the three frozen dev Traces")
    if _sha(ACTIVE_SKILL_PATH) != active_sha:
        raise CandidateGateError("active Skill changed during Candidate Gate A")

    patch_artifact = patching.diff_artifact(preview) | {"api_calls": side_query.api_calls}
    frozen_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task045-candidate-gate-a-", dir=runtime_root) as directory:
        stage = Path(directory)
        stage_patch = stage / patch_path.name
        _atomic_json(stage_patch, patch_artifact)
        candidate_base = {
            "schema_version": "healthvault-task-045-candidate-v1",
            "candidate_id": f"TASK045-{preview.candidate_skill_sha256[:20]}",
            "status": "frozen_unpublished",
            "active_skill_sha256": active_sha,
            "candidate_content": preview.candidate_content,
            "candidate_sha256": preview.candidate_skill_sha256,
            "patch_artifact_sha256": _sha(stage_patch),
            "source_trace_ids": list(CANDIDATE_SOURCE_TRACE_IDS),
            "source_splits": ["dev"],
            "verifier_status": "verified",
            "gate_a_preflight_sha256": preflight["preflight_sha256"],
            "frozen_at_utc": frozen_at,
        }
        stage_candidate = stage / candidate_path.name
        _atomic_json(stage_candidate, candidate_base | {"patch_artifact_path": str(stage_patch)})

        # Held-out data is first opened here, after the generator request and local
        # TASK-044 verification have completed; it is never copied into model input.
        leakage = build_leakage_audit(
            load_dataset(),
            candidate_path=stage_candidate,
            require_gate_consumption=False,
        )
        if not leakage["passed"]:
            raise CandidateGateError(
                "Candidate Gate A failed TASK-045 leakage audit: "
                + ",".join(leakage["candidate_audit_reasons"])
            )

        final_candidate = candidate_base | {"patch_artifact_path": str(patch_path)}
        try:
            _atomic_json(patch_path, patch_artifact)
            final_stage_candidate = stage / "final-candidate-check.json"
            _atomic_json(final_stage_candidate, final_candidate)
            load_frozen_candidate(final_stage_candidate, require_gate_consumption=False)
            _atomic_json(candidate_path, final_candidate)
            load_frozen_candidate(candidate_path, require_gate_consumption=False)
        except Exception:
            patch_path.unlink(missing_ok=True)
            candidate_path.unlink(missing_ok=True)
            raise

    if _sha(ACTIVE_SKILL_PATH) != active_sha:
        patch_path.unlink(missing_ok=True)
        candidate_path.unlink(missing_ok=True)
        raise CandidateGateError("active Skill changed while freezing Candidate Gate A output")
    return {
        "schema_version": "healthvault-task-045-candidate-gate-a-result-v1",
        "status": "frozen_unpublished",
        "preflight_sha256": preflight["preflight_sha256"],
        "candidate_path": str(candidate_path),
        "candidate_sha256": preview.candidate_skill_sha256,
        "patch_artifact_path": str(patch_path),
        "patch_artifact_sha256": _sha(patch_path),
        "source_trace_ids": list(CANDIDATE_SOURCE_TRACE_IDS),
        "source_splits": ["dev"],
        "generator_calls": preview.generator_calls,
        "api_calls": side_query.api_calls,
        "transport_attempts": side_query.transport_requests,
        "total_tokens": side_query.total_tokens,
        "active_skill_sha256": active_sha,
        "active_skill_unchanged": True,
        "formal_ab_executed": False,
        "published": False,
    }


async def execute_candidate_gate_a_once(
    preflight: Mapping[str, Any],
    *,
    approved: bool,
    supplied_hash: str,
    api_key: str,
    transport: Any,
    runtime_root: Path,
) -> dict[str, Any]:
    """Consume one approved hash before transport, including rejected generations."""
    runtime_root = Path(runtime_root)
    patch_path = (runtime_root / "task-045-patch-artifact.json").resolve()
    candidate_path = (runtime_root / "task-045-frozen-candidate.json").resolve()
    if patch_path.exists() or candidate_path.exists():
        raise ApiGateError("TASK-045 Candidate Gate A output already exists; second generation is forbidden")
    assert_candidate_gate_a(
        preflight,
        approved=approved,
        supplied_hash=supplied_hash,
        api_key=api_key,
    )
    current = build_candidate_gate_a_preflight(
        {"HEALTHVAULT_OPENAI_BASE_URL": preflight["candidate_model"]["api_base"]},
        auth_probe_hash=_candidate_gate_a_auth_probe_hash(preflight),
    )
    if current["preflight_sha256"] != preflight["preflight_sha256"]:
        raise ApiGateError("TASK-045 Candidate Gate A inputs changed after approval")

    consumption_path = _candidate_gate_a_consumption_path(
        str(preflight["preflight_sha256"])
    )
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record: dict[str, Any] = {
        "schema_version": "healthvault-task-045-candidate-gate-a-consumption-v1",
        "preflight_sha256": preflight["preflight_sha256"],
        "status": "started",
        "disposition": "generation_consumed",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "active_skill_sha256": _sha(ACTIVE_SKILL_PATH),
        "transport_attempts": [],
        "api_calls": 0,
        "total_tokens": 0,
        "validation_status": "pending",
        "failure_class": None,
        "failure_reason": None,
        "provider_error": None,
        "candidate_artifact_path": None,
        "candidate_artifact_sha256": None,
        "candidate_sha256": None,
        "patch_artifact_path": None,
        "patch_artifact_sha256": None,
        "validation_failure_artifact_path": None,
        "validation_failure_artifact_sha256": None,
    }
    _create_candidate_gate_a_consumption(consumption_path, record)
    state: dict[str, Any] = {}
    try:
        result = await _execute_candidate_gate_a_generation(
            preflight,
            approved=True,
            supplied_hash=supplied_hash,
            api_key=api_key,
            transport=transport,
            runtime_root=runtime_root,
            execution_state=state,
        )
        side_query = state.get("side_query")
        load_frozen_candidate(candidate_path, require_gate_consumption=False)
        record.update({
            "status": "completed",
            "disposition": "frozen_unpublished",
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "transport_attempts": list(getattr(side_query, "attempts", [])),
            "api_calls": int(getattr(side_query, "api_calls", 0)),
            "total_tokens": int(getattr(side_query, "total_tokens", 0)),
            "validation_status": "passed",
            "candidate_artifact_path": str(candidate_path),
            "candidate_artifact_sha256": _sha(candidate_path),
            "candidate_sha256": result["candidate_sha256"],
            "patch_artifact_path": str(patch_path),
            "patch_artifact_sha256": result["patch_artifact_sha256"],
        })
        _atomic_json(consumption_path, record)
        load_frozen_candidate(candidate_path)
        return result | {"consumption_record_path": str(consumption_path)}
    except BaseException as exc:
        side_query = state.get("side_query")
        failure_artifact_path: str | None = None
        failure_artifact_sha256: str | None = None
        failure_artifact = _write_candidate_gate_a_validation_failure(
            preflight_hash=str(preflight["preflight_sha256"]),
            side_query=side_query,
            preview=state.get("preview"),
            failure=exc,
        )
        if failure_artifact is not None:
            failure_artifact_path = str(failure_artifact[0])
            failure_artifact_sha256 = failure_artifact[1]
        record.update({
            "status": "failed",
            "disposition": "rejected_consumed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "transport_attempts": list(getattr(side_query, "attempts", [])),
            "api_calls": int(getattr(side_query, "api_calls", 0)),
            "total_tokens": int(getattr(side_query, "total_tokens", 0)),
            "validation_status": "failed",
            "failure_class": type(exc).__name__,
            "failure_reason": str(exc),
            "provider_error": _safe_transport_diagnostics(exc),
            "validation_failure_artifact_path": failure_artifact_path,
            "validation_failure_artifact_sha256": failure_artifact_sha256,
        })
        _atomic_json(consumption_path, record)
        raise


def build_offline_readiness() -> dict[str, Any]:
    dataset = load_dataset()
    audit = build_leakage_audit(dataset)
    if not audit["baseline_isolation_passed"]:
        raise ValueError("TASK-045 held-out isolation audit failed")
    estimated_model_calls = 2 * sum(
        len(case["expected"]["required_tools"]) + 1
        for case in dataset["cases"]
    )
    return {
        "schema_version": "healthvault-task-045-offline-readiness-v1",
        "task": "TASK-045",
        "status": "CANDIDATE_FREEZE_PENDING",
        "api_calls_made": 0,
        "formal_preflight_created": False,
        "dataset_version": dataset["dataset_version"],
        "case_count": len(dataset["cases"]),
        "attempt_count_after_approval": 40,
        "estimated_model_calls_after_approval": estimated_model_calls,
        "maximum_model_calls_after_approval": 160,
        "publish_thresholds": dict(PUBLISH_THRESHOLDS),
        "bindings": {
            "dataset_sha256": _sha(DATASET_PATH),
            "fixture_sha256": _sha(FIXTURE_PATH),
            "split_registry_sha256": _sha(SPLIT_PATH),
            "active_skill_sha256": _sha(ACTIVE_SKILL_PATH),
            "runner_sha256": _sha(Path(__file__)),
            "oracle_sha256": _sha(Path(oracle.__file__).resolve()),
            "patch_pipeline_sha256": _sha(PATCH_PIPELINE_PATH),
            "patch_schema_sha256": _sha(PATCH_SCHEMA_PATH),
            "authorized_source_trace_manifest_sha256": _sha(SOURCE_TRACE_MANIFEST_PATH),
            "candidate_verifier_sha256": sha256_bytes(inspect.getsource(CandidateVerifier).encode("utf-8")),
            "signoff_template_sha256": _sha(SIGNOFF_TEMPLATE_PATH),
            "artifact_template_sha256": _sha(ARTIFACT_TEMPLATE_PATH),
        },
        "leakage": {
            "baseline_isolation_passed": audit["baseline_isolation_passed"],
            "candidate_audit_status": audit["candidate_audit_status"],
            "patient_profile_split_conflict_count": audit["patient_profile_split_conflict_count"],
            "text_collision_count": audit["text_collision_count"],
        },
        "next_gate": "freeze one verified unpublished TASK-044 local Patch candidate from discovery/dev evidence",
    }


def write_offline_artifacts() -> dict[str, Any]:
    dataset = load_dataset()
    audit = build_leakage_audit(dataset)
    matrix = build_case_matrix(dataset)
    readiness = build_offline_readiness()
    _atomic_json(LEAKAGE_AUDIT_PATH, audit)
    _atomic_json(CASE_MATRIX_PATH, matrix)
    _atomic_json(READINESS_PATH, readiness)
    return readiness


def write_preflight(candidate_path: str | Path, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    preflight = build_preflight(candidate_path, environ=environ)
    _atomic_json(PREFLIGHT_PATH, preflight)
    return preflight


def write_candidate_gate_a_preflight(
    environ: Mapping[str, str] | None = None,
    *,
    auth_probe_hash: str,
) -> dict[str, Any]:
    preflight = build_candidate_gate_a_preflight(
        environ,
        auth_probe_hash=auth_probe_hash,
    )
    _atomic_json(CANDIDATE_GATE_A_PREFLIGHT_PATH, preflight)
    return preflight


def write_candidate_gate_a_auth_probe_preflight(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    preflight = build_candidate_gate_a_auth_probe_preflight(environ)
    _atomic_json(CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH, preflight)
    return preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate")
    parser.add_argument("--candidate-gate-a", action="store_true")
    parser.add_argument("--execute-candidate-gate-a", action="store_true")
    parser.add_argument("--candidate-gate-a-auth-probe", action="store_true")
    parser.add_argument("--execute-candidate-gate-a-auth-probe", action="store_true")
    parser.add_argument("--auth-probe-hash", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--preflight-hash", default="")
    parser.add_argument("--run-root", default=str(ROOT / ".task_outputs" / "TASK-045"))
    arguments = parser.parse_args(argv)
    if arguments.candidate_gate_a_auth_probe or arguments.execute_candidate_gate_a_auth_probe:
        if (
            arguments.candidate
            or arguments.execute
            or arguments.candidate_gate_a
            or arguments.execute_candidate_gate_a
        ):
            parser.error("Candidate Gate A auth probe cannot be combined with another run path")
        preflight = write_candidate_gate_a_auth_probe_preflight(os.environ)
        if not arguments.execute_candidate_gate_a_auth_probe:
            print(json.dumps({
                "status": preflight["status"],
                "preflight_hash": preflight["preflight_sha256"],
                "model": preflight["model"]["model"],
                "maximum_api_calls": preflight["limits"]["maximum_api_calls"],
                "maximum_transport_attempts": preflight["limits"]["maximum_transport_attempts"],
                "maximum_output_tokens": preflight["limits"]["maximum_output_tokens"],
                "candidate_generation": False,
                "api_calls_made": 0,
            }, ensure_ascii=False, sort_keys=True))
            return 0
        key = str(os.environ.get(CANDIDATE_GATE_A_MODEL["key_environment"], "") or "").strip()
        from healthvault.evaluation.skill.benchmark.task_022 import UrllibJsonTransport
        result = asyncio.run(execute_candidate_gate_a_auth_probe_once(
            preflight,
            approved=arguments.approved,
            supplied_hash=arguments.preflight_hash,
            api_key=key,
            transport=UrllibJsonTransport(),
        ))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if arguments.candidate_gate_a or arguments.execute_candidate_gate_a:
        if arguments.candidate or arguments.execute:
            parser.error("Candidate Gate A cannot be combined with the formal A/B path")
        if not arguments.auth_probe_hash:
            parser.error("Candidate Gate A requires a passed --auth-probe-hash")
        preflight = write_candidate_gate_a_preflight(
            os.environ,
            auth_probe_hash=arguments.auth_probe_hash,
        )
        if not arguments.execute_candidate_gate_a:
            print(json.dumps({
                "status": preflight["status"],
                "preflight_hash": preflight["preflight_sha256"],
                "model": preflight["candidate_model"]["model"],
                "estimated_model_calls": preflight["limits"]["candidate_generation_calls"],
                "maximum_model_calls": preflight["limits"]["maximum_model_calls"],
                "maximum_transport_attempts": preflight["limits"]["maximum_transport_attempts"],
                "api_calls_made": 0,
            }, ensure_ascii=False, sort_keys=True))
            return 0
        key = str(os.environ.get(CANDIDATE_GATE_A_MODEL["key_environment"], "") or "").strip()
        from healthvault.evaluation.skill.benchmark.task_022 import UrllibJsonTransport
        result = asyncio.run(execute_candidate_gate_a_once(
            preflight,
            approved=arguments.approved,
            supplied_hash=arguments.preflight_hash,
            api_key=key,
            transport=UrllibJsonTransport(),
            runtime_root=Path(arguments.run_root) / "candidate-gate-a" / preflight["preflight_sha256"],
        ))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if not arguments.candidate:
        readiness = write_offline_artifacts()
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 0
    preflight = write_preflight(arguments.candidate, environ=os.environ)
    if not arguments.execute:
        print(json.dumps({
            "status": preflight["status"],
            "preflight_hash": preflight["preflight_sha256"],
            "estimated_model_calls": preflight["limits"]["estimated_model_calls"],
            "maximum_model_calls": preflight["limits"]["maximum_model_calls"],
            "api_calls_made": 0,
        }, ensure_ascii=False, sort_keys=True))
        return 0
    key = str(os.environ.get(MODEL_CONFIG["key_environment"], "") or "").strip()
    assert_api_gate(
        preflight,
        approved=arguments.approved,
        supplied_hash=arguments.preflight_hash,
        api_key=key,
    )
    from healthvault.evaluation.skill.benchmark.task_022 import UrllibJsonTransport
    summary = asyncio.run(execute_ab_once(
        preflight,
        candidate_path=arguments.candidate,
        approved=True,
        supplied_hash=arguments.preflight_hash,
        api_key=key,
        transport=UrllibJsonTransport(),
        run_root=Path(arguments.run_root),
    ))
    print(json.dumps({
        "status": summary["run_status"],
        "publish_eligible": summary["publish_eligible"],
        "candidate_disposition": summary["candidate_disposition"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
