"""Strict data binding, fake execution, and deterministic TASK-021 metrics."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.evaluation.used_cases import normalized_query_sha256, validate_used_case_registry
from healthvault.evolution.synthetic_demo import SKILL_V2
from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository


ROOT = Path(__file__).resolve().parents[4]
SKILL_ROOT = ROOT / "healthvault" / "evaluation" / "skill"
DEFAULT_DATASET_PATH = SKILL_ROOT / "task-021-heldout-dataset.json"
DEFAULT_MANIFEST_PATH = SKILL_ROOT / "task-021-run-manifest.json"
DEFAULT_SUMMARY_PATH = SKILL_ROOT / "task-021-mock-summary.json"
DEFAULT_DATABASE_FIXTURE_PATH = SKILL_ROOT / "task-021-synthetic-db-v1.json"
ACTIVE_SKILL_PATH = ROOT / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"
CORPUS_ARTIFACT_PATH = ROOT / "healthvault" / "knowledge" / "r1-2026-08-24-v1" / "chunks" / "hybrid_v1.json"

DATASET_FIELDS = {
    "schema_version", "dataset_version", "data_classification", "held_out",
    "frozen_on", "leakage_sources", "cases",
}
CASE_FIELDS = {
    "case_id", "case_family_id", "task_type", "prompt", "fixture",
    "expected", "validator", "boundary_tags",
}
FIXTURE_FIELDS = {
    "fixture_id", "data_classification", "profile_id", "database_snapshot_id",
    "corpus_snapshot_id", "evidence", "retryable_failures",
}
EXPECTED_FIELDS = {
    "tool_set", "tool_order", "final_status", "citations",
    "refusal_required", "prohibited_writes",
}
VALIDATOR_FIELDS = {
    "kind", "exact_tool_order", "exact_final_status", "citation_mode",
    "refusal_mode", "reject_hard_failure", "reject_invalid_calls", "reject_any_write",
}
SOURCE_FIELDS = {"track", "case_version", "purpose", "path", "sha256", "extractor"}
DATABASE_FIELDS = {"schema_version", "snapshot_id", "data_classification", "profiles", "report_count"}
PROFILE_FIELDS = {"profile_id", "reports"}
REPORT_FIELDS = {"source_id", "report_id", "report_date", "facility", "indicators"}
INDICATOR_FIELDS = {"code", "value", "unit", "lower", "upper"}
ARM_IDS = ("no_skill", "active_v1", "candidate_v2")
TASK_TYPES = ("timeline", "indicator_comparison", "knowledge_retrieval", "visit_brief")
SHA256 = set("0123456789abcdef")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} must use the exact schema")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _load_json(value: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, dict):
        return json.loads(json.dumps(value, ensure_ascii=False)), None
    path = Path(value)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed, path


def _source_entries(source: dict[str, Any], workspace_root: Path) -> list[dict[str, Any]]:
    path = workspace_root / source["path"]
    if not path.is_file() or sha256_bytes(path.read_bytes()) != source["sha256"]:
        raise ValueError(f"leakage source hash mismatch: {source['path']}")
    extractor = source["extractor"]
    values: list[tuple[str, str, str]] = []
    if extractor == "rag_queries":
        document = json.loads(path.read_text(encoding="utf-8"))
        values = [(item["query_id"], f"rag-{item['query_id']}", item["query"]) for item in document["queries"]]
    elif extractor == "task019_lessons":
        document = json.loads(path.read_text(encoding="utf-8"))
        for batch in document["batches"]:
            for signal in batch["signals"]:
                values.append((signal["trace_id"], f"task019-{signal['trace_id']}", signal["lesson"]))
    elif extractor == "task020_events":
        document = json.loads(path.read_text(encoding="utf-8"))
        for event in document["events"]:
            query = f"Synthetic usage evaluation {event['task_id']} for {event['task_type']}."
            values.append((event["task_id"], f"task020-{event['task_id']}", query))
    elif extractor in {"agent_e2e_inventory", "long_fault_inventory"}:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        fault_markers = ("reject", "block", "cannot", "requires", "failure", "fault", "retry")
        for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            for method in class_node.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or not method.name.startswith("test_"):
                    continue
                is_fault = any(marker in method.name for marker in fault_markers)
                if (extractor == "long_fault_inventory") is not is_fault:
                    continue
                prompts = []
                for node in ast.walk(method):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "run"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        prompts.append(node.args[0].value)
                query = prompts[0] if prompts else f"{class_node.name}.{method.name}"
                case_id = f"{class_node.name}.{method.name}"
                values.append((case_id, f"{source['track']}-{case_id}", query))
    else:
        raise ValueError(f"unsupported leakage extractor: {extractor}")
    return [
        {
            "case_id": f"{source['track']}:{case_id}",
            "case_family_id": family,
            "track": source["track"],
            "case_version": source["case_version"],
            "split": "test",
            "held_out": True,
            "query": query,
            "query_sha256": normalized_query_sha256(query),
            "upstream_dataset": source["path"],
            "upstream_id": case_id,
            "status": "frozen",
            "source": source["path"],
        }
        for case_id, family, query in values
    ]


def _build_used_case_registry(raw: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    sources = raw["leakage_sources"]
    tracks = [{"track": "skill_benchmark", "case_version": raw["dataset_version"], "purpose": "held_out_skill_benchmark"}]
    entries = []
    for source in sources:
        tracks.append({field: source[field] for field in ("track", "case_version", "purpose")})
        entries.extend(_source_entries(source, workspace_root))
    for case in raw["cases"]:
        entries.append({
            "case_id": case["case_id"],
            "case_family_id": case["case_family_id"],
            "track": "skill_benchmark",
            "case_version": raw["dataset_version"],
            "split": "test",
            "held_out": True,
            "query": case["prompt"],
            "query_sha256": normalized_query_sha256(case["prompt"]),
            "upstream_dataset": raw["dataset_version"],
            "upstream_id": case["case_id"],
            "status": "frozen",
            "source": "synthetic-task-021",
        })
    return validate_used_case_registry({
        "schema_version": 1,
        "registry_version": "used-cases-r1-task-021",
        "created_on": raw["frozen_on"].split("T", 1)[0],
        "normalization": "nfkc-casefold-alnum-v1",
        "tracks": tracks,
        "entries": entries,
    })


def load_dataset(value: str | Path | dict[str, Any], *, workspace_root: str | Path = ROOT) -> dict[str, Any]:
    document, _path = _load_json(value)
    prebuilt_registry = document.pop("used_case_registry", None)
    _exact(document, DATASET_FIELDS, "held-out dataset")
    if document["schema_version"] != "healthvault-skill-heldout-dataset-v1":
        raise ValueError("unsupported held-out dataset schema")
    if document["dataset_version"] != "skill-heldout-r1-v1" or document["data_classification"] != "synthetic" or document["held_out"] is not True:
        raise ValueError("held-out dataset identity is not frozen")
    if not isinstance(document["leakage_sources"], list) or len(document["leakage_sources"]) != 5:
        raise ValueError("five leakage sources are required")
    seen_tracks = set()
    for source in document["leakage_sources"]:
        _exact(source, SOURCE_FIELDS, "leakage source")
        for field in SOURCE_FIELDS - {"sha256"}:
            _nonempty(source[field], f"leakage source {field}")
        if not _is_sha256(source["sha256"]):
            raise ValueError("leakage source sha256 is invalid")
        key = (source["track"], source["case_version"])
        if key in seen_tracks:
            raise ValueError("duplicate leakage track/version")
        seen_tracks.add(key)
    cases = document["cases"]
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError("held-out dataset must contain exactly 20 cases")
    case_ids, family_ids = set(), set()
    counts: Counter[str] = Counter()
    for case in cases:
        _exact(case, CASE_FIELDS, "held-out case")
        for field in ("case_id", "case_family_id", "task_type", "prompt"):
            _nonempty(case[field], f"case {field}")
        if case["case_id"] in case_ids or case["case_family_id"] in family_ids:
            raise ValueError("case and family IDs must be unique")
        case_ids.add(case["case_id"]); family_ids.add(case["case_family_id"])
        if case["task_type"] not in TASK_TYPES:
            raise ValueError("unsupported task type")
        counts[case["task_type"]] += 1
        fixture = _exact(case["fixture"], FIXTURE_FIELDS, "case fixture")
        expected = _exact(case["expected"], EXPECTED_FIELDS, "case expectation")
        validator = _exact(case["validator"], VALIDATOR_FIELDS, "case validator")
        if fixture["data_classification"] != "synthetic" or not isinstance(fixture["evidence"], list):
            raise ValueError("fixtures must be synthetic with evidence")
        if not isinstance(fixture["retryable_failures"], list):
            raise ValueError("retryable failures must be a list")
        if not isinstance(expected["tool_set"], list) or not expected["tool_set"] or len(expected["tool_set"]) != len(set(expected["tool_set"])):
            raise ValueError("expected tool_set must be unique and non-empty")
        if not isinstance(expected["tool_order"], list) or set(expected["tool_order"]) != set(expected["tool_set"]):
            raise ValueError("tool order must use exactly the expected tool set")
        try:
            retry_indices = [int(index) for index in fixture["retryable_failures"]]
        except (TypeError, ValueError) as error:
            raise ValueError("retryable failure positions must be integer strings") from error
        if len(retry_indices) != len(set(retry_indices)) or any(str(index) not in fixture["retryable_failures"] for index in retry_indices):
            raise ValueError("retryable failure positions must be unique canonical integer strings")
        if any(
            index < 0
            or index + 1 >= len(expected["tool_order"])
            or expected["tool_order"][index + 1] != expected["tool_order"][index]
            or index + 1 in retry_indices
            for index in retry_indices
        ):
            raise ValueError("retryable failure must be followed by one successful retry of the same tool")
        if not isinstance(expected["citations"], list) or not isinstance(expected["refusal_required"], bool):
            raise ValueError("citation/refusal expectations are invalid")
        if not isinstance(expected["prohibited_writes"], list) or not expected["prohibited_writes"]:
            raise ValueError("prohibited writes must be non-empty")
        if any(citation not in fixture["evidence"] for citation in expected["citations"]):
            raise ValueError("expected citations must be backed by fixture evidence")
        if validator != {
            "kind": "deterministic_exact_v1", "exact_tool_order": True,
            "exact_final_status": True, "citation_mode": "exact",
            "refusal_mode": "exact", "reject_hard_failure": True,
            "reject_invalid_calls": True, "reject_any_write": True,
        }:
            raise ValueError("validator must use the frozen deterministic policy")
        if not isinstance(case["boundary_tags"], list) or not case["boundary_tags"]:
            raise ValueError("boundary tags must be non-empty")
    if counts != Counter({task_type: 5 for task_type in TASK_TYPES}):
        raise ValueError("held-out cases must be balanced four by five")
    workspace = Path(workspace_root)
    database_fixture = load_database_fixture(workspace / DEFAULT_DATABASE_FIXTURE_PATH.relative_to(ROOT))
    profile_ids = {profile["profile_id"] for profile in database_fixture["profiles"]}
    record_citations = {
        f"REC:{report['report_id']}:{indicator['code']}"
        for profile in database_fixture["profiles"]
        for report in profile["reports"]
        for indicator in report["indicators"]
    }
    corpus_path = workspace / CORPUS_ARTIFACT_PATH.relative_to(ROOT)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_citations = {chunk["citation_id"] for chunk in corpus["chunks"]}
    for case in cases:
        if case["fixture"]["profile_id"] not in profile_ids:
            raise ValueError(f"fixture profile does not resolve: {case['case_id']}")
        if case["fixture"]["database_snapshot_id"] != database_fixture["snapshot_id"]:
            raise ValueError("case database snapshot does not match the loadable fixture")
        for citation in case["expected"]["citations"]:
            if citation not in record_citations and citation not in corpus_citations:
                raise ValueError(f"citation does not resolve: {citation}")
    registry = _build_used_case_registry(document, workspace)
    if prebuilt_registry is not None:
        validate_used_case_registry(prebuilt_registry)
        if canonical_json_bytes(prebuilt_registry) != canonical_json_bytes(registry):
            raise ValueError("stale used-case registry does not match the current dataset")
    result = dict(document)
    result["used_case_registry"] = registry
    return result


def _dataset_payload(dataset: dict[str, Any]) -> dict[str, Any]:
    return {key: dataset[key] for key in DATASET_FIELDS}


def dataset_sha256(dataset: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_dataset_payload(dataset)))


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "run_config_sha256"}


def build_manifest(dataset: dict[str, Any], *, workspace_root: str | Path = ROOT) -> dict[str, Any]:
    workspace = Path(workspace_root)
    active_bytes = (workspace / ACTIVE_SKILL_PATH.relative_to(ROOT)).read_bytes()
    empty_hash = sha256_bytes(b"")
    arms = [
        {"arm_id": "no_skill", "skill_source": "none", "skill_sha256": empty_hash, "content_sha256": empty_hash},
        {"arm_id": "active_v1", "skill_source": ".bear/skills/HealthVault-Task-Execution/SKILL.md", "skill_sha256": sha256_bytes(active_bytes), "content_sha256": sha256_bytes(active_bytes)},
        {"arm_id": "candidate_v2", "skill_source": "healthvault.evolution.synthetic_demo.SKILL_V2", "skill_sha256": sha256_bytes(SKILL_V2.encode("utf-8")), "content_sha256": sha256_bytes(SKILL_V2.encode("utf-8"))},
    ]
    manifest = {
        "schema_version": "healthvault-skill-benchmark-manifest-v1",
        "benchmark_version": "skill-benchmark-r1-v1",
        "run_kind": "mock_smoke",
        "claim_status": "not_claimable",
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": dataset_sha256(dataset),
        "arms": arms,
        "shared_config": {
            "model": "fake-deterministic-no-model",
            "temperature": 0.0,
            "tools": ["query_health_timeline", "compare_health_indicators", "search_health_knowledge", "generate_visit_brief"],
            "database_snapshot_id": "task-021-synthetic-db-v1",
            "database_fixture_sha256": sha256_bytes((workspace / DEFAULT_DATABASE_FIXTURE_PATH.relative_to(ROOT)).read_bytes()),
            "corpus_snapshot_id": "r1-2026-08-24-v1",
            "corpus_artifact_sha256": sha256_bytes((workspace / CORPUS_ARTIFACT_PATH.relative_to(ROOT)).read_bytes()),
            "case_order": [case["case_id"] for case in dataset["cases"]],
            "run_parameters": {"max_attempts_per_arm_case": 1, "parallelism": 1, "seed": 21},
            "prompt_template_sha256": sha256_bytes(b"TASK-021 prompt = dataset case prompt; no arm-specific additions"),
            "validator_sha256": sha256_bytes(inspect.getsource(validate_attempt).encode("utf-8")),
        },
    }
    manifest["run_config_sha256"] = sha256_bytes(canonical_json_bytes(_manifest_payload(manifest)))
    return manifest


def load_manifest(value: str | Path | dict[str, Any], dataset: dict[str, Any], *, workspace_root: str | Path = ROOT) -> dict[str, Any]:
    manifest, _path = _load_json(value)
    required = {"schema_version", "benchmark_version", "run_kind", "claim_status", "dataset_version", "dataset_sha256", "arms", "shared_config", "run_config_sha256"}
    _exact(manifest, required, "run manifest")
    if manifest["schema_version"] != "healthvault-skill-benchmark-manifest-v1" or manifest["run_kind"] != "mock_smoke" or manifest["claim_status"] != "not_claimable":
        raise ValueError("unsupported benchmark manifest")
    if [arm.get("arm_id") for arm in manifest["arms"]] != list(ARM_IDS):
        raise ValueError("manifest must contain the three benchmark arms")
    if manifest["dataset_version"] != dataset["dataset_version"] or manifest["dataset_sha256"] != dataset_sha256(dataset):
        raise ValueError("manifest dataset hash does not match the frozen dataset")
    expected = build_manifest(dataset, workspace_root=workspace_root)
    if manifest != expected:
        if manifest.get("run_config_sha256") != sha256_bytes(canonical_json_bytes(_manifest_payload(manifest))):
            raise ValueError("manifest run config hash mismatch")
        raise ValueError("manifest configuration or Skill hash mismatch")
    return manifest


def validate_attempt(case: dict[str, Any], attempt: dict[str, Any]) -> dict[str, bool]:
    expected = case["expected"]
    calls = attempt.get("tool_calls", [])
    call_names = [call.get("name") for call in calls]
    retry_indices = {int(index) for index in case["fixture"]["retryable_failures"]}
    expected_statuses = ["retryable_failure" if index in retry_indices else "ok" for index in range(len(expected["tool_order"]))]
    retry_chain_ok = all(
        index + 1 < len(calls)
        and calls[index + 1].get("name") == calls[index].get("name")
        and calls[index + 1].get("status") == "ok"
        for index in retry_indices
    )
    tools_ok = (
        call_names == expected["tool_order"]
        and [call.get("status") for call in calls] == expected_statuses
        and retry_chain_ok
    )
    citations_ok = attempt.get("citations") == expected["citations"]
    refusal_ok = attempt.get("refusal") is expected["refusal_required"]
    status_ok = attempt.get("final_status") == expected["final_status"]
    safe = (
        attempt.get("hard_failure") is False
        and attempt.get("invalid_calls") == 0
        and attempt.get("write_attempts") == []
    )
    return {
        "tool_call_correct": tools_ok,
        "citation_correct": citations_ok,
        "refusal_correct": refusal_ok,
        "status_correct": status_ok,
        "task_success": tools_ok and citations_ok and refusal_ok and status_ok and safe,
    }


class FakeArmRunner:
    """Return the same frozen successful script for every arm without model calls."""

    def __call__(self, arm: dict[str, Any], case: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
        failures = set(case["fixture"]["retryable_failures"])
        calls = []
        for index, name in enumerate(case["expected"]["tool_order"]):
            calls.append({"name": name, "status": "retryable_failure" if str(index) in failures else "ok"})
        return {
            "arm_id": arm["arm_id"], "case_id": case["case_id"],
            "dataset_sha256": manifest["dataset_sha256"], "run_config_sha256": manifest["run_config_sha256"],
            "skill_sha256": arm["skill_sha256"], "tool_calls": calls,
            "final_status": case["expected"]["final_status"], "citations": list(case["expected"]["citations"]),
            "refusal": case["expected"]["refusal_required"], "write_attempts": [],
            "hard_failure": False, "invalid_calls": 0, "steps": len(calls) + 1,
            "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0, "cost_usd": 0.0, "api_calls": 0,
        }


ArmRunner = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]


def run_benchmark(dataset: dict[str, Any], manifest: dict[str, Any], arm_runner: ArmRunner) -> list[dict[str, Any]]:
    attempts = []
    case_by_id = {case["case_id"]: case for case in dataset["cases"]}
    for arm in manifest["arms"]:
        for case_id in manifest["shared_config"]["case_order"]:
            attempts.append(arm_runner(arm, case_by_id[case_id], manifest))
    validate_attempts(dataset, manifest, attempts)
    return attempts


def validate_attempts(dataset: dict[str, Any], manifest: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
    expected_pairs = {(arm_id, case["case_id"]) for arm_id in ARM_IDS for case in dataset["cases"]}
    seen = set()
    arm_hashes = {arm["arm_id"]: arm["skill_sha256"] for arm in manifest["arms"]}
    for attempt in attempts:
        pair = (attempt.get("arm_id"), attempt.get("case_id"))
        if pair in seen:
            raise ValueError(f"duplicate attempt: {pair}")
        seen.add(pair)
        if pair not in expected_pairs:
            raise ValueError(f"unexpected attempt: {pair}")
        if attempt.get("dataset_sha256") != manifest["dataset_sha256"]:
            raise ValueError("attempt dataset hash mismatch")
        if attempt.get("run_config_sha256") != manifest["run_config_sha256"]:
            raise ValueError("attempt run config hash mismatch")
        if attempt.get("skill_sha256") != arm_hashes[pair[0]]:
            raise ValueError("attempt Skill hash mismatch")
    missing = expected_pairs - seen
    if missing:
        raise ValueError(f"missing attempt: {sorted(missing)[0]}")


def _average(attempts: list[dict[str, Any]], field: str) -> float:
    return sum(float(item[field]) for item in attempts) / len(attempts)


def summarize_attempts(dataset: dict[str, Any], manifest: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    validate_attempts(dataset, manifest, attempts)
    cases = {case["case_id"]: case for case in dataset["cases"]}
    arm_metrics = {}
    for arm_id in ARM_IDS:
        group = [attempt for attempt in attempts if attempt["arm_id"] == arm_id]
        validations = [validate_attempt(cases[item["case_id"]], item) for item in group]
        arm_metrics[arm_id] = {
            "attempts": len(group),
            "task_success_rate": sum(item["task_success"] for item in validations) / len(group),
            "correct_tool_call_rate": sum(item["tool_call_correct"] for item in validations) / len(group),
            "citation_correctness_rate": sum(item["citation_correct"] for item in validations) / len(group),
            "hard_failure_rate": sum(item["hard_failure"] for item in group) / len(group),
            "invalid_calls": sum(item["invalid_calls"] for item in group),
            "average_steps": _average(group, "steps"),
            "average_tokens": _average(group, "input_tokens") + _average(group, "output_tokens"),
            "average_latency_ms": _average(group, "latency_ms"),
            "average_cost_usd": _average(group, "cost_usd"),
        }
    outcomes = {
        arm_id: {
            item["case_id"]: validate_attempt(cases[item["case_id"]], item)["task_success"]
            for item in attempts if item["arm_id"] == arm_id
        }
        for arm_id in ARM_IDS
    }
    baseline = outcomes["no_skill"]
    regressions = {}
    for arm_id in ARM_IDS[1:]:
        regressed = sum(baseline[case_id] and not outcomes[arm_id][case_id] for case_id in baseline)
        improved = sum(not baseline[case_id] and outcomes[arm_id][case_id] for case_id in baseline)
        regressions[arm_id] = {
            "regressed_cases": regressed,
            "regression_rate": regressed / len(baseline),
            "improved_cases": improved,
            "improvement_rate": improved / len(baseline),
        }
    return {
        "schema_version": "healthvault-skill-benchmark-summary-v1",
        "dataset_version": dataset["dataset_version"], "dataset_sha256": manifest["dataset_sha256"],
        "run_config_sha256": manifest["run_config_sha256"], "run_kind": "mock_smoke",
        "claim_status": "not_claimable", "data_classification": "synthetic",
        "attempts": len(attempts), "arms": arm_metrics,
        "regression_vs_baseline": regressions,
        "api_calls": sum(item["api_calls"] for item in attempts),
        "tokens": sum(item["input_tokens"] + item["output_tokens"] for item in attempts),
        "cost_usd": sum(float(item["cost_usd"]) for item in attempts),
        "candidate_generation_calls": 0, "judge_calls": 0,
    }


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


def load_database_fixture(path: str | Path = DEFAULT_DATABASE_FIXTURE_PATH) -> dict[str, Any]:
    document, _ = _load_json(path)
    _exact(document, DATABASE_FIELDS, "database fixture")
    if document["schema_version"] != "healthvault-skill-database-fixture-v1" or document["snapshot_id"] != "task-021-synthetic-db-v1" or document["data_classification"] != "synthetic":
        raise ValueError("unsupported database fixture")
    if not isinstance(document["profiles"], list) or not document["profiles"]:
        raise ValueError("database fixture profiles must be non-empty")
    profiles, reports = set(), 0
    report_ids, source_keys = set(), set()
    for profile in document["profiles"]:
        _exact(profile, PROFILE_FIELDS, "database profile")
        profile_id = _nonempty(profile["profile_id"], "database profile_id")
        if profile_id in profiles or not isinstance(profile["reports"], list):
            raise ValueError("database profile IDs must be unique")
        profiles.add(profile_id)
        for report in profile["reports"]:
            _exact(report, REPORT_FIELDS, "database report")
            for field in ("source_id", "report_id", "report_date", "facility"):
                _nonempty(report[field], f"database report {field}")
            if report["report_id"] in report_ids or (profile_id, report["source_id"]) in source_keys:
                raise ValueError("database report identity must be unique")
            report_ids.add(report["report_id"]); source_keys.add((profile_id, report["source_id"])); reports += 1
            if not isinstance(report["indicators"], list) or not report["indicators"]:
                raise ValueError("database reports must contain indicators")
            codes = set()
            for indicator in report["indicators"]:
                _exact(indicator, INDICATOR_FIELDS, "database indicator")
                if indicator["code"] in codes or not all(isinstance(indicator[field], (int, float)) for field in ("value", "lower", "upper")):
                    raise ValueError("database indicators must be unique and numeric")
                codes.add(_nonempty(indicator["code"], "indicator code"))
                _nonempty(indicator["unit"], "indicator unit")
    if document["report_count"] != reports:
        raise ValueError("database fixture report_count mismatch")
    return document


def materialize_database_fixture(
    database_path: str | Path,
    fixture_path: str | Path = DEFAULT_DATABASE_FIXTURE_PATH,
) -> dict[str, Any]:
    fixture = load_database_fixture(fixture_path)
    target = Path(database_path)
    if target.exists():
        raise ValueError("database fixture target must not already exist")
    repository = SQLiteHealthRecordRepository(target)
    confirmed = 0
    for profile in fixture["profiles"]:
        for report in profile["reports"]:
            lines = [
                "# Synthetic TASK-021 benchmark data only; no real person",
                f"report_id: {report['report_id']}",
                f"report_date: {report['report_date']}",
                f"facility: {report['facility']}",
            ]
            lines.extend(
                f"result|{item['code']}|{item['value']}|{item['unit']}|{item['lower']}|{item['upper']}"
                for item in report["indicators"]
            )
            report_text = "\n".join(lines) + "\n"
            result = confirm_health_records(
                {
                    "profile_id": profile["profile_id"],
                    "source_id": report["source_id"],
                    "report_text": report_text,
                    "expected_sha256": sha256_bytes(report_text.encode("utf-8")),
                    "user_confirmed": True,
                },
                repository,
            ).to_dict()
            if result["status"] != "confirmed":
                raise ValueError(f"database fixture report failed to load: {report['report_id']}")
            confirmed += 1
    return {"snapshot_id": fixture["snapshot_id"], "profile_count": len(fixture["profiles"]), "report_count": confirmed}


def write_default_outputs(*, workspace_root: str | Path = ROOT) -> dict[str, Any]:
    dataset = load_dataset(DEFAULT_DATASET_PATH, workspace_root=workspace_root)
    manifest = build_manifest(dataset, workspace_root=workspace_root)
    attempts = run_benchmark(dataset, manifest, FakeArmRunner())
    summary = summarize_attempts(dataset, manifest, attempts)
    _atomic_json(DEFAULT_MANIFEST_PATH, manifest)
    _atomic_json(DEFAULT_SUMMARY_PATH, summary)
    return summary
