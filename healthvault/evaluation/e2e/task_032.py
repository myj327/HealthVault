"""TASK-032 R2 held-out, two-arm citation-contract benchmark."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from healthvault.evaluation.e2e import task_030 as v1
from healthvault.evaluation.skill.benchmark.core import canonical_json_bytes, sha256_bytes
from healthvault.evaluation.used_cases import (
    find_held_out_collisions,
    normalized_query_sha256,
    validate_used_case_registry,
)


ROOT = v1.ROOT
E2E_ROOT = v1.E2E_ROOT
DATASET_PATH = E2E_ROOT / "agent-e2e-r2-v1.json"
FIXTURE_PATH = E2E_ROOT / "agent-e2e-r2-v1-fixture.json"
PREFLIGHT_PATH = E2E_ROOT / "task-032-preflight.json"
MANIFEST_PATH = E2E_ROOT / "task-032-run-manifest.json"
RUNTIME_OUTPUT_ROOT = ROOT / ".task_outputs" / "TASK-032"
ARM_IDS = ("legacy_contract", "canonical_contract")
CASE_KEYS = {
    "case_id", "case_family_id", "category", "profile_id", "prompt", "scenario",
    "retryable_failure_indices", "boundary_tags",
}
DATASET_KEYS = {
    "schema_version", "dataset_version", "data_classification", "held_out", "frozen_on",
    "fixture_path", "leakage_sources", "cases",
}
FIXTURE_KEYS = {"schema_version", "snapshot_id", "data_classification", "profiles", "report_count"}
CATEGORY_COUNTS = {
    "timeline": 4, "comparison": 4, "knowledge_refusal": 4,
    "visit_brief": 4, "permissions_failure": 4,
}
TOOL_FOR_CATEGORY = {
    "timeline": "query_health_timeline",
    "comparison": "compare_health_indicators",
    "knowledge_refusal": "search_health_knowledge",
    "visit_brief": "generate_visit_brief",
}


class ApiGateError(v1.ApiGateError):
    """TASK-032 cannot create a live transport until the current hash is approved."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    fixture = _json(path)
    if set(fixture) != FIXTURE_KEYS or fixture["schema_version"] != "healthvault-agent-e2e-fixture-v1":
        raise ValueError("TASK-032 fixture schema is invalid")
    if fixture["snapshot_id"] != "agent-e2e-r2-fixture-v1" or fixture["data_classification"] != "synthetic":
        raise ValueError("TASK-032 fixture identity is invalid")
    reports = 0
    profile_ids: set[str] = set()
    report_ids: set[str] = set()
    source_ids: set[str] = set()
    for profile in fixture["profiles"]:
        if set(profile) != {"profile_id", "reports"} or not isinstance(profile["profile_id"], str):
            raise ValueError("TASK-032 fixture profile is invalid")
        if profile["profile_id"] in profile_ids or not isinstance(profile["reports"], list):
            raise ValueError("fixture profile IDs must be unique")
        profile_ids.add(profile["profile_id"])
        for report in profile["reports"]:
            if set(report) != {"source_id", "report_id", "report_date", "facility", "indicators"}:
                raise ValueError("TASK-032 fixture report is invalid")
            if report["report_id"] in report_ids or report["source_id"] in source_ids:
                raise ValueError("fixture report/source IDs must be unique")
            report_ids.add(report["report_id"]); source_ids.add(report["source_id"]); reports += 1
            if not isinstance(report["indicators"], list) or not report["indicators"]:
                raise ValueError("fixture reports need indicators")
    if reports != fixture["report_count"]:
        raise ValueError("fixture report count is invalid")
    return fixture


def _record_ids(reports: list[dict[str, Any]], code: str | None = None) -> list[str]:
    return [
        f"REC:{report['report_id']}:{indicator['code']}"
        for report in reports
        for indicator in report["indicators"]
        if code is None or indicator["code"] == code
    ]


def _knowledge_ids(query: str) -> list[str]:
    from healthvault.infrastructure.local_health_knowledge import LocalHealthKnowledgeRepository

    return [hit.document.citation_id for hit in LocalHealthKnowledgeRepository(v1.CORPUS_PATH).search(query, 3)]


def derive_expected(case: Mapping[str, Any], fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Build all Oracle rules from the frozen fixture, tool states, and policy rules."""
    profile = next(item for item in fixture["profiles"] if item["profile_id"] == case["profile_id"])
    reports = profile["reports"]
    category = str(case["category"])
    all_tools = set(v1.READ_ONLY_TOOL_NAMES)
    if category == "permissions_failure":
        return {
            "required_tools": [], "allowed_tools": [], "forbidden_tools": sorted(all_tools),
            "max_calls": 0, "required_tool_statuses": [], "required_tool_result_statuses": [],
            "final_status": "rejected", "refusal_required": True, "allowed_citation_ids": [],
            "min_citations": 0, "max_citations": 0,
            "prohibited_writes": ["confirm_health_records", "file", "shell", "network"],
        }
    tool = TOOL_FOR_CATEGORY[category]
    allowed = [tool]
    required = [tool]
    statuses = ["ok"]
    result_status = "ok"
    final_status = "ok"
    citations: list[str] = []
    minimum = maximum = 0
    scenario = case["scenario"]
    if category == "timeline":
        citations = _record_ids(reports)
        if not reports:
            result_status, final_status = "empty", "insufficient_evidence"
        else:
            minimum, maximum = 1, min(2, len(citations))
    elif category == "comparison":
        code = str(scenario["indicator_code"])
        citations = _record_ids(reports, code)
        if len(citations) < 2:
            result_status, final_status, citations = "insufficient_data", "insufficient_evidence", []
        else:
            minimum = maximum = 2
    elif category == "knowledge_refusal":
        rule = str(scenario["rule"])
        if rule == "supported":
            citations = _knowledge_ids(str(scenario["query"]))
            if not citations:
                raise ValueError(f"supported knowledge scenario has no structured corpus hit: {case['case_id']}")
            minimum = maximum = 1
        elif rule == "unsafe":
            result_status, final_status = "empty", "rejected"
        elif rule == "unsupported":
            result_status, final_status = "empty", "insufficient_evidence"
        else:
            raise ValueError("unknown knowledge rule")
    elif category == "visit_brief":
        citations = _record_ids(reports)
        if not reports:
            result_status, final_status = "empty", "insufficient_evidence"
        else:
            observation_counts = Counter(
                indicator["code"] for report in reports for indicator in report["indicators"]
            )
            result_status = "partial" if any(count < 2 for count in observation_counts.values()) else "ok"
            minimum, maximum = 1, min(2, len(citations))
    if final_status != "ok":
        citations, minimum, maximum = [], 0, 0
    if case["retryable_failure_indices"]:
        required = [tool, tool]
        statuses = ["retryable_failure", "ok"]
        result_statuses = ["retryable_failure", result_status]
    else:
        result_statuses = [result_status]
    return {
        "required_tools": required, "allowed_tools": allowed,
        "forbidden_tools": sorted(all_tools - set(allowed)), "max_calls": len(required),
        "required_tool_statuses": statuses, "required_tool_result_statuses": result_statuses,
        "final_status": final_status, "refusal_required": final_status != "ok",
        "allowed_citation_ids": citations, "min_citations": minimum, "max_citations": maximum,
        "prohibited_writes": ["confirm_health_records", "file", "shell", "network"],
    }


def _source_path(source: Mapping[str, Any]) -> Path:
    path = ROOT / str(source["path"])
    if not path.is_file():
        raise ValueError(f"missing leakage source: {path}")
    return path


def build_used_case_registry(dataset: Mapping[str, Any]) -> dict[str, Any]:
    tracks: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for source in dataset["leakage_sources"]:
        if set(source) != {"track", "case_version", "purpose", "path"}:
            raise ValueError("leakage source schema is invalid")
        tracks.append({key: str(source[key]) for key in ("track", "case_version", "purpose")})
        for source_id, query in v1._cases_from_source(_source_path(source)):
            entries.append({
                "case_id": f"{source['track']}:{source['case_version']}:{source_id}",
                "track": str(source["track"]), "case_version": str(source["case_version"]),
                "case_family_id": f"{source['track']}:{source['case_version']}:{source_id}",
                "split": "held_out", "held_out": True, "query": query,
                "query_sha256": normalized_query_sha256(query), "upstream_dataset": str(source["path"]),
                "upstream_id": source_id, "status": "frozen", "source": str(source["purpose"]),
            })
    tracks.append({"track": "agent-e2e-r2-v1", "case_version": "agent-e2e-r2-v1", "purpose": "task-032-independent-e2e"})
    for case in dataset["cases"]:
        entries.append({
            "case_id": f"agent-e2e-r2-v1:{case['case_id']}", "track": "agent-e2e-r2-v1",
            "case_version": "agent-e2e-r2-v1", "case_family_id": f"agent-e2e-r2-v1:{case['case_family_id']}",
            "split": "held_out", "held_out": True, "query": str(case["prompt"]),
            "query_sha256": normalized_query_sha256(str(case["prompt"])), "upstream_dataset": "agent-e2e-r2-v1",
            "upstream_id": str(case["case_id"]), "status": "frozen", "source": "TASK-032",
        })
    registry = {"schema_version": 1, "registry_version": "task-032-local-cross-track-v1", "created_on": "2026-09-06", "normalization": "nfkc-casefold-alnum-v1", "tracks": tracks, "entries": entries}
    validate_used_case_registry(registry)
    return registry


def _semantic_payload(dataset: Mapping[str, Any]) -> dict[str, Any]:
    return {key: dataset[key] for key in sorted(DATASET_KEYS)}


def semantic_dataset_sha256(dataset: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_semantic_payload(dataset)))


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    dataset = _json(path)
    fixture = load_fixture(ROOT / str(dataset.get("fixture_path", "")))
    if set(dataset) != DATASET_KEYS or dataset["schema_version"] != "healthvault-agent-e2e-dataset-v2":
        raise ValueError("TASK-032 dataset schema is invalid")
    if dataset["dataset_version"] != "agent-e2e-r2-v1" or dataset["data_classification"] != "synthetic" or dataset["held_out"] is not True:
        raise ValueError("TASK-032 dataset identity is invalid")
    if dataset["fixture_path"] != str(FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/") or len(dataset["leakage_sources"]) != 4:
        raise ValueError("TASK-032 fixture or leakage binding is invalid")
    if len(dataset["cases"]) != 20 or Counter(case.get("category") for case in dataset["cases"]) != CATEGORY_COUNTS:
        raise ValueError("TASK-032 requires five balanced categories")
    profile_ids = {profile["profile_id"] for profile in fixture["profiles"]}
    result_cases = []
    for case in dataset["cases"]:
        if set(case) != CASE_KEYS or case["profile_id"] not in profile_ids:
            raise ValueError("TASK-032 case schema or profile is invalid")
        if not isinstance(case["prompt"], str) or not case["prompt"].strip() or not isinstance(case["scenario"], dict):
            raise ValueError("TASK-032 case prompt/scenario is invalid")
        expected = derive_expected(case, fixture)
        result_cases.append(dict(case) | {"expected": expected})
    if len({case["case_id"] for case in result_cases}) != 20 or len({normalized_query_sha256(case["prompt"]) for case in result_cases}) != 20:
        raise ValueError("TASK-032 case IDs/prompts must be unique")
    result = dict(dataset) | {"cases": result_cases}
    result["used_case_registry"] = build_used_case_registry(dataset)
    if find_held_out_collisions(result["used_case_registry"]["entries"]):
        raise ValueError("TASK-032 held-out collision detected")
    return result


def materialize_fixture(database_path: str | Path, fixture_path: Path = FIXTURE_PATH) -> dict[str, Any]:
    from healthvault.application.confirm_health_records import confirm_health_records
    from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository

    fixture = load_fixture(fixture_path)
    target = Path(database_path)
    if target.exists():
        raise ValueError("TASK-032 database target already exists")
    repository = SQLiteHealthRecordRepository(target)
    count = 0
    for profile in fixture["profiles"]:
        for report in profile["reports"]:
            lines = ["# Synthetic TASK-032 benchmark data only; no real person", f"report_id: {report['report_id']}", f"report_date: {report['report_date']}", f"facility: {report['facility']}"]
            lines.extend(f"result|{item['code']}|{item['value']}|{item['unit']}|{item['lower']}|{item['upper']}" for item in report["indicators"])
            report_text = "\n".join(lines) + "\n"
            outcome = confirm_health_records({"profile_id": profile["profile_id"], "source_id": report["source_id"], "report_text": report_text, "expected_sha256": sha256_bytes(report_text.encode("utf-8")), "user_confirmed": True}, repository).to_dict()
            if outcome["status"] != "confirmed":
                raise ValueError(f"fixture report did not confirm: {report['report_id']}")
            count += 1
    return {"snapshot_id": fixture["snapshot_id"], "report_count": count}


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8")); digest.update(b"\0"); digest.update(item.read_bytes())
    return digest.hexdigest()


def build_run_manifest(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    sources = {str(source["path"]): _sha256_file(_source_path(source)) for source in dataset["leakage_sources"]}
    return {
        "schema_version": "healthvault-agent-e2e-r2-run-manifest-v1", "task": "TASK-032",
        "dataset_version": dataset["dataset_version"], "dataset_sha256": semantic_dataset_sha256(dataset),
        "used_case_registry_sha256": sha256_bytes(canonical_json_bytes(dataset["used_case_registry"])),
        "arms": {"order": list(ARM_IDS), "fixed_alternating_by_case": True, "legacy_contract": {"top_level_available_citation_ids": False}, "canonical_contract": {"top_level_available_citation_ids": True}},
        "runtime": {"factory": "build_healthvault_agent_runtime", "active_skill_path": str(v1.ACTIVE_SKILL_PATH.relative_to(ROOT)).replace("\\", "/"), "active_skill_sha256": _tree_sha256(v1.ACTIVE_SKILL_PATH), "read_only_tools": list(v1.READ_ONLY_TOOL_NAMES), "final_rag_gate": str(v1.FINAL_GATE_PATH.relative_to(ROOT)).replace("\\", "/"), "final_rag_gate_sha256": _sha256_file(v1.FINAL_GATE_PATH)},
        "frozen_inputs": {"fixture_path": str(FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/"), "fixture_sha256": _sha256_file(FIXTURE_PATH), "corpus_path": str(v1.CORPUS_PATH.relative_to(ROOT)).replace("\\", "/"), "corpus_sha256": _sha256_file(v1.CORPUS_PATH), "leakage_source_sha256": sources},
        "protocol": {"structured_final": "TASK-025 parse_final_v3", "parser_sha256": _sha256_file(v1.TASK025_PATH), "oracle_order": ["outcome", "behavior", "response"], "prompt_template_sha256": sha256_bytes(v1.PROMPT_TEMPLATE.encode("utf-8")), "temperature": 0, "max_turns": 4, "max_transport_retries": 1},
        "model": {key: copy.deepcopy(model_config[key]) for key in ("provider", "protocol", "api_base", "model", "key_environment", "thinking", "reasoning_effort", "max_output_tokens")} | {"temperature": 0},
        "code_sha256": _sha256_file(Path(__file__)), "git": v1._git_provenance(),
    }


def _call_estimates(dataset: Mapping[str, Any]) -> tuple[int, int]:
    single = sum(len(case["expected"]["required_tools"]) + 1 for case in dataset["cases"])
    return single * len(ARM_IDS), len(dataset["cases"]) * len(ARM_IDS) * 4


def build_preflight(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = build_run_manifest(dataset, model_config)
    expected, maximum = _call_estimates(dataset)
    return {"schema_version": "healthvault-agent-e2e-r2-preflight-v1", "task": "TASK-032", "status": "API_GATE_PENDING", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dataset_version": dataset["dataset_version"], "dataset_sha256": semantic_dataset_sha256(dataset), "code_sha256": _sha256_file(Path(__file__)), "run_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)), "model": manifest["model"], "estimated_model_calls": expected, "maximum_model_calls": maximum, "maximum_transport_attempts": maximum * 2, "api_calls_made": 0, "offline_only": True, "api_gate": {"requires": ["--execute", "--approved", "--preflight-hash <this hash>"], "approval_binding": "R2 dataset, fixture, arm adapters, runtime, tool contract, model, and code hashes"}, "bindings": {"fixture_sha256": manifest["frozen_inputs"]["fixture_sha256"], "corpus_sha256": manifest["frozen_inputs"]["corpus_sha256"], "active_skill_sha256": manifest["runtime"]["active_skill_sha256"], "final_rag_gate_sha256": manifest["runtime"]["final_rag_gate_sha256"], "structured_final_parser_sha256": manifest["protocol"]["parser_sha256"], "arms_sha256": sha256_bytes(canonical_json_bytes(manifest["arms"]))}, "git": manifest["git"]}


def preflight_sha256(preflight: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes({key: value for key, value in preflight.items() if key != "created_at_utc"}))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp"); temporary.write_bytes(canonical_json_bytes(value)); temporary.replace(path)


def write_preflight_outputs(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, preflight = build_run_manifest(dataset, model_config), build_preflight(dataset, model_config)
    _atomic_json(MANIFEST_PATH, manifest); _atomic_json(PREFLIGHT_PATH, preflight)
    return manifest, preflight


def assert_api_gate(*, preflight: Mapping[str, Any], supplied_hash: str | None, approved: bool, api_key: str | None) -> None:
    if not approved or supplied_hash != preflight_sha256(preflight) or not api_key:
        raise ApiGateError("API Gate closed: require explicit current-hash approval and configured key")


def _database_fingerprint(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return hashlib.sha256("\n".join(connection.iterdump()).encode("utf-8")).hexdigest()


def _adapt_payload(payload: Any, arm_id: str) -> Any:
    if arm_id == "canonical_contract" or not isinstance(payload, dict):
        return payload
    adapted = copy.deepcopy(payload)
    adapted.pop("available_citation_ids", None)
    return adapted


async def run_e2e_case(case: Mapping[str, Any], *, arm_id: str, model_config: Mapping[str, Any], transport: Any, ledger: Any, runtime_root: Path, api_key: str = "fake") -> dict[str, Any]:
    from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
    from healthvault.evaluation.skill.benchmark.task_022 import OpenAICompatibleController, _extract_citations
    from healthvault.model_config import HealthVaultModelConfig, OPENAI_COMPATIBLE
    from healthvault.runtime_context import RuntimeContext

    started = time.perf_counter(); workdir = Path(tempfile.mkdtemp(prefix=f"task032-{arm_id}-{case['case_id'].lower()}-", dir=runtime_root)); database_path = workdir / "fixture.sqlite3"
    materialize_fixture(database_path); before = _database_fingerprint(database_path)
    calls: list[dict[str, Any]] = []; observed: set[str] = set(); invalid_calls = tool_index = 0
    runtime_error: str | None = None; turn_exhausted = transport_exhausted = False; parser_error: str | None = None; final = None
    expected = case["expected"]
    runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(context=RuntimeContext(profile_id=str(case["profile_id"]), request_id=f"{arm_id}-{case['case_id']}"), database_path=database_path, knowledge_path=v1.CORPUS_PATH, model_config=HealthVaultModelConfig(OPENAI_COMPATIBLE, str(model_config["model"]), str(model_config["api_base"]), api_key), skill_root=v1.ACTIVE_SKILL_PATH.parent, skill_names=(v1.ACTIVE_SKILL_PATH.name,), max_turns=4, max_cost_usd=999.0))
    runtime.agent.is_sub_agent = True
    # Vary only the model-visible contract; runtime citation authority uses real tool output.
    controller = OpenAICompatibleController(transport=transport, api_key=api_key, model_config=dict(model_config), ledger=ledger, tool_result_view=lambda payload: _adapt_payload(payload, arm_id))
    runtime.agent._call_openai_stream = lambda: controller.call(runtime.agent)
    original_execute = runtime.agent._execute_tool_call

    async def traced_execute(name: str, arguments: dict[str, Any]) -> str:
        nonlocal invalid_calls, tool_index
        retry = tool_index in set(case["retryable_failure_indices"]); tool_index += 1
        if name not in expected["allowed_tools"]: invalid_calls += 1
        if retry:
            calls.append({"name": name, "execution_status": "retryable_failure", "result_status": "retryable_failure", "available_citation_ids": []})
            return json.dumps(v1._retry_failure_payload(name))
        raw = await original_execute(name, arguments)
        payload = json.loads(raw); citation_ids = sorted(_extract_citations(payload)); observed.update(citation_ids)
        calls.append({"name": name, "execution_status": "ok", "result_status": v1._tool_result_status(payload), "available_citation_ids": citation_ids})
        return raw

    runtime.agent._execute_tool_call = traced_execute
    try:
        execution = await runtime.run(v1.PROMPT_TEMPLATE.format(case_id=case["case_id"], prompt=case["prompt"]))
        parsed = v1._parse_task025_final(execution.text)
        if parsed.failure_code: parser_error = parsed.failure_code
        else: final = parsed
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"; transport_exhausted = "transport" in str(exc).lower(); turn_exhausted = "turn limit" in str(exc).lower()
    finally:
        unchanged = before == _database_fingerprint(database_path); shutil.rmtree(workdir, ignore_errors=True)
    attempt = {"calls": calls, "final": final, "parser_error": parser_error, "runtime_error": runtime_error, "turn_exhausted": turn_exhausted, "transport_exhausted": transport_exhausted, "database_unchanged": unchanged, "invalid_calls": invalid_calls, "observed_citation_ids": sorted(observed), "latency_ms": round((time.perf_counter() - started) * 1000, 3), "model_turns": controller.model_calls, "api_calls": controller.api_calls, "input_tokens": controller.input_tokens, "output_tokens": controller.output_tokens, "total_tokens": controller.total_tokens}
    oracle = v1.evaluate_three_layer_oracle(case, attempt)
    public_attempt = {key: value for key, value in attempt.items() if key != "final"} | {"final": None if final is None else {"final_status": final.final_status, "citations": list(final.citations), "refusal": final.refusal}}
    return {"arm_id": arm_id, "case_id": case["case_id"], "category": case["category"], "passed": oracle["passed"], "hard_failure": oracle["hard_failure"], "oracle": oracle, "attempt": public_attempt}


def summarize(dataset: Mapping[str, Any], results: list[Mapping[str, Any]]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        group = [item for item in results if item["arm_id"] == arm_id]
        if len(group) != 20: raise ValueError("incomplete TASK-032 arm")
        arms[arm_id] = {"attempts": len(group), "task_success_rate": sum(item["passed"] for item in group) / len(group), "hard_failure_rate": sum(item["hard_failure"] for item in group) / len(group), "correct_tool_call_rate": sum(item["oracle"]["behavior"]["passed"] for item in group) / len(group), "citation_correctness_rate": sum(item["oracle"]["response"]["citation_correct"] for item in group) / len(group), "api_calls": sum(item["attempt"]["api_calls"] for item in group)}
    return {"schema_version": "healthvault-agent-e2e-r2-summary-v1", "case_count": 20, "arms": arms, "canonical_minus_legacy": {key: arms["canonical_contract"][key] - arms["legacy_contract"][key] for key in ("task_success_rate", "hard_failure_rate", "correct_tool_call_rate", "citation_correctness_rate")}}


async def execute_benchmark(dataset: Mapping[str, Any], *, model_config: Mapping[str, Any], api_key: str, run_root: Path, run_config_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from healthvault.evaluation.skill.benchmark.task_022 import CallLedger, ResumeStore, UrllibJsonTransport

    ledger = CallLedger(run_root / "transport-ledger.jsonl", run_config_sha256=run_config_sha256, max_transport_attempts=320); store = ResumeStore(run_root / "resume.jsonl", run_config_sha256); transport = UrllibJsonTransport(); results: list[dict[str, Any]] = []
    for index, case in enumerate(dataset["cases"]):
        order = ARM_IDS if index % 2 == 0 else tuple(reversed(ARM_IDS))
        for arm_id in order:
            previous = next((item for item in store.records if item.get("case_id") == case["case_id"] and item.get("arm_id") == arm_id), None)
            if previous and previous.get("complete") is True: results.append(dict(previous)); continue
            result = await run_e2e_case(case, arm_id=arm_id, model_config=model_config, transport=transport, ledger=ledger, runtime_root=run_root, api_key=api_key)
            store.append(dict(result) | {"complete": True}); results.append(result)
    return results, summarize(dataset, results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TASK-032 R2 contract blind benchmark (offline by default)"); parser.add_argument("--execute", action="store_true"); parser.add_argument("--approved", action="store_true"); parser.add_argument("--preflight-hash"); args = parser.parse_args(argv)
    dataset = load_dataset(); model = v1.resolve_model_config(dict(os.environ)); manifest, preflight = write_preflight_outputs(dataset, model); digest = preflight_sha256(preflight)
    report = {"mode": "offline_preflight" if not args.execute else "api_gate_checked", "preflight_hash": digest, "model": manifest["model"], "estimated_model_calls": preflight["estimated_model_calls"], "maximum_model_calls": preflight["maximum_model_calls"], "api_calls_made": 0, "dataset_sha256": preflight["dataset_sha256"], "code_sha256": preflight["code_sha256"]}
    if not args.execute: print(json.dumps(report, ensure_ascii=False, sort_keys=True)); return 0
    api_key = str(os.environ.get(str(model["key_environment"]), "") or "").strip(); assert_api_gate(preflight=preflight, supplied_hash=args.preflight_hash, approved=args.approved, api_key=api_key)
    root = RUNTIME_OUTPUT_ROOT / digest; root.mkdir(parents=True, exist_ok=True); results, summary = asyncio.run(execute_benchmark(dataset, model_config=model, api_key=api_key, run_root=root, run_config_sha256=digest)); _atomic_json(root / "task-032-case-matrix.json", {"results": results}); _atomic_json(root / "task-032-summary.json", summary); print(json.dumps(summary, ensure_ascii=False, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
