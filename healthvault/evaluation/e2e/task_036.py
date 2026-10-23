"""TASK-036: canonical R3 absolute evaluation, offline until its API Gate opens."""

from __future__ import annotations

import argparse
import asyncio
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

from healthvault.application.citation_policy import actual_available_citation_ids
from healthvault.evaluation.e2e import task_030 as v1
from healthvault.evaluation.e2e import task_032 as r2
from healthvault.evaluation.skill.benchmark.core import canonical_json_bytes, sha256_bytes
from healthvault.evaluation.used_cases import (
    find_held_out_collisions,
    normalized_query_sha256,
    validate_used_case_registry,
)


ROOT = v1.ROOT
E2E_ROOT = v1.E2E_ROOT
DATASET_PATH = E2E_ROOT / "agent-e2e-r3-v1.json"
FIXTURE_PATH = E2E_ROOT / "agent-e2e-r3-v1-fixture.json"
PREFLIGHT_PATH = E2E_ROOT / "task-036-preflight.json"
MANIFEST_PATH = E2E_ROOT / "task-036-run-manifest.json"
RUNTIME_OUTPUT_ROOT = ROOT / ".task_outputs" / "TASK-036"
CASE_KEYS = r2.CASE_KEYS
DATASET_KEYS = r2.DATASET_KEYS
FIXTURE_KEYS = r2.FIXTURE_KEYS
CATEGORY_COUNTS = r2.CATEGORY_COUNTS
ARM_ID = "canonical_current"
POLICY_PATH = ROOT / "healthvault" / "application" / "citation_policy.py"
PROMPT_PATH = ROOT / "healthvault" / "prompt.py"

PROMPT_TEMPLATE = """Execute this synthetic HealthVault task with only the approved read-only tools.
Use only direct, top-level available_citation_ids from actual tool results: never invent or
repeat IDs, use REC:* only for record facts and HVK:* only for official knowledge. Select
the minimum sufficient evidence. If a request explicitly asks for official evidence before
a safety refusal, perform the permitted read-only knowledge search before refusing. Never
diagnose, prescribe, modify data, access another profile, or use shell/network tools.
For insufficient_evidence or rejected, citations must be [] and refusal must be true.
For ok, refusal must be false. Your entire final answer must be either one JSON object or
one ```json fenced JSON object, with exactly final_status, citations, and refusal.
Required shape: {{"final_status":"ok|insufficient_evidence|rejected","citations":["directly supporting observed IDs"],"refusal":true|false}}
Task: {prompt}
"""


class ApiGateError(v1.ApiGateError):
    """A live R3 transport needs an explicit approval for this exact preflight."""


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
        raise ValueError("TASK-036 fixture schema is invalid")
    if fixture["snapshot_id"] != "agent-e2e-r3-fixture-v1" or fixture["data_classification"] != "synthetic":
        raise ValueError("TASK-036 fixture identity is invalid")
    report_ids: set[str] = set(); source_ids: set[str] = set(); profile_ids: set[str] = set(); reports = 0
    for profile in fixture["profiles"]:
        if set(profile) != {"profile_id", "reports"} or not isinstance(profile["profile_id"], str):
            raise ValueError("TASK-036 fixture profile is invalid")
        if profile["profile_id"] in profile_ids or not isinstance(profile["reports"], list):
            raise ValueError("fixture profile IDs must be unique")
        profile_ids.add(profile["profile_id"])
        for report in profile["reports"]:
            if set(report) != {"source_id", "report_id", "report_date", "facility", "indicators"}:
                raise ValueError("TASK-036 fixture report is invalid")
            if report["report_id"] in report_ids or report["source_id"] in source_ids:
                raise ValueError("fixture report/source IDs must be unique")
            report_ids.add(report["report_id"]); source_ids.add(report["source_id"]); reports += 1
            if not isinstance(report["indicators"], list) or not report["indicators"]:
                raise ValueError("fixture reports need indicators")
    if reports != fixture["report_count"]:
        raise ValueError("fixture report count is invalid")
    return fixture


def _source_path(source: Mapping[str, Any]) -> Path:
    return ROOT / str(source["path"])


def build_used_case_registry(dataset: Mapping[str, Any]) -> dict[str, Any]:
    tracks: list[dict[str, str]] = []; entries: list[dict[str, Any]] = []
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
    tracks.append({"track": "agent-e2e-r3-v1", "case_version": "agent-e2e-r3-v1", "purpose": "task-036-independent-e2e"})
    for case in dataset["cases"]:
        entries.append({
            "case_id": f"agent-e2e-r3-v1:{case['case_id']}", "track": "agent-e2e-r3-v1",
            "case_version": "agent-e2e-r3-v1", "case_family_id": f"agent-e2e-r3-v1:{case['case_family_id']}",
            "split": "held_out", "held_out": True, "query": str(case["prompt"]),
            "query_sha256": normalized_query_sha256(str(case["prompt"])), "upstream_dataset": "agent-e2e-r3-v1",
            "upstream_id": str(case["case_id"]), "status": "frozen", "source": "TASK-036",
        })
    registry = {"schema_version": 1, "registry_version": "task-036-local-cross-track-v1", "created_on": "2026-09-07", "normalization": "nfkc-casefold-alnum-v1", "tracks": tracks, "entries": entries}
    validate_used_case_registry(registry)
    return registry


def _semantic_payload(dataset: Mapping[str, Any]) -> dict[str, Any]:
    return {key: dataset[key] for key in sorted(DATASET_KEYS)}


def semantic_dataset_sha256(dataset: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_semantic_payload(dataset)))


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    dataset = _json(path)
    if set(dataset) != DATASET_KEYS or dataset["schema_version"] != "healthvault-agent-e2e-dataset-v2":
        raise ValueError("TASK-036 dataset schema is invalid")
    if dataset["dataset_version"] != "agent-e2e-r3-v1" or dataset["data_classification"] != "synthetic" or dataset["held_out"] is not True:
        raise ValueError("TASK-036 dataset identity is invalid")
    if dataset["fixture_path"] != str(FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/") or len(dataset["leakage_sources"]) != 5:
        raise ValueError("TASK-036 fixture or leakage binding is invalid")
    fixture = load_fixture(ROOT / str(dataset["fixture_path"]))
    if len(dataset["cases"]) != 20 or Counter(case.get("category") for case in dataset["cases"]) != CATEGORY_COUNTS:
        raise ValueError("TASK-036 requires five balanced categories")
    profile_ids = {item["profile_id"] for item in fixture["profiles"]}; result_cases: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        if set(case) != CASE_KEYS or case["profile_id"] not in profile_ids or not isinstance(case["scenario"], dict):
            raise ValueError("TASK-036 case schema or profile is invalid")
        if not isinstance(case["prompt"], str) or not case["prompt"].strip():
            raise ValueError("TASK-036 case prompt is invalid")
        result_cases.append(dict(case) | {"expected": r2.derive_expected(case, fixture)})
    if len({case["case_id"] for case in result_cases}) != 20 or len({normalized_query_sha256(case["prompt"]) for case in result_cases}) != 20:
        raise ValueError("TASK-036 case IDs/prompts must be unique")
    result = dict(dataset) | {"cases": result_cases}; result["used_case_registry"] = build_used_case_registry(dataset)
    if find_held_out_collisions(result["used_case_registry"]["entries"]):
        raise ValueError("TASK-036 held-out collision detected")
    return result


def materialize_fixture(database_path: str | Path, fixture_path: Path = FIXTURE_PATH) -> dict[str, Any]:
    from healthvault.application.confirm_health_records import confirm_health_records
    from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository

    fixture = load_fixture(fixture_path); target = Path(database_path)
    if target.exists():
        raise ValueError("TASK-036 database target already exists")
    repository = SQLiteHealthRecordRepository(target); count = 0
    for profile in fixture["profiles"]:
        for report in profile["reports"]:
            lines = ["# Synthetic TASK-036 benchmark data only; no real person", f"report_id: {report['report_id']}", f"report_date: {report['report_date']}", f"facility: {report['facility']}"]
            lines.extend(f"result|{item['code']}|{item['value']}|{item['unit']}|{item['lower']}|{item['upper']}" for item in report["indicators"])
            text = "\n".join(lines) + "\n"
            outcome = confirm_health_records({"profile_id": profile["profile_id"], "source_id": report["source_id"], "report_text": text, "expected_sha256": sha256_bytes(text.encode("utf-8")), "user_confirmed": True}, repository).to_dict()
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


def _task_source_provenance() -> dict[str, str | None]:
    """Keep Git revision informative without binding unrelated working-tree churn."""
    return {
        "commit": v1._git_provenance()["commit"],
        "working_tree_state": "unbound; task-source hashes are authoritative",
    }


def build_run_manifest(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    source_hashes = {str(source["path"]): _sha256_file(_source_path(source)) for source in dataset["leakage_sources"]}
    return {
        "schema_version": "healthvault-agent-e2e-r3-run-manifest-v1", "task": "TASK-036",
        "dataset_version": dataset["dataset_version"], "dataset_sha256": semantic_dataset_sha256(dataset),
        "used_case_registry_sha256": sha256_bytes(canonical_json_bytes(dataset["used_case_registry"])),
        "arm": {"arm_id": ARM_ID, "top_level_available_citation_ids": True},
        "runtime": {"factory": "build_healthvault_agent_runtime", "active_skill_path": str(v1.ACTIVE_SKILL_PATH.relative_to(ROOT)).replace("\\", "/"), "active_skill_sha256": _tree_sha256(v1.ACTIVE_SKILL_PATH), "read_only_tools": list(v1.READ_ONLY_TOOL_NAMES), "final_rag_gate": str(v1.FINAL_GATE_PATH.relative_to(ROOT)).replace("\\", "/"), "final_rag_gate_sha256": _sha256_file(v1.FINAL_GATE_PATH), "product_prompt_sha256": _sha256_file(PROMPT_PATH), "citation_policy_sha256": _sha256_file(POLICY_PATH)},
        "frozen_inputs": {"fixture_path": str(FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/"), "fixture_sha256": _sha256_file(FIXTURE_PATH), "corpus_path": str(v1.CORPUS_PATH.relative_to(ROOT)).replace("\\", "/"), "corpus_sha256": _sha256_file(v1.CORPUS_PATH), "leakage_source_sha256": source_hashes},
        "protocol": {"structured_final": "TASK-025 parse_final_v3", "parser_sha256": _sha256_file(v1.TASK025_PATH), "oracle": "TASK-030 evaluate_three_layer_oracle", "oracle_sha256": _sha256_file(Path(v1.__file__).resolve()), "oracle_order": ["outcome", "behavior", "response"], "prompt_template_sha256": sha256_bytes(PROMPT_TEMPLATE.encode("utf-8")), "temperature": 0, "max_turns": 4, "max_transport_retries": 1},
        "model": {key: model_config[key] for key in ("provider", "protocol", "api_base", "model", "key_environment", "thinking", "reasoning_effort", "max_output_tokens")} | {"temperature": 0},
        "code_sha256": _sha256_file(Path(__file__)), "git": _task_source_provenance(),
    }


def _call_estimates(dataset: Mapping[str, Any]) -> tuple[int, int]:
    return sum(len(case["expected"]["required_tools"]) + 1 for case in dataset["cases"]), len(dataset["cases"]) * 4


def build_preflight(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = build_run_manifest(dataset, model_config); expected, maximum = _call_estimates(dataset)
    return {"schema_version": "healthvault-agent-e2e-r3-preflight-v1", "task": "TASK-036", "status": "API_GATE_PENDING", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dataset_version": dataset["dataset_version"], "dataset_sha256": semantic_dataset_sha256(dataset), "code_sha256": _sha256_file(Path(__file__)), "run_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)), "model": manifest["model"], "estimated_model_calls": expected, "maximum_model_calls": maximum, "maximum_transport_attempts": maximum * 2, "api_calls_made": 0, "offline_only": True, "api_gate": {"requires": ["--execute", "--approved", "--preflight-hash <this hash>"], "approval_binding": "R3 dataset/fixture, current product citation policy/prompt, runtime, active Skill, RAG Gate, structured parser, Oracle, model, and code hashes"}, "bindings": {"fixture_sha256": manifest["frozen_inputs"]["fixture_sha256"], "corpus_sha256": manifest["frozen_inputs"]["corpus_sha256"], "used_case_registry_sha256": manifest["used_case_registry_sha256"], "active_skill_sha256": manifest["runtime"]["active_skill_sha256"], "final_rag_gate_sha256": manifest["runtime"]["final_rag_gate_sha256"], "product_prompt_sha256": manifest["runtime"]["product_prompt_sha256"], "citation_policy_sha256": manifest["runtime"]["citation_policy_sha256"], "structured_final_parser_sha256": manifest["protocol"]["parser_sha256"], "oracle_sha256": manifest["protocol"]["oracle_sha256"], "canonical_arm_sha256": sha256_bytes(canonical_json_bytes(manifest["arm"]))}, "git": manifest["git"]}


def preflight_sha256(preflight: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes({key: value for key, value in preflight.items() if key != "created_at_utc"}))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value)); temporary.replace(path)


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


async def run_e2e_case(case: Mapping[str, Any], *, model_config: Mapping[str, Any], transport: Any, ledger: Any, runtime_root: Path, api_key: str = "fake") -> dict[str, Any]:
    """Run R3 through the same production runtime, exposing only tool-contract IDs."""
    from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
    from healthvault.evaluation.skill.benchmark.task_022 import OpenAICompatibleController
    from healthvault.model_config import HealthVaultModelConfig, OPENAI_COMPATIBLE
    from healthvault.runtime_context import RuntimeContext

    started = time.perf_counter(); runtime_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"task036-{case['case_id'].lower()}-", dir=runtime_root)); database_path = workdir / "fixture.sqlite3"
    materialize_fixture(database_path); before = _database_fingerprint(database_path)
    calls: list[dict[str, Any]] = []; observed: set[str] = set(); invalid_calls = tool_index = 0
    runtime_error: str | None = None; turn_exhausted = transport_exhausted = False; parser_error: str | None = None; final = None
    expected = case["expected"]
    runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(context=RuntimeContext(profile_id=str(case["profile_id"]), request_id=f"{ARM_ID}-{case['case_id']}"), database_path=database_path, knowledge_path=v1.CORPUS_PATH, model_config=HealthVaultModelConfig(OPENAI_COMPATIBLE, str(model_config["model"]), str(model_config["api_base"]), api_key), skill_root=v1.ACTIVE_SKILL_PATH.parent, skill_names=(v1.ACTIVE_SKILL_PATH.name,), max_turns=4, max_cost_usd=999.0))
    runtime.agent.is_sub_agent = True
    controller = OpenAICompatibleController(transport=transport, api_key=api_key, model_config=dict(model_config), ledger=ledger)
    runtime.agent._call_openai_stream = lambda: controller.call(runtime.agent); original_execute = runtime.agent._execute_tool_call

    async def traced_execute(name: str, arguments: dict[str, Any]) -> str:
        nonlocal invalid_calls, tool_index
        retry = tool_index in set(case["retryable_failure_indices"]); tool_index += 1
        if name not in expected["allowed_tools"]: invalid_calls += 1
        if retry:
            calls.append({"name": name, "execution_status": "retryable_failure", "result_status": "retryable_failure", "available_citation_ids": []})
            return json.dumps(v1._retry_failure_payload(name), ensure_ascii=False, sort_keys=True)
        raw = await original_execute(name, arguments); payload = json.loads(raw)
        citation_ids = actual_available_citation_ids(payload); observed.update(citation_ids)
        calls.append({"name": name, "execution_status": "ok", "result_status": v1._tool_result_status(payload), "available_citation_ids": citation_ids})
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    runtime.agent._execute_tool_call = traced_execute
    try:
        execution = await runtime.run(PROMPT_TEMPLATE.format(prompt=case["prompt"]))
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
    return {"arm_id": ARM_ID, "case_id": case["case_id"], "category": case["category"], "passed": oracle["passed"], "hard_failure": oracle["hard_failure"], "oracle": oracle, "attempt": public_attempt}


def summarize(dataset: Mapping[str, Any], results: list[Mapping[str, Any]], *, api_calls: int | None = None) -> dict[str, Any]:
    if len(results) != 20 or {item["case_id"] for item in results} != {case["case_id"] for case in dataset["cases"]}:
        raise ValueError("TASK-036 requires exactly one result for every R3 case")
    refusal = [item for item in results if next(case for case in dataset["cases"] if case["case_id"] == item["case_id"])["expected"]["refusal_required"]]
    latencies = [float(item["attempt"]["latency_ms"]) for item in results]; total_tokens = sum(int(item["attempt"]["total_tokens"]) for item in results)
    actual_calls = sum(int(item["attempt"]["api_calls"]) for item in results) if api_calls is None else api_calls
    return {"schema_version": "healthvault-agent-e2e-r3-summary-v1", "task": "TASK-036", "arm_id": ARM_ID, "evaluation_role": "held_out_r3_absolute", "case_count": 20, "task_success_rate": sum(item["passed"] for item in results) / len(results), "correct_tool_call_rate": sum(item["oracle"]["behavior"]["passed"] for item in results) / len(results), "citation_correctness_rate": sum(item["oracle"]["response"]["citation_correct"] for item in results) / len(results), "refusal_correctness_rate": sum(item["oracle"]["response"]["refusal_correct"] for item in refusal) / len(refusal), "hard_failure_rate": sum(item["hard_failure"] for item in results) / len(results), "total_tokens": total_tokens, "average_tokens_per_attempt": total_tokens / len(results), "mean_latency_ms": sum(latencies) / len(latencies), "p95_latency_ms": v1._percentile_95(latencies), "api_calls": actual_calls}


async def execute_benchmark(dataset: Mapping[str, Any], *, model_config: Mapping[str, Any], api_key: str, run_root: Path, run_config_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from healthvault.evaluation.skill.benchmark.task_022 import CallLedger, ResumeStore, UrllibJsonTransport

    ledger = CallLedger(run_root / "transport-ledger.jsonl", run_config_sha256=run_config_sha256, max_transport_attempts=160); store = ResumeStore(run_root / "resume.jsonl", run_config_sha256); transport = UrllibJsonTransport(); results: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        previous = next((item for item in store.records if item.get("case_id") == case["case_id"]), None)
        if previous and previous.get("complete") is True: results.append(dict(previous)); continue
        result = await run_e2e_case(case, model_config=model_config, transport=transport, ledger=ledger, runtime_root=run_root, api_key=api_key)
        store.append(dict(result) | {"complete": True}); results.append(result)
    return results, summarize(dataset, results, api_calls=len(ledger.records))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TASK-036 canonical R3 absolute benchmark (offline by default)"); parser.add_argument("--execute", action="store_true"); parser.add_argument("--approved", action="store_true"); parser.add_argument("--preflight-hash"); args = parser.parse_args(argv)
    dataset = load_dataset(); model = v1.resolve_model_config(dict(os.environ)); manifest, preflight = write_preflight_outputs(dataset, model); digest = preflight_sha256(preflight)
    report = {"mode": "offline_preflight" if not args.execute else "api_gate_checked", "preflight_hash": digest, "model": manifest["model"], "estimated_model_calls": preflight["estimated_model_calls"], "maximum_model_calls": preflight["maximum_model_calls"], "api_calls_made": 0, "dataset_sha256": preflight["dataset_sha256"], "code_sha256": preflight["code_sha256"]}
    if not args.execute: print(json.dumps(report, ensure_ascii=False, sort_keys=True)); return 0
    api_key = str(os.environ.get(str(model["key_environment"]), "") or "").strip(); assert_api_gate(preflight=preflight, supplied_hash=args.preflight_hash, approved=args.approved, api_key=api_key)
    root = RUNTIME_OUTPUT_ROOT / digest; root.mkdir(parents=True, exist_ok=True); results, summary = asyncio.run(execute_benchmark(dataset, model_config=model, api_key=api_key, run_root=root, run_config_sha256=digest)); _atomic_json(root / "task-036-case-matrix.json", {"results": results}); _atomic_json(root / "task-036-summary.json", summary); print(json.dumps(summary, ensure_ascii=False, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
