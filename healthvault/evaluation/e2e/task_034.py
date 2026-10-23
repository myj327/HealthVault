"""TASK-034: one-shot absolute metrics for TASK-032's canonical R2 system."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from healthvault.evaluation.e2e import task_032 as r2
from healthvault.evaluation.skill.benchmark.core import canonical_json_bytes, sha256_bytes


ROOT = r2.ROOT
DATASET_PATH = r2.DATASET_PATH
PREFLIGHT_PATH = r2.E2E_ROOT / "task-034-preflight.json"
MANIFEST_PATH = r2.E2E_ROOT / "task-034-run-manifest.json"
RUNTIME_OUTPUT_ROOT = ROOT / ".task_outputs" / "TASK-034"
ARM_ID = "canonical_contract"
TASK032_RUNNER_PATH = Path(r2.__file__).resolve()
TASK030_ORACLE_PATH = Path(r2.v1.__file__).resolve()
CONSOLE_OUTPUT_BINDING = {"encoding": "utf-8", "errors": "backslashreplace"}


class ApiGateError(r2.ApiGateError):
    """A TASK-034 run requires a new explicit, hash-bound approval."""


def configure_utf8_console_output() -> None:
    """Keep Rich tool rendering out of a Windows GBK stream before Agent imports."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(**CONSOLE_OUTPUT_BINDING)


def load_dataset() -> dict[str, Any]:
    """Load TASK-032's frozen R2 input without altering its stored bytes."""
    return r2.load_dataset(DATASET_PATH)


async def run_e2e_case(
    case: Mapping[str, Any], *, model_config: Mapping[str, Any], transport: Any,
    ledger: Any, runtime_root: Path, api_key: str = "fake",
) -> dict[str, Any]:
    """The only contract view visible to this task is the current canonical one."""
    configure_utf8_console_output()
    return await r2.run_e2e_case(
        case, arm_id=ARM_ID, model_config=model_config, transport=transport,
        ledger=ledger, runtime_root=runtime_root, api_key=api_key,
    )


def build_run_manifest(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    base = r2.build_run_manifest(dataset, model_config)
    protocol = dict(base["protocol"]) | {"console_output": CONSOLE_OUTPUT_BINDING}
    delegated_sources = {
        "task_032_runner": r2._sha256_file(TASK032_RUNNER_PATH),
        "task_030_oracle": r2._sha256_file(TASK030_ORACLE_PATH),
    }
    return {
        "schema_version": "healthvault-agent-e2e-absolute-run-manifest-v1",
        "task": "TASK-034",
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": r2.semantic_dataset_sha256(dataset),
        "used_case_registry_sha256": base["used_case_registry_sha256"],
        "arm": {"arm_id": ARM_ID, "top_level_available_citation_ids": True},
        "runtime": base["runtime"],
        "frozen_inputs": base["frozen_inputs"],
        "protocol": protocol,
        "model": base["model"],
        "oracle": {
            "implementation": "TASK-030 evaluate_three_layer_oracle",
            "oracle_order": base["protocol"]["oracle_order"],
            "source_sha256": delegated_sources["task_030_oracle"],
        },
        "delegated_source_sha256": delegated_sources,
        "code_sha256": r2._sha256_file(Path(__file__)),
        "git": base["git"],
    }


def _call_estimates(dataset: Mapping[str, Any]) -> tuple[int, int]:
    return sum(len(case["expected"]["required_tools"]) + 1 for case in dataset["cases"]), len(dataset["cases"]) * 4


def build_preflight(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = build_run_manifest(dataset, model_config)
    expected, maximum = _call_estimates(dataset)
    return {
        "schema_version": "healthvault-agent-e2e-absolute-preflight-v1",
        "task": "TASK-034",
        "status": "API_GATE_PENDING",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_version": dataset["dataset_version"],
        "dataset_sha256": r2.semantic_dataset_sha256(dataset),
        "code_sha256": r2._sha256_file(Path(__file__)),
        "run_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "model": manifest["model"],
        "estimated_model_calls": expected,
        "maximum_model_calls": maximum,
        "maximum_transport_attempts": maximum * 2,
        "api_calls_made": 0,
        "offline_only": True,
        "api_gate": {
            "requires": ["--execute", "--approved", "--preflight-hash <this hash>"],
            "approval_binding": "TASK-032 R2 dataset/fixture/oracle/runtime/Skill/model/prompt/code hashes",
        },
        "bindings": {
            "fixture_sha256": manifest["frozen_inputs"]["fixture_sha256"],
            "corpus_sha256": manifest["frozen_inputs"]["corpus_sha256"],
            "used_case_registry_sha256": manifest["used_case_registry_sha256"],
            "active_skill_sha256": manifest["runtime"]["active_skill_sha256"],
            "final_rag_gate_sha256": manifest["runtime"]["final_rag_gate_sha256"],
            "structured_final_parser_sha256": manifest["protocol"]["parser_sha256"],
            "canonical_arm_sha256": sha256_bytes(canonical_json_bytes(manifest["arm"])),
            "console_output_sha256": sha256_bytes(canonical_json_bytes(manifest["protocol"]["console_output"])),
            "task_032_runner_sha256": manifest["delegated_source_sha256"]["task_032_runner"],
            "task_030_oracle_sha256": manifest["delegated_source_sha256"]["task_030_oracle"],
        },
        "git": manifest["git"],
    }


def preflight_sha256(preflight: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes({key: value for key, value in preflight.items() if key != "created_at_utc"}))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def write_preflight_outputs(dataset: Mapping[str, Any], model_config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    configure_utf8_console_output()
    manifest, preflight = build_run_manifest(dataset, model_config), build_preflight(dataset, model_config)
    _atomic_json(MANIFEST_PATH, manifest)
    _atomic_json(PREFLIGHT_PATH, preflight)
    return manifest, preflight


def assert_api_gate(*, preflight: Mapping[str, Any], supplied_hash: str | None, approved: bool, api_key: str | None) -> None:
    if not approved or supplied_hash != preflight_sha256(preflight) or not api_key:
        raise ApiGateError("API Gate closed: require explicit current-hash approval and configured key")


def summarize(dataset: Mapping[str, Any], results: list[Mapping[str, Any]], *, api_calls: int | None = None) -> dict[str, Any]:
    if len(results) != 20 or {item["case_id"] for item in results} != {case["case_id"] for case in dataset["cases"]}:
        raise ValueError("TASK-034 requires exactly one result for every frozen R2 case")
    refusal_ids = {
        case["case_id"] for case in dataset["cases"] if case["expected"]["refusal_required"]
    }
    refusal = [item for item in results if item["case_id"] in refusal_ids]
    latencies = [float(item["attempt"]["latency_ms"]) for item in results]
    input_tokens = sum(int(item["attempt"]["input_tokens"]) for item in results)
    output_tokens = sum(int(item["attempt"]["output_tokens"]) for item in results)
    total_tokens = sum(int(item["attempt"]["total_tokens"]) for item in results)
    actual_api_calls = sum(int(item["attempt"]["api_calls"]) for item in results) if api_calls is None else api_calls
    return {
        "schema_version": "healthvault-agent-e2e-absolute-summary-v1",
        "task": "TASK-034",
        "arm_id": ARM_ID,
        "case_count": len(results),
        "refusal_case_count": len(refusal),
        "task_success_rate": sum(item["passed"] for item in results) / len(results),
        "correct_tool_call_rate": sum(item["oracle"]["behavior"]["passed"] for item in results) / len(results),
        "citation_correctness_rate": sum(item["oracle"]["response"]["citation_correct"] for item in results) / len(results),
        "refusal_correctness_rate": sum(item["oracle"]["response"]["refusal_correct"] for item in refusal) / len(refusal),
        "hard_failure_rate": sum(item["hard_failure"] for item in results) / len(results),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "average_tokens_per_attempt": total_tokens / len(results),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "p95_latency_ms": r2.v1._percentile_95(latencies),
        "api_calls": actual_api_calls,
    }


async def execute_benchmark(dataset: Mapping[str, Any], *, model_config: Mapping[str, Any], api_key: str, run_root: Path, run_config_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from healthvault.evaluation.skill.benchmark.task_022 import CallLedger, ResumeStore, UrllibJsonTransport

    ledger = CallLedger(run_root / "transport-ledger.jsonl", run_config_sha256=run_config_sha256, max_transport_attempts=160)
    store = ResumeStore(run_root / "resume.jsonl", run_config_sha256)
    transport = UrllibJsonTransport()
    results: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        previous = next((item for item in store.records if item.get("case_id") == case["case_id"]), None)
        if previous is not None and previous.get("complete") is True:
            results.append(dict(previous))
            continue
        result = await run_e2e_case(case, model_config=model_config, transport=transport, ledger=ledger, runtime_root=run_root, api_key=api_key)
        store.append(dict(result) | {"complete": True})
        results.append(result)
    return results, summarize(dataset, results, api_calls=len(ledger.records))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TASK-034 canonical R2 absolute benchmark (offline by default)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--preflight-hash")
    args = parser.parse_args(argv)
    configure_utf8_console_output()
    dataset = load_dataset()
    model = r2.v1.resolve_model_config(dict(os.environ))
    manifest, preflight = write_preflight_outputs(dataset, model)
    digest = preflight_sha256(preflight)
    report = {"mode": "offline_preflight" if not args.execute else "api_gate_checked", "preflight_hash": digest, "model": manifest["model"], "estimated_model_calls": preflight["estimated_model_calls"], "maximum_model_calls": preflight["maximum_model_calls"], "api_calls_made": 0, "dataset_sha256": preflight["dataset_sha256"], "code_sha256": preflight["code_sha256"]}
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    api_key = str(os.environ.get(str(model["key_environment"]), "") or "").strip()
    assert_api_gate(preflight=preflight, supplied_hash=args.preflight_hash, approved=args.approved, api_key=api_key)
    run_root = RUNTIME_OUTPUT_ROOT / digest
    run_root.mkdir(parents=True, exist_ok=True)
    results, summary = asyncio.run(execute_benchmark(dataset, model_config=model, api_key=api_key, run_root=run_root, run_config_sha256=digest))
    _atomic_json(run_root / "task-034-case-matrix.json", {"results": results})
    _atomic_json(run_root / "task-034-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
