"""Current native Agent regression on unchanged R5 cases, without publishing Skills."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from unittest.mock import patch

from healthvault.evaluation.current_delivery import ROOT, digest, implementation_binding, save
from healthvault.evaluation.current_skill_run import CurrentTransport, STRUCTURED_OUTPUT
from healthvault.evaluation.e2e import task_043 as legacy
from healthvault.evaluation.skill.benchmark.task_025 import FinalParse
from healthvault.evaluation.web_rag_run import ModelHTTPClient, value_hash
from healthvault.tools.directory import HARNESS_LOCAL_READ_ONLY_TOOL_NAMES
from healthvault.web.model_runtime import configured_web_model
from healthvault.rag.releases import active_corpus_root, configured_release_root
from healthvault.rag.retrieval_config import selection
from healthvault.rag.vector_cache import model_fingerprint

PROTOCOL = "healthvault-current-e2e-2026-10-04-v2"
MAX_CALLS = 80
APPROVAL_ROOT = ROOT / ".task_outputs/task-075-e2e-approvals"


def load_current_dataset(corpus_path: Path) -> dict:
    # Re-derive source membership from the frozen corpus; keep all three Oracles.
    with patch.object(legacy, "CORPUS_PATH", corpus_path):
        dataset = legacy.load_dataset()
    sources = {chunk["source_id"] for chunk in json.loads(corpus_path.read_text(encoding="utf-8"))["chunks"]}
    required_sources = {"R5E-010": "medlineplus-ferritin-blood-test",
                        "R5E-011": "medlineplus-tsh-test", "R5E-012": "medlineplus-tsh-test"}
    adjustments = []
    for case in dataset["cases"]:
        source = required_sources.get(case["case_id"])
        if not source or source in sources:
            continue
        if case["scenario"]["rule"] == "supported":
            unavailable = {**case, "scenario": {**case["scenario"], "rule": "unsupported"}}
            case["expected"] = legacy.derive_expected(unavailable, legacy.load_fixture())
        else:
            case["expected"]["required_tool_result_statuses"] = ["empty"]
        adjustments.append({"case_id": case["case_id"], "missing_source_id": source,
                            "reason": "The frozen current corpus lacks this official source; no model result was used to change the expectation."})
    return {**dataset, "dataset_version": "agent-e2e-r5-hybrid-2026-10-04-v2",
            "source_scope_adjustments": adjustments}


def prepare_run(output: Path, *, api_base: str) -> dict:
    endpoint = urlparse(api_base)
    if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.query or endpoint.fragment:
        raise ValueError("An HTTPS endpoint without embedded credentials is required.")
    corpus = active_corpus_root()
    settings = selection(corpus, configured_release_root())
    if settings["knowledge_mode"] != "hybrid" or not settings["hybrid_index_path"] or not settings["hybrid_model_path"]:
        raise ValueError("The demo hybrid index and model must be configured before E2E evaluation.")
    corpus_path = corpus / "chunks/hybrid_v1.json"
    index_path, model_path = Path(settings["hybrid_index_path"]), Path(settings["hybrid_model_path"])
    dataset = load_current_dataset(corpus_path)
    fixture = legacy.load_fixture()
    matrix = legacy.build_case_matrix(dataset, fixture)
    output.mkdir(parents=True, exist_ok=False)
    paths = [legacy.DATASET_PATH, legacy.FIXTURE_PATH, corpus_path,
             corpus / "aliases/aliases.json",
             legacy.ACTIVE_SKILL_PATH / "SKILL.md", legacy.FORMAL_SPLIT_PATH, legacy.DEV_SPLIT_PATH]
    paths.extend(path for _, _, _, path in legacy.LEAKAGE_SOURCES)
    paths.extend([legacy.DEV_SAFETY_MATRIX_PATH, legacy.DEV_BADCASES_PATH,
                  legacy.DEV_CASES_PATH, legacy.DEV_TEST_SOURCE_PATH])
    save(output / "case-matrix.json", matrix)
    old = {case["case_id"]: case for case in legacy.load_dataset()["cases"]}
    save(output / "source-compatibility.json", {"source_dataset": "agent-e2e-r5-v1", "new_dataset": dataset["dataset_version"],
         "rule": "Citation membership and three missing-source expectations derived from frozen corpus metadata, not model answers; prompts/tools/permissions and Oracle logic unchanged.",
         "source_scope_adjustments": dataset["source_scope_adjustments"],
         "cases": [{"case_id": case["case_id"], "prompt_unchanged": case["prompt"] == old[case["case_id"]]["prompt"],
                    "old_expected": old[case["case_id"]]["expected"], "current_expected": case["expected"]} for case in dataset["cases"]]})
    paths.extend(index_path / name for name in ("manifest.json", "index.faiss", "chunk_ids.json"))
    plan = {
        "protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": implementation_binding()["sha256"],
        "bindings": {str(path.resolve()): digest(path) for path in paths + [output / "case-matrix.json", output / "source-compatibility.json"]},
        "dataset_semantic_sha256": legacy.semantic_dataset_sha256(dataset),
        "case_order": [case["case_id"] for case in dataset["cases"]],
        "endpoint": api_base.rstrip("/") + "/chat/completions",
        "model_config": {"model": "glm-5.3-flash", "api_base": api_base.rstrip("/"),
                         "thinking": {"type": "enabled"}, "reasoning_effort": "low", "max_output_tokens": 2048},
        "limits": {"maximum_calls": MAX_CALLS, "model_turns_per_attempt": 4, "transport_retries": 0,
                   "input_bytes_per_call": 160_000, "timeout_seconds": 60},
        "tool_names": list(HARNESS_LOCAL_READ_ONLY_TOOL_NAMES),
        "scope": "current_native_agent_single_arm_r5_contract_regression",
        "source_dataset_version": dataset["dataset_version"], "untouched_held_out": False,
        "knowledge_mode": "hybrid", "corpus_path": str(corpus_path.resolve()),
        "hybrid_index_path": str(index_path.resolve()), "hybrid_model_path": str(model_path.resolve()),
        "embedding_model_sha256": model_fingerprint(model_path),
        "structured_output": copy.deepcopy(STRUCTURED_OUTPUT),
        "final_contract": "fixed_JSON; free_text_semantics_evaluated_in_separate_RAG_track",
        "human_review_min_cases": 10, "publishes_skill": False, "approval_required": True, "api_calls": 0,
    }
    plan["sha256"] = value_hash(plan)
    save(output / "preflight.json", plan)
    return plan


def validate_preflight(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan["protocol"] != PROTOCOL or plan["sha256"] != value_hash({key: value for key, value in plan.items() if key != "sha256"}):
        raise ValueError("Preflight integrity failed.")
    if implementation_binding()["sha256"] != plan["implementation_sha256"]:
        raise ValueError("Implementation changed after preflight.")
    if any(digest(Path(name)) != expected for name, expected in plan["bindings"].items()):
        raise ValueError("Bound input changed after preflight.")
    if plan["limits"] != {"maximum_calls": MAX_CALLS, "model_turns_per_attempt": 4, "transport_retries": 0,
                          "input_bytes_per_call": 160_000, "timeout_seconds": 60}:
        raise ValueError("Frozen limits changed.")
    if plan["knowledge_mode"] != "hybrid" or plan["tool_names"] != list(HARNESS_LOCAL_READ_ONLY_TOOL_NAMES) or plan.get("structured_output") != STRUCTURED_OUTPUT:
        raise ValueError("Frozen runtime changed.")
    if plan["embedding_model_sha256"] != model_fingerprint(Path(plan["hybrid_model_path"])):
        raise ValueError("Frozen embedding model changed.")
    dataset = load_current_dataset(Path(plan["corpus_path"]))
    if plan["dataset_semantic_sha256"] != legacy.semantic_dataset_sha256(dataset) or plan["case_order"] != [case["case_id"] for case in dataset["cases"]]:
        raise ValueError("Frozen cases changed.")
    return plan


class E2ETransport(CurrentTransport):
    async def post_json(self, url, headers, payload, timeout):
        if len(self.calls) >= MAX_CALLS:
            raise RuntimeError("Actual E2E call budget exhausted.")
        return await super().post_json(url, headers, payload, timeout)


async def run_case(case: dict, plan: dict, transport: E2ETransport, root: Path, *, api_key: str) -> dict:
    import healthvault.agent_runtime as native
    original_world_state = legacy.world_state
    original_build = native.build_healthvault_agent_runtime
    searches, runtime_metadata = [], {}

    def build_runtime(config):
        runtime = original_build(replace(config, knowledge_path=plan["corpus_path"], knowledge_mode="hybrid",
                                         hybrid_index_path=plan["hybrid_index_path"], hybrid_model_path=plan["hybrid_model_path"]))
        repository = runtime.components.knowledge_repository
        runtime_metadata.update(repository.index_metadata)
        search = repository.search

        def traced_search(query, limit):
            hits = search(query, limit)
            searches.append({"query": query, "limit": limit, **copy.deepcopy(repository.last_search_metadata or {}),
                             "returned_citation_ids": [hit.document.citation_id for hit in hits]})
            return hits

        repository.search = traced_search
        return runtime

    def closed_world_state(path):
        # SQLite's transaction context commits but does not close the handle.
        connections = []

        def connect(*args, **kwargs):
            connection = sqlite3.connect(*args, **kwargs)
            connections.append(connection)
            return connection

        try:
            with patch.object(legacy, "sqlite3", SimpleNamespace(connect=connect, Row=sqlite3.Row)):
                return original_world_state(path)
        finally:
            for connection in connections:
                connection.close()

    # The older runner cleans its own fixture; check that absolute target before deletion.
    def checked_remove(path, *args, **kwargs):
        target = Path(path).resolve()
        if not target.is_relative_to(root.resolve()) or target == root.resolve():
            raise ValueError("Fixture cleanup escaped the intended run directory.")
        return shutil.rmtree(path, *args, **kwargs)

    with patch.object(legacy, "world_state", closed_world_state), \
         patch.object(legacy, "shutil", SimpleNamespace(rmtree=checked_remove)), \
         patch.object(native, "build_healthvault_agent_runtime", build_runtime):
        result = await legacy.run_e2e_case(case, model_config=plan["model_config"], transport=transport,
                     ledger=None, runtime_root=root, api_key=api_key)
    group = [row for row in transport.calls if row["case_id"] == case["case_id"] and row['arm_id'] == transport.arm_id]
    terminal = next((row["raw_final"] for row in reversed(group) if "raw_final" in row), None)
    attempt = result["attempt"]
    attempt["retrieval"] = {"requested_mode": "hybrid", "runtime_metadata": runtime_metadata, "searches": searches}
    if terminal is not None:
        raw = FinalParse(terminal["final_status"], tuple(terminal["citations"]),
                         terminal["refusal"], terminal["failure_code"])
        attempt["parser_error"] = raw.failure_code or attempt["parser_error"]
        result["oracle"] = legacy.evaluate_three_layer_oracle(case, {**attempt, "final": raw})
        result.update(passed=result["oracle"]["passed"], hard_failure=result["oracle"]["hard_failure"])
        attempt["final"] = {key: terminal[key] for key in ("final_status", "citations", "refusal")}
    attempt["raw_final_validation"] = "failed" if terminal and terminal["failure_code"] else "passed" if terminal else "application_terminal_or_runtime_failure"
    known = [row["usage"] for row in group if row.get("usage") and all(name in row["usage"] for name in ("prompt_tokens", "completion_tokens"))]
    attempt.update(api_calls=sum(row["transport_kind"] == "real" for row in group),
                   input_tokens=sum(item["prompt_tokens"] for item in known),
                   output_tokens=sum(item["completion_tokens"] for item in known),
                   total_tokens=sum(item["prompt_tokens"] + item["completion_tokens"] for item in known),
                   usage_unknown_calls=len(group) - len(known))
    # A lexical fallback is useful operationally, but cannot pass a hybrid evaluation.
    hybrid_ok = runtime_metadata.get("retrieval_mode") == "hybrid" and all(
        search.get("retrieval_mode") == "hybrid" and search.get("dense_participated") is True
        and not search.get("dense_fallback_reason") for search in searches)
    attempt["hybrid_path_passed"] = hybrid_ok
    attempt["failure_categories"] = sorted({row["failure_category"] for row in group if "failure_category" in row})
    if not hybrid_ok:
        result["passed"] = False
        result["oracle"]["passed"] = False
        result["oracle"]["hybrid_path_passed"] = False
    return result


async def execute_run(path: Path, output: Path, *, approved_sha256: str, reviewed_sha256: str,
                      client=None, approval_root: Path = APPROVAL_ROOT) -> dict:
    plan = validate_preflight(path)
    if approved_sha256 != plan["sha256"] or reviewed_sha256 != plan["sha256"]:
        raise ValueError("Current independent review and explicit paid approval are required.")
    if client is not None and client.is_real:
        raise ValueError("Injected clients are offline-only.")
    config = configured_web_model() if client is None else None
    if config and (config.api_base.rstrip("/") + "/chat/completions" != plan["endpoint"] or config.model != plan["model_config"]["model"]):
        raise ValueError("Configured provider changed after approval.")
    if client is None and not output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Real run artifacts must stay within the project.")
    if output.exists():
        raise FileExistsError("Existing results must not be overwritten.")
    save(approval_root / (plan["sha256"] + ".json"), {"preflight_sha256": plan["sha256"], "output": str(output.resolve())})
    output.mkdir(parents=True, exist_ok=False)
    transport = E2ETransport(plan, output, client or ModelHTTPClient(plan["endpoint"], config.api_key))
    transport.arm_id = legacy.ARM_ID
    cases = {case["case_id"]: case for case in load_current_dataset(Path(plan["corpus_path"]))["cases"]}
    results, status = [], "completed"
    for identifier in plan["case_order"]:
        transport.case_id = identifier
        result = await run_case(cases[identifier], plan, transport, output / "cases",
                                api_key="fake" if config is None else config.api_key)
        results.append(result)
        save(output / (identifier + ".json"), result)
        print(f"E2E {len(results)}/{len(cases)}: {identifier}, passed={result['passed']}", flush=True)
        if transport.failed:
            status = "stopped_on_error"
            break
    api_calls = sum(row["transport_kind"] == "real" for row in transport.calls)
    summary = legacy.summarize({"dataset_version": plan["source_dataset_version"], "cases": list(cases.values())}, results, api_calls=api_calls) if status == "completed" else None
    if summary:
        summary.update(schema_version=PROTOCOL, task="TASK-078", untouched_held_out=False,
                       usage_unknown_calls=sum(item["attempt"]["usage_unknown_calls"] for item in results))
    value = {"protocol": PROTOCOL, "preflight_sha256": plan["sha256"], "status": status,
             "mode": "real_model_regression" if transport.is_real else "offline_contract_check",
             "untouched_held_out": False, "api_calls": api_calls, "transport_attempts": len(transport.calls),
             "completed_cases": len(results), "planned_cases": len(cases), "summary": summary,
             "results": results, "human_review_status": "pending", "publishes_skill": False}
    save(output / "results.json", value)
    selected = set(sorted(cases, key=lambda identifier: value_hash([plan["sha256"], identifier]))[:10])
    selected.update(item["case_id"] for item in results if not item["passed"])
    save(output / "human-review.json", {"status": "pending", "preflight_sha256": plan["sha256"], "cases": [
        {"case_id": item["case_id"], "task": cases[item["case_id"]]["prompt"],
         "expected": cases[item["case_id"]]["expected"], "result": item, "human_verdict": None}
        for item in results if item["case_id"] in selected]})
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-sha256")
    parser.add_argument("--reviewed-sha256")
    args = parser.parse_args()
    if args.prepare:
        plan = prepare_run(args.output, api_base=configured_web_model().api_base)
        value = {"status": "prepared", "preflight_sha256": plan["sha256"], "maximum_api_calls": MAX_CALLS, "api_calls": 0}
    else:
        if not args.preflight or not args.approved_sha256 or not args.reviewed_sha256:
            parser.error("Execution requires current independent review and explicit paid approval.")
        result = asyncio.run(execute_run(args.preflight, args.output, approved_sha256=args.approved_sha256, reviewed_sha256=args.reviewed_sha256))
        value = {key: result[key] for key in ("status", "mode", "api_calls", "transport_attempts", "completed_cases")}
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return 0 if value["status"] in {"prepared", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
