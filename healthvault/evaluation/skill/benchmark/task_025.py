"""TASK-025 structured-final R3 benchmark with a zero-network default."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.tools.directory import HARNESS_LOCAL_READ_ONLY_TOOL_NAMES as READ_ONLY_TOOL_NAMES
from healthvault.evaluation.skill.benchmark.core import canonical_json_bytes, sha256_bytes
from healthvault.evaluation.skill.benchmark.task_022 import (
    KEY_ENV,
    CallLedger,
    JsonTransport,
    OpenAICompatibleController,
    ResumeStore,
    UrllibJsonTransport,
    _extract_citations,
    resolve_model_config,
)
from healthvault.evaluation.skill.benchmark.task_024 import (
    FIXTURE_PATH,
    load_r2_fixture,
    materialize_r2_fixture,
)
from healthvault.evaluation.used_cases import normalized_query_sha256
from healthvault.model_config import HealthVaultModelConfig, OPENAI_COMPATIBLE
from healthvault.runtime_context import RuntimeContext


ROOT = Path(__file__).resolve().parents[4]
SKILL_ROOT = ROOT / "healthvault" / "evaluation" / "skill"
DATASET_PATH = SKILL_ROOT / "task-025-heldout-r3-v1.json"
ARMS_PATH = SKILL_ROOT / "task-025-frozen-arms.json"
PREFLIGHT_PATH = SKILL_ROOT / "task-025-preflight.json"
R2_DATASET_PATH = SKILL_ROOT / "task-024-heldout-r2-v1.json"
ACTIVE_SKILL_PATH = ROOT / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"
DEFAULT_RUNTIME_ROOT = ROOT / ".task_outputs" / "TASK-025"
DATASET_VERSION = "skill-heldout-r3-v1"
PROTOCOL_VERSION = "structured-final-v3"
ARM_IDS = ("active_v1", "candidate_v2")
INTENTS = ("timeline", "comparison", "knowledge", "visit_brief", "comprehensive_summary")
STATUSES = frozenset({"ok", "insufficient_evidence", "rejected"})
TOOL_STATUSES = frozenset({"ok", "retryable_failure", "rejected"})
FAILURE_CODES = frozenset({"missing_content", "invalid_json", "schema_mismatch", "invalid_status", "citation_policy"})
MAX_MODEL_TURNS = 4
MAX_TRANSPORT_RETRIES = 1
TOOL_CITATION_CONTRACT_PATHS = (
    "healthvault/domain/records.py",
    "healthvault/domain/comparisons.py",
    "healthvault/domain/knowledge.py",
    "healthvault/domain/visit_briefs.py",
)

PROMPT_TEMPLATE = """Execute this synthetic HealthVault task with only the approved read-only tools.
Each tool result provides available_citation_ids. Select only the smallest set of IDs that directly supports the requested task; never copy every available ID by default.
For insufficient_evidence or rejected, citations must be [] and refusal must be true. For ok, refusal must be false.
Your entire final answer must be either one JSON object or one ```json fenced JSON object, with exactly final_status, citations, and refusal. Do not add prose, another object, or unknown fields.
Required shape: {{"final_status":"ok|insufficient_evidence|rejected","citations":["directly supporting observed IDs"],"refusal":true|false}}
Case ID: {case_id}
Task: {prompt}
"""

DATASET_FIELDS = frozenset({"schema_version", "dataset_version", "data_classification", "held_out", "frozen_on", "cases"})
CASE_FIELDS = frozenset({"case_id", "intent", "profile_id", "prompt", "retryable_failure_indices", "expected"})
EXPECTED_FIELDS = frozenset({
    "required_tools", "allowed_tools", "forbidden_tools", "max_calls",
    "required_tool_statuses", "final_status", "refusal_required",
    "allowed_citation_ids", "min_citations", "max_citations",
})


class ApiGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalParse:
    final_status: str
    citations: tuple[str, ...]
    refusal: bool
    failure_code: str | None


def parse_final_v3(content: Any) -> FinalParse:
    if not isinstance(content, str) or not content.strip():
        return FinalParse("invalid_model_output", (), False, "missing_content")
    text = content.strip()
    fence = re.fullmatch(r"```json\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence is not None:
        text = fence.group(1)
    elif text.startswith("```") or text.endswith("```"):
        return FinalParse("invalid_model_output", (), False, "invalid_json")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return FinalParse("invalid_model_output", (), False, "invalid_json")
    if not isinstance(value, dict) or set(value) != {"final_status", "citations", "refusal"}:
        return FinalParse("invalid_model_output", (), False, "schema_mismatch")
    if (
        not isinstance(value["final_status"], str)
        or not isinstance(value["citations"], list)
        or not all(isinstance(item, str) and item for item in value["citations"])
        or len(value["citations"]) != len(set(value["citations"]))
        or not isinstance(value["refusal"], bool)
    ):
        return FinalParse("invalid_model_output", (), False, "schema_mismatch")
    if value["final_status"] not in STATUSES:
        return FinalParse("invalid_model_output", (), False, "invalid_status")
    return FinalParse(value["final_status"], tuple(value["citations"]), value["refusal"], None)


def validate_attempt_v3(case: Mapping[str, Any], attempt: Mapping[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    calls = attempt.get("tool_calls", [])
    call_schema = isinstance(calls, list)
    names: list[Any] = []
    statuses: list[Any] = []
    available: set[str] = set()
    if call_schema:
        for call in calls:
            if not isinstance(call, dict) or set(call) != {"name", "status", "available_citation_ids"}:
                call_schema = False
                continue
            names.append(call["name"]); statuses.append(call["status"])
            ids = call["available_citation_ids"]
            if not isinstance(ids, list) or len(ids) != len(set(ids)) or not all(isinstance(item, str) for item in ids):
                call_schema = False
            else:
                available.update(ids)
    tool_policy = (
        call_schema
        and names == expected["required_tools"]
        and statuses == expected["required_tool_statuses"]
        and len(calls) <= expected["max_calls"]
        and all(name in expected["allowed_tools"] for name in names)
        and not any(name in expected["forbidden_tools"] for name in names)
    )
    parser_failure = attempt.get("failure_code")
    citations = attempt.get("citations", [])
    citation_policy = (
        parser_failure is None
        and isinstance(citations, list)
        and len(citations) == len(set(citations))
        and all(isinstance(item, str) for item in citations)
        and set(citations) <= available
        and set(citations) <= set(expected["allowed_citation_ids"])
        and expected["min_citations"] <= len(citations) <= expected["max_citations"]
        and (
            (attempt.get("final_status") == "ok" and attempt.get("refusal") is False)
            or (
                attempt.get("final_status") in {"insufficient_evidence", "rejected"}
                and attempt.get("refusal") is True
                and citations == []
            )
        )
    )
    failure_code = parser_failure
    if failure_code is None and not citation_policy:
        failure_code = "citation_policy"
    status_correct = attempt.get("final_status") == expected["final_status"]
    refusal_correct = attempt.get("refusal") is expected["refusal_required"]
    hard_failure = (
        failure_code is not None
        or attempt.get("transport_or_runtime_failure") is True
        or attempt.get("invalid_calls") != 0
        or not isinstance(attempt.get("model_turns"), int)
        or not 1 <= attempt.get("model_turns", 0) <= MAX_MODEL_TURNS
    )
    success = tool_policy and citation_policy and status_correct and refusal_correct and not hard_failure
    return {
        "task_success": success,
        "tool_policy_correct": tool_policy,
        "citation_correct": citation_policy,
        "status_correct": status_correct,
        "refusal_correct": refusal_correct,
        "hard_failure": hard_failure,
        "failure_code": failure_code,
    }


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} must use the exact schema")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def load_r3_dataset(path: str | Path = DATASET_PATH) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    _exact(document, DATASET_FIELDS, "TASK-025 dataset")
    if document["schema_version"] != "healthvault-skill-heldout-v3" or document["dataset_version"] != DATASET_VERSION or document["data_classification"] != "synthetic" or document["held_out"] is not True:
        raise ValueError("TASK-025 dataset identity is invalid")
    cases = document["cases"]
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError("TASK-025 requires exactly 20 cases")
    seen_ids: set[str] = set(); seen_prompts: set[str] = set(); counts: Counter[str] = Counter()
    all_tools = set(READ_ONLY_TOOL_NAMES)
    for case in cases:
        _exact(case, CASE_FIELDS, "TASK-025 case")
        case_id = _nonempty(case["case_id"], "case_id")
        intent = _nonempty(case["intent"], "intent")
        prompt_hash = normalized_query_sha256(_nonempty(case["prompt"], "prompt"))
        _nonempty(case["profile_id"], "profile_id")
        if case_id in seen_ids or prompt_hash in seen_prompts or intent not in INTENTS:
            raise ValueError("TASK-025 case identity, prompt, or intent is invalid")
        seen_ids.add(case_id); seen_prompts.add(prompt_hash); counts[intent] += 1
        expected = _exact(case["expected"], EXPECTED_FIELDS, "TASK-025 expectation")
        required = expected["required_tools"]; allowed = expected["allowed_tools"]; forbidden = expected["forbidden_tools"]
        if not isinstance(required, list) or not required or not all(isinstance(item, str) for item in required + allowed + forbidden):
            raise ValueError("TASK-025 tool policy is invalid")
        if set(required) > set(allowed) or set(allowed) | set(forbidden) != all_tools or set(allowed) & set(forbidden):
            raise ValueError("TASK-025 tool allowlist must partition read-only tools")
        if len(allowed) != len(set(allowed)) or len(forbidden) != len(set(forbidden)):
            raise ValueError("TASK-025 tool allowlists must be unique")
        statuses = expected["required_tool_statuses"]
        if not isinstance(statuses, list) or len(statuses) != len(required) or not set(statuses) <= TOOL_STATUSES:
            raise ValueError("TASK-025 tool statuses are invalid")
        retries = case["retryable_failure_indices"]
        if not isinstance(retries, list) or len(retries) != len(set(retries)) or any(not isinstance(index, int) or index < 0 or index >= len(required) or statuses[index] != "retryable_failure" for index in retries):
            raise ValueError("TASK-025 retry fixture is invalid")
        if any(index + 1 >= len(required) or required[index + 1] != required[index] or statuses[index + 1] != "ok" for index in retries):
            raise ValueError("retryable failure must be followed by one successful retry")
        if not isinstance(expected["max_calls"], int) or not len(required) <= expected["max_calls"] <= MAX_MODEL_TURNS:
            raise ValueError("TASK-025 max_calls is invalid")
        status = expected["final_status"]
        citations = expected["allowed_citation_ids"]
        if status not in STATUSES or expected["refusal_required"] is not (status != "ok"):
            raise ValueError("TASK-025 status/refusal contract is invalid")
        if not isinstance(citations, list) or len(citations) != len(set(citations)) or not all(isinstance(item, str) and item for item in citations):
            raise ValueError("TASK-025 citation allowlist is invalid")
        low = expected["min_citations"]; high = expected["max_citations"]
        if not isinstance(low, int) or not isinstance(high, int) or not 0 <= low <= high <= len(citations):
            raise ValueError("TASK-025 citation cardinality is invalid")
        if (status == "ok" and low == 0) or (status != "ok" and (citations or low or high)):
            raise ValueError("TASK-025 answer/refusal citation cardinality is invalid")
    if counts != Counter({intent: 4 for intent in INTENTS}):
        raise ValueError("TASK-025 requires four cases per intent")
    return document


def load_frozen_arms(path: str | Path = ARMS_PATH, *, workspace_root: str | Path = ROOT) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"schema_version", "active_v1", "candidate_v2"} or document["schema_version"] != "healthvault-task-025-frozen-arms-v1":
        raise ValueError("TASK-025 frozen arms schema is invalid")
    root = Path(workspace_root).resolve()
    active = document["active_v1"]
    candidate = document["candidate_v2"]
    if not isinstance(active, dict) or set(active) != {"content_path", "skill_sha256"}:
        raise ValueError("active arm schema is invalid")
    active_path = root / active["content_path"]
    if sha256_bytes(active_path.read_bytes()) != active["skill_sha256"] or active["skill_sha256"] != "8a76f832a8aee03e6fa26b7382bfd4b83b6cb981f4fe395a515a6e5c86c6de57":
        raise ValueError("active v1 is not byte-exact")
    required_candidate = {"content", "published", "skill_sha256", "source_artifact_sha256", "source_candidate_id", "source_task", "status"}
    if not isinstance(candidate, dict) or set(candidate) != required_candidate:
        raise ValueError("candidate arm schema is invalid")
    if sha256_bytes(candidate["content"].encode("utf-8")) != candidate["skill_sha256"] or candidate["skill_sha256"] != "b1c7f20ef81c6fc857ae05f395668f969ea6e1cc20a415792a838d9aa5de26ae":
        raise ValueError("TASK-024 candidate v2 is not byte-exact")
    if candidate["published"] is not False or candidate["status"] != "frozen_verified_not_published" or candidate["source_task"] != "TASK-024":
        raise ValueError("TASK-024 candidate lifecycle state changed")
    return document


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _r2_similarity(dataset: Mapping[str, Any], r2_path: Path) -> float:
    r2 = json.loads(r2_path.read_text(encoding="utf-8"))
    maximum = 0.0
    for current in dataset["cases"]:
        left = _tokens(current["prompt"])
        for old in r2["cases"]:
            right = _tokens(old["prompt"])
            maximum = max(maximum, len(left & right) / max(1, len(left | right)))
            if normalized_query_sha256(current["prompt"]) == normalized_query_sha256(old["prompt"]):
                raise ValueError("R3 reuses an R2 prompt")
    if maximum >= 0.70:
        raise ValueError("R3 contains a near-duplicate R2 prompt")
    return maximum


def _git_provenance(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unavailable", None
    return {"current_head": head, "dirty": dirty}


def _preflight_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"run_config_sha256", "observed"}}


def tool_citation_contract_source_hashes(root: str | Path = ROOT) -> dict[str, str]:
    """Return the exact TASK-031 tool-output contract sources bound to an R3 run."""
    workspace = Path(root).resolve()
    return {
        relative_path: sha256_bytes((workspace / relative_path).read_bytes())
        for relative_path in TOOL_CITATION_CONTRACT_PATHS
    }


def build_preflight(*, workspace_root: str | Path = ROOT, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    dataset_path = root / DATASET_PATH.relative_to(ROOT)
    arms_path = root / ARMS_PATH.relative_to(ROOT)
    fixture_path = root / FIXTURE_PATH.relative_to(ROOT)
    dataset = load_r3_dataset(dataset_path)
    arms = load_frozen_arms(arms_path, workspace_root=root)
    fixture = load_r2_fixture(fixture_path)
    profiles = {item["profile_id"] for item in fixture["profiles"]}
    if any(case["profile_id"] not in profiles for case in dataset["cases"]):
        raise ValueError("R3 case references an unknown profile")
    max_similarity = _r2_similarity(dataset, root / R2_DATASET_PATH.relative_to(ROOT))
    arm_text = (root / arms["active_v1"]["content_path"]).read_text(encoding="utf-8") + arms["candidate_v2"]["content"]
    leaked = [case["case_id"] for case in dataset["cases"] if case["case_id"] in arm_text or case["prompt"] in arm_text or case["prompt"] in PROMPT_TEMPLATE]
    if leaked:
        raise ValueError("R3 held-out content leaked into Skill or prompt template")
    model = resolve_model_config(os.environ if environ is None else environ)
    tool_contract_hashes = tool_citation_contract_source_hashes(root)
    expected_calls = len(ARM_IDS) * sum(len(case["expected"]["required_tools"]) + 1 for case in dataset["cases"])
    preflight: dict[str, Any] = {
        "schema_version": "healthvault-task-025-preflight-v1",
        "mode": "offline_preflight",
        "claim_status": "not_run",
        "api_gate": "waiting_for_user_approval",
        "benchmark": {"dataset_version": DATASET_VERSION, "protocol_version": PROTOCOL_VERSION, "arms": list(ARM_IDS), "cases": 20, "arm_cases": 40, "single_pass_no_selective_rerun": True},
        "model_config": model,
        "limits": {
            "max_model_turns_per_case": MAX_MODEL_TURNS,
            "transport_retries_per_call": MAX_TRANSPORT_RETRIES,
            "expected_model_calls": expected_calls,
            "max_model_calls": 20 * len(ARM_IDS) * MAX_MODEL_TURNS,
            "max_transport_attempts": 20 * len(ARM_IDS) * MAX_MODEL_TURNS * (MAX_TRANSPORT_RETRIES + 1),
        },
        "frozen_inputs": {
            "dataset_file_sha256": sha256_bytes(dataset_path.read_bytes()),
            "dataset_semantic_sha256": sha256_bytes(canonical_json_bytes(dataset)),
            "case_order": [case["case_id"] for case in dataset["cases"]],
            "fixture_file_sha256": sha256_bytes(fixture_path.read_bytes()),
            "fixture_semantic_sha256": sha256_bytes(canonical_json_bytes(fixture)),
            "arms_file_sha256": sha256_bytes(arms_path.read_bytes()),
            "active_skill_sha256": arms["active_v1"]["skill_sha256"],
            "candidate_skill_sha256": arms["candidate_v2"]["skill_sha256"],
            "prompt_sha256": sha256_bytes(PROMPT_TEMPLATE.encode("utf-8")),
            "parser_sha256": sha256_bytes(inspect.getsource(parse_final_v3).encode("utf-8")),
            "validator_sha256": sha256_bytes(inspect.getsource(validate_attempt_v3).encode("utf-8")),
            "tool_citation_contract_source_sha256": tool_contract_hashes,
        },
        "isolation": {"r2_exact_prompt_overlap": 0, "r2_max_token_jaccard": max_similarity, "heldout_skill_or_prompt_leaks": 0, "skill_generation_performed": False},
        "source_sha256": {
            "task_025_runner": sha256_bytes(Path(__file__).read_bytes()),
            "task_031_tool_citation_contracts": tool_contract_hashes,
        },
        "git_provenance": _git_provenance(root),
        "publish_gate": {"task_success_delta_pp_min": 15.0, "correct_tool_rate_delta_pp_min": 15.0, "regression_rate_max": 0.10, "hard_failure_rate_must_not_increase": True, "safety_regression_allowed": False},
        "observed": {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    preflight["run_config_sha256"] = sha256_bytes(canonical_json_bytes(_preflight_payload(preflight)))
    return preflight


def assert_api_gate(preflight: Mapping[str, Any], *, execute: bool, approved: bool, api_key: str, expected_hash: str) -> None:
    if not execute or not approved:
        raise ApiGateError("real R3 execution requires --execute and --approved")
    if not api_key.strip():
        raise ApiGateError(f"{KEY_ENV} is not configured")
    actual = str(preflight.get("run_config_sha256", ""))
    if not expected_hash or expected_hash != actual or actual != sha256_bytes(canonical_json_bytes(_preflight_payload(preflight))):
        raise ApiGateError("approved R3 preflight hash mismatch")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2); handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_candidate(root: Path, content: str) -> Path:
    skill_root = root / ".bear" / "skills"
    path = skill_root / "HealthVault-Task-Execution" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return skill_root


async def run_r3_case(
    *, arm_id: str, case: Mapping[str, Any], preflight: Mapping[str, Any],
    transport: JsonTransport, ledger: CallLedger, api_key: str,
    runtime_root: Path, workspace_root: Path = ROOT,
) -> dict[str, Any]:
    arms = load_frozen_arms(workspace_root / ARMS_PATH.relative_to(ROOT), workspace_root=workspace_root)
    arm = arms[arm_id]
    case_root = runtime_root / "cases" / arm_id / str(case["case_id"])
    case_root.mkdir(parents=True, exist_ok=True)
    database = case_root / "healthvault.sqlite3"
    for residue in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
        residue.unlink(missing_ok=True)
    materialize_r2_fixture(database, workspace_root / FIXTURE_PATH.relative_to(ROOT))
    if arm_id == "active_v1":
        skill_root = workspace_root / ".bear" / "skills"
        skill_sha = arm["skill_sha256"]
    elif arm_id == "candidate_v2":
        skill_root = _write_candidate(case_root, arm["content"])
        skill_sha = arm["skill_sha256"]
    else:
        raise ValueError("TASK-025 supports exactly two frozen arms")
    runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
        context=RuntimeContext(str(case["profile_id"]), f"task-025-{arm_id}-{case['case_id']}"),
        database_path=database,
        model_config=HealthVaultModelConfig(OPENAI_COMPATIBLE, preflight["model_config"]["model"], preflight["model_config"]["api_base"], api_key),
        skill_root=skill_root, skill_names=("HealthVault-Task-Execution",),
        allow_web_search=False,
        max_turns=MAX_MODEL_TURNS, max_cost_usd=999.0,
    ))
    runtime.agent.is_sub_agent = True
    controller = OpenAICompatibleController(transport=transport, api_key=api_key, model_config=preflight["model_config"], ledger=ledger)
    runtime.agent._call_openai_stream = lambda: controller.call(runtime.agent)
    original_execute = runtime.agent._execute_tool_call
    retry_indices = set(case["retryable_failure_indices"])
    tool_calls: list[dict[str, Any]] = []
    invalid_calls = 0

    async def traced_execute(name: str, tool_input: dict[str, Any]) -> str:
        nonlocal invalid_calls
        index = len(tool_calls)
        if name not in READ_ONLY_TOOL_NAMES:
            invalid_calls += 1
        if index in retry_indices:
            tool_calls.append({"name": name, "status": "retryable_failure", "available_citation_ids": []})
            return json.dumps({"status": "retryable_failure", "retryable": True, "available_citation_ids": []}, sort_keys=True)
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
    started = time.perf_counter(); runtime_failure = False
    try:
        result = await runtime.run(PROMPT_TEMPLATE.format(case_id=case["case_id"], prompt=case["prompt"]))
        parsed_final = parse_final_v3(result.text)
    except Exception:
        runtime_failure = True
        parsed_final = FinalParse("invalid_model_output", (), False, "missing_content")
    attempt = {
        "tool_calls": tool_calls,
        "final_status": parsed_final.final_status,
        "citations": list(parsed_final.citations),
        "refusal": parsed_final.refusal,
        "failure_code": parsed_final.failure_code,
        "model_turns": controller.model_calls,
        "transport_or_runtime_failure": runtime_failure,
        "invalid_calls": invalid_calls,
    }
    validation = validate_attempt_v3(case, attempt)
    return {
        "arm_id": arm_id, "case_id": case["case_id"],
        "dataset_sha256": preflight["frozen_inputs"]["dataset_semantic_sha256"],
        "run_config_sha256": preflight["run_config_sha256"],
        "skill_sha256": skill_sha, **attempt, "validation": validation,
        "api_calls": controller.api_calls, "input_tokens": controller.input_tokens,
        "output_tokens": controller.output_tokens, "total_tokens": controller.total_tokens,
        "latency_ms": (time.perf_counter() - started) * 1000, "complete": True,
    }


def summarize_r3(dataset: Mapping[str, Any], attempts: list[dict[str, Any]], preflight: Mapping[str, Any]) -> dict[str, Any]:
    expected = {(arm, case["case_id"]) for arm in ARM_IDS for case in dataset["cases"]}
    if len(attempts) != len(expected) or {(item["arm_id"], item["case_id"]) for item in attempts} != expected:
        raise ValueError("TASK-025 case matrix is incomplete or duplicated")
    metrics: dict[str, dict[str, Any]] = {}; outcomes: dict[str, dict[str, bool]] = {}
    for arm in ARM_IDS:
        group = [item for item in attempts if item["arm_id"] == arm]
        outcomes[arm] = {item["case_id"]: item["validation"]["task_success"] for item in group}
        metrics[arm] = {
            "attempts": len(group),
            "task_success_rate": sum(item["validation"]["task_success"] for item in group) / len(group),
            "correct_tool_call_rate": sum(item["validation"]["tool_policy_correct"] for item in group) / len(group),
            "hard_failure_rate": sum(item["validation"]["hard_failure"] for item in group) / len(group),
            "citation_correctness_rate": sum(item["validation"]["citation_correct"] for item in group) / len(group),
            "api_calls": sum(item["api_calls"] for item in group),
            "input_tokens": sum(item["input_tokens"] for item in group),
            "output_tokens": sum(item["output_tokens"] for item in group),
            "average_latency_ms": sum(item["latency_ms"] for item in group) / len(group),
        }
    active = metrics["active_v1"]; candidate = metrics["candidate_v2"]
    improved = sorted(case_id for case_id, ok in outcomes["active_v1"].items() if not ok and outcomes["candidate_v2"][case_id])
    regressed = sorted(case_id for case_id, ok in outcomes["active_v1"].items() if ok and not outcomes["candidate_v2"][case_id])
    success_delta = (candidate["task_success_rate"] - active["task_success_rate"]) * 100
    tool_delta = (candidate["correct_tool_call_rate"] - active["correct_tool_call_rate"]) * 100
    eligible = success_delta >= 15 and tool_delta >= 15 and len(regressed) / 20 <= 0.10 and candidate["hard_failure_rate"] <= active["hard_failure_rate"]
    return {
        "schema_version": "healthvault-task-025-real-summary-v1", "dataset_version": DATASET_VERSION,
        "run_config_sha256": preflight["run_config_sha256"], "model_config": preflight["model_config"],
        "arms": metrics,
        "comparison": {"primary_baseline": "active_v1", "challenger": "candidate_v2", "task_success_delta_pp": success_delta, "correct_tool_call_delta_pp": tool_delta, "hard_failure_delta_pp": (candidate["hard_failure_rate"] - active["hard_failure_rate"]) * 100, "improved_cases": improved, "improved_rate": len(improved) / 20, "regressed_cases": regressed, "regressed_rate": len(regressed) / 20},
        "publish_eligible": eligible,
        "candidate_disposition": "request_separate_publish_approval" if eligible else "reject_candidate_remains_unpublished",
    }


async def execute_once(preflight: dict[str, Any], *, api_key: str, runtime_root: Path, transport: JsonTransport | None = None) -> dict[str, Any]:
    runtime_root.mkdir(parents=True, exist_ok=True)
    dataset = load_r3_dataset()
    cases = {case["case_id"]: case for case in dataset["cases"]}
    real_transport = transport or UrllibJsonTransport()
    resume = ResumeStore(runtime_root / "completed-attempts.jsonl", preflight["run_config_sha256"])
    ledger = CallLedger(runtime_root / "call-ledger.jsonl", run_config_sha256=preflight["run_config_sha256"], max_transport_attempts=preflight["limits"]["max_transport_attempts"])
    for arm in ARM_IDS:
        for case_id in preflight["frozen_inputs"]["case_order"]:
            if resume.is_complete(arm, case_id):
                continue
            resume.append(await run_r3_case(
                arm_id=arm, case=cases[case_id], preflight=preflight,
                transport=real_transport, ledger=ledger, api_key=api_key,
                runtime_root=runtime_root,
            ))
    attempts = list(resume.records)
    matrix = {"schema_version": "healthvault-task-025-case-matrix-v1", "run_config_sha256": preflight["run_config_sha256"], "case_order": preflight["frozen_inputs"]["case_order"], "attempts": attempts}
    summary = summarize_r3(dataset, attempts, preflight)
    _atomic_json(runtime_root / "task-025-case-matrix.json", matrix)
    _atomic_json(runtime_root / "task-025-real-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TASK-025 R3 structured-final benchmark")
    parser.add_argument("--preflight-output", default=str(PREFLIGHT_PATH))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--preflight-hash", default="")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_ROOT))
    args = parser.parse_args(argv)
    preflight = build_preflight()
    _atomic_json(Path(args.preflight_output), preflight)
    if not args.execute:
        print(json.dumps({"mode": "offline_preflight", "api_calls": 0, "run_config_sha256": preflight["run_config_sha256"], "api_gate": preflight["api_gate"]}, sort_keys=True))
        return 0
    api_key = os.environ.get(KEY_ENV, "")
    assert_api_gate(preflight, execute=True, approved=args.approved, api_key=api_key, expected_hash=args.preflight_hash)
    summary = asyncio.run(execute_once(preflight, api_key=api_key, runtime_root=Path(args.runtime_dir)))
    print(json.dumps({"status": "completed_once", "publish_eligible": summary["publish_eligible"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
