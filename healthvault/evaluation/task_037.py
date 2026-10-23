"""TASK-037 Gate A: one candidate only, offline by default."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from healthvault.adapters.bearcode.final_response_normalizer import FinalResponseNormalizer
from healthvault.evaluation.skill.benchmark.core import canonical_json_bytes, sha256_bytes
from healthvault.evolution.signal_extractor import SignalExtractor, TaskObservation
from healthvault.evolution.signal_extractor import generate_candidate_once
from healthvault.evolution.candidates import CANDIDATE_SCHEMA, CandidateGenerator
from healthvault.evolution.verifier import CandidateVerifier
from healthvault.evaluation.used_cases import find_held_out_collisions, normalized_query_sha256
from healthvault.evaluation.e2e import task_030 as e2e_oracle
from healthvault.evaluation.e2e import task_032 as fixture_oracle


ROOT = Path(__file__).resolve().parents[2]
SKILL_R4_PATH = ROOT / "healthvault/evaluation/skill/skill-heldout-r4-v1.json"
E2E_R4_PATH = ROOT / "healthvault/evaluation/e2e/agent-e2e-r4-v1.json"
R4_FIXTURE_PATH = ROOT / "healthvault/evaluation/e2e/agent-e2e-r4-v1-fixture.json"
DEV_TRACE_PATH = ROOT / "healthvault/evaluation/skill/task-037-dev-traces.json"
PREFLIGHT_PATH = ROOT / "healthvault/evaluation/task-037-gate-a-preflight.json"
GATE_B_PREFLIGHT_PATH = ROOT / "healthvault/evaluation/task-037-gate-b-preflight.json"
GATE_C_PREFLIGHT_PATH = ROOT / "healthvault/evaluation/task-037-gate-c-preflight.json"
ACTIVE_SKILL_PATH = ROOT / ".bear/skills/HealthVault-Task-Execution/SKILL.md"
WORKFLOW_DISTILLATION_PATH = ROOT / "healthvault/evolution/workflow_distillation.py"


class ApiGateError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dev_observations() -> tuple[TaskObservation, ...]:
    """Load the frozen, test-isolated dev traces as extractor observations."""
    skill_sha = _sha(ACTIVE_SKILL_PATH)
    document = json.loads(DEV_TRACE_PATH.read_text(encoding="utf-8"))
    if (document.get("schema_version") != 1
            or document.get("role") != "candidate_input_only"
            or not isinstance(document.get("feedback"), str)
            or not document["feedback"].strip()):
        raise ValueError("invalid TASK-037 isolated dev trace artifact")
    observations: list[TaskObservation] = []
    for item in document.get("traces", []):
        if not isinstance(item, Mapping):
            raise ValueError("invalid TASK-037 dev trace")
        required = ("trace_id", "task_id", "intent", "preconditions", "expected_outcome",
                    "validator_status", "contract_checks", "tool_calls")
        if any(key not in item for key in required):
            raise ValueError("incomplete TASK-037 dev trace")
        observations.append(TaskObservation(
            trace={"trace_id": item["trace_id"], "task_id": item["task_id"],
                   "skill": {"sha256": skill_sha}, "result_status": "succeeded",
                   "failure_code": None, "tool_calls": item["tool_calls"],
                   "feedback": document["feedback"]},
            intent=str(item["intent"]), preconditions=tuple(item["preconditions"]),
            expected_outcome=str(item["expected_outcome"]),
            validator_status=str(item["validator_status"]),
            contract_checks=dict(item["contract_checks"]),
        ))
    if len(observations) != 3:
        raise ValueError("TASK-037 requires exactly three isolated dev traces")
    return tuple(observations)


async def extract_verified_dev_cards() -> tuple[dict[str, Any], ...]:
    report = await SignalExtractor().extract(dev_observations())
    cards = [card for card in report.cards if card.verification_strength == "strong"]
    if len(cards) != 1 or cards[0].evidence_count != 3:
        raise ValueError("TASK-037 requires one isolated verified dev ExperienceCard")
    return tuple({"card_id": card.card_id, "trace_ids": list(card.trace_ids), "task_ids": list(card.task_ids), "strength": card.verification_strength, "source": card.verification_source} for card in cards)


def build_gate_a_preflight(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    values = dict(environ or {})
    model = {
        "provider": "zhipu", "protocol": "OpenAI Chat Completions",
        "api_base": str(values.get("HEALTHVAULT_OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")),
        "model": str(values.get("HEALTHVAULT_CANDIDATE_MODEL", "glm-5.3")),
        "key_environment": "HEALTHVAULT_OPENAI_API_KEY", "do_sample": False,
        "temperature": 0.0, "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "high", "max_output_tokens": 4096, "timeout_seconds": 120.0,
    }
    cards = __import__("asyncio").run(extract_verified_dev_cards())
    payload: dict[str, Any] = {
        "schema_version": "healthvault-task-037-gate-a-v1", "task": "TASK-037", "gate": "A",
        "status": "API_GATE_PENDING", "api_calls_made": 0,
        "candidate_model": model,
        "limits": {"candidate_generation_calls": 1, "maximum_model_calls": 1, "maximum_transport_attempts": 2},
        "frozen_inputs": {"active_skill_sha256": _sha(ACTIVE_SKILL_PATH), "normalizer_sha256": sha256_bytes(inspect.getsource(FinalResponseNormalizer).encode("utf-8")), "skill_r4_sha256": _sha(SKILL_R4_PATH), "e2e_r4_sha256": _sha(E2E_R4_PATH), "r4_fixture_sha256": _sha(R4_FIXTURE_PATH), "dev_trace_sha256": _sha(DEV_TRACE_PATH), "verified_dev_cards": cards, "candidate_schema_sha256": sha256_bytes(canonical_json_bytes(CANDIDATE_SCHEMA)), "generator_sha256": sha256_bytes(inspect.getsource(CandidateGenerator).encode("utf-8")), "verifier_sha256": sha256_bytes(inspect.getsource(CandidateVerifier).encode("utf-8")), "candidate_prompt_sha256": _sha(WORKFLOW_DISTILLATION_PATH), "task_runner_sha256": _sha(Path(__file__))},
        "isolation": {"dev_trace_role": "candidate_input_only", "heldout_cases_exposed_to_candidate": False, "candidate_limit": 1},
    }
    payload["preflight_sha256"] = sha256_bytes(canonical_json_bytes({key: value for key, value in payload.items() if key != "preflight_sha256"}))
    return payload


def assert_gate_a(preflight: Mapping[str, Any], *, approved: bool, supplied_hash: str, api_key: str) -> None:
    expected = sha256_bytes(canonical_json_bytes({key: value for key, value in preflight.items() if key != "preflight_sha256"}))
    if not approved or not api_key or supplied_hash != expected or preflight.get("preflight_sha256") != expected:
        raise ApiGateError("TASK-037 Gate A requires explicit approval of the current hash")


async def execute_gate_a_once(preflight: Mapping[str, Any], *, approved: bool, supplied_hash: str, api_key: str, transport: Any, runtime_root: Path) -> dict[str, Any]:
    """Generate, verify, and freeze exactly one candidate after Gate A approval."""
    runtime_root = Path(runtime_root)
    artifact_path = runtime_root / "task-037-frozen-candidate.json"
    # This check intentionally precedes approval and every model/transport path:
    # one Gate-A approval can never create a second candidate artifact.
    if artifact_path.exists():
        raise ApiGateError("TASK-037 candidate already frozen; a second generation is forbidden")
    assert_gate_a(preflight, approved=approved, supplied_hash=supplied_hash, api_key=api_key)
    from healthvault.evolution.workflow_distillation import CandidateModelSideQuery
    report = await SignalExtractor().extract(dev_observations())
    side_query = CandidateModelSideQuery(transport=transport, api_key=api_key, model_config=preflight["candidate_model"], active_skill_content=ACTIVE_SKILL_PATH.read_text(encoding="utf-8"))
    result = await generate_candidate_once(report, generator=CandidateGenerator(runtime_root, side_query), action="evolve", verifier=CandidateVerifier("HealthVault-Task-Execution"), active_content=ACTIVE_SKILL_PATH.read_text(encoding="utf-8"))
    if result.candidate is None or result.generator_calls != 1 or result.verification is None or result.verification.status != "verified":
        raise ApiGateError("candidate generation or verification failed; do not create another candidate")
    artifact = {"preflight_sha256": preflight["preflight_sha256"], "candidate_id": result.candidate.candidate_id, "candidate_sha256": result.verification.content_sha256, "candidate_content": result.candidate.content, "trace_ids": list(result.candidate.trace_ids), "status": "frozen_unpublished"}
    runtime_root.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(canonical_json_bytes(artifact))
    return artifact


def write_gate_a_preflight(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    value = build_gate_a_preflight(environ)
    PREFLIGHT_PATH.write_bytes(canonical_json_bytes(value))
    return value


def r4_collision_check() -> list[dict[str, str]]:
    """Check both fresh R4 sets against each other and the frozen evaluation tracks."""
    sources = [
        ROOT / "healthvault/evaluation/rag/task-026/task-026-human-reviewed-gold.json",
        ROOT / "healthvault/evaluation/skill/task-024-heldout-r2-v1.json",
        ROOT / "healthvault/evaluation/skill/task-025-heldout-r3-v1.json",
        ROOT / "healthvault/evaluation/e2e/agent-e2e-r1-v1.json",
        ROOT / "healthvault/evaluation/e2e/agent-e2e-r2-v1.json",
        ROOT / "healthvault/evaluation/e2e/agent-e2e-r3-v1.json",
        SKILL_R4_PATH, E2E_R4_PATH,
    ]
    entries: list[dict[str, Any]] = []
    for path in sources:
        document = json.loads(path.read_text(encoding="utf-8")); cases = document.get("cases", document.get("items", []))
        for index, case in enumerate(cases):
            if not isinstance(case, Mapping):
                continue
            prompt = str(case.get("prompt", case.get("query", case.get("question", "")))).strip()
            if not prompt:
                continue
            version = str(document.get("dataset_version", path.stem)); case_id = str(case.get("case_id", case.get("id", index)))
            entries.append({"case_id": f"{version}:{case_id}", "case_family_id": f"{version}:{case_id}", "track": "task037", "case_version": version, "split": "held_out", "held_out": True, "query": prompt, "query_sha256": normalized_query_sha256(prompt), "upstream_dataset": str(path.relative_to(ROOT)), "upstream_id": case_id, "status": "frozen", "source": "TASK-037"})
    return find_held_out_collisions(entries)


def load_r4_fixture(path: Path = R4_FIXTURE_PATH) -> dict[str, Any]:
    """Validate the resettable synthetic snapshot used by both new R4 gates."""
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict) or set(fixture) != fixture_oracle.FIXTURE_KEYS:
        raise ValueError("TASK-037 R4 fixture schema is invalid")
    if fixture.get("schema_version") != "healthvault-agent-e2e-fixture-v1" or fixture.get("snapshot_id") != "agent-e2e-r4-fixture-v1" or fixture.get("data_classification") != "synthetic":
        raise ValueError("TASK-037 R4 fixture identity is invalid")
    profiles = fixture.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("TASK-037 R4 fixture profiles are invalid")
    profile_ids: set[str] = set(); report_ids: set[str] = set(); source_ids: set[str] = set(); reports = 0
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) != {"profile_id", "reports"} or not isinstance(profile["profile_id"], str) or profile["profile_id"] in profile_ids or not isinstance(profile["reports"], list):
            raise ValueError("TASK-037 R4 fixture profile is invalid")
        profile_ids.add(profile["profile_id"])
        for report in profile["reports"]:
            if not isinstance(report, dict) or set(report) != {"source_id", "report_id", "report_date", "facility", "indicators"} or report["report_id"] in report_ids or report["source_id"] in source_ids or not report["indicators"]:
                raise ValueError("TASK-037 R4 fixture report is invalid")
            report_ids.add(report["report_id"]); source_ids.add(report["source_id"]); reports += 1
    if fixture.get("report_count") != reports:
        raise ValueError("TASK-037 R4 fixture report count is invalid")
    return fixture


def load_r4_dataset(path: Path) -> dict[str, Any]:
    """Generate deterministic three-layer Oracle rules from fixture/tool policy only."""
    document = json.loads(path.read_text(encoding="utf-8")); fixture = load_r4_fixture()
    if not isinstance(document, dict) or set(document) != fixture_oracle.DATASET_KEYS:
        raise ValueError("TASK-037 R4 dataset schema is invalid")
    version = "skill-heldout-r4-v1" if path == SKILL_R4_PATH else "agent-e2e-r4-v1"
    if document.get("dataset_version") != version or document.get("schema_version") != "healthvault-agent-e2e-dataset-v2" or document.get("data_classification") != "synthetic" or document.get("held_out") is not True or document.get("fixture_path") != str(R4_FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/") or document.get("leakage_sources") != []:
        raise ValueError("TASK-037 R4 dataset identity or fixture binding is invalid")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != 20 or Counter(case.get("category") for case in cases) != fixture_oracle.CATEGORY_COUNTS:
        raise ValueError("TASK-037 R4 requires twenty balanced cases")
    profile_ids = {item["profile_id"] for item in fixture["profiles"]}; result: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != fixture_oracle.CASE_KEYS or case.get("profile_id") not in profile_ids or not isinstance(case.get("prompt"), str) or not case["prompt"].strip() or not isinstance(case.get("scenario"), dict):
            raise ValueError("TASK-037 R4 case is invalid")
        # The existing R2 helper deliberately treats a declared supported query
        # with no local official hit as malformed.  R4 instead derives that
        # terminal state from the same structured corpus: no hit is an
        # insufficient-evidence outcome, not hand-authored Gold.
        oracle_case = dict(case)
        if (case["category"] == "knowledge_refusal"
                and case["scenario"].get("rule") == "supported"
                and not fixture_oracle._knowledge_ids(str(case["scenario"].get("query", "")))):
            oracle_case["scenario"] = dict(case["scenario"]) | {"rule": "unsupported"}
        result.append(dict(case) | {"expected": fixture_oracle.derive_expected(oracle_case, fixture)})
    if len({item["case_id"] for item in result}) != 20 or len({normalized_query_sha256(item["prompt"]) for item in result}) != 20:
        raise ValueError("TASK-037 R4 case IDs/prompts must be unique")
    return dict(document) | {"cases": result}


def evaluate_r4_three_layer_oracle(case: Mapping[str, Any], attempt: Mapping[str, Any]) -> dict[str, Any]:
    """Outcome, tool-behavior, and final-response checks share the frozen R2 Oracle."""
    return e2e_oracle.evaluate_three_layer_oracle(case, attempt)


def _candidate_artifact(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "frozen_unpublished" or not isinstance(value.get("candidate_sha256"), str) or len(value["candidate_sha256"]) != 64 or not isinstance(value.get("candidate_content"), str) or hashlib.sha256(value["candidate_content"].encode("utf-8")).hexdigest() != value["candidate_sha256"]:
        raise ValueError("TASK-037 requires the unique verified Gate-A candidate artifact")
    return value


def _flash_model(environ: Mapping[str, str] | None) -> dict[str, Any]:
    values = dict(environ or {})
    return {"provider": "zhipu", "protocol": "OpenAI Chat Completions", "api_base": str(values.get("HEALTHVAULT_OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")), "model": "glm-5.3-flash", "key_environment": "HEALTHVAULT_OPENAI_API_KEY", "temperature": 0.0, "thinking": {"type": "enabled", "clear_thinking": False}, "reasoning_effort": "low", "max_output_tokens": 4096}


def _gate_preflight(gate: str, dataset: Mapping[str, Any], candidate: Mapping[str, Any], environ: Mapping[str, str] | None = None, selected_arm: str | None = None) -> dict[str, Any]:
    case_multiplier = 2 if gate == "B" else 1
    estimated = len(dataset["cases"]) * case_multiplier
    payload: dict[str, Any] = {"schema_version": f"healthvault-task-037-gate-{gate.lower()}-v1", "task": "TASK-037", "gate": gate, "status": "API_GATE_PENDING", "api_calls_made": 0, "dataset_version": dataset["dataset_version"], "dataset_sha256": sha256_bytes(canonical_json_bytes({key: dataset[key] for key in fixture_oracle.DATASET_KEYS})), "model": _flash_model(environ), "limits": {"estimated_model_calls": estimated, "maximum_model_calls": estimated * 4, "maximum_transport_attempts": estimated * 8, "max_turns": 4, "max_transport_retries": 1}, "frozen_inputs": {"fixture_sha256": _sha(R4_FIXTURE_PATH), "active_skill_sha256": _sha(ACTIVE_SKILL_PATH), "candidate_preflight_sha256": candidate.get("preflight_sha256"), "candidate_sha256": candidate["candidate_sha256"], "normalizer_sha256": sha256_bytes(inspect.getsource(FinalResponseNormalizer).encode("utf-8")), "oracle_sha256": _sha(Path(fixture_oracle.__file__).resolve()), "three_layer_oracle_sha256": _sha(Path(e2e_oracle.__file__).resolve()), "task_runner_sha256": _sha(Path(__file__)), "prompt_sha256": _sha(ROOT / "healthvault/prompt.py")}, "protocol": {"structured_final": "TASK-025 parse_final_v3", "parser_sha256": _sha(ROOT / "healthvault/evaluation/skill/benchmark/task_025.py"), "oracle_order": ["outcome", "behavior", "response"], "same_runtime_and_prompt": True, "active_candidate_common_normalizer": True}}
    if gate == "B":
        payload["arms"] = {"active": payload["frozen_inputs"]["active_skill_sha256"], "candidate": candidate["candidate_sha256"], "run_once": "20 cases x 2 arms"}
    else:
        if selected_arm not in {"active", "candidate"}:
            raise ValueError("Gate C requires the independently selected Gate-B arm")
        payload["selected_arm"] = selected_arm
    payload["preflight_sha256"] = sha256_bytes(canonical_json_bytes({key: value for key, value in payload.items() if key != "preflight_sha256"}))
    return payload


def build_gate_b_preflight(candidate_path: Path, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    return _gate_preflight("B", load_r4_dataset(SKILL_R4_PATH), _candidate_artifact(candidate_path), environ)


def build_gate_c_preflight(candidate_path: Path, *, gate_b_summary_path: Path, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    summary_path = Path(gate_b_summary_path); summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_arm = summary.get("selected_arm") if isinstance(summary, dict) else None
    preflight = _gate_preflight("C", load_r4_dataset(E2E_R4_PATH), _candidate_artifact(candidate_path), environ, selected_arm)
    preflight["frozen_inputs"]["gate_b_summary_sha256"] = _sha(summary_path)
    preflight["frozen_inputs"]["gate_b_selected_arm"] = selected_arm
    preflight["preflight_sha256"] = sha256_bytes(canonical_json_bytes({key: value for key, value in preflight.items() if key != "preflight_sha256"}))
    return preflight


def assert_gate_preflight(preflight: Mapping[str, Any], *, approved: bool, supplied_hash: str, api_key: str) -> None:
    expected = sha256_bytes(canonical_json_bytes({key: value for key, value in preflight.items() if key != "preflight_sha256"}))
    if not approved or not api_key or supplied_hash != expected or preflight.get("preflight_sha256") != expected:
        raise ApiGateError("TASK-037 Gate requires explicit approval of its current hash")


def materialize_r4_fixture(database_path: Path) -> dict[str, Any]:
    """Create a new database from the frozen R4 snapshot for one attempt."""
    from healthvault.domain.records import (
        ConfirmedIndicator,
        ConfirmedReportBundle,
        build_idempotency_key,
    )
    from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository

    if database_path.exists():
        raise ValueError("R4 fixture target already exists")
    fixture = load_r4_fixture(); repository = SQLiteHealthRecordRepository(database_path); count = 0
    for profile in fixture["profiles"]:
        for report in profile["reports"]:
            lines = ["# Synthetic TASK-037 benchmark data only; no real person", f"report_id: {report['report_id']}", f"report_date: {report['report_date']}", f"facility: {report['facility']}"]
            lines.extend(f"result|{item['code']}|{item['value']}|{item['unit']}|{item['lower']}|{item['upper']}" for item in report["indicators"])
            text = "\n".join(lines) + "\n"
            source_sha256 = sha256_bytes(text.encode("utf-8"))
            indicators = tuple(
                ConfirmedIndicator(
                    code=str(item["code"]),
                    display_name=str(item["code"]).replace("_", " ").title(),
                    value=float(item["value"]),
                    unit=str(item["unit"]),
                    reference_lower=float(item["lower"]),
                    reference_upper=float(item["upper"]),
                    flag=(
                        "low" if float(item["value"]) < float(item["lower"])
                        else "high" if float(item["value"]) > float(item["upper"])
                        else "normal"
                    ),
                    evidence_source_id=str(report["source_id"]),
                    evidence_line_start=index + 5,
                    evidence_line_end=index + 5,
                    evidence_quote=lines[index + 4],
                )
                for index, item in enumerate(report["indicators"])
            )
            outcome = repository.confirm_report(ConfirmedReportBundle(
                idempotency_key=build_idempotency_key(
                    str(profile["profile_id"]), str(report["source_id"]), source_sha256,
                ),
                profile_id=str(profile["profile_id"]),
                source_id=str(report["source_id"]),
                source_sha256=source_sha256,
                report_id=str(report["report_id"]),
                report_date=str(report["report_date"]),
                facility=str(report["facility"]),
                raw_report_text=text,
                indicators=indicators,
            ))
            if outcome.status != "confirmed":
                raise ValueError("TASK-037 fixture report did not confirm")
            count += 1
    return {"snapshot_id": fixture["snapshot_id"], "report_count": count}


async def run_r4_case(case: Mapping[str, Any], *, arm_id: str, skill_content: str, model: Mapping[str, Any], transport: Any, ledger: Any, runtime_root: Path, api_key: str) -> dict[str, Any]:
    """One real production-runtime R4 attempt; caller supplies only an approved arm."""
    import shutil
    import sqlite3
    import tempfile
    from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
    from healthvault.evaluation.skill.benchmark.task_022 import OpenAICompatibleController
    from healthvault.model_config import HealthVaultModelConfig, OPENAI_COMPATIBLE
    from healthvault.runtime_context import RuntimeContext

    started = time.perf_counter(); runtime_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"task037-{arm_id}-{case['case_id'].lower()}-", dir=runtime_root)); database = workdir / "fixture.sqlite3"
    materialize_r4_fixture(database)
    connection = sqlite3.connect(database)
    try:
        before = hashlib.sha256("\n".join(connection.iterdump()).encode("utf-8")).hexdigest()
    finally:
        connection.close()
    skill_root = workdir / ".bear" / "skills"; skill_path = skill_root / "HealthVault-Task-Execution" / "SKILL.md"; skill_path.parent.mkdir(parents=True); skill_path.write_text(skill_content, encoding="utf-8")
    runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(context=RuntimeContext(profile_id=str(case["profile_id"]), request_id=f"task037-{arm_id}-{case['case_id']}"), database_path=database, knowledge_path=e2e_oracle.CORPUS_PATH, model_config=HealthVaultModelConfig(OPENAI_COMPATIBLE, str(model["model"]), str(model["api_base"]), api_key), skill_root=skill_root, skill_names=("HealthVault-Task-Execution",), max_turns=4, max_cost_usd=999.0))
    runtime.agent.is_sub_agent = True; controller = OpenAICompatibleController(transport=transport, api_key=api_key, model_config=dict(model), ledger=ledger); runtime.agent._call_openai_stream = lambda: controller.call(runtime.agent)
    original_execute = runtime.agent._execute_tool_call; calls: list[dict[str, Any]] = []; observed: set[str] = set(); invalid_calls = tool_index = 0; runtime_error = None; parser_error = None; final = None; turn_exhausted = transport_exhausted = False

    async def traced_execute(name: str, arguments: dict[str, Any]) -> str:
        nonlocal invalid_calls, tool_index
        retry = tool_index in set(case["retryable_failure_indices"]); tool_index += 1
        if name not in case["expected"]["allowed_tools"]: invalid_calls += 1
        if retry:
            calls.append({"name": name, "execution_status": "retryable_failure", "result_status": "retryable_failure", "available_citation_ids": []})
            return json.dumps(e2e_oracle._retry_failure_payload(name), ensure_ascii=False, sort_keys=True)
        raw = await original_execute(name, arguments)
        try: payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError): payload = {"status": "rejected"}
        ids = sorted(set(payload.get("available_citation_ids", []))) if isinstance(payload, dict) else []
        observed.update(ids); calls.append({"name": name, "execution_status": "ok", "result_status": e2e_oracle._tool_result_status(payload), "available_citation_ids": ids})
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    runtime.agent._execute_tool_call = traced_execute
    try:
        result = await runtime.run(e2e_oracle.PROMPT_TEMPLATE.format(case_id=case["case_id"], prompt=case["prompt"]))
        parsed = e2e_oracle._parse_task025_final(result.text)
        if parsed.failure_code: parser_error = parsed.failure_code
        else: final = parsed
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"; transport_exhausted = "transport" in str(exc).lower(); turn_exhausted = "turn limit" in str(exc).lower()
    finally:
        connection = sqlite3.connect(database)
        try:
            unchanged = before == hashlib.sha256("\n".join(connection.iterdump()).encode("utf-8")).hexdigest()
        finally:
            connection.close()
        shutil.rmtree(workdir, ignore_errors=True)
    attempt = {"calls": calls, "final": final, "parser_error": parser_error, "runtime_error": runtime_error, "turn_exhausted": turn_exhausted, "transport_exhausted": transport_exhausted, "database_unchanged": unchanged, "invalid_calls": invalid_calls, "observed_citation_ids": sorted(observed), "latency_ms": round((time.perf_counter() - started) * 1000, 3), "model_turns": controller.model_calls, "api_calls": controller.api_calls, "total_tokens": controller.total_tokens}
    oracle = evaluate_r4_three_layer_oracle(case, attempt)
    public = {key: value for key, value in attempt.items() if key != "final"} | {"final": None if final is None else {"final_status": final.final_status, "citations": list(final.citations), "refusal": final.refusal}}
    return {"arm_id": arm_id, "case_id": case["case_id"], "passed": oracle["passed"], "hard_failure": oracle["hard_failure"], "oracle": oracle, "attempt": public}


def summarize_gate_b(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(results) != 40 or {(item["arm_id"], item["case_id"]) for item in results} != {(arm, f"S4-{index:02d}") for arm in ("active", "candidate") for index in range(1, 21)}:
        raise ValueError("TASK-037 Gate B requires exactly one result per arm/case")
    arms: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, bool]] = {}
    for arm in ("active", "candidate"):
        group = [item for item in results if item["arm_id"] == arm]; outcomes[arm] = {item["case_id"]: bool(item["passed"]) for item in group}
        arms[arm] = {"task_success_rate": sum(item["passed"] for item in group) / 20, "correct_tool_call_rate": sum(item["oracle"]["behavior"]["passed"] for item in group) / 20, "hard_failure_rate": sum(item["hard_failure"] for item in group) / 20, "average_calls": sum(len(item["attempt"]["calls"]) for item in group) / 20, "total_tokens": sum(item["attempt"]["total_tokens"] for item in group)}
    regressions = [case_id for case_id, ok in outcomes["active"].items() if ok and not outcomes["candidate"][case_id]]
    candidate = arms["candidate"]; active = arms["active"]
    eligible = candidate["task_success_rate"] - active["task_success_rate"] >= .10 and len(regressions) / 20 <= .05 and candidate["hard_failure_rate"] <= active["hard_failure_rate"] and candidate["correct_tool_call_rate"] >= active["correct_tool_call_rate"]
    return {"schema_version": "healthvault-task-037-gate-b-summary-v1", "case_count": 20, "arms": arms, "regressed_cases": sorted(regressions), "publish_eligible": eligible, "selected_arm": "candidate" if eligible else "active", "candidate_disposition": "publish_eligible_requires_separate_approval" if eligible else "candidate_remains_unpublished"}


async def execute_gate_b_once(preflight: Mapping[str, Any], *, candidate_path: Path, approved: bool, supplied_hash: str, api_key: str, transport: Any, runtime_root: Path) -> dict[str, Any]:
    """The single, ordered 20x2 Skill comparison; quality failures are recorded, never rerun."""
    output = Path(runtime_root) / "task-037-gate-b-summary.json"
    if output.exists():
        raise ApiGateError("TASK-037 Gate B already has a terminal result; rerun is forbidden")
    assert_gate_preflight(preflight, approved=approved, supplied_hash=supplied_hash, api_key=api_key)
    candidate = _candidate_artifact(candidate_path); dataset = load_r4_dataset(SKILL_R4_PATH)
    from healthvault.evaluation.skill.benchmark.task_022 import CallLedger
    ledger = CallLedger(Path(runtime_root) / "task-037-gate-b-ledger.jsonl", run_config_sha256=preflight["preflight_sha256"], max_transport_attempts=int(preflight["limits"]["maximum_transport_attempts"]))
    results: list[dict[str, Any]] = []
    for index, case in enumerate(dataset["cases"]):
        order = ("active", "candidate") if index % 2 == 0 else ("candidate", "active")
        for arm in order:
            content = ACTIVE_SKILL_PATH.read_text(encoding="utf-8") if arm == "active" else candidate["candidate_content"]
            results.append(await run_r4_case(case, arm_id=arm, skill_content=content, model=preflight["model"], transport=transport, ledger=ledger, runtime_root=Path(runtime_root), api_key=api_key))
    summary = summarize_gate_b(results) | {"api_calls": len(ledger.records), "preflight_sha256": preflight["preflight_sha256"]}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(canonical_json_bytes(summary))
    return summary


async def execute_gate_c_once(preflight: Mapping[str, Any], *, candidate_path: Path, approved: bool, supplied_hash: str, api_key: str, transport: Any, runtime_root: Path) -> dict[str, Any]:
    """Run the selected Gate-B arm on the independent R4 E2E set exactly once."""
    output = Path(runtime_root) / "task-037-gate-c-summary.json"
    if output.exists():
        raise ApiGateError("TASK-037 Gate C already has a terminal result; rerun is forbidden")
    assert_gate_preflight(preflight, approved=approved, supplied_hash=supplied_hash, api_key=api_key)
    candidate = _candidate_artifact(candidate_path); dataset = load_r4_dataset(E2E_R4_PATH); arm = str(preflight["selected_arm"])
    from healthvault.evaluation.skill.benchmark.task_022 import CallLedger
    ledger = CallLedger(Path(runtime_root) / "task-037-gate-c-ledger.jsonl", run_config_sha256=preflight["preflight_sha256"], max_transport_attempts=int(preflight["limits"]["maximum_transport_attempts"]))
    content = ACTIVE_SKILL_PATH.read_text(encoding="utf-8") if arm == "active" else candidate["candidate_content"]
    results = [await run_r4_case(case, arm_id=arm, skill_content=content, model=preflight["model"], transport=transport, ledger=ledger, runtime_root=Path(runtime_root), api_key=api_key) for case in dataset["cases"]]
    summary = summarize_gate_c(dataset, results, selected_arm=arm, api_calls=len(ledger.records)) | {"preflight_sha256": preflight["preflight_sha256"]}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(canonical_json_bytes(summary))
    return summary


def summarize_gate_c(dataset: Mapping[str, Any], results: list[Mapping[str, Any]], *, selected_arm: str, api_calls: int) -> dict[str, Any]:
    """Emit the complete frozen E2E metric set from exactly one R4 result matrix."""
    cases = list(dataset["cases"])
    if len(cases) != 20 or len(results) != 20 or {item.get("case_id") for item in results} != {case["case_id"] for case in cases}:
        raise ValueError("TASK-037 Gate C requires exactly one result for each R4 E2E case")
    by_id = {case["case_id"]: case for case in cases}
    refusal_results = [item for item in results if by_id[str(item["case_id"])]["expected"]["refusal_required"]]
    if not refusal_results:
        raise ValueError("TASK-037 Gate C needs refusal cases")
    latencies = [float(item["attempt"]["latency_ms"]) for item in results]
    tokens = [int(item["attempt"]["total_tokens"]) for item in results]
    return {"schema_version": "healthvault-task-037-gate-c-summary-v1", "case_count": 20, "selected_arm": selected_arm, "task_success_rate": sum(bool(item["passed"]) for item in results) / 20, "correct_tool_call_rate": sum(bool(item["oracle"]["behavior"]["passed"]) for item in results) / 20, "citation_correctness_rate": sum(bool(item["oracle"]["response"]["citation_correct"]) for item in results) / 20, "refusal_correctness_rate": sum(bool(item["oracle"]["response"]["refusal_correct"]) for item in refusal_results) / len(refusal_results), "hard_failure_rate": sum(bool(item["hard_failure"]) for item in results) / 20, "total_tokens": sum(tokens), "average_tokens_per_attempt": sum(tokens) / 20, "mean_latency_ms": sum(latencies) / 20, "p95_latency_ms": e2e_oracle._percentile_95(latencies), "api_calls": int(api_calls)}
