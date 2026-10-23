"""Approved-only TASK-026 generation and pairwise Judge execution."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from healthvault.evaluation.rag.generation import _deterministic_generation_metrics
from healthvault.evaluation.rag.task_026 import (
    DEFAULT_GOLD_PATH,
    DEFAULT_OUTPUT_DIR,
    GENERATION_INSTRUCTIONS,
    JUDGE_INSTRUCTIONS,
    _canonical_bytes,
    _load_json,
    _sha256,
    _validate_judge_output,
    build_generation_judge_preflight,
    build_generator_request_payload,
    build_generator_system_prompt,
    build_judge_system_prompt,
)
from healthvault.evaluation.skill.benchmark.task_022 import (
    JsonTransport,
    TransportError,
    UrllibJsonTransport,
)


MAX_MODEL_CALLS = 104
MAX_TRANSPORT_ATTEMPTS = 208
MAX_TRANSPORT_RETRIES = 1
DEFAULT_EXECUTION_DIR = DEFAULT_OUTPUT_DIR / "generation-judge-run-v5"


class Task026ExecutionGateError(RuntimeError):
    """Raised before either model transport can be called."""


class Task026ProtocolError(RuntimeError):
    """Raised when a provider response fails the frozen local protocol."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


class Task026AttemptLedger:
    """Append-only attempt ledger bound to one preflight and hard attempt cap."""

    def __init__(self, path: str | Path, run_config_sha256: str) -> None:
        self.path = Path(path)
        self.run_config_sha256 = run_config_sha256
        self.records: list[dict[str, Any]] = []
        if self.path.is_file():
            self.records = [
                json.loads(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if any(
            item.get("run_config_sha256") != run_config_sha256
            for item in self.records
        ):
            raise ValueError("attempt ledger is bound to another preflight")

    def append(
        self,
        *,
        call_id: str,
        phase: str,
        client_id: str,
        model: str,
        transport_attempt: int,
        status: str,
        usage: dict[str, Any] | None,
        latency_ms: float,
        is_real: bool,
    ) -> None:
        if len(self.records) >= MAX_TRANSPORT_ATTEMPTS:
            raise RuntimeError("TASK-026 transport attempt limit exceeded")
        known = (
            isinstance(usage, dict)
            and isinstance(usage.get("prompt_tokens"), int)
            and isinstance(usage.get("completion_tokens"), int)
            and isinstance(usage.get("total_tokens"), int)
            and usage["total_tokens"]
            == usage["prompt_tokens"] + usage["completion_tokens"]
        )
        record = {
            "schema_version": 1,
            "run_config_sha256": self.run_config_sha256,
            "sequence": len(self.records) + 1,
            "call_id": call_id,
            "phase": phase,
            "client_id": client_id,
            "model": model,
            "transport_attempt": transport_attempt,
            "transport_kind": "real" if is_real else "fake",
            "status": status,
            "input_tokens": usage["prompt_tokens"] if known else None,
            "output_tokens": usage["completion_tokens"] if known else None,
            "total_tokens": usage["total_tokens"] if known else None,
            "usage_source": "provider" if known else "unknown",
            "latency_ms": round(float(latency_ms), 6),
        }
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class Task026ChatClient:
    """One model-specific client with fresh messages for every call."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        transport: JsonTransport,
        api_key: str,
        ledger: Task026AttemptLedger,
        max_model_calls: int,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.transport = transport
        self.api_key = api_key
        self.ledger = ledger
        self.max_model_calls = max_model_calls
        self.model_calls = 0

    async def complete_json(
        self,
        *,
        call_id: str,
        phase: str,
        messages: list[dict[str, str]],
        validator: Any,
    ) -> dict[str, Any]:
        if self.model_calls >= self.max_model_calls:
            raise RuntimeError(f"{self.config['client_id']} model call limit exceeded")
        self.model_calls += 1
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": copy.deepcopy(messages),
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_output_tokens"],
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if "do_sample" in self.config:
            payload["do_sample"] = self.config["do_sample"]
        if "thinking" in self.config:
            payload["thinking"] = copy.deepcopy(self.config["thinking"])
        if "reasoning_effort" in self.config:
            payload["reasoning_effort"] = self.config["reasoning_effort"]

        for attempt in range(1, MAX_TRANSPORT_RETRIES + 2):
            started = time.perf_counter()
            try:
                response = await self.transport.post_json(
                    f"{self.config['api_base'].rstrip('/')}/chat/completions",
                    {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    payload,
                    60.0,
                )
            except TransportError as exc:
                self.ledger.append(
                    call_id=call_id,
                    phase=phase,
                    client_id=self.config["client_id"],
                    model=self.config["model"],
                    transport_attempt=attempt,
                    status="transport_error",
                    usage=None,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    is_real=self.transport.is_real,
                )
                if not exc.retryable or attempt > MAX_TRANSPORT_RETRIES:
                    raise
                continue

            latency_ms = (time.perf_counter() - started) * 1000
            usage = response.get("usage") if isinstance(response, dict) else None
            try:
                message = response["choices"][0]["message"]
                content = message["content"]
                parsed = json.loads(content)
                validated = validator(parsed)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
                self.ledger.append(
                    call_id=call_id,
                    phase=phase,
                    client_id=self.config["client_id"],
                    model=self.config["model"],
                    transport_attempt=attempt,
                    status="protocol_error",
                    usage=usage,
                    latency_ms=latency_ms,
                    is_real=self.transport.is_real,
                )
                if attempt > MAX_TRANSPORT_RETRIES:
                    raise Task026ProtocolError(
                        "model response failed local JSON validation"
                    ) from exc
                payload["messages"] = copy.deepcopy(messages) + [
                    {
                        "role": "system",
                        "content": (
                            "Your previous response failed the local JSON schema. "
                            f"Validation error: {exc}. Return only a corrected JSON "
                            "object; do not change the underlying answer or use new "
                            "evidence."
                        ),
                    }
                ]
                continue
            self.ledger.append(
                call_id=call_id,
                phase=phase,
                client_id=self.config["client_id"],
                model=self.config["model"],
                transport_attempt=attempt,
                status="ok",
                usage=usage,
                latency_ms=latency_ms,
                is_real=self.transport.is_real,
            )
            return validated
        raise RuntimeError("unreachable transport loop")


def _validate_generation_json(value: Any) -> dict[str, Any]:
    exact = {"status", "answer", "claims", "refusal_reason"}
    if not isinstance(value, dict) or set(value) != exact:
        raise ValueError("Generator output must use the exact schema")
    if value["status"] not in {"answered", "refused"} or not isinstance(value["answer"], str):
        raise ValueError("Generator status or answer is invalid")
    if not isinstance(value["claims"], list):
        raise ValueError("Generator claims must be a list")
    claim_ids = []
    for claim in value["claims"]:
        if (
            not isinstance(claim, dict)
            or set(claim) != {"claim_id", "text", "citation_ids"}
            or not isinstance(claim["claim_id"], str)
            or not claim["claim_id"]
            or not isinstance(claim["text"], str)
            or not claim["text"].strip()
            or not isinstance(claim["citation_ids"], list)
            or not claim["citation_ids"]
            or not all(isinstance(item, str) and item for item in claim["citation_ids"])
            or len(claim["citation_ids"]) != len(set(claim["citation_ids"]))
        ):
            raise ValueError("Generator claim is invalid")
        claim_ids.append(claim["claim_id"])
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Generator claim IDs must be unique")
    if value["status"] == "answered":
        if not value["answer"].strip() or not value["claims"] or value["refusal_reason"] is not None:
            raise ValueError("answered output is invalid")
    elif (
        value["claims"]
        or not isinstance(value["refusal_reason"], str)
        or not value["refusal_reason"].strip()
    ):
        raise ValueError("refused output is invalid")
    return value


def _messages(system: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_task026_execution_gate(
    preflight: dict[str, Any],
    *,
    preflight_path: str | Path,
    provided_preflight_sha256: str,
    execute: bool,
    approved: bool,
    api_key: str,
) -> None:
    """Rebuild all frozen bindings and fail before transport construction."""

    if not execute:
        raise Task026ExecutionGateError("TASK-026 real execution requires --execute")
    if not approved:
        raise Task026ExecutionGateError("TASK-026 real execution requires --approved")
    if not isinstance(api_key, str) or not api_key.strip():
        raise Task026ExecutionGateError("HEALTHVAULT_OPENAI_API_KEY is not configured")
    current_file_sha256 = _file_sha256(preflight_path)
    if provided_preflight_sha256 != current_file_sha256:
        raise Task026ExecutionGateError("current preflight file hash mismatch")
    expected = build_generation_judge_preflight(
        _load_json(DEFAULT_GOLD_PATH),
        _load_json(DEFAULT_OUTPUT_DIR / "task-026-retrieval-results.json"),
    )
    if preflight != expected or preflight.get("run_config_sha256") != expected.get(
        "run_config_sha256"
    ):
        raise Task026ExecutionGateError("preflight configuration is stale or modified")
    if preflight.get("maximum_model_calls") != MAX_MODEL_CALLS or preflight.get(
        "maximum_transport_attempts"
    ) != MAX_TRANSPORT_ATTEMPTS:
        raise Task026ExecutionGateError("TASK-026 call limits are not frozen")
    contexts_path = Path(preflight_path).with_name("task-026-generation-contexts-v3.json")
    contexts = _load_json(contexts_path)
    if _sha256(_canonical_bytes(contexts)) != preflight.get("contexts_sha256"):
        raise Task026ExecutionGateError("generation context hash mismatch")
    if _file_sha256(DEFAULT_GOLD_PATH) != preflight["context_bindings"]["gold_sha256"]:
        raise Task026ExecutionGateError("Gold hash mismatch")
    if preflight["generator"]["model"] != "glm-5.3-flash" or preflight["judge"][
        "model"
    ] != "glm-5.3":
        raise Task026ExecutionGateError("TASK-026 model configuration mismatch")
    if (
        preflight["generator"]["temperature"] != 0.0
        or preflight["judge"]["temperature"] != 0.0
        or preflight["generator"]["client_id"] == preflight["judge"]["client_id"]
    ):
        raise Task026ExecutionGateError("model separation or temperature mismatch")


def _combined_evidence(context: dict[str, Any]) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for arm in ("baseline", "variant"):
        for item in context[arm]:
            combined.setdefault(item["citation_id"], item)
    return list(combined.values())


def _usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "transport_attempts": len(records),
        "api_calls": sum(item["transport_kind"] == "real" for item in records),
        "input_tokens": sum(item["input_tokens"] or 0 for item in records),
        "output_tokens": sum(item["output_tokens"] or 0 for item in records),
        "total_tokens": sum(item["total_tokens"] or 0 for item in records),
        "unknown_usage_attempts": sum(item["usage_source"] == "unknown" for item in records),
        "latency_ms": round(sum(item["latency_ms"] for item in records), 6),
    }


async def execute_approved_task026_run(
    *,
    preflight_path: str | Path,
    provided_preflight_sha256: str,
    execute: bool,
    approved: bool,
    api_key: str,
    output_dir: str | Path,
    generator_transport: JsonTransport | None = None,
    judge_transport: JsonTransport | None = None,
) -> dict[str, Any]:
    """Run calibration, generation, and final pairwise Judge exactly once."""

    preflight = _load_json(preflight_path)
    assert_task026_execution_gate(
        preflight,
        preflight_path=preflight_path,
        provided_preflight_sha256=provided_preflight_sha256,
        execute=execute,
        approved=approved,
        api_key=api_key,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "task-026-generation-judge-results.json"
    if result_path.exists():
        raise Task026ExecutionGateError("TASK-026 result already exists; execution is once-only")
    contexts = _load_json(Path(preflight_path).with_name("task-026-generation-contexts-v3.json"))
    calibration = _load_json(Path(preflight_path).with_name("task-026-judge-dev-calibration-v1.json"))
    gold = _load_json(DEFAULT_GOLD_PATH)
    gold_by_id = {item["query_id"]: item for item in gold["queries"]}
    context_by_id = {item["query_id"]: item for item in contexts["queries"]}
    ledger = Task026AttemptLedger(
        output / "task-026-attempt-ledger.jsonl", preflight["run_config_sha256"]
    )
    generator = Task026ChatClient(
        config=preflight["generator"],
        transport=generator_transport or UrllibJsonTransport(),
        api_key=api_key,
        ledger=ledger,
        max_model_calls=48,
    )
    judge = Task026ChatClient(
        config=preflight["judge"],
        transport=judge_transport or UrllibJsonTransport(),
        api_key=api_key,
        ledger=ledger,
        max_model_calls=56,
    )

    calibration_results = []
    pairs = {item["pair_id"]: item for item in calibration["pairs"]}
    for item in calibration["judgments"]:
        pair = pairs[item["pair_id"]]
        payload = {
            "dimension": item["dimension"],
            "query": pair["query"],
            "evidence": pair["evidence"],
            "answer_a": item["answer_a"],
            "answer_b": item["answer_b"],
        }
        if item["dimension"] == "completeness":
            payload["reference_claims"] = pair["reference_claims"]
            payload["refusal_target"] = pair["refusal_target"]
        citations = {evidence["citation_id"] for evidence in pair["evidence"]}
        value = await judge.complete_json(
            call_id=f"calibration:{item['pair_id']}:{item['dimension']}",
            phase="judge_calibration",
            messages=_messages(build_judge_system_prompt(item["dimension"]), payload),
            validator=lambda output_value, dimension=item["dimension"], available=citations: _validate_judge_output(
                output_value, dimension, available
            ),
        )
        calibration_results.append(
            {
                "pair_id": item["pair_id"],
                "dimension": item["dimension"],
                "expected_winner": item["expected_winner"],
                "output": value,
                "passed": value["winner"] == item["expected_winner"],
            }
        )
    if not all(item["passed"] for item in calibration_results):
        raise Task026ProtocolError("Judge failed dev calibration")

    generations: dict[tuple[str, str], dict[str, Any]] = {}
    for context in contexts["queries"]:
        for arm in ("baseline", "variant"):
            request_payload = build_generator_request_payload(
                context["query"], context[arm]
            )
            if set(request_payload) != {"query", "evidence"}:
                raise Task026ProtocolError("Generator input leaked non-evidence fields")
            value = await generator.complete_json(
                call_id=f"generation:{context['query_id']}:{arm}",
                phase="generation",
                messages=_messages(build_generator_system_prompt(), request_payload),
                validator=_validate_generation_json,
            )
            gold_query = gold_by_id[context["query_id"]]
            case = {
                "answerable": gold_query["answerable"],
                "expected_citations": gold_query["qrels"],
                "reference_claims": gold_query["reference_claims"],
            }
            metrics, _parsed = _deterministic_generation_metrics(
                case, context[arm], json.dumps(value, ensure_ascii=False)
            )
            generations[(context["query_id"], arm)] = {
                "query_id": context["query_id"],
                "arm": arm,
                "output": value,
                "deterministic_validation": metrics,
            }

    final_judgments = []
    for assignment in preflight["pairwise_order"]["schedule"]:
        query_id = assignment["query_id"]
        dimension = assignment["dimension"]
        context = context_by_id[query_id]
        gold_query = gold_by_id[query_id]
        evidence = _combined_evidence(context)
        payload: dict[str, Any] = {
            "dimension": dimension,
            "query": context["query"],
            "evidence": evidence,
            "answer_a": generations[(query_id, assignment["answer_a"])]["output"],
            "answer_b": generations[(query_id, assignment["answer_b"])]["output"],
        }
        if dimension == "completeness":
            payload["reference_claims"] = gold_query["reference_claims"]
            payload["refusal_target"] = gold_query["refusal_target"]
        citations = {item["citation_id"] for item in evidence}
        value = await judge.complete_json(
            call_id=f"judge:{query_id}:{dimension}",
            phase="final_judge",
            messages=_messages(build_judge_system_prompt(dimension), payload),
            validator=lambda output_value, current=dimension, available=citations: _validate_judge_output(
                output_value, current, available
            ),
        )
        final_judgments.append({**assignment, "output": value})

    if generator.model_calls + judge.model_calls != MAX_MODEL_CALLS:
        raise RuntimeError("TASK-026 did not execute exactly 104 model calls")
    generation_rows = list(generations.values())
    results = {
        "schema_version": 1,
        "status": "completed_once",
        "run_config_sha256": preflight["run_config_sha256"],
        "preflight_file_sha256": provided_preflight_sha256,
        "calibration": calibration_results,
        "generations": generation_rows,
        "judgments": final_judgments,
        "usage": _usage_summary(ledger.records),
    }
    _write_json(result_path, results)

    pairwise = {}
    for dimension in ("faithfulness", "completeness"):
        counts = {"baseline": 0, "variant": 0, "tie": 0}
        for item in final_judgments:
            if item["dimension"] != dimension:
                continue
            winner = item["output"]["winner"]
            counts["tie" if winner == "tie" else item[f"answer_{winner.casefold()}"]] += 1
        pairwise[dimension] = counts
    summary = {
        "schema_version": 1,
        "status": "completed_once",
        "run_config_sha256": preflight["run_config_sha256"],
        "models": {
            "generator": preflight["generator"]["model"],
            "judge": preflight["judge"]["model"],
        },
        "generation_count": len(generation_rows),
        "judge_calibration_count": len(calibration_results),
        "final_judge_count": len(final_judgments),
        "pairwise": pairwise,
        "deterministic": {
            arm: {
                "citation_correct_count": sum(
                    item["deterministic_validation"]["citation_valid"]
                    for item in generation_rows
                    if item["arm"] == arm
                ),
                "refusal_correct_count": sum(
                    item["deterministic_validation"]["refusal_correct"]
                    for item in generation_rows
                    if item["arm"] == arm
                ),
            }
            for arm in ("baseline", "variant")
        },
        "usage": results["usage"],
    }
    _write_json(output / "task-026-generation-judge-summary.json", summary)

    sample_ids = sorted(
        context_by_id,
        key=lambda query_id: _sha256(
            f"task-026-human-review-v1|{query_id}".encode("utf-8")
        ),
    )[:10]
    review = {
        "schema_version": 1,
        "status": "pending_human_review",
        "selection": "10 query IDs by deterministic SHA-256 order",
        "queries": [
            {
                "query_id": query_id,
                "query": context_by_id[query_id]["query"],
                "evidence": _combined_evidence(context_by_id[query_id]),
                "answer_a": generations[(query_id, "baseline")]["output"],
                "answer_b": generations[(query_id, "variant")]["output"],
                "reference_claims": gold_by_id[query_id]["reference_claims"],
                "refusal_target": gold_by_id[query_id]["refusal_target"],
                "judgments": [
                    item for item in final_judgments if item["query_id"] == query_id
                ],
            }
            for query_id in sample_ids
        ],
    }
    _write_json(output / "task-026-human-review-sample.json", review)
    manifest = {
        "schema_version": 1,
        "status": "completed_once",
        "approved": True,
        "api_key_recorded": False,
        "preflight_file_sha256": provided_preflight_sha256,
        "run_config_sha256": preflight["run_config_sha256"],
        "gold_sha256": preflight["context_bindings"]["gold_sha256"],
        "contexts_sha256": preflight["contexts_sha256"],
        "generator_config": preflight["generator"],
        "judge_config": preflight["judge"],
        "limits": {
            "max_model_calls": MAX_MODEL_CALLS,
            "max_transport_attempts": MAX_TRANSPORT_ATTEMPTS,
            "max_transport_retries_per_call": MAX_TRANSPORT_RETRIES,
        },
        "files": {
            path.name: _file_sha256(path)
            for path in (
                output / "task-026-attempt-ledger.jsonl",
                result_path,
                output / "task-026-generation-judge-summary.json",
                output / "task-026-human-review-sample.json",
            )
        },
    }
    _write_json(output / "task-026-execution-run-manifest.json", manifest)
    return {"results": results, "summary": summary, "review": review, "manifest": manifest}


def execute_task026_sync(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(execute_approved_task026_run(**kwargs))
