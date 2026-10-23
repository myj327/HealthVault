"""Provider-independent generation/Judge runner with a deterministic mock."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from healthvault.evaluation.rag.dataset import QueryRelevanceDataset
from healthvault.rag.corpus import (
    DEFAULT_CORPUS_ROOT,
    load_chunk_artifact,
    parse_chunk_id,
)
from healthvault.evaluation.rag.run_control import (
    AttemptLedger,
    BudgetExceededError,
    RunLimits,
    ensure_run_manifest,
    model_config_dict,
)


GENERATION_PROMPT_VERSION = "rag-generation-r1-v2"
JUDGE_PROMPT_VERSION = "rag-judge-r1-v2"
GENERATION_INSTRUCTIONS = """Use only the supplied evidence chunks. Return one JSON object with exactly status, answer, claims, and refusal_reason. status is answered or refused. Each claim must have exactly claim_id, text, and citation_ids; claim_id must be unique and citation_ids may contain only supplied citation IDs. For a refusal, claims must be empty and refusal_reason must explain that the supplied evidence is insufficient. Do not use patient records or outside knowledge."""
JUDGE_INSTRUCTIONS = """Evaluate only answer relevance, evidence faithfulness, completeness, and expression quality. Do not decide whether a medical fact is true; that belongs to deterministic evidence checks and human review. Return one JSON object with exactly relevance, faithfulness, completeness, expression, reason, and evidence_citation_ids. Each score is an integer from 1 to 4, and evidence_citation_ids may contain only supplied citation IDs."""
_CLAIM_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    temperature: float
    max_output_tokens: int
    api_format: str = "mock"
    endpoint_path: str = "/chat/completions"
    thinking_type: str = "disabled"
    response_format: str = "json_object"
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float = 0.0
    cost_source: str = "mock"


class ModelClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        call_type: str,
        request_id: str,
        config: ModelConfig,
    ) -> ModelResponse:
        ...


class MockModelClient:
    """Scripted local client used to test success, timeout, and retry paths."""

    def __init__(self, outcomes: dict[str, list[ModelResponse | Exception]]) -> None:
        self._outcomes = {key: list(values) for key, values in outcomes.items()}
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        prompt: str,
        *,
        call_type: str,
        request_id: str,
        config: ModelConfig,
    ) -> ModelResponse:
        del prompt, config
        self.calls.append((call_type, request_id))
        outcomes = self._outcomes.get(request_id, [])
        if not outcomes:
            raise RuntimeError(f"No mock outcome for {request_id}")
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _call_model(
    client: ModelClient,
    prompt: str,
    call_type: str,
    request_id: str,
    config: ModelConfig,
    max_retries: int,
    strategy: str,
    ledger: AttemptLedger,
) -> tuple[dict[str, Any], ModelResponse | None]:
    retries = 0
    attempts = []
    call_id = f"{strategy}:{request_id}:{call_type}"
    while True:
        reservation = ledger.preflight(prompt, config)
        attempt_number = retries + 1
        try:
            response = client.complete(
                prompt,
                call_type=call_type,
                request_id=request_id,
                config=config,
            )
            attempt = {
                "schema_version": 2,
                "run_config_sha256": ledger.run_config_sha256,
                "sequence": len(ledger.records) + 1,
                "call_id": call_id,
                "call_type": call_type,
                "strategy": strategy,
                "query_id": request_id,
                "attempt_number": attempt_number,
                "provider": config.provider,
                "model": config.model,
                "model_config_sha256": hashlib.sha256(
                    json.dumps(
                        model_config_dict(config), sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "prompt_sha256": _prompt_hash(prompt),
                "estimated_input_tokens": reservation.estimated_input_tokens,
                "reserved_input_tokens": reservation.input_tokens,
                "reserved_output_tokens": reservation.output_tokens,
                "reserved_cost_usd": reservation.cost_usd,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "success": True,
                "failure": None,
                "cost_usd": response.cost_usd,
                "cost_source": response.cost_source,
                "usage_status": "actual",
                "budget_input_tokens": max(
                    reservation.input_tokens, response.input_tokens
                ),
                "budget_output_tokens": max(
                    reservation.output_tokens, response.output_tokens
                ),
                "budget_cost_usd": max(reservation.cost_usd, response.cost_usd),
            }
            ledger.append(attempt)
            attempts.append(attempt)
            return (
                {
                    "call_type": call_type,
                    "provider": config.provider,
                    "model": config.model,
                    "config": {
                        "temperature": config.temperature,
                        "max_output_tokens": config.max_output_tokens,
                    },
                    "prompt_sha256": _prompt_hash(prompt),
                    "input_tokens": sum((item["input_tokens"] or 0) for item in attempts),
                    "output_tokens": sum((item["output_tokens"] or 0) for item in attempts),
                    "latency_ms": sum(item["latency_ms"] for item in attempts),
                    "success": True,
                    "failure": None,
                    "retries": retries,
                    "cost_usd": round(
                        sum((item["cost_usd"] or 0.0) for item in attempts), 10
                    ),
                    "cost_source": response.cost_source,
                    "attempt_count": len(attempts),
                    "unknown_usage_attempts": sum(
                        item["usage_status"] == "conservative_unknown"
                        for item in attempts
                    ),
                    "budget_input_tokens": sum(
                        item["budget_input_tokens"] for item in attempts
                    ),
                    "budget_output_tokens": sum(
                        item["budget_output_tokens"] for item in attempts
                    ),
                    "budget_cost_usd": round(
                        sum(item["budget_cost_usd"] for item in attempts), 10
                    ),
                },
                response,
            )
        except BudgetExceededError:
            raise
        except (TimeoutError, ConnectionError, RuntimeError) as exc:
            attempt = {
                "schema_version": 2,
                "run_config_sha256": ledger.run_config_sha256,
                "sequence": len(ledger.records) + 1,
                "call_id": call_id,
                "call_type": call_type,
                "strategy": strategy,
                "query_id": request_id,
                "attempt_number": attempt_number,
                "provider": config.provider,
                "model": config.model,
                "model_config_sha256": hashlib.sha256(
                    json.dumps(
                        model_config_dict(config), sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "prompt_sha256": _prompt_hash(prompt),
                "estimated_input_tokens": reservation.estimated_input_tokens,
                "reserved_input_tokens": reservation.input_tokens,
                "reserved_output_tokens": reservation.output_tokens,
                "reserved_cost_usd": reservation.cost_usd,
                "input_tokens": None,
                "output_tokens": None,
                "latency_ms": 0,
                "success": False,
                "failure": type(exc).__name__,
                "cost_usd": None,
                "cost_source": "unavailable",
                "usage_status": "conservative_unknown",
                "budget_input_tokens": reservation.input_tokens,
                "budget_output_tokens": reservation.output_tokens,
                "budget_cost_usd": reservation.cost_usd,
            }
            ledger.append(attempt)
            attempts.append(attempt)
            if retries < max_retries:
                retries += 1
                continue
            return (
                {
                    "call_type": call_type,
                    "provider": config.provider,
                    "model": config.model,
                    "config": {
                        "temperature": config.temperature,
                        "max_output_tokens": config.max_output_tokens,
                    },
                    "prompt_sha256": _prompt_hash(prompt),
                    "input_tokens": sum((item["input_tokens"] or 0) for item in attempts),
                    "output_tokens": sum((item["output_tokens"] or 0) for item in attempts),
                    "latency_ms": sum(item["latency_ms"] for item in attempts),
                    "success": False,
                    "failure": type(exc).__name__,
                    "retries": retries,
                    "cost_usd": round(
                        sum((item["cost_usd"] or 0.0) for item in attempts), 10
                    ),
                    "cost_source": "unavailable",
                    "attempt_count": len(attempts),
                    "unknown_usage_attempts": sum(
                        item["usage_status"] == "conservative_unknown"
                        for item in attempts
                    ),
                    "budget_input_tokens": sum(
                        item["budget_input_tokens"] for item in attempts
                    ),
                    "budget_output_tokens": sum(
                        item["budget_output_tokens"] for item in attempts
                    ),
                    "budget_cost_usd": round(
                        sum(item["budget_cost_usd"] for item in attempts), 10
                    ),
                },
                None,
            )


def _generation_prompt(case: dict[str, Any], context: list[dict[str, str]]) -> str:
    return "\n".join(
        (
            GENERATION_INSTRUCTIONS,
            f"Query: {case['query']}",
            "Evidence:",
            *(
                f"[{item['citation_id']}] {item['body']}"
                for item in context
            ),
        )
    )


def _judge_prompt(case: dict[str, Any], context: list[dict[str, str]], answer: Any) -> str:
    return "\n".join(
        (
            JUDGE_INSTRUCTIONS,
            f"Query: {case['query']}",
            f"Answer JSON: {json.dumps(answer, ensure_ascii=False, sort_keys=True)}",
            "Evidence:",
            *(f"[{item['citation_id']}] {item['body']}" for item in context),
        )
    )


def _deterministic_generation_metrics(
    case: dict[str, Any],
    context: list[dict[str, str]],
    response_text: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        parsed = None
    exact = {"status", "answer", "claims", "refusal_reason"}
    parse_valid = (
        isinstance(parsed, dict)
        and set(parsed) == exact
        and parsed.get("status") in {"answered", "refused"}
        and isinstance(parsed.get("answer"), str)
        and isinstance(parsed.get("claims"), list)
        and (parsed.get("refusal_reason") is None or isinstance(parsed.get("refusal_reason"), str))
    )
    if parse_valid:
        claim_ids: list[str] = []
        for claim in parsed["claims"]:
            if (
                not isinstance(claim, dict)
                or set(claim) != {"claim_id", "text", "citation_ids"}
                or not isinstance(claim.get("claim_id"), str)
                or not _CLAIM_ID_RE.fullmatch(claim["claim_id"])
                or not isinstance(claim.get("text"), str)
                or not claim["text"].strip()
                or not isinstance(claim.get("citation_ids"), list)
                or not claim["citation_ids"]
                or not all(isinstance(item, str) and item for item in claim["citation_ids"])
                or len(claim["citation_ids"]) != len(set(claim["citation_ids"]))
            ):
                parse_valid = False
                break
            claim_ids.append(claim["claim_id"])
        if len(claim_ids) != len(set(claim_ids)):
            parse_valid = False
        if parsed["status"] == "answered":
            parse_valid = (
                parse_valid
                and bool(parsed["answer"].strip())
                and bool(parsed["claims"])
                and parsed["refusal_reason"] is None
            )
        else:
            parse_valid = (
                parse_valid
                and not parsed["claims"]
                and isinstance(parsed["refusal_reason"], str)
                and bool(parsed["refusal_reason"].strip())
            )
    if not parse_valid:
        return {
            "json_valid": False,
            "answer_or_refusal_correct": False,
            "refusal_correct": False,
            "reference_claim_coverage": 0.0,
            "citation_format_valid": False,
            "citation_resolvable": False,
            "expected_citation_hit": False,
            "citation_valid": False,
        }, None
    citations = [
        citation
        for claim in parsed["claims"]
        for citation in claim["citation_ids"]
    ]
    available = {item["citation_id"] for item in context}
    expected = set(case["expected_citations"])
    citation_format_valid = len(citations) == len(set(citations))
    for citation in citations:
        try:
            parse_chunk_id(citation)
        except ValueError:
            citation_format_valid = False
    citation_resolvable = all(citation in available for citation in citations)
    expected_hit = bool(expected.intersection(citations)) if case["answerable"] else not citations
    refusal_correct = (parsed["status"] == "refused") == (not case["answerable"])
    claims = case["reference_claims"]
    reference_texts = [
        claim["text"] if isinstance(claim, dict) else claim
        for claim in claims
    ]
    generated_text = " ".join(
        (parsed["answer"], *(claim["text"] for claim in parsed["claims"]))
    ).casefold()
    coverage = (
        sum(claim.casefold() in generated_text for claim in reference_texts)
        / len(reference_texts)
        if reference_texts
        else 1.0
    )
    return {
        "json_valid": True,
        "answer_or_refusal_correct": refusal_correct,
        "refusal_correct": refusal_correct,
        "reference_claim_coverage": round(coverage, 6),
        "citation_format_valid": citation_format_valid,
        "citation_resolvable": citation_resolvable,
        "expected_citation_hit": expected_hit,
        "citation_valid": citation_format_valid and citation_resolvable and expected_hit,
    }, parsed


def _load_records(path: Path, run_config_sha256: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if any(
        record.get("run_config_sha256") != run_config_sha256
        for record in records
    ):
        raise ValueError("incompatible result resume")
    return records


class GenerationJudgeRunner:
    def __init__(
        self,
        generator: ModelClient,
        judge: ModelClient,
        generation_config: ModelConfig,
        judge_config: ModelConfig,
        *,
        max_retries: int,
        limits: RunLimits,
        run_manifest: dict[str, Any],
        manifest_path: str | Path,
        attempt_ledger_path: str | Path,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self._generator = generator
        self._judge = judge
        self._generation_config = generation_config
        self._judge_config = judge_config
        self._max_retries = max_retries
        if (
            run_manifest.get("generation_config") != model_config_dict(generation_config)
            or run_manifest.get("judge_config") != model_config_dict(judge_config)
            or run_manifest.get("max_retries") != max_retries
            or run_manifest.get("limits") != asdict(limits)
        ):
            raise ValueError("incompatible run manifest and runner configuration")
        self._run_manifest = ensure_run_manifest(manifest_path, run_manifest)
        self._ledger = AttemptLedger(
            attempt_ledger_path,
            limits,
            self._run_manifest["run_config_sha256"],
        )

    def run(
        self,
        cases: tuple[dict[str, Any], ...],
        contexts: dict[str, list[dict[str, str]]],
        strategy: str,
        output_path: str | Path,
    ) -> list[dict[str, Any]]:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = _load_records(path, self._run_manifest["run_config_sha256"])
        completed = {(item["query_id"], item["strategy"]) for item in records}
        for case in cases:
            if case.get("split", "test") != "test":
                raise ValueError("Real generation/Judge runs are restricted to the test split")
            key = (case["query_id"], strategy)
            if key in completed:
                continue
            context = contexts[case["query_id"]]
            generation_prompt = _generation_prompt(case, context)
            generation_call, generation_response = _call_model(
                self._generator,
                generation_prompt,
                "generation",
                case["query_id"],
                self._generation_config,
                self._max_retries,
                strategy,
                self._ledger,
            )
            record: dict[str, Any] = {
                "schema_version": 1,
                "run_config_sha256": self._run_manifest["run_config_sha256"],
                "query_id": case["query_id"],
                "strategy": strategy,
                "prompt_versions": {
                    "generation": GENERATION_PROMPT_VERSION,
                    "judge": JUDGE_PROMPT_VERSION,
                },
                "generation_call": generation_call,
                "generation": None,
                "judge_call": None,
                "deterministic": None,
                "judge": None,
            }
            if generation_response is not None:
                deterministic, parsed = _deterministic_generation_metrics(
                    case, context, generation_response.text
                )
                record["deterministic"] = deterministic
                record["generation"] = parsed
                if parsed is not None:
                    judge_call, judge_response = _call_model(
                        self._judge,
                        _judge_prompt(case, context, parsed),
                        "judge",
                        case["query_id"],
                        self._judge_config,
                        self._max_retries,
                        strategy,
                        self._ledger,
                    )
                    record["judge_call"] = judge_call
                    if judge_response is not None:
                        try:
                            judge_value = json.loads(judge_response.text)
                            if (
                                not isinstance(judge_value, dict)
                                or set(judge_value)
                                != {
                                    "faithfulness", "relevance", "completeness",
                                    "expression", "reason", "evidence_citation_ids",
                                }
                                or not isinstance(judge_value["faithfulness"], int)
                                or not 1 <= judge_value["faithfulness"] <= 4
                                or not isinstance(judge_value["relevance"], int)
                                or not 1 <= judge_value["relevance"] <= 4
                                or not isinstance(judge_value["completeness"], int)
                                or not 1 <= judge_value["completeness"] <= 4
                                or not isinstance(judge_value["expression"], int)
                                or not 1 <= judge_value["expression"] <= 4
                                or not isinstance(judge_value["reason"], str)
                                or not judge_value["reason"].strip()
                                or not isinstance(judge_value["evidence_citation_ids"], list)
                                or not all(
                                    isinstance(item, str)
                                    and item in {entry["citation_id"] for entry in context}
                                    for item in judge_value["evidence_citation_ids"]
                                )
                                or len(judge_value["evidence_citation_ids"])
                                != len(set(judge_value["evidence_citation_ids"]))
                            ):
                                raise ValueError("Invalid Judge rubric response")
                            record["judge"] = judge_value
                        except (json.JSONDecodeError, ValueError):
                            record["judge"] = {"parse_error": True}
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            records.append(record)
            completed.add(key)
        return records


def summarize_usage(records: list[dict[str, Any]]) -> dict[str, int | float]:
    calls = [
        call
        for record in records
        for call in (record.get("generation_call"), record.get("judge_call"))
        if call is not None
    ]
    return {
        "call_count": len(calls),
        "successful_calls": sum(call["success"] for call in calls),
        "failed_calls": sum(not call["success"] for call in calls),
        "input_tokens": sum(call["input_tokens"] for call in calls),
        "output_tokens": sum(call["output_tokens"] for call in calls),
        "latency_ms": sum(call["latency_ms"] for call in calls),
        "retries": sum(call["retries"] for call in calls),
        "cost_usd": round(sum(call["cost_usd"] for call in calls), 8),
    }


def summarize_generation_quality(records: list[dict[str, Any]]) -> dict[str, int | float | None]:
    deterministic = [
        record["deterministic"]
        for record in records
        if record.get("deterministic") is not None
    ]
    judges = [
        record["judge"]
        for record in records
        if isinstance(record.get("judge"), dict)
        and isinstance(record["judge"].get("faithfulness"), int)
        and isinstance(record["judge"].get("relevance"), int)
    ]

    def rate(field: str) -> float | None:
        if not deterministic:
            return None
        return round(sum(bool(item[field]) for item in deterministic) / len(deterministic), 6)

    def mean(field: str, values: list[dict[str, Any]]) -> float | None:
        if not values:
            return None
        return round(sum(float(item[field]) for item in values) / len(values), 6)

    return {
        "query_count": len(records),
        "evaluated_count": len(deterministic),
        "generation_failure_count": sum(
            not record["generation_call"]["success"] for record in records
        ),
        "json_valid_rate": rate("json_valid"),
        "answer_or_refusal_accuracy": rate("answer_or_refusal_correct"),
        "reference_claim_coverage": mean("reference_claim_coverage", deterministic),
        "citation_format_valid_rate": rate("citation_format_valid"),
        "citation_resolvable_rate": rate("citation_resolvable"),
        "expected_citation_hit_rate": rate("expected_citation_hit"),
        "judge_count": len(judges),
        "judge_faithfulness_mean": mean("faithfulness", judges),
        "judge_relevance_mean": mean("relevance", judges),
        "judge_completeness_mean": mean("completeness", judges),
        "judge_expression_mean": mean("expression", judges),
    }


def prepare_test_generation_inputs(
    dataset: QueryRelevanceDataset,
    retrieval_result: dict[str, Any],
    strategy: str,
    *,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
) -> tuple[tuple[dict[str, Any], ...], dict[str, list[dict[str, str]]]]:
    """Bind frozen test queries to the exact retrieved chunks supplied to a model."""

    if strategy not in {"natural_section", "hybrid_v1"}:
        raise ValueError("Unsupported generation strategy")
    strategy_result = retrieval_result[strategy]
    rankings = {
        item["query_id"]: item["ranked_chunk_ids"]
        for item in strategy_result["queries"]
    }
    artifact = load_chunk_artifact(
        Path(corpus_root) / "chunks" / f"{strategy}.json"
    )
    chunks = {chunk.chunk_id: chunk for chunk in artifact.chunks}
    cases = []
    contexts = {}
    for query in dataset.queries:
        if query.split != "test":
            continue
        cases.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "split": query.split,
                "answerable": query.answerable,
                "reference_claims": list(query.reference_claims),
                "expected_citations": list(query.relevance[strategy].citation_ids),
            }
        )
        contexts[query.query_id] = [
            {"citation_id": chunk_id, "body": chunks[chunk_id].body}
            for chunk_id in rankings[query.query_id]
        ]
    return tuple(cases), contexts
