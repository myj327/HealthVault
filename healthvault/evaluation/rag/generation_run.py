"""Auditable preflight and explicit execution entry for generation/Judge evals."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from healthvault.evaluation.rag.dataset import (
    DEFAULT_DATASET_PATH,
    load_query_relevance_dataset,
)
from healthvault.evaluation.rag.deepseek import (
    DEEPSEEK_GENERATION_CONFIG,
    DEEPSEEK_JUDGE_CONFIG,
    DEEPSEEK_PRICING_URL,
    DeepSeekChatCompletionsClient,
    JsonTransport,
)
from healthvault.evaluation.rag.generation import (
    GENERATION_INSTRUCTIONS,
    GENERATION_PROMPT_VERSION,
    JUDGE_INSTRUCTIONS,
    JUDGE_PROMPT_VERSION,
    GenerationJudgeRunner,
    prepare_test_generation_inputs,
    summarize_generation_quality,
    summarize_usage,
    _deterministic_generation_metrics,
    _generation_prompt,
    _judge_prompt,
)
from healthvault.evaluation.rag.run_control import (
    AttemptLedger,
    DEFAULT_REAL_RUN_LIMITS,
    RunLimits,
    build_run_manifest,
    canonical_json_bytes,
    ensure_run_manifest,
)
from healthvault.rag.chunking import estimate_tokens


DEFAULT_RETRIEVAL_RESULT_PATH = Path(__file__).parent / "results" / "retrieval-results.json"
DEFAULT_GENERATION_OUTPUT_DIR = Path(__file__).parent / "results" / "generation"
STRATEGIES = ("natural_section", "hybrid_v1")
MAX_RETRIES = 1
PRICING_VERIFIED_AT = "2026-08-27T12:00:00+08:00"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def capture_implementation_provenance(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Bind a run to the current Git state and exact evaluation implementation."""

    root = Path(repository_root) if repository_root else Path(__file__).parents[3]
    root = root.resolve()

    def git_output(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("cannot capture evaluation Git provenance") from exc
        return completed.stdout.strip()

    current_head = git_output("rev-parse", "HEAD")
    worktree_dirty = bool(
        git_output("status", "--porcelain=v1", "--untracked-files=all")
    )
    evaluation_root = root / "healthvault" / "evaluation"
    source_files = {
        path.relative_to(root).as_posix(): _sha256(path.read_bytes())
        for path in sorted(evaluation_root.rglob("*.py"))
    }
    if not source_files:
        raise RuntimeError("evaluation source tree is empty")
    source_tree_sha256 = _sha256(
        json.dumps(
            source_files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    prompt_sha256 = _sha256(
        json.dumps(
            {
                "generation_instructions": GENERATION_INSTRUCTIONS,
                "generation_version": GENERATION_PROMPT_VERSION,
                "judge_instructions": JUDGE_INSTRUCTIONS,
                "judge_version": JUDGE_PROMPT_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    validator_sha256 = _sha256(
        inspect.getsource(_deterministic_generation_metrics).encode("utf-8")
    )
    return {
        "schema_version": 1,
        "current_head": current_head,
        "worktree_dirty": worktree_dirty,
        "evaluation_source_sha256": source_tree_sha256,
        "evaluation_source_files": source_files,
        "prompt_sha256": prompt_sha256,
        "validator_sha256": validator_sha256,
    }


def _pricing_snapshot() -> dict[str, Any]:
    return {
        "provider": "deepseek",
        "currency": "USD",
        "unit": "per_million_tokens",
        "billing_basis": "cache_miss_input_and_output",
        "source_url": DEEPSEEK_PRICING_URL,
        "verified_at": PRICING_VERIFIED_AT,
        "models": {
            "generation": {
                "model": DEEPSEEK_GENERATION_CONFIG.model,
                "input_usd": DEEPSEEK_GENERATION_CONFIG.input_cost_per_million,
                "output_usd": DEEPSEEK_GENERATION_CONFIG.output_cost_per_million,
            },
            "judge": {
                "model": DEEPSEEK_JUDGE_CONFIG.model,
                "input_usd": DEEPSEEK_JUDGE_CONFIG.input_cost_per_million,
                "output_usd": DEEPSEEK_JUDGE_CONFIG.output_cost_per_million,
            },
        },
        "peak_price_multiplier": 2.0,
        "peak_risk": (
            "Official pricing warns that peak Beijing-time windows may charge 2x; "
            "re-verify the effective date immediately before an authorized run."
        ),
    }


def _normal_response(
    case: dict[str, Any], context: list[dict[str, str]]
) -> dict[str, Any]:
    if case["answerable"] and context:
        citation = context[0]["citation_id"]
        return {
            "status": "answered",
            "answer": "Supported by the supplied evidence.",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Supported by the supplied evidence.",
                    "citation_ids": [citation],
                }
            ],
            "refusal_reason": None,
        }
    return {
        "status": "refused",
        "answer": "Insufficient evidence.",
        "claims": [],
        "refusal_reason": "The supplied evidence is insufficient.",
    }


def _fixed_refusal_response() -> dict[str, Any]:
    return {
        "status": "refused",
        "answer": "Insufficient evidence.",
        "claims": [],
        "refusal_reason": "The supplied evidence is insufficient.",
    }


def _near_limit_response(max_tokens: int) -> dict[str, Any]:
    response = _fixed_refusal_response()
    low, high = 0, max_tokens * 4
    while low < high:
        middle = (low + high + 1) // 2
        response["answer"] = "x" * middle
        encoded = json.dumps(response, ensure_ascii=False, sort_keys=True)
        if estimate_tokens(encoded) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    response["answer"] = "x" * low
    return response


def _strategy_estimate(
    cases: tuple[dict[str, Any], ...],
    contexts: dict[str, list[dict[str, str]]],
    *,
    near_output_limits: bool,
) -> dict[str, int | float]:
    generation_input = 0
    judge_input = 0
    generation_output = 0
    for case in cases:
        context = contexts[case["query_id"]]
        generation_input += estimate_tokens(_generation_prompt(case, context))
        response = (
            _near_limit_response(DEEPSEEK_GENERATION_CONFIG.max_output_tokens)
            if near_output_limits
            else _normal_response(case, context)
        )
        judge_input += estimate_tokens(_judge_prompt(case, context, response))
        generation_output += (
            DEEPSEEK_GENERATION_CONFIG.max_output_tokens
            if near_output_limits
            else estimate_tokens(json.dumps(response, ensure_ascii=False, sort_keys=True))
        )
    judge_output = (
        len(cases) * DEEPSEEK_JUDGE_CONFIG.max_output_tokens
        if near_output_limits
        else sum(
            estimate_tokens(
                json.dumps(
                    {
                        "relevance": 4,
                        "faithfulness": 4,
                        "completeness": 4,
                        "expression": 4,
                        "reason": "Evidence-bound response.",
                        "evidence_citation_ids": [],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            for _case in cases
        )
    )
    cost = (
        generation_input * DEEPSEEK_GENERATION_CONFIG.input_cost_per_million
        + generation_output * DEEPSEEK_GENERATION_CONFIG.output_cost_per_million
        + judge_input * DEEPSEEK_JUDGE_CONFIG.input_cost_per_million
        + judge_output * DEEPSEEK_JUDGE_CONFIG.output_cost_per_million
    ) / 1_000_000
    return {
        "generation_input_tokens": generation_input,
        "judge_input_tokens": judge_input,
        "generation_output_tokens": generation_output,
        "judge_output_tokens": judge_output,
        "input_tokens": generation_input + judge_input,
        "output_tokens": generation_output + judge_output,
        "cost_usd": round(cost, 10),
        "peak_2x_cost_usd": round(cost * 2, 10),
    }


def _scenario_totals(strategies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cost = sum(item["cost_usd"] for item in strategies.values())
    return {
        "input_tokens": sum(item["input_tokens"] for item in strategies.values()),
        "output_tokens": sum(item["output_tokens"] for item in strategies.values()),
        "cost_usd": round(cost, 10),
        "peak_2x_cost_usd": round(cost * 2, 10),
    }


def _double_estimate(value: dict[str, int | float]) -> dict[str, int | float]:
    doubled = {
        field: item * 2
        for field, item in value.items()
        if field not in {"cost_usd", "peak_2x_cost_usd"}
    }
    doubled["cost_usd"] = round(value["cost_usd"] * 2, 10)
    doubled["peak_2x_cost_usd"] = round(value["peak_2x_cost_usd"] * 2, 10)
    return doubled


def _token_cost_plan(dataset: Any, retrieval: dict[str, Any]) -> dict[str, Any]:
    inputs = {
        strategy: prepare_test_generation_inputs(dataset, retrieval, strategy)
        for strategy in STRATEGIES
    }
    prompt_inputs = {}
    normal_strategies = {}
    near_strategies = {}
    fixed_refusal = _fixed_refusal_response()
    for strategy, (cases, contexts) in inputs.items():
        prompt_inputs[strategy] = {
            "generation_input_tokens": sum(
                estimate_tokens(_generation_prompt(case, contexts[case["query_id"]]))
                for case in cases
            ),
            "judge_input_tokens_with_fixed_refusal": sum(
                estimate_tokens(
                    _judge_prompt(case, contexts[case["query_id"]], fixed_refusal)
                )
                for case in cases
            ),
        }
        normal_strategies[strategy] = _strategy_estimate(
            cases, contexts, near_output_limits=False
        )
        near_strategies[strategy] = _strategy_estimate(
            cases, contexts, near_output_limits=True
        )
    retry_strategies = {
        strategy: _double_estimate(estimate)
        for strategy, estimate in near_strategies.items()
    }
    scenarios = {
        "normal": {
            "assumption": "one successful attempt with short schema-valid responses",
            "strategies": normal_strategies,
            "totals": _scenario_totals(normal_strategies),
        },
        "near_output_limits": {
            "assumption": "one successful attempt with generation and Judge outputs near configured maxima",
            "strategies": near_strategies,
            "totals": _scenario_totals(near_strategies),
        },
        "all_once_retry": {
            "assumption": "every near-limit generation and Judge request is attempted twice",
            "strategies": retry_strategies,
            "totals": _scenario_totals(retry_strategies),
        },
    }
    return {
        "prompt_input_estimates": prompt_inputs,
        "token_cost_scenarios": scenarios,
        "estimated_range": {
            "input_tokens": [
                scenarios["normal"]["totals"]["input_tokens"],
                scenarios["all_once_retry"]["totals"]["input_tokens"],
            ],
            "output_tokens": [
                scenarios["normal"]["totals"]["output_tokens"],
                scenarios["all_once_retry"]["totals"]["output_tokens"],
            ],
            "cost_usd": [
                scenarios["normal"]["totals"]["cost_usd"],
                scenarios["all_once_retry"]["totals"]["cost_usd"],
            ],
            "peak_2x_cost_usd": [
                scenarios["normal"]["totals"]["peak_2x_cost_usd"],
                scenarios["all_once_retry"]["totals"]["peak_2x_cost_usd"],
            ],
        },
    }


def _load_bound_inputs(
    dataset_path: str | Path,
    retrieval_result_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    dataset = load_query_relevance_dataset(dataset_path)
    retrieval = json.loads(Path(retrieval_result_path).read_text(encoding="utf-8"))
    expected = {
        "eval_version": dataset.eval_version,
        "dataset_sha256": dataset.dataset_sha256,
        "frozen_test_sha256": dataset.frozen_test_sha256,
        "corpus_sha256": dataset.corpus_sha256,
        "artifact_sha256": dataset.artifact_sha256,
    }
    for field, value in expected.items():
        if retrieval.get(field) != value:
            raise ValueError(f"retrieval result is not bound to dataset field {field}")
    if any(strategy not in retrieval for strategy in STRATEGIES):
        raise ValueError("retrieval result is missing a required strategy")
    return dataset, retrieval


def _manifest(
    dataset: Any,
    *,
    implementation_provenance: dict[str, Any],
    created_at: str,
    limits: RunLimits,
) -> dict[str, Any]:
    query_ids = tuple(
        query.query_id for query in dataset.queries if query.split == "test"
    )
    return build_run_manifest(
        eval_version=dataset.eval_version,
        dataset_sha256=dataset.dataset_sha256,
        frozen_test_sha256=dataset.frozen_test_sha256,
        corpus_sha256=dataset.corpus_sha256,
        artifact_sha256=dataset.artifact_sha256,
        aliases_sha256=dataset.aliases_sha256,
        query_ids=query_ids,
        generation_config=DEEPSEEK_GENERATION_CONFIG,
        judge_config=DEEPSEEK_JUDGE_CONFIG,
        max_retries=MAX_RETRIES,
        limits=limits,
        implementation_provenance=implementation_provenance,
        created_at=created_at,
        run_type="deepseek_generation_judge",
        generation_prompt_version=GENERATION_PROMPT_VERSION,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
    )


def prepare_generation_preflight(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    retrieval_result_path: str | Path = DEFAULT_RETRIEVAL_RESULT_PATH,
    output_dir: str | Path = DEFAULT_GENERATION_OUTPUT_DIR,
    implementation_provenance: dict[str, Any] | None = None,
    key_configured: bool,
    created_at: str | None = None,
    limits: RunLimits = DEFAULT_REAL_RUN_LIMITS,
) -> dict[str, Any]:
    """Validate frozen inputs and write a zero-call execution plan."""

    dataset, retrieval = _load_bound_inputs(dataset_path, retrieval_result_path)
    output = Path(output_dir)
    timestamp = created_at or datetime.now().astimezone().isoformat(timespec="seconds")
    provenance = implementation_provenance or capture_implementation_provenance()
    manifest = _manifest(
        dataset,
        implementation_provenance=provenance,
        created_at=timestamp,
        limits=limits,
    )
    ensure_run_manifest(output / "generation-run-manifest.json", manifest)
    test_count = sum(query.split == "test" for query in dataset.queries)
    estimates = _token_cost_plan(dataset, retrieval)
    plan = {
        "schema_version": 2,
        "mode": "preflight_only",
        "api_key_environment": "DEEPSEEK_API_KEY",
        "api_key_status": "configured" if key_configured else "not_configured",
        "api_key_value_recorded": False,
        "api_calls_executed": 0,
        "eval_version": dataset.eval_version,
        "dataset_sha256": dataset.dataset_sha256,
        "frozen_test_sha256": dataset.frozen_test_sha256,
        "corpus_sha256": dataset.corpus_sha256,
        "artifact_sha256": dataset.artifact_sha256,
        "strategies": list(STRATEGIES),
        "test_queries_per_strategy": test_count,
        "logical_generation_requests": test_count * len(STRATEGIES),
        "logical_judge_requests": test_count * len(STRATEGIES),
        "maximum_attempts_with_retries": limits.max_attempts,
        "hard_limits": {
            "input_tokens": limits.max_input_tokens,
            "output_tokens": limits.max_output_tokens,
            "cost_usd": limits.max_cost_usd,
        },
        "models": {
            "generation": DEEPSEEK_GENERATION_CONFIG.model,
            "judge": DEEPSEEK_JUDGE_CONFIG.model,
        },
        "thinking": {"type": "disabled"},
        "pricing_snapshot": _pricing_snapshot(),
        **estimates,
        "execute_requires_explicit_flag": True,
        "run_config_sha256": manifest["run_config_sha256"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation-preflight.json").write_bytes(canonical_json_bytes(plan))
    return plan


def execute_generation_run(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    retrieval_result_path: str | Path = DEFAULT_RETRIEVAL_RESULT_PATH,
    output_dir: str | Path = DEFAULT_GENERATION_OUTPUT_DIR,
    implementation_provenance: dict[str, Any] | None = None,
    api_key: str,
    transport: JsonTransport | None = None,
    created_at: str | None = None,
    limits: RunLimits = DEFAULT_REAL_RUN_LIMITS,
) -> dict[str, Any]:
    """Execute only after an explicit caller supplies a configured credential."""

    if not api_key:
        raise ValueError("DeepSeek API key is not configured")
    output = Path(output_dir)
    provenance = implementation_provenance or capture_implementation_provenance()
    timestamp = created_at or datetime.now().astimezone().isoformat(timespec="seconds")
    prepare_generation_preflight(
        dataset_path=dataset_path,
        retrieval_result_path=retrieval_result_path,
        output_dir=output,
        implementation_provenance=provenance,
        key_configured=True,
        created_at=timestamp,
        limits=limits,
    )
    dataset, retrieval = _load_bound_inputs(dataset_path, retrieval_result_path)
    manifest = _manifest(
        dataset,
        implementation_provenance=provenance,
        created_at=timestamp,
        limits=limits,
    )
    client = DeepSeekChatCompletionsClient(api_key=api_key, transport=transport)
    runner = GenerationJudgeRunner(
        client,
        client,
        DEEPSEEK_GENERATION_CONFIG,
        DEEPSEEK_JUDGE_CONFIG,
        max_retries=MAX_RETRIES,
        limits=limits,
        run_manifest=manifest,
        manifest_path=output / "generation-run-manifest.json",
        attempt_ledger_path=output / "attempt-ledger.jsonl",
    )
    strategy_summaries = {}
    for strategy in STRATEGIES:
        cases, contexts = prepare_test_generation_inputs(dataset, retrieval, strategy)
        records = runner.run(
            cases,
            contexts,
            strategy,
            output / f"{strategy}-results.jsonl",
        )
        strategy_summaries[strategy] = {
            "records": len(records),
            "usage": summarize_usage(records),
            "quality": summarize_generation_quality(records),
        }
    ledger = AttemptLedger(
        output / "attempt-ledger.jsonl", limits, manifest["run_config_sha256"]
    )
    summary = {
        "schema_version": 1,
        "run_config_sha256": manifest["run_config_sha256"],
        "strategies": strategy_summaries,
        "ledger": ledger.totals(),
    }
    (output / "generation-run-summary.json").write_bytes(canonical_json_bytes(summary))
    return summary
