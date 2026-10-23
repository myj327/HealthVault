"""TASK-027 deterministic RAG support Gate and zero-API recomputation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from healthvault.application.knowledge_support_gate import (
    SUPPORT_GATE_RULES,
    assess_health_knowledge_evidence,
    assess_health_knowledge_support,
    search_with_health_knowledge_gate,
)
from healthvault.evaluation.rag.metrics import aggregate_retrieval_metrics
from healthvault.evaluation.rag.task_026 import deterministic_query_rewrite
from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    SQLiteFTSHealthKnowledgeRepository,
)
from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT


TASK_VERSION = "task-027-gate-v3"
EVAL_VERSION = "rag-eval-r1-v3-human-rewrite-gate-v3"
TOP_K = 5
ROOT = Path(__file__).parents[3]
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "task-027"
DEFAULT_GOLD_PATH = Path(__file__).parent / "task-026/task-026-human-reviewed-gold.json"
DEFAULT_OLD_RANKINGS_PATH = Path(__file__).parent / "task-026/task-026-rankings.json"
DEFAULT_OLD_RESULT_PATH = Path(__file__).parent / "task-026/task-026-retrieval-results.json"
DEFAULT_DEV_CASES_PATH = Path(__file__).parent / "task-027/gate-dev-cases.json"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _render_summary(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return "\n".join(
        (
            "# TASK-027 RAG support Gate recomputation",
            "",
            "- Label: iterative evaluation after test exposure; not a new untouched held-out test",
            f"- Answerable Top-5 unchanged: {str(result['answerable_top5_unchanged']).lower()} ({result['answerable_top5_checked']}/16)",
            f"- Recall@5: {metrics['recall_at_5']}",
            f"- MRR@5: {metrics['mrr_at_5']}",
            f"- Unanswerable false-positive rate: {metrics['unanswerable_false_positive_rate']}",
            f"- Blocked by Gate: {result['gate_blocked_count']}/{result['query_count']}",
            f"- Pre/post retrieval blocks: {result['pre_gate_blocked_count']}/{result['post_gate_blocked_count']}",
            "- API calls / input tokens / output tokens / cost USD: 0 / 0 / 0 / 0.0",
            "",
        )
    )


def run_task_027(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    gold_path: str | Path = DEFAULT_GOLD_PATH,
    old_rankings_path: str | Path = DEFAULT_OLD_RANKINGS_PATH,
    old_result_path: str | Path = DEFAULT_OLD_RESULT_PATH,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
) -> dict[str, Any]:
    """Run one fresh rewrite+Gate retrieval against the signed TASK-026 Gold."""

    gold = _load(gold_path)
    old_rankings = _load(old_rankings_path)
    old_result = _load(old_result_path)
    if gold.get("gold_status") != "signed" or len(gold.get("queries", ())) != 24:
        raise ValueError("TASK-027 requires the signed 24-query TASK-026 Gold")
    old_by_id = {item["query_id"]: item for item in old_rankings["queries"]}
    root = Path(corpus_root)
    repository = SQLiteFTSHealthKnowledgeRepository(
        artifact_path=root / "chunks/hybrid_v1.json",
        aliases_path=root / "aliases/aliases.json",
    )
    query_results: list[dict[str, Any]] = []
    try:
        for query in gold["queries"]:
            raw_query = query["query"]
            rewritten = deterministic_query_rewrite(raw_query)
            decision, hits = search_with_health_knowledge_gate(
                raw_query,
                repository,
                TOP_K,
                retrieval_query=rewritten,
            )
            query_results.append(
                {
                    "query_id": query["query_id"],
                    "answerable": query["answerable"],
                    "status": "ok" if hits else "empty",
                    "gate_allowed": decision.allowed,
                    "gate_reason_code": decision.reason_code,
                    "gate_stage": decision.stage,
                    "citation_ids": [hit.document.citation_id for hit in hits],
                    "scores": [hit.score for hit in hits],
                }
            )
        repository_config = dict(repository.index_metadata)
    finally:
        repository.close()

    result_by_id = {item["query_id"]: item for item in query_results}
    mismatches = []
    answerable_count = 0
    for query in gold["queries"]:
        if not query["answerable"]:
            continue
        answerable_count += 1
        current = result_by_id[query["query_id"]]
        previous = old_by_id[query["query_id"]]["variant"]
        if (
            current["citation_ids"] != previous["citation_ids"]
            or current["scores"] != previous["scores"]
        ):
            mismatches.append(query["query_id"])
    if mismatches:
        raise ValueError("Gate changed answerable Top-5 retrieval")

    gold_by_id = {query["query_id"]: query for query in gold["queries"]}
    aggregate = aggregate_retrieval_metrics(
        (set(gold_by_id[item["query_id"]]["qrels"]), item["citation_ids"])
        for item in query_results
    )
    metrics = {
        key: aggregate[key]
        for key in (
            "query_count",
            "answerable_count",
            "unanswerable_count",
            "recall_at_5",
            "mrr_at_5",
            "unanswerable_false_positive_rate",
        )
    }
    old_variant = old_result["arms"]["variant"]["metrics"]
    if (
        metrics["recall_at_5"] != old_variant["recall_at_5"]
        or metrics["mrr_at_5"] != old_variant["mrr_at_5"]
    ):
        raise ValueError("Gate changed answerable retrieval metrics")

    gate_source_sha256 = _sha256(
        inspect.getsource(search_with_health_knowledge_gate).encode("utf-8")
        + inspect.getsource(assess_health_knowledge_support).encode("utf-8")
        + inspect.getsource(assess_health_knowledge_evidence).encode("utf-8")
    )
    rules_sha256 = _sha256(_canonical_bytes(SUPPORT_GATE_RULES))
    result = {
        "schema_version": 1,
        "task_version": TASK_VERSION,
        "eval_version": EVAL_VERSION,
        "status": "completed_zero_api",
        "validity_label": "iterative_evaluation_after_test_exposure",
        "query_count": len(query_results),
        "corpus": gold["corpus"],
        "repository": repository_config,
        "top_k": TOP_K,
        "query_rewrite_unchanged": True,
        "answerable_top5_unchanged": not mismatches,
        "answerable_top5_checked": answerable_count,
        "answerable_top5_mismatches": mismatches,
        "gate_blocked_count": sum(not item["gate_allowed"] for item in query_results),
        "pre_gate_blocked_count": sum(
            item["gate_stage"] == "pre_retrieval" for item in query_results
        ),
        "post_gate_blocked_count": sum(
            item["gate_stage"] == "post_retrieval" for item in query_results
        ),
        "gate_reason_counts": {
            reason: sum(item["gate_reason_code"] == reason for item in query_results)
            for reason in sorted(
                {item["gate_reason_code"] for item in query_results if item["gate_reason_code"]}
            )
        },
        "metrics": metrics,
        "rewrite_to_gate_delta_percentage_points": {
            "recall_at_5": round((metrics["recall_at_5"] - old_variant["recall_at_5"]) * 100, 2),
            "mrr_at_5": round((metrics["mrr_at_5"] - old_variant["mrr_at_5"]) * 100, 2),
            "unanswerable_false_positive_rate": round(
                (metrics["unanswerable_false_positive_rate"] - old_variant["unanswerable_false_positive_rate"]) * 100,
                2,
            ),
        },
        "gate_rules_sha256": rules_sha256,
        "gate_code_sha256": gate_source_sha256,
        "source_hashes": {
            "gold_sha256": _sha256(Path(gold_path).read_bytes()),
            "old_rankings_sha256": _sha256(Path(old_rankings_path).read_bytes()),
            "old_result_sha256": _sha256(Path(old_result_path).read_bytes()),
            "dev_cases_sha256": _sha256(DEFAULT_DEV_CASES_PATH.read_bytes()),
        },
        "queries": query_results,
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "task-027-v3-retrieval-results.json"
    summary_path = output / "task-027-v3-retrieval-summary.md"
    result_path.write_bytes(_canonical_bytes(result))
    summary_path.write_text(_render_summary(result), encoding="utf-8", newline="\n")
    config = {
        "task_version": TASK_VERSION,
        "eval_version": EVAL_VERSION,
        "top_k": TOP_K,
        "corpus": gold["corpus"],
        "gate_rules_sha256": rules_sha256,
        "gate_code_sha256": gate_source_sha256,
        "query_rewrite": "task-026 deterministic_query_rewrite unchanged",
        "dev_cases_sha256": _sha256(DEFAULT_DEV_CASES_PATH.read_bytes()),
    }
    manifest = {
        "schema_version": 3,
        "status": "completed_zero_api",
        "run_config_sha256": _sha256(_canonical_bytes(config)),
        "gate_rules_sha256": rules_sha256,
        "gate_code_sha256": gate_source_sha256,
        "files": {
            result_path.name: _sha256(result_path.read_bytes()),
            summary_path.name: _sha256(summary_path.read_bytes()),
        },
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    (output / "task-027-v3-run-manifest.json").write_bytes(_canonical_bytes(manifest))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TASK-027 zero-API RAG Gate evaluation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = run_task_027(args.output)
    print(
        f"status={result['status']} api_calls=0 "
        f"answerable_top5_unchanged={str(result['answerable_top5_unchanged']).lower()} "
        f"unanswerable_false_positive_rate={result['metrics']['unanswerable_false_positive_rate']}"
    )
    print(f"result={args.output / 'task-027-v3-retrieval-results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
