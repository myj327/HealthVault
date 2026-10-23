"""Deterministic natural-section versus hybrid retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from healthvault.evaluation.rag.dataset import QueryRelevanceDataset, RagQuery
from healthvault.evaluation.rag.metrics import aggregate_retrieval_metrics
from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    SQLiteFTSHealthKnowledgeRepository,
)
from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT


DEFAULT_K = 5
TASK_BASELINE_HEAD = "320254658709b05e285d7d547e52881cccb0c679"


def _group_queries(queries: tuple[RagQuery, ...]) -> dict[str, tuple[RagQuery, ...]]:
    groups: dict[str, tuple[RagQuery, ...]] = {"overall": queries}
    dimensions: tuple[tuple[str, Callable[[RagQuery], str]], ...] = (
        ("split", lambda query: query.split),
        ("answerable", lambda query: str(query.answerable).lower()),
        ("language", lambda query: query.language),
        ("category", lambda query: query.category),
        ("rewrite_category", lambda query: query.rewrite_category),
    )
    for dimension, getter in dimensions:
        for value in sorted({getter(query) for query in queries}):
            groups[f"{dimension}/{value}"] = tuple(
                query for query in queries if getter(query) == value
            )
    return groups


def evaluate_retrieval_strategy(
    dataset: QueryRelevanceDataset,
    strategy: str,
    *,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    if strategy not in {"natural_section", "hybrid_v1"}:
        raise ValueError("Unsupported retrieval strategy")
    root = Path(corpus_root)
    repository = SQLiteFTSHealthKnowledgeRepository(
        artifact_path=root / "chunks" / f"{strategy}.json",
        aliases_path=root / "aliases" / "aliases.json",
    )
    try:
        query_results = []
        ranked_by_id = {}
        for query in dataset.queries:
            hits = repository.search(query.query, k)
            ranked = tuple(hit.document.chunk_id for hit in hits)
            ranked_by_id[query.query_id] = ranked
            query_results.append(
                {
                    "query_id": query.query_id,
                    "split": query.split,
                    "answerable": query.answerable,
                    "ranked_chunk_ids": list(ranked),
                    "scores": [hit.score for hit in hits],
                    "expected_chunk_ids": list(query.relevance[strategy].chunk_ids),
                }
            )
    finally:
        repository.close()
    groups = {}
    for group_name, group_queries in _group_queries(dataset.queries).items():
        groups[group_name] = aggregate_retrieval_metrics(
            (
                set(query.relevance[strategy].chunk_ids),
                ranked_by_id[query.query_id],
            )
            for query in group_queries
        )
    return {
        "strategy": strategy,
        "k": k,
        "repository": {
            "engine": "fts5",
            "ranking": "bm25",
            "tokenizer": "trigram",
            "alias_sha256": dataset.aliases_sha256,
        },
        "groups": groups,
        "queries": query_results,
    }


def _metric_deltas(
    natural: dict[str, Any], hybrid: dict[str, Any]
) -> dict[str, dict[str, float | int | None]]:
    deltas = {}
    for group_name in natural["groups"]:
        natural_metrics = natural["groups"][group_name]
        hybrid_metrics = hybrid["groups"][group_name]
        group_delta = {}
        for metric, natural_value in natural_metrics.items():
            hybrid_value = hybrid_metrics[metric]
            if (
                isinstance(natural_value, (int, float))
                and not isinstance(natural_value, bool)
                and isinstance(hybrid_value, (int, float))
                and not isinstance(hybrid_value, bool)
            ):
                group_delta[metric] = round(hybrid_value - natural_value, 6)
            else:
                group_delta[metric] = None
        deltas[group_name] = group_delta
    return deltas


def compare_retrieval_strategies(
    dataset: QueryRelevanceDataset,
    *,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    natural = evaluate_retrieval_strategy(
        dataset, "natural_section", corpus_root=corpus_root, k=k
    )
    hybrid = evaluate_retrieval_strategy(
        dataset, "hybrid_v1", corpus_root=corpus_root, k=k
    )
    return {
        "schema_version": 1,
        "validity_status": dataset.validity_status,
        "supersedes": {
            "eval_version": "rag-eval-r1-v1",
            "metrics_status": "invalidated",
            "reason": "held-out leakage and incomplete qrels",
        },
        "eval_version": dataset.eval_version,
        "frozen_on": dataset.frozen_on,
        "corpus_version": dataset.corpus_version,
        "corpus_sha256": dataset.corpus_sha256,
        "dataset_sha256": dataset.dataset_sha256,
        "frozen_test_sha256": dataset.frozen_test_sha256,
        "artifact_sha256": dataset.artifact_sha256,
        "evaluation_config": {
            "k": k,
            "engine": "sqlite_fts5",
            "ranking": "bm25",
            "tokenizer": "trigram",
            "metric_cutoffs": [1, 3, 5],
        },
        "run_command": (
            "python -m healthvault.evaluation.rag retrieval "
            "--dataset tests/fixtures/rag_eval/rag-eval-r1-v2.json "
            "--output healthvault/evaluation/rag/results"
        ),
        "natural_section": natural,
        "hybrid_v1": hybrid,
        "hybrid_minus_natural": _metric_deltas(natural, hybrid),
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_retrieval_run_manifest(
    result: dict[str, Any], *, executed_at: str, code_head: str
) -> dict[str, Any]:
    """Keep volatile run provenance separate from byte-stable metric results."""

    config_bytes = json.dumps(
        result["evaluation_config"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    result_bytes = canonical_json_bytes(result)
    return {
        "schema_version": 1,
        "run_type": "offline_retrieval",
        "executed_at": executed_at,
        "code_head": code_head,
        "eval_version": result["eval_version"],
        "dataset_sha256": result["dataset_sha256"],
        "frozen_test_sha256": result["frozen_test_sha256"],
        "corpus_sha256": result["corpus_sha256"],
        "artifact_sha256": result["artifact_sha256"],
        "configuration": result["evaluation_config"],
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "query_executions": {
            "successful": sum(
                len(result[strategy]["queries"])
                for strategy in ("natural_section", "hybrid_v1")
            ),
            "failed": 0,
        },
        "api_attempts": {"successful": 0, "failed": 0},
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }


def render_markdown_summary(result: dict[str, Any]) -> str:
    rows = []
    metrics = (
        "recall_at_1", "recall_at_3", "recall_at_5", "precision_at_3",
        "mrr_at_5", "ndcg_at_3", "no_answer_accuracy",
        "unanswerable_false_positive_rate",
    )
    for metric in metrics:
        natural = result["natural_section"]["groups"]["overall"][metric]
        hybrid = result["hybrid_v1"]["groups"]["overall"][metric]
        delta = result["hybrid_minus_natural"]["overall"][metric]
        rows.append(f"| {metric} | {natural} | {hybrid} | {delta} |")
    return "\n".join(
        (
            f"# RAG retrieval evaluation: {result['eval_version']}",
            "",
            f"- Frozen on: `{result['frozen_on']}`",
            f"- Corpus: `{result['corpus_version']}`",
            f"- Dataset SHA-256: `{result['dataset_sha256']}`",
            f"- Frozen test SHA-256: `{result['frozen_test_sha256']}`",
            "- Supersedes: `rag-eval-r1-v1`; all v1 metrics are invalidated and must not be used as quantitative conclusions",
            "- Retrieval: identical SQLite FTS5 / BM25 / trigram / alias configuration, k=5",
            "- API/model calls: 0",
            "",
            "| Metric | natural_section | hybrid_v1 | absolute delta |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            "Detailed overall/dev/test/language/category/rewrite groups and per-query rankings are in `retrieval-results.json`.",
            "",
        )
    )
