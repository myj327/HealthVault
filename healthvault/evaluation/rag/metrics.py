"""Small, deterministic information-retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def rank_scored_documents(scored: Iterable[tuple[str, float]]) -> tuple[str, ...]:
    """Sort by descending score then ID, retaining the best duplicate only."""

    best: dict[str, float] = {}
    for document_id, score in scored:
        if document_id not in best or score > best[document_id]:
            best[document_id] = score
    return tuple(
        document_id
        for document_id, _score in sorted(best.items(), key=lambda item: (-item[1], item[0]))
    )


def _recall(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def _precision(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / k


def _reciprocal_rank(relevant: set[str], ranked: Sequence[str]) -> float:
    for rank, document_id in enumerate(ranked, 1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, document_id in enumerate(ranked[:k], 1)
        if document_id in relevant
    )
    ideal_count = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def aggregate_retrieval_metrics(
    evaluations: Iterable[tuple[set[str], Sequence[str]]],
) -> dict[str, float | int | None]:
    """Aggregate answerable retrieval metrics and explicit no-answer metrics."""

    pairs = tuple((set(relevant), tuple(ranked)) for relevant, ranked in evaluations)
    answerable = tuple(pair for pair in pairs if pair[0])
    unanswerable = tuple(pair for pair in pairs if not pair[0])
    answerable_hits = sum(bool(relevant.intersection(ranked)) for relevant, ranked in answerable)
    no_answer_hits = sum(not ranked for _relevant, ranked in unanswerable)
    answerability_correct = answerable_hits + no_answer_hits
    return {
        "query_count": len(pairs),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "recall_at_1": _mean([_recall(rel, ranked, 1) for rel, ranked in answerable]),
        "recall_at_3": _mean([_recall(rel, ranked, 3) for rel, ranked in answerable]),
        "recall_at_5": _mean([_recall(rel, ranked, 5) for rel, ranked in answerable]),
        "precision_at_3": _mean([_precision(rel, ranked, 3) for rel, ranked in answerable]),
        "mrr_at_5": _mean(
            [_reciprocal_rank(rel, ranked[:5]) for rel, ranked in answerable]
        ),
        "ndcg_at_3": _mean([_ndcg(rel, ranked, 3) for rel, ranked in answerable]),
        "answerable_hit_rate_at_5": (
            round(answerable_hits / len(answerable), 6) if answerable else None
        ),
        "no_answer_accuracy": (
            round(no_answer_hits / len(unanswerable), 6) if unanswerable else None
        ),
        "unanswerable_false_positive_rate": (
            round(1.0 - no_answer_hits / len(unanswerable), 6)
            if unanswerable
            else None
        ),
        "answerability_accuracy": (
            round(answerability_correct / len(pairs), 6) if pairs else None
        ),
    }
