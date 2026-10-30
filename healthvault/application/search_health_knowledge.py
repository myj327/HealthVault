"""Application service for curated offline health knowledge search."""

from __future__ import annotations

from typing import Any

from healthvault.domain.knowledge import (
    KnowledgeSearchIssue,
    KnowledgeSearchResult,
)
from healthvault.repositories.health_knowledge import HealthKnowledgeRepository
from healthvault.application.knowledge_support_gate import (
    assess_health_knowledge_support,
)


def _invalid(
    code: str,
    message: str,
    *,
    query: str | None = None,
    limit: int | None = 3,
) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        status="invalid_request",
        query=query,
        limit=limit,
        hits=(),
        issues=(KnowledgeSearchIssue(code, message),),
    )


def search_health_knowledge(
    tool_input: Any,
    repository: HealthKnowledgeRepository,
) -> KnowledgeSearchResult:
    """Validate one query and return deterministic local source cards."""
    if not isinstance(tool_input, dict):
        return _invalid("invalid_request", "Tool input must be an object.")
    if "query" not in tool_input:
        return _invalid(
            "missing_required_field",
            "Missing required field: query.",
        )
    query = tool_input["query"]
    if not isinstance(query, str) or not query.strip():
        return _invalid(
            "invalid_query",
            "query must be a non-empty string.",
        )

    limit = tool_input.get("limit", 3)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5:
        return _invalid(
            "invalid_limit",
            "limit must be an integer from 1 to 5.",
            query=query,
            limit=None,
        )

    from healthvault.application.query_rewrite import prepare_query
    prepared = prepare_query(query)
    if prepared.status == "needs_clarification":
        return _invalid("ambiguous_query", "Specify the indicator or supply explicit conversation context.", query=query, limit=limit)
    decision = assess_health_knowledge_support(prepared.search_query)
    # Search returns candidate sources. Whole-answer checking determines support;
    # conversational query words need not appear literally in English source text.
    hits = repository.search(prepared.search_query, limit) if decision.allowed else ()
    if not decision.allowed:
        return KnowledgeSearchResult(
            status="empty",
            query=query,
            limit=limit,
            hits=(),
            issues=(
                KnowledgeSearchIssue(
                    decision.reason_code or "unsupported_query",
                    decision.message or "The query is outside the supported corpus.",
                ),
            ),
        )
    return KnowledgeSearchResult(
        status="ok" if hits else "empty",
        query=query,
        limit=limit,
        hits=hits,
        issues=(),
    )
