"""Tool definition and handler for offline health knowledge search."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from healthvault.application.search_health_knowledge import (
    search_health_knowledge,
)
from healthvault.repositories.health_knowledge import HealthKnowledgeRepository
from healthvault.tools.directory import tool_definition


SEARCH_HEALTH_KNOWLEDGE_TOOL = tool_definition("search_health_knowledge")


def build_search_health_knowledge_handler(
    repository: HealthKnowledgeRepository,
) -> Callable[[Any], Awaitable[str]]:
    """Bind a caller-supplied Repository to the async search handler."""

    async def handler(tool_input: Any) -> str:
        result = search_health_knowledge(tool_input, repository)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    return handler
