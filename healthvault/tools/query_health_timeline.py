"""Tool definition and handler for confirmed health timeline queries."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from healthvault.application.query_health_timeline import query_health_timeline
from healthvault.repositories.health_records import HealthRecordRepository
from healthvault.tools.directory import tool_definition


QUERY_HEALTH_TIMELINE_TOOL = tool_definition("query_health_timeline")


def build_query_health_timeline_handler(
    repository: HealthRecordRepository,
) -> Callable[[Any], Awaitable[str]]:
    """Bind a caller-supplied Repository to the async timeline handler."""

    async def handler(tool_input: Any) -> str:
        result = query_health_timeline(tool_input, repository)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    return handler
