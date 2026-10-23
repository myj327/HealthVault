"""Tool definition and handler for deterministic indicator comparisons."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from healthvault.application.compare_health_indicators import (
    compare_health_indicators,
)
from healthvault.repositories.health_records import HealthRecordRepository
from healthvault.tools.directory import tool_definition


COMPARE_HEALTH_INDICATORS_TOOL = tool_definition("compare_health_indicators")


def build_compare_health_indicators_handler(
    repository: HealthRecordRepository,
) -> Callable[[Any], Awaitable[str]]:
    """Bind a caller-supplied Repository to the async comparison handler."""

    async def handler(tool_input: Any) -> str:
        result = compare_health_indicators(tool_input, repository)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    return handler
