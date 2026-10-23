"""Tool definition and repository-bound handler for record confirmation."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.repositories.health_records import HealthRecordRepository
from healthvault.tools.directory import tool_definition


CONFIRM_HEALTH_RECORDS_TOOL = tool_definition("confirm_health_records")


def build_confirm_health_records_handler(
    repository: HealthRecordRepository,
) -> Callable[[Any], Awaitable[str]]:
    """Bind a caller-supplied Repository to the async tool handler."""

    async def handler(tool_input: Any) -> str:
        result = confirm_health_records(tool_input, repository)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    return handler
