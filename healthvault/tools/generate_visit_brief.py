"""Tool definition and handler for deterministic visit preparation briefs."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from healthvault.application.generate_visit_brief import generate_visit_brief
from healthvault.repositories.health_knowledge import HealthKnowledgeRepository
from healthvault.repositories.health_records import HealthRecordRepository
from healthvault.tools.directory import tool_definition


GENERATE_VISIT_BRIEF_TOOL = tool_definition("generate_visit_brief")


def build_generate_visit_brief_handler(
    record_repository: HealthRecordRepository,
    knowledge_repository: HealthKnowledgeRepository,
) -> Callable[[Any], Awaitable[str]]:
    """Bind both Repository ports to the async visit brief handler."""

    async def handler(tool_input: Any) -> str:
        result = generate_visit_brief(
            tool_input,
            record_repository,
            knowledge_repository,
        )
        return json.dumps(result.to_dict(), ensure_ascii=False)

    return handler
