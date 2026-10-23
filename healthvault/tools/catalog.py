"""Bind the authoritative HealthVault tool directory to runtime handlers."""

from collections.abc import Collection

from healthvault.tools.registry import ToolRegistry
from healthvault.repositories.health_knowledge import HealthKnowledgeRepository
from healthvault.repositories.health_records import HealthRecordRepository
from healthvault.tools.directory import TOOL_SPECS


def build_healthvault_tool_registry(
    record_repository: HealthRecordRepository | None = None,
    knowledge_repository: HealthKnowledgeRepository | None = None,
    *,
    allowed_names: Collection[str] | None = None,
) -> ToolRegistry:
    """Build an isolated registry containing implemented domain tools."""
    from healthvault.tools.preview_health_report import preview_health_report
    from healthvault.tools.confirm_health_records import build_confirm_health_records_handler
    from healthvault.tools.query_health_timeline import build_query_health_timeline_handler
    from healthvault.tools.compare_health_indicators import build_compare_health_indicators_handler
    from healthvault.tools.search_health_knowledge import build_search_health_knowledge_handler
    from healthvault.tools.search_official_health_web import build_search_official_health_web_handler
    from healthvault.tools.generate_visit_brief import build_generate_visit_brief_handler

    handlers = {"preview_health_report": preview_health_report}
    if record_repository is not None:
        handlers.update({
            "confirm_health_records": build_confirm_health_records_handler(record_repository),
            "query_health_timeline": build_query_health_timeline_handler(record_repository),
            "compare_health_indicators": build_compare_health_indicators_handler(record_repository),
        })
    if knowledge_repository is not None:
        handlers.update({
            "search_health_knowledge": build_search_health_knowledge_handler(knowledge_repository),
            "search_official_health_web": build_search_official_health_web_handler(knowledge_repository),
        })
        if record_repository is not None:
            handlers["generate_visit_brief"] = build_generate_visit_brief_handler(
                record_repository, knowledge_repository,
            )

    requested = set(allowed_names) if allowed_names is not None else set(handlers)
    unknown = requested - {spec.name for spec in TOOL_SPECS}
    unavailable = requested - set(handlers)
    if unknown:
        raise ValueError(f"Unknown HealthVault tools: {', '.join(sorted(unknown))}")
    if unavailable:
        raise ValueError(
            "HealthVault tool dependencies are unavailable: "
            + ", ".join(sorted(unavailable))
        )
    registry = ToolRegistry()
    for spec in TOOL_SPECS:
        if spec.name in requested:
            registry.register(spec.definition(), handlers[spec.name])
    return registry
