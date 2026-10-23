"""Optional integration switches for application-specific Agent runtimes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRuntimePolicy:
    """Keep BearCode defaults while allowing an application to opt out."""

    mcp_enabled: bool = True
    memory_enabled: bool = True
    global_skills_enabled: bool = True
    session_persistence_enabled: bool = True
    allowed_tool_names: frozenset[str] | None = None
