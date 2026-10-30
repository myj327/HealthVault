"""Tool interface for application-injected tools; no filesystem or shell tools."""
from __future__ import annotations

ToolDef = dict
tool_definitions: list[ToolDef] = []
# Applications may explicitly opt safe tools into concurrency; default is sequential.
CONCURRENCY_SAFE_TOOLS: set[str] = set()


def get_active_tool_definitions(all_tools: list[ToolDef] | None = None) -> list[ToolDef]:
    return list(all_tools or [])


def check_permission(tool_name: str, inp: dict, mode: str = "default") -> dict:
    """Domain authorization lives in the application policy and injected executor."""
    return {"action": "allow"}


async def execute_tool(name: str, inp: dict, read_file_state: dict | None = None) -> str:
    return f"Tool unavailable: {name}"
