"""Registration and dispatch for HealthVault domain tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


ToolDefinition = dict[str, Any]
ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


class ToolRegistry:
    """Keep HealthVault tool definitions and handlers together by name."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolHandler]] = {}

    @property
    def definitions(self) -> list[ToolDefinition]:
        """Return registered definitions in registration order."""
        return [definition for definition, _handler in self._tools.values()]

    def register(
        self,
        tool_definition: ToolDefinition,
        handler: ToolHandler,
    ) -> None:
        """Register one tool, rejecting duplicate names immediately."""
        name = tool_definition["name"]
        if name in self._tools:
            raise ValueError(f"HealthVault tool already registered: {name}")
        self._tools[name] = (tool_definition, handler)

    def contains(self, name: str) -> bool:
        """Return whether a HealthVault handler is registered for ``name``."""
        return name in self._tools

    async def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        """Execute a registered handler or raise a clear unknown-tool error."""
        if name not in self._tools:
            raise KeyError(f"Unknown HealthVault tool: {name}")
        _definition, handler = self._tools[name]
        return await handler(tool_input)
