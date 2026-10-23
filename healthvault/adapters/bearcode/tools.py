"""Connect HealthVault domain tools to BearCode's tool interface."""

from __future__ import annotations

import copy
from collections.abc import Collection
from typing import Any

from agents.tools import ToolDef, execute_tool, tool_definitions
from healthvault.runtime_context import RuntimeContext
from healthvault.tools.directory import PROFILE_BOUND_TOOL_NAMES
from healthvault.tools.registry import ToolRegistry


CONTEXT_PROFILE_TOOLS = frozenset(PROFILE_BOUND_TOOL_NAMES)


class BearCodeToolAdapter:
    """Combine tool definitions and route calls to the correct executor."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        include_native_tools: bool = True,
        allowed_domain_tools: Collection[str] | None = None,
        runtime_context: RuntimeContext | None = None,
    ) -> None:
        native_names = {definition["name"] for definition in tool_definitions}
        domain_definitions = registry.definitions
        registered_domain_names = {
            definition["name"] for definition in domain_definitions
        }
        if allowed_domain_tools is None:
            exposed_domain_names = registered_domain_names
        else:
            exposed_domain_names = set(allowed_domain_tools)
            missing_names = exposed_domain_names - registered_domain_names
            if missing_names:
                names = ", ".join(sorted(missing_names))
                raise ValueError(
                    "Unknown HealthVault domain tools in allowlist: "
                    f"{names}"
                )
            domain_definitions = [
                definition
                for definition in domain_definitions
                if definition["name"] in exposed_domain_names
            ]

        if runtime_context is not None:
            domain_definitions = copy.deepcopy(domain_definitions)
            for definition in domain_definitions:
                if definition["name"] not in CONTEXT_PROFILE_TOOLS:
                    continue
                schema = definition["input_schema"]
                schema.get("properties", {}).pop("profile_id", None)
                if "required" in schema:
                    schema["required"] = [
                        name
                        for name in schema["required"]
                        if name != "profile_id"
                    ]

        conflicts = native_names & exposed_domain_names
        if conflicts:
            conflict_names = ", ".join(sorted(conflicts))
            raise ValueError(
                "HealthVault tool name conflicts with BearCode tool: "
                f"{conflict_names}"
            )

        self._registry = registry
        self._include_native_tools = include_native_tools
        self._exposed_domain_names = exposed_domain_names
        self._runtime_context = runtime_context
        self.combined_definitions: list[ToolDef] = (
            (list(tool_definitions) if include_native_tools else [])
            + domain_definitions
        )

    async def execute(
        self,
        name: str,
        tool_input: dict[str, Any],
        read_file_state: dict[str, float] | None = None,
    ) -> str:
        """Route domain calls to the registry and all others to BearCode."""
        if name in self._exposed_domain_names:
            prepared_input = dict(tool_input)
            if (
                self._runtime_context is not None
                and name in CONTEXT_PROFILE_TOOLS
            ):
                if "profile_id" in prepared_input:
                    return (
                        "Tool input rejected: profile_id is supplied by "
                        "RuntimeContext."
                    )
                prepared_input["profile_id"] = self._runtime_context.profile_id
            return await self._registry.execute(name, prepared_input)
        if not self._include_native_tools:
            return f"Tool unavailable in isolated HealthVault adapter: {name}"
        return await execute_tool(name, tool_input, read_file_state)
