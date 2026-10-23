"""Opt-in routing from a HealthVault runtime to its approved MCP client."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from healthvault.agent_runtime import HealthVaultAgentRuntime
from healthvault.mcp.client import ApprovedMCPClient
from healthvault.mcp.server import READ_ONLY_TOOLS, SERVER_NAME


class AttachedMCP:
    def __init__(
        self,
        runtime: HealthVaultAgentRuntime,
        client: ApprovedMCPClient,
        *,
        original_tools: list[dict[str, Any]],
        original_executor: Any,
        original_policy: Any,
        original_manifest: dict[str, Any],
    ) -> None:
        self.runtime = runtime
        self.client = client
        self.original_tools = original_tools
        self.original_executor = original_executor
        self.original_policy = original_policy
        self.original_manifest = original_manifest
        self.closed = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        agent = self.runtime.agent
        agent.tools = self.original_tools
        agent._tool_executor = self.original_executor
        agent.runtime_policy = self.original_policy
        for key in ("tool_names", "mcp_opt_in_enabled", "mcp_server_names"):
            if key in self.original_manifest:
                self.runtime.manifest[key] = self.original_manifest[key]
            else:
                self.runtime.manifest.pop(key, None)
        await self.client.disconnect_all()

    async def __aenter__(self) -> "AttachedMCP":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()


async def attach_approved_mcp(
    runtime: HealthVaultAgentRuntime,
    *,
    config_path: str | Path | None = None,
    server_configs: dict[str, dict[str, Any]] | None = None,
    allow_web_search: bool = False,
    mcp_only: bool = False,
) -> AttachedMCP:
    """Attach only explicit approved local server settings; generic MCP stays disabled.

    The caller must await this and then await attachment.close() in a finally
    block, or use "async with await attach_approved_mcp(...)".
    """
    agent = runtime.agent
    if agent.runtime_policy.mcp_enabled:
        raise ValueError("Generic BearCode MCP discovery must remain disabled.")
    allowed = agent.runtime_policy.allowed_tool_names
    if allowed is None:
        raise ValueError("HealthVault runtime requires an explicit tool allowlist.")
    if (config_path is None) == (server_configs is None):
        raise ValueError("Provide exactly one approved MCP configuration source.")
    client = (
        ApprovedMCPClient.from_config_file(
            config_path,
            profile_id=runtime.context.profile_id,
            database_path=runtime.components.record_repository.database_path,
        )
        if config_path is not None else
        ApprovedMCPClient(
            server_configs,
            profile_id=runtime.context.profile_id,
            database_path=runtime.components.record_repository.database_path,
        )
    )
    await client.connect()
    definitions = client.get_tool_definitions()
    expected = {f"mcp__{SERVER_NAME}__{name}" for name in READ_ONLY_TOOLS}
    exposed = {definition["name"] for definition in definitions}
    if exposed != expected:
        await client.disconnect_all()
        raise ValueError("Approved HealthVault MCP server did not expose all read-only tools.")
    if not allow_web_search:
        official_name = f"mcp__{SERVER_NAME}__search_official_health_web"
        definitions = [definition for definition in definitions if definition["name"] != official_name]
        exposed.remove(official_name)
    if exposed & {definition["name"] for definition in agent.tools}:
        await client.disconnect_all()
        raise ValueError("Approved MCP tool names conflict with existing Agent tools.")
    original_tools = agent.tools
    original_executor = agent._tool_executor
    original_policy = agent.runtime_policy
    original_manifest = dict(runtime.manifest)

    async def restricted_executor(
        name: str,
        arguments: dict[str, Any],
        read_file_state: dict[str, float] | None = None,
    ) -> str:
        if name in exposed:
            return await client.call_tool(name, arguments)
        if mcp_only:
            return f"Tool unavailable under Agent runtime policy: {name}"
        return await original_executor(name, arguments, read_file_state)

    agent.tools = definitions if mcp_only else list(original_tools) + definitions
    agent._tool_executor = restricted_executor
    agent.runtime_policy = replace(
        original_policy,
        allowed_tool_names=frozenset(exposed if mcp_only else set(allowed) | exposed),
    )
    runtime.manifest["tool_names"] = [
        definition["name"] for definition in agent.tools
    ]
    runtime.manifest["mcp_opt_in_enabled"] = True
    runtime.manifest["mcp_server_names"] = [SERVER_NAME]
    return AttachedMCP(
        runtime,
        client,
        original_tools=original_tools,
        original_executor=original_executor,
        original_policy=original_policy,
        original_manifest=original_manifest,
    )
