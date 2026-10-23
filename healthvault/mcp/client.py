"""MCP client with exact HealthVault server and read-only tool allowlists."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable

from agents.mcp_client import McpConnection
from healthvault.mcp.server import PROFILE_PATTERN, PROFILE_TOOLS, READ_ONLY_TOOLS, SERVER_NAME


APPROVED_SERVER_TOOLS = {SERVER_NAME: frozenset(READ_ONLY_TOOLS)}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SafeMcpConnection(McpConnection):
    """Register a JSON-RPC waiter before a fast local server can answer."""

    async def _read_loop(self) -> None:
        try:
            await super()._read_loop()
        finally:
            pending = self._pending
            self._pending = {}
            for future in pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("HealthVault MCP server disconnected."))

    async def _send_request(self, method: str, params: dict | None = None) -> Any:
        assert self._process and self._process.stdin
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        })
        try:
            self._process.stdin.write((message + "\n").encode("utf-8"))
            await self._process.stdin.drain()
        except Exception:
            self._pending.pop(request_id, None)
            future.cancel()
            raise
        return await future


ConnectionFactory = Callable[[str, str, list[str], dict[str, str]], McpConnection]


class ApprovedMCPClient:
    """Connect only one explicitly configured local server for one profile/DB."""

    def __init__(
        self,
        server_configs: dict[str, dict[str, Any]],
        *,
        profile_id: str,
        database_path: str | Path,
        connection_factory: ConnectionFactory = SafeMcpConnection,
    ) -> None:
        if not PROFILE_PATTERN.fullmatch(profile_id):
            raise ValueError("MCP client profile_id is invalid.")
        self.profile_id = profile_id
        self.database_path = Path(database_path).resolve()
        if not self.database_path.is_file():
            raise ValueError("MCP client database must already exist.")
        self.server_configs = server_configs
        self.connection_factory = connection_factory
        self.connections: dict[str, McpConnection] = {}
        self.routes: dict[str, tuple[str, str, dict[str, Any]]] = {}

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        *,
        profile_id: str,
        database_path: str | Path,
        connection_factory: ConnectionFactory = SafeMcpConnection,
    ) -> "ApprovedMCPClient":
        """Read only the path explicitly supplied by the caller."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        servers = raw.get("mcpServers")
        if not isinstance(servers, dict):
            raise ValueError("Opt-in config must contain an mcpServers object.")
        return cls(
            servers,
            profile_id=profile_id,
            database_path=database_path,
            connection_factory=connection_factory,
        )

    def _validated_config(self, config: Any) -> tuple[str, list[str]]:
        if not isinstance(config, dict):
            raise ValueError("Approved MCP server config must be an object.")
        command = config.get("command")
        args = config.get("args")
        if not isinstance(command, str) or not isinstance(args, list):
            raise ValueError("Approved MCP command and args are required.")
        if Path.cwd().resolve() != PROJECT_ROOT:
            raise ValueError("Start the MCP client from the BearCode project root.")
        if Path(command).resolve() != Path(sys.executable).resolve():
            raise ValueError("Approved MCP command must be the current Python interpreter.")
        if len(args) != 7 or args[:6] != [
            "-B", "-m", "healthvault.mcp.server",
            "--profile-id", self.profile_id, "--database",
        ]:
            raise ValueError("Approved MCP args must bind the exact module and profile.")
        if not isinstance(args[6], str) or Path(args[6]).resolve() != self.database_path:
            raise ValueError("Approved MCP database does not match runtime database.")
        if config.get("env", {}) != {}:
            raise ValueError("Approved MCP config cannot override environment variables.")
        return command, args

    async def connect(self) -> None:
        for server_name, allowed_tools in APPROVED_SERVER_TOOLS.items():
            config = self.server_configs.get(server_name)
            if config is None:
                continue
            command, args = self._validated_config(config)
            connection = self.connection_factory(server_name, command, args, {})
            try:
                await asyncio.wait_for(connection.connect(), timeout=15)
                await asyncio.wait_for(connection.initialize(), timeout=15)
                advertised = await asyncio.wait_for(connection.list_tools(), timeout=15)
            except Exception:
                await self._close(connection)
                raise
            self.connections[server_name] = connection
            for tool in advertised:
                name = tool.get("name")
                if name in allowed_tools:
                    prefixed = f"mcp__{server_name}__{name}"
                    self.routes[prefixed] = (server_name, name, tool)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": prefixed,
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema") or {
                    "type": "object", "properties": {},
                },
            }
            for prefixed, (_server, _name, tool) in self.routes.items()
        ]

    async def call_tool(self, prefixed_name: str, arguments: dict[str, Any]) -> str:
        route = self.routes.get(prefixed_name)
        if route is None:
            raise KeyError(f"MCP tool is not approved: {prefixed_name}")
        if not isinstance(arguments, dict):
            raise ValueError("MCP tool arguments must be an object.")
        server_name, tool_name, _tool = route
        claimed_profile = arguments.get("profile_id")
        if (
            tool_name in PROFILE_TOOLS
            and claimed_profile is not None
            and claimed_profile != self.profile_id
        ):
            raise ValueError("MCP tool profile does not match runtime context.")
        try:
            return await asyncio.wait_for(
                self.connections[server_name].call_tool(tool_name, arguments),
                timeout=30,
            )
        except asyncio.TimeoutError as exc:
            await self.disconnect_all()
            raise RuntimeError("HealthVault MCP tool timed out.") from exc

    @staticmethod
    async def _close(connection: McpConnection) -> None:
        process = getattr(connection, "_process", None)
        reader_task = getattr(connection, "_reader_task", None)
        connection.close()
        if reader_task is not None:
            await asyncio.gather(reader_task, return_exceptions=True)
        if process is not None:
            await asyncio.wait_for(process.wait(), timeout=5)

    async def disconnect_all(self) -> None:
        for connection in self.connections.values():
            await self._close(connection)
        self.connections.clear()
        self.routes.clear()
