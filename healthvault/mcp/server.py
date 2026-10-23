"""Explicit read-only MCP stdio server for one synthetic HealthVault profile."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO

from healthvault.infrastructure.local_health_knowledge import LocalHealthKnowledgeRepository
from healthvault.infrastructure.sqlite_health_records import (
    ReadOnlySQLiteHealthRecordRepository,
    SQLiteHealthRecordRepository,
)
from healthvault.tools.catalog import build_healthvault_tool_registry
from healthvault.tools.directory import MCP_TOOL_NAMES, PROFILE_BOUND_TOOL_NAMES
from healthvault.tools.registry import ToolRegistry
from healthvault.ui.controller import configured_database_path
from healthvault.rag.releases import active_corpus_root


SERVER_NAME = "healthvault"
READ_ONLY_TOOLS = MCP_TOOL_NAMES
PROFILE_TOOLS = frozenset(PROFILE_BOUND_TOOL_NAMES) & frozenset(READ_ONLY_TOOLS)
PROFILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


def build_read_only_registry(
    record_repository: SQLiteHealthRecordRepository,
    knowledge_repository: LocalHealthKnowledgeRepository,
) -> ToolRegistry:
    """Register approved read-only handlers; confirmation is never imported."""
    registry = build_healthvault_tool_registry(
        record_repository,
        knowledge_repository,
        allowed_names=READ_ONLY_TOOLS,
    )
    assert tuple(definition["name"] for definition in registry.definitions) == READ_ONLY_TOOLS
    return registry


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


class HealthVaultMCPServer:
    def __init__(self, profile_id: str, registry: ToolRegistry) -> None:
        if not PROFILE_PATTERN.fullmatch(profile_id):
            raise ValueError("A single synthetic profile_id is required at server startup.")
        registered = tuple(definition["name"] for definition in registry.definitions)
        if registered != READ_ONLY_TOOLS:
            raise ValueError("MCP registry must contain exactly the approved read-only tools.")
        self.profile_id = profile_id
        self.registry = registry

    def advertised_tools(self) -> list[dict[str, Any]]:
        result = []
        for definition in self.registry.definitions:
            schema = deepcopy(definition["input_schema"])
            if definition["name"] in PROFILE_TOOLS:
                schema["properties"].pop("profile_id", None)
                schema["required"] = [
                    item for item in schema.get("required", []) if item != "profile_id"
                ]
            schema["additionalProperties"] = False
            result.append({
                "name": definition["name"],
                "description": definition["description"],
                "inputSchema": schema,
                "annotations": {"readOnlyHint": True},
            })
        return result

    async def handle(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            request_id = message.get("id") if isinstance(message, dict) else None
            return _error(request_id, -32600, "Invalid JSON-RPC request.")
        if "id" not in message:
            # Notifications cannot execute tools.
            return None
        request_id = message["id"]
        method = message.get("method")
        params = message.get("params", {})
        if not isinstance(params, dict):
            return _error(request_id, -32602, "params must be an object.")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": self.advertised_tools()},
            }
        if method != "tools/call":
            return _error(request_id, -32601, "Method not found.")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name not in READ_ONLY_TOOLS or not self.registry.contains(name):
            return _error(request_id, -32602, "Tool is not approved.")
        if not isinstance(arguments, dict):
            return _error(request_id, -32602, "Tool arguments must be an object.")
        if name in PROFILE_TOOLS:
            claimed_profile = arguments.get("profile_id")
            if claimed_profile is not None and claimed_profile != self.profile_id:
                return _error(request_id, -32602, "Profile does not match server binding.")
            arguments = {**arguments, "profile_id": self.profile_id}
        try:
            output = await self.registry.execute(name, arguments)
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": output}], "isError": False},
        }


def serve_stdio(server: HealthVaultMCPServer, reader: TextIO, writer: TextIO) -> None:
    """Emit one JSON-RPC response per input line and nothing else on stdout."""
    for line in reader:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response = _error(None, -32700, "Parse error.")
        else:
            response = asyncio.run(server.handle(request))
        if response is not None:
            writer.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            writer.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HealthVault read-only MCP stdio server")
    parser.add_argument(
        "--profile-id",
        default=os.environ.get("HEALTHVAULT_MCP_PROFILE_ID"),
        help="Synthetic profile bound to this server process.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=configured_database_path(),
        help="Local confirmed-record SQLite database.",
    )
    args = parser.parse_args(argv)
    if not args.profile_id or not PROFILE_PATTERN.fullmatch(args.profile_id):
        parser.error("--profile-id or HEALTHVAULT_MCP_PROFILE_ID is required.")
    if not args.database.is_file():
        parser.error("--database must name an existing confirmed-record SQLite file.")
    record_repository = ReadOnlySQLiteHealthRecordRepository(args.database)
    from healthvault.rag.releases import configured_release_root
    from healthvault.rag.retrieval_config import selection
    from healthvault.infrastructure.hybrid_health_knowledge import HybridHealthKnowledgeRepository
    corpus = active_corpus_root()
    settings = selection(corpus, configured_release_root())
    artifact = corpus / "chunks" / "hybrid_v1.json"
    knowledge_repository = LocalHealthKnowledgeRepository(artifact)
    if settings['knowledge_mode'] == 'hybrid':
        knowledge_repository = HybridHealthKnowledgeRepository.from_local_artifacts(
            knowledge_repository, artifact_path=artifact,
            index_directory=settings['hybrid_index_path'], model_path=settings['hybrid_model_path'])
    registry = build_read_only_registry(record_repository, knowledge_repository)
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    serve_stdio(HealthVaultMCPServer(args.profile_id, registry), sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
