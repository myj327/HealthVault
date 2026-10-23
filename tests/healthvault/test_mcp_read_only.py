import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agents.runtime_policy import AgentRuntimePolicy

from healthvault.infrastructure.local_health_knowledge import LocalHealthKnowledgeRepository
from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
from healthvault.mcp.client import ApprovedMCPClient
from healthvault.mcp.runtime_bridge import attach_approved_mcp
from healthvault.runtime_context import RuntimeContext
from healthvault.mcp.server import (
    HealthVaultMCPServer,
    READ_ONLY_TOOLS,
    build_read_only_registry,
)
from healthvault.tools.catalog import build_healthvault_tool_registry


ROOT = Path(__file__).parents[2]
REPORT = (ROOT / "tests/fixtures/health_reports/synthetic_checkup_2024.txt").read_text(
    encoding="utf-8"
)


def request(request_id, method, params=None):
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


class FakeConnection:
    created = []

    def __init__(self, server_name, command, args, env):
        self.server_name = server_name
        self.connected = False
        self.closed = False
        self.calls = []
        self.created.append(self)

    async def connect(self):
        self.connected = True

    async def initialize(self):
        pass

    async def list_tools(self):
        return [
            {"name": "query_health_timeline", "description": "read", "inputSchema": {"type": "object"}},
            {"name": "confirm_health_records", "description": "write", "inputSchema": {"type": "object"}},
            {"name": "shell", "description": "unsafe", "inputSchema": {"type": "object"}},
        ]

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return "read-only result"

    def close(self):
        self.closed = True


class MCPIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.directory.name) / "records.sqlite3"
        self.records = SQLiteHealthRecordRepository(self.database_path)
        self.knowledge = LocalHealthKnowledgeRepository()
        self.registry = build_read_only_registry(self.records, self.knowledge)
        self.server = HealthVaultMCPServer("bound-demo", self.registry)

    def tearDown(self):
        self.directory.cleanup()

    def call(self, name, arguments=None):
        return asyncio.run(self.server.handle(
            request(3, "tools/call", {"name": name, "arguments": arguments or {}})
        ))

    def test_tool_list_contains_only_approved_read_only_handlers(self):
        listed = asyncio.run(self.server.handle(request(1, "tools/list")))["result"]["tools"]
        self.assertEqual(list(READ_ONLY_TOOLS), [tool["name"] for tool in listed])
        self.assertTrue(all(tool["annotations"]["readOnlyHint"] for tool in listed))
        self.assertFalse(self.registry.contains("confirm_health_records"))
        with self.assertRaisesRegex(ValueError, "exactly the approved"):
            HealthVaultMCPServer(
                "bound-demo",
                build_healthvault_tool_registry(self.records, self.knowledge),
            )

    def test_bound_profile_and_unknown_tools_are_rejected(self):
        different = self.call("query_health_timeline", {"profile_id": "other-demo"})
        self.assertEqual(-32602, different["error"]["code"])
        unknown = self.call("confirm_health_records", {"profile_id": "bound-demo"})
        self.assertEqual(-32602, unknown["error"]["code"])
        unknown = self.call("not_a_tool")
        self.assertEqual(-32602, unknown["error"]["code"])
        accepted = self.call("query_health_timeline")
        payload = json.loads(accepted["result"]["content"][0]["text"])
        self.assertEqual("empty", payload["status"])
        self.assertEqual("bound-demo", payload["profile_id"])

    def test_preview_tool_preserves_existing_registry_result(self):
        preview = self.call(
            "preview_health_report",
            {"source_id": "mcp-synthetic", "report_text": REPORT},
        )
        payload = json.loads(preview["result"]["content"][0]["text"])
        self.assertTrue(payload["can_confirm"])
        self.assertEqual("mcp-synthetic", payload["source"]["source_id"])

    def test_explicit_client_ignores_unapproved_server_and_tools(self):
        FakeConnection.created.clear()
        client = ApprovedMCPClient(
            {
                "healthvault": {"command": sys.executable, "args": [
                    "-B", "-m", "healthvault.mcp.server", "--profile-id",
                    "bound-demo", "--database", str(self.database_path),
                ], "env": {}},
                "unapproved": {"command": "should-never-run", "args": [], "env": {}},
            },
            profile_id="bound-demo",
            database_path=self.database_path,
            connection_factory=FakeConnection,
        )
        asyncio.run(client.connect())
        self.assertEqual(["healthvault"], [item.server_name for item in FakeConnection.created])
        self.assertEqual(
            ["mcp__healthvault__query_health_timeline"],
            [item["name"] for item in client.get_tool_definitions()],
        )
        with self.assertRaises(KeyError):
            asyncio.run(client.call_tool("mcp__healthvault__confirm_health_records", {}))
        with self.assertRaises(KeyError):
            asyncio.run(client.call_tool("mcp__unapproved__query_health_timeline", {}))
        self.assertEqual(
            "read-only result",
            asyncio.run(client.call_tool("mcp__healthvault__query_health_timeline", {})),
        )
        asyncio.run(client.disconnect_all())
        self.assertTrue(FakeConnection.created[0].closed)

    def test_stdio_json_rpc_handshake_and_tool_call(self):
        messages = [
            request(1, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "smoke", "version": "1"},
            }),
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            request(2, "tools/list"),
            request(3, "tools/call", {
                "name": "preview_health_report",
                "arguments": {"source_id": "stdio-synthetic", "report_text": REPORT},
            }),
        ]
        run = subprocess.run(
            [
                sys.executable, "-B", "-m", "healthvault.mcp.server",
                "--profile-id", "bound-demo",
                "--database", str(self.database_path),
            ],
            input="\n".join(json.dumps(item) for item in messages) + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
            timeout=20,
        )
        self.assertEqual(0, run.returncode, run.stderr)
        responses = [json.loads(line) for line in run.stdout.splitlines()]
        self.assertEqual([1, 2, 3], [item["id"] for item in responses])
        self.assertEqual("healthvault", responses[0]["result"]["serverInfo"]["name"])
        self.assertEqual(len(READ_ONLY_TOOLS), len(responses[1]["result"]["tools"]))
        preview = json.loads(responses[2]["result"]["content"][0]["text"])
        self.assertTrue(preview["can_confirm"])


    def test_spoofed_command_or_profile_is_never_started(self):
        FakeConnection.created.clear()
        valid_args = [
            "-B", "-m", "healthvault.mcp.server", "--profile-id",
            "bound-demo", "--database", str(self.database_path),
        ]
        for command, args in (
            ("not-python.exe", valid_args),
            (sys.executable, [*valid_args[:4], "other-demo", *valid_args[5:]]),
        ):
            client = ApprovedMCPClient(
                {"healthvault": {"command": command, "args": args, "env": {}}},
                profile_id="bound-demo",
                database_path=self.database_path,
                connection_factory=FakeConnection,
            )
            with self.assertRaises(ValueError):
                asyncio.run(client.connect())
        self.assertEqual([], FakeConnection.created)

    def test_stdio_server_requires_existing_database(self):
        missing = Path(self.directory.name) / "missing.sqlite3"
        run = subprocess.run(
            [
                sys.executable, "-B", "-m", "healthvault.mcp.server",
                "--profile-id", "bound-demo", "--database", str(missing),
            ],
            input="",
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=20,
        )
        self.assertNotEqual(0, run.returncode)
        self.assertFalse(missing.exists())
        self.assertEqual("", run.stdout)

    def test_real_approved_client_and_runtime_bridge(self):
        config_path = Path(self.directory.name) / "approved-mcp.json"
        config_path.write_text(json.dumps({
            "mcpServers": {
                "healthvault": {
                    "command": sys.executable,
                    "args": [
                        "-B", "-m", "healthvault.mcp.server",
                        "--profile-id", "bound-demo",
                        "--database", str(self.database_path),
                    ],
                    "env": {},
                },
                "unapproved": {
                    "command": "never-launch",
                    "args": [],
                },
            }
        }), encoding="utf-8")

        async def original_executor(name, arguments, read_file_state=None):
            return "local"

        agent = SimpleNamespace(
            tools=[{"name": "query_health_timeline"}],
            _tool_executor=original_executor,
            runtime_policy=AgentRuntimePolicy(
                mcp_enabled=False,
                memory_enabled=False,
                global_skills_enabled=False,
                session_persistence_enabled=False,
                allowed_tool_names=frozenset(READ_ONLY_TOOLS),
            ),
        )
        runtime = SimpleNamespace(
            context=RuntimeContext("bound-demo", "mcp-smoke"),
            components=SimpleNamespace(record_repository=self.records),
            agent=agent,
            manifest={"tool_names": ["query_health_timeline"]},
        )

        async def scenario():
            async with await attach_approved_mcp(
                runtime, config_path=config_path, allow_web_search=True
            ):
                self.assertFalse(agent.runtime_policy.mcp_enabled)
                self.assertEqual(1 + len(READ_ONLY_TOOLS), len(agent.tools))
                self.assertIn(
                    "mcp__healthvault__query_health_timeline",
                    agent.runtime_policy.allowed_tool_names,
                )
                payload = json.loads(await agent._tool_executor(
                    "mcp__healthvault__query_health_timeline", {}, None,
                ))
                self.assertEqual("empty", payload["status"])
                self.assertEqual("bound-demo", payload["profile_id"])
                runtime.manifest["task_memory_last_write_status"] = "saved"
            self.assertEqual(1, len(agent.tools))
            self.assertEqual(
                "saved", runtime.manifest["task_memory_last_write_status"]
            )
            self.assertEqual("local", await agent._tool_executor("local", {}, None))
            self.assertEqual(
                ["query_health_timeline"], runtime.manifest["tool_names"]
            )
            async with await attach_approved_mcp(
                runtime, config_path=config_path, allow_web_search=False,
                mcp_only=True,
            ):
                names = {definition["name"] for definition in agent.tools}
                self.assertEqual(len(READ_ONLY_TOOLS) - 1, len(names))
                self.assertNotIn("query_health_timeline", names)
                self.assertNotIn("mcp__healthvault__search_official_health_web", names)
                self.assertNotIn(
                    "mcp__healthvault__search_official_health_web",
                    agent.runtime_policy.allowed_tool_names,
                )
                self.assertEqual(
                    "Tool unavailable under Agent runtime policy: query_health_timeline",
                    await agent._tool_executor("query_health_timeline", {}, None),
                )

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
