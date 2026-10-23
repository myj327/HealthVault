import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent
from agents.tools import tool_definitions
from healthvault.adapters.bearcode.tools import BearCodeToolAdapter
from healthvault.runtime_context import RuntimeContext
from healthvault.tools.registry import ToolRegistry


SYNTHETIC_TOOL = {
    "name": "synthetic_healthvault_probe",
    "description": "Return a deterministic synthetic probe result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "marker": {"type": "string"},
        },
        "required": ["marker"],
    },
}


def make_agent(adapter):
    with patch.object(agent_module.openai, "AsyncOpenAI", return_value=MagicMock()):
        return Agent(
            api_base="https://invalid.local/v1",
            api_key="synthetic-test-key",
            custom_system_prompt="Synthetic test prompt.",
            custom_tools=adapter.combined_definitions,
            tool_executor=adapter.execute,
        )


class BearCodeToolAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_combines_definitions_without_mutating_globals_and_rejects_conflicts(self):
        native_snapshot = copy.deepcopy(tool_definitions)
        registry = ToolRegistry()

        async def handler(tool_input):
            return str(tool_input)

        registry.register(SYNTHETIC_TOOL, handler)
        adapter = BearCodeToolAdapter(registry)
        combined_names = [definition["name"] for definition in adapter.combined_definitions]

        self.assertIsNot(adapter.combined_definitions, tool_definitions)
        self.assertEqual(native_snapshot, tool_definitions)
        self.assertEqual(
            [definition["name"] for definition in tool_definitions]
            + ["synthetic_healthvault_probe"],
            combined_names,
        )
        self.assertEqual(len(combined_names), len(set(combined_names)))

        conflicting_registry = ToolRegistry()
        conflicting_registry.register(
            next(definition for definition in tool_definitions if definition["name"] == "read_file"),
            handler,
        )
        with self.assertRaisesRegex(ValueError, "read_file"):
            BearCodeToolAdapter(conflicting_registry)

    async def test_real_agent_routes_domain_tool_to_registry_handler(self):
        handler = AsyncMock(return_value="TASK004_DOMAIN_ROUTE_OK")
        registry = ToolRegistry()
        registry.register(SYNTHETIC_TOOL, handler)
        adapter = BearCodeToolAdapter(registry)
        agent = make_agent(adapter)
        tool_input = {"marker": "TASK004_SYNTHETIC"}

        with patch(
            "healthvault.adapters.bearcode.tools.execute_tool",
            new=AsyncMock(return_value="unexpected native result"),
        ) as native_executor:
            result = await agent._execute_tool_call(
                "synthetic_healthvault_probe",
                tool_input,
            )

        self.assertIn(SYNTHETIC_TOOL, agent.tools)
        self.assertTrue(any(tool["name"] == "read_file" for tool in agent.tools))
        handler.assert_awaited_once_with(tool_input)
        native_executor.assert_not_awaited()
        self.assertEqual("TASK004_DOMAIN_ROUTE_OK", result)

    async def test_real_agent_falls_back_to_native_tool_with_same_read_state(self):
        handler = AsyncMock(return_value="unexpected domain result")
        registry = ToolRegistry()
        registry.register(SYNTHETIC_TOOL, handler)
        adapter = BearCodeToolAdapter(registry)
        agent = make_agent(adapter)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "synthetic.txt"
            file_path.write_text("TASK004_NATIVE_FALLBACK_OK", encoding="utf-8")

            result = await agent._execute_tool_call(
                "read_file",
                {"file_path": str(file_path)},
            )

            self.assertIn(str(file_path.resolve()), agent._read_file_state)

        handler.assert_not_awaited()
        self.assertIn("TASK004_NATIVE_FALLBACK_OK", result)

    async def test_unknown_tool_keeps_bearcode_fallback_behavior(self):
        adapter = BearCodeToolAdapter(ToolRegistry())
        read_file_state = {}

        result = await adapter.execute(
            "synthetic_unknown_tool",
            {},
            read_file_state,
        )

        self.assertEqual("Unknown tool: synthetic_unknown_tool", result)

    async def test_domain_only_allowlist_preserves_order_and_blocks_fallback(self):
        registry = ToolRegistry()
        first_handler = AsyncMock(return_value="first-result")
        hidden_handler = AsyncMock(return_value="hidden-result")
        last_handler = AsyncMock(return_value="last-result")
        for name, handler in (
            ("domain_first", first_handler),
            ("confirm_health_records", hidden_handler),
            ("domain_last", last_handler),
        ):
            registry.register(
                {
                    "name": name,
                    "description": f"Synthetic definition for {name}.",
                    "input_schema": {"type": "object", "properties": {}},
                },
                handler,
            )

        adapter = BearCodeToolAdapter(
            registry,
            include_native_tools=False,
            allowed_domain_tools={"domain_last", "domain_first"},
        )

        self.assertEqual(
            ["domain_first", "domain_last"],
            [definition["name"] for definition in adapter.combined_definitions],
        )
        with patch(
            "healthvault.adapters.bearcode.tools.execute_tool",
            new=AsyncMock(return_value="unexpected native result"),
        ) as native_executor:
            allowed_result = await adapter.execute("domain_last", {})
            hidden_result = await adapter.execute("confirm_health_records", {})
            native_result = await adapter.execute("read_file", {})
            unknown_result = await adapter.execute("unknown_tool", {})

        self.assertEqual("last-result", allowed_result)
        self.assertEqual(
            "Tool unavailable in isolated HealthVault adapter: confirm_health_records",
            hidden_result,
        )
        self.assertEqual(
            "Tool unavailable in isolated HealthVault adapter: read_file",
            native_result,
        )
        self.assertEqual(
            "Tool unavailable in isolated HealthVault adapter: unknown_tool",
            unknown_result,
        )
        hidden_handler.assert_not_awaited()
        native_executor.assert_not_awaited()

    def test_invalid_domain_allowlist_fails_with_sorted_names(self):
        registry = ToolRegistry()
        registry.register(SYNTHETIC_TOOL, AsyncMock(return_value="unused"))

        with self.assertRaisesRegex(
            ValueError,
            "missing_a, missing_z",
        ):
            BearCodeToolAdapter(
                registry,
                include_native_tools=False,
                allowed_domain_tools={
                    "synthetic_healthvault_probe",
                    "missing_z",
                    "missing_a",
                },
            )

    async def test_runtime_context_hides_injects_and_protects_profile_id(self):
        registry = ToolRegistry()
        captured = {}

        async def handler(tool_input):
            captured.update(tool_input)
            return "context-result"

        definition = {
            "name": "query_health_timeline",
            "description": "Synthetic context-bound tool.",
            "input_schema": {
                "type": "object",
                "properties": {"profile_id": {"type": "string"}},
                "required": ["profile_id"],
            },
        }
        registry.register(definition, handler)
        adapter = BearCodeToolAdapter(
            registry,
            include_native_tools=False,
            runtime_context=RuntimeContext("profile-from-context", "request-1"),
        )

        exposed_schema = adapter.combined_definitions[0]["input_schema"]
        self.assertNotIn("profile_id", exposed_schema["properties"])
        self.assertNotIn("profile_id", exposed_schema.get("required", []))
        self.assertIn("profile_id", definition["input_schema"]["properties"])

        result = await adapter.execute("query_health_timeline", {})
        rejected = await adapter.execute(
            "query_health_timeline",
            {"profile_id": "model-attempt"},
        )

        self.assertEqual("context-result", result)
        self.assertEqual({"profile_id": "profile-from-context"}, captured)
        self.assertEqual(
            "Tool input rejected: profile_id is supplied by RuntimeContext.",
            rejected,
        )


if __name__ == "__main__":
    unittest.main()
