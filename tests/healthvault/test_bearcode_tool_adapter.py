import unittest
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
    def test_adapter_exposes_only_registered_domain_tools(self):
        registry = ToolRegistry()
        registry.register(SYNTHETIC_TOOL, AsyncMock())
        adapter = BearCodeToolAdapter(registry)
        self.assertEqual([], tool_definitions)
        self.assertEqual([SYNTHETIC_TOOL], adapter.combined_definitions)

    async def test_real_agent_routes_domain_tool_and_rejects_undeclared_tools(self):
        handler = AsyncMock(return_value="domain-result")
        registry = ToolRegistry()
        registry.register(SYNTHETIC_TOOL, handler)
        adapter = BearCodeToolAdapter(registry)
        agent = make_agent(adapter)
        self.assertEqual("domain-result", await agent._execute_tool_call(SYNTHETIC_TOOL["name"], {}))
        handler.assert_awaited_once_with({})
        for name in ("read_file", "write_file", "run_shell", "agent", "skill", "enter_plan_mode", "unknown"):
            self.assertIn("unavailable", await agent._execute_tool_call(name, {}))
            self.assertIn("unavailable", await adapter.execute(name, {}))
        handler.assert_awaited_once()

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
            allowed_domain_tools={"domain_last", "domain_first"},
        )

        self.assertEqual(
            ["domain_first", "domain_last"],
            [definition["name"] for definition in adapter.combined_definitions],
        )
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

    def test_invalid_domain_allowlist_fails_with_sorted_names(self):
        registry = ToolRegistry()
        registry.register(SYNTHETIC_TOOL, AsyncMock(return_value="unused"))

        with self.assertRaisesRegex(
            ValueError,
            "missing_a, missing_z",
        ):
            BearCodeToolAdapter(
                registry,
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
