import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent
from healthvault.adapters.bearcode.tools import BearCodeToolAdapter
from healthvault.application.report_preview import build_report_preview
from healthvault.tools.catalog import build_healthvault_tool_registry
from healthvault.tools.preview_health_report import (
    PREVIEW_HEALTH_REPORT_TOOL,
    preview_health_report,
)


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "health_reports"


def load_fixture(name="synthetic_checkup_2024.txt"):
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def make_agent(adapter):
    with patch.object(agent_module.openai, "AsyncOpenAI", return_value=MagicMock()):
        return Agent(
            api_base="https://invalid.local/v1",
            api_key="synthetic-test-key",
            custom_system_prompt="Synthetic test prompt.",
            custom_tools=adapter.combined_definitions,
            tool_executor=adapter.execute,
        )


class PreviewHealthReportToolTests(unittest.IsolatedAsyncioTestCase):
    def test_tool_definition_has_exact_name_and_required_input_fields(self):
        self.assertEqual("preview_health_report", PREVIEW_HEALTH_REPORT_TOOL["name"])
        schema = PREVIEW_HEALTH_REPORT_TOOL["input_schema"]
        self.assertEqual("object", schema["type"])
        self.assertEqual(["source_id", "report_text"], schema["required"])
        self.assertEqual(
            {"source_id", "report_text"},
            set(schema["properties"]),
        )
        self.assertEqual("string", schema["properties"]["source_id"]["type"])
        self.assertEqual("string", schema["properties"]["report_text"]["type"])

    async def test_handler_returns_json_equal_to_application_result(self):
        report_text = load_fixture()
        tool_input = {
            "source_id": "synthetic-tool-direct",
            "report_text": report_text,
        }

        encoded = await preview_health_report(tool_input)

        self.assertEqual(
            build_report_preview(
                tool_input["source_id"],
                report_text,
            ).to_dict(),
            json.loads(encoded),
        )

    async def test_handler_rejects_missing_fields_and_non_dictionary_input(self):
        for tool_input in ({}, {"source_id": "source"}, None, "not-a-dict"):
            with self.subTest(tool_input=tool_input):
                result = json.loads(await preview_health_report(tool_input))
                self.assertEqual("invalid", result["status"])
                self.assertFalse(result["can_confirm"])

    async def test_registry_builds_independent_single_tool_instances(self):
        first = build_healthvault_tool_registry()
        second = build_healthvault_tool_registry()

        self.assertIsNot(first, second)
        self.assertEqual(
            ["preview_health_report"],
            [definition["name"] for definition in first.definitions],
        )
        self.assertEqual(
            ["preview_health_report"],
            [definition["name"] for definition in second.definitions],
        )

        async def synthetic_handler(tool_input):
            return str(tool_input)

        first.register(
            {
                "name": "synthetic_registry_isolation_probe",
                "description": "Test-only registry isolation probe.",
                "input_schema": {"type": "object", "properties": {}},
            },
            synthetic_handler,
        )
        self.assertFalse(second.contains("synthetic_registry_isolation_probe"))

    async def test_real_agent_routes_formal_tool_without_native_or_api_call(self):
        report_text = load_fixture()
        tool_input = {
            "source_id": "synthetic-agent-route",
            "report_text": report_text,
        }
        registry = build_healthvault_tool_registry()
        adapter = BearCodeToolAdapter(registry)
        agent = make_agent(adapter)

        with patch(
            "agents.agent.execute_tool",
            new=AsyncMock(return_value="unexpected native result"),
        ) as native_executor:
            result_text = await agent._execute_tool_call(
                "preview_health_report",
                tool_input,
            )

        tool_names = {definition["name"] for definition in agent.tools}
        self.assertNotIn("read_file", tool_names)
        self.assertIn("preview_health_report", tool_names)
        self.assertEqual(
            build_report_preview(
                tool_input["source_id"],
                report_text,
            ).to_dict(),
            json.loads(result_text),
        )
        native_executor.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
