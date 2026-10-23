import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent
from healthvault.adapters.bearcode.tools import BearCodeToolAdapter
from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.application.query_health_timeline import query_health_timeline
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.tools.catalog import build_healthvault_tool_registry
from healthvault.tools.query_health_timeline import (
    QUERY_HEALTH_TIMELINE_TOOL,
    build_query_health_timeline_handler,
)


FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "health_reports"
    / "synthetic_checkup_2024.txt"
)


def make_agent(adapter):
    with patch.object(agent_module.openai, "AsyncOpenAI", return_value=MagicMock()):
        return Agent(
            api_base="https://invalid.local/v1",
            api_key="synthetic-test-key",
            custom_system_prompt="Synthetic test prompt.",
            custom_tools=adapter.combined_definitions,
            tool_executor=adapter.execute,
        )


class QueryHealthTimelineToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "timeline-tool.sqlite3"
        self.repository = SQLiteHealthRecordRepository(self.database_path)
        self.profile_id = "synthetic-timeline-tool-profile"
        report_text = FIXTURE_PATH.read_text(encoding="utf-8")
        result = confirm_health_records(
            {
                "profile_id": self.profile_id,
                "source_id": "synthetic-timeline-tool-source",
                "report_text": report_text,
                "expected_sha256": hashlib.sha256(
                    report_text.encode("utf-8")
                ).hexdigest(),
                "user_confirmed": True,
            },
            self.repository,
        ).to_dict()
        if result["status"] != "confirmed":
            raise AssertionError(f"Synthetic setup confirmation failed: {result}")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tool_definition_only_accepts_profile_id(self):
        self.assertEqual(
            "query_health_timeline",
            QUERY_HEALTH_TIMELINE_TOOL["name"],
        )
        schema = QUERY_HEALTH_TIMELINE_TOOL["input_schema"]
        self.assertEqual("object", schema["type"])
        self.assertEqual({"profile_id"}, set(schema["properties"]))
        self.assertEqual(["profile_id"], schema["required"])
        self.assertEqual("string", schema["properties"]["profile_id"]["type"])
        for forbidden in ("database_path", "sql", "date_range", "sort"):
            self.assertNotIn(forbidden, schema["properties"])

    async def test_handler_json_matches_application_result(self):
        handler = build_query_health_timeline_handler(self.repository)
        tool_input = {"profile_id": self.profile_id}

        result_text = await handler(tool_input)

        self.assertEqual(
            query_health_timeline(tool_input, self.repository).to_dict(),
            json.loads(result_text),
        )

    async def test_catalog_registers_four_tools_in_fixed_order(self):
        preview_only = build_healthvault_tool_registry()
        with_repository = build_healthvault_tool_registry(
            record_repository=self.repository
        )

        self.assertEqual(
            ["preview_health_report"],
            [definition["name"] for definition in preview_only.definitions],
        )
        self.assertEqual(
            [
                "preview_health_report",
                "confirm_health_records",
                "query_health_timeline",
                "compare_health_indicators",
            ],
            [definition["name"] for definition in with_repository.definitions],
        )

    async def test_real_agent_routes_query_without_native_or_api_call(self):
        registry = build_healthvault_tool_registry(
            record_repository=self.repository
        )
        adapter = BearCodeToolAdapter(registry)
        agent = make_agent(adapter)

        with patch(
            "healthvault.adapters.bearcode.tools.execute_tool",
            new=AsyncMock(return_value="unexpected native result"),
        ) as native_executor:
            result_text = await agent._execute_tool_call(
                "query_health_timeline",
                {"profile_id": self.profile_id},
            )

        result = json.loads(result_text)
        tool_names = [definition["name"] for definition in agent.tools]
        self.assertIn("read_file", tool_names)
        self.assertIn("preview_health_report", tool_names)
        self.assertIn("confirm_health_records", tool_names)
        self.assertIn("query_health_timeline", tool_names)
        self.assertEqual("ok", result["status"])
        self.assertEqual(1, result["report_count"])
        self.assertEqual(3, result["indicator_count"])
        native_executor.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
