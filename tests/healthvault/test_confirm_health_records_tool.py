import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent
from healthvault.adapters.bearcode.tools import BearCodeToolAdapter
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.tools.catalog import build_healthvault_tool_registry
from healthvault.tools.confirm_health_records import (
    CONFIRM_HEALTH_RECORDS_TOOL,
    build_confirm_health_records_handler,
)


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "health_reports"


def load_fixture():
    return (FIXTURE_DIR / "synthetic_checkup_2024.txt").read_text(
        encoding="utf-8"
    )


def tool_input():
    report_text = load_fixture()
    return {
        "profile_id": "synthetic-profile-tool",
        "source_id": "synthetic-tool-source",
        "report_text": report_text,
        "expected_sha256": hashlib.sha256(
            report_text.encode("utf-8")
        ).hexdigest(),
        "user_confirmed": True,
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


class ConfirmHealthRecordsToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "tool-test.sqlite3"
        self.repository = SQLiteHealthRecordRepository(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def counts(self):
        connection = sqlite3.connect(self.database_path)
        try:
            report_count = connection.execute(
                "SELECT COUNT(*) FROM confirmed_reports"
            ).fetchone()[0]
            record_count = connection.execute(
                "SELECT COUNT(*) FROM health_records"
            ).fetchone()[0]
            return report_count, record_count
        finally:
            connection.close()

    def test_tool_definition_has_only_required_confirmation_inputs(self):
        self.assertEqual(
            "confirm_health_records",
            CONFIRM_HEALTH_RECORDS_TOOL["name"],
        )
        schema = CONFIRM_HEALTH_RECORDS_TOOL["input_schema"]
        expected_fields = {
            "profile_id",
            "source_id",
            "report_text",
            "expected_sha256",
            "user_confirmed",
        }
        self.assertEqual(expected_fields, set(schema["properties"]))
        self.assertEqual(
            [
                "profile_id",
                "source_id",
                "report_text",
                "expected_sha256",
                "user_confirmed",
            ],
            schema["required"],
        )
        self.assertEqual(
            "boolean",
            schema["properties"]["user_confirmed"]["type"],
        )
        self.assertNotIn("database_path", schema["properties"])
        self.assertNotIn("indicators", schema["properties"])

    async def test_handler_returns_json_and_rejected_call_does_not_write(self):
        handler = build_confirm_health_records_handler(self.repository)
        rejected_input = tool_input()
        rejected_input["user_confirmed"] = False

        result = json.loads(await handler(rejected_input))

        self.assertEqual("confirmation_required", result["status"])
        self.assertFalse(result["confirmed"])
        self.assertEqual((0, 0), self.counts())

    async def test_catalog_is_backward_compatible_and_registries_are_independent(self):
        preview_only = build_healthvault_tool_registry()
        first = build_healthvault_tool_registry(record_repository=self.repository)
        second = build_healthvault_tool_registry(record_repository=self.repository)

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
            [definition["name"] for definition in first.definitions],
        )
        self.assertIsNot(first, second)

        async def synthetic_handler(value):
            return str(value)

        first.register(
            {
                "name": "synthetic_catalog_isolation_probe",
                "description": "Test-only catalog isolation probe.",
                "input_schema": {"type": "object", "properties": {}},
            },
            synthetic_handler,
        )
        self.assertFalse(second.contains("synthetic_catalog_isolation_probe"))

    async def test_real_agent_routes_confirmation_without_native_or_api_call(self):
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
                "confirm_health_records",
                tool_input(),
            )

        result = json.loads(result_text)
        tool_names = {definition["name"] for definition in agent.tools}
        self.assertIn("read_file", tool_names)
        self.assertIn("preview_health_report", tool_names)
        self.assertIn("confirm_health_records", tool_names)
        self.assertEqual("confirmed", result["status"])
        self.assertTrue(result["write_performed"])
        self.assertEqual(3, result["indicator_count"])
        self.assertEqual((1, 3), self.counts())
        native_executor.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
