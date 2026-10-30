import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent
from healthvault.adapters.bearcode.tools import BearCodeToolAdapter
from healthvault.application.compare_health_indicators import (
    compare_health_indicators,
)
from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.repositories.health_records import HealthRecordQueryError
from healthvault.tools.catalog import build_healthvault_tool_registry
from healthvault.tools.compare_health_indicators import (
    COMPARE_HEALTH_INDICATORS_TOOL,
    build_compare_health_indicators_handler,
)


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "health_reports"


def make_agent(adapter):
    with patch.object(agent_module.openai, "AsyncOpenAI", return_value=MagicMock()):
        return Agent(
            api_base="https://invalid.local/v1",
            api_key="synthetic-test-key",
            custom_system_prompt="Synthetic test prompt.",
            custom_tools=adapter.combined_definitions,
            tool_executor=adapter.execute,
        )


class FailingRepository:
    def list_timeline(self, profile_id):
        raise HealthRecordQueryError(f"query failed for {profile_id}")


class CompareHealthIndicatorsToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "compare-tool.sqlite3"
        self.repository = SQLiteHealthRecordRepository(self.database_path)
        self.profile_id = "synthetic-compare-tool-profile"
        for year in (2025, 2024):
            report_text = (
                FIXTURE_DIR / f"synthetic_checkup_{year}.txt"
            ).read_text(encoding="utf-8")
            result = confirm_health_records(
                {
                    "profile_id": self.profile_id,
                    "source_id": f"synthetic-compare-{year}",
                    "report_text": report_text,
                    "expected_sha256": hashlib.sha256(
                        report_text.encode("utf-8")
                    ).hexdigest(),
                    "user_confirmed": True,
                },
                self.repository,
            ).to_dict()
            if result["status"] != "confirmed":
                raise AssertionError(
                    f"Synthetic setup confirmation failed: {result}"
                )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tool_definition_accepts_year_pair_and_indicator_filter(self):
        self.assertEqual(
            "compare_health_indicators",
            COMPARE_HEALTH_INDICATORS_TOOL["name"],
        )
        schema = COMPARE_HEALTH_INDICATORS_TOOL["input_schema"]
        self.assertEqual("object", schema["type"])
        self.assertEqual({"profile_id", "start_year", "end_year", "indicator_code"}, set(schema["properties"]))
        self.assertEqual(["profile_id"], schema["required"])
        self.assertEqual("string", schema["properties"]["profile_id"]["type"])
        for forbidden in (
            "database_path",
            "sql",
            "indicator_codes",
            "thresholds",
            "date_range",
            "sort",
            "medical_parameters",
        ):
            self.assertNotIn(forbidden, schema["properties"])

    async def test_handler_json_matches_application_and_preserves_query_error(self):
        handler = build_compare_health_indicators_handler(self.repository)
        tool_input = {"profile_id": self.profile_id}

        result_text = await handler(tool_input)

        self.assertEqual(
            compare_health_indicators(tool_input, self.repository).to_dict(),
            json.loads(result_text),
        )
        failing_handler = build_compare_health_indicators_handler(
            FailingRepository()
        )
        with self.assertRaises(HealthRecordQueryError):
            await failing_handler({"profile_id": "synthetic-error"})

    async def test_catalog_registers_four_independent_tools_in_fixed_order(self):
        preview_only = build_healthvault_tool_registry()
        first = build_healthvault_tool_registry(
            record_repository=self.repository
        )
        second = build_healthvault_tool_registry(
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
            [definition["name"] for definition in first.definitions],
        )
        self.assertIsNot(first, second)
        self.assertEqual(
            [definition["name"] for definition in first.definitions],
            [definition["name"] for definition in second.definitions],
        )

    async def test_real_agent_routes_compare_without_native_or_api_call(self):
        registry = build_healthvault_tool_registry(
            record_repository=self.repository
        )
        adapter = BearCodeToolAdapter(registry)
        agent = make_agent(adapter)

        with patch(
            "agents.agent.execute_tool",
            new=AsyncMock(return_value="unexpected native result"),
        ) as native_executor:
            result_text = await agent._execute_tool_call(
                "compare_health_indicators",
                {"profile_id": self.profile_id},
            )

        result = json.loads(result_text)
        tool_names = [definition["name"] for definition in agent.tools]
        self.assertNotIn("read_file", tool_names)
        self.assertIn("preview_health_report", tool_names)
        self.assertIn("confirm_health_records", tool_names)
        self.assertIn("query_health_timeline", tool_names)
        self.assertIn("compare_health_indicators", tool_names)
        self.assertEqual("ok", result["status"])
        self.assertEqual(2, result["report_count"])
        self.assertEqual(3, result["comparison_count"])
        native_executor.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
