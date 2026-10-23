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
from healthvault.application.generate_visit_brief import generate_visit_brief
from healthvault.infrastructure.local_health_knowledge import (
    LocalHealthKnowledgeRepository,
)
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.repositories.health_knowledge import HealthKnowledgeQueryError
from healthvault.repositories.health_records import HealthRecordQueryError
from healthvault.tools.catalog import build_healthvault_tool_registry
from healthvault.tools.directory import TOOL_SPECS
from healthvault.tools.generate_visit_brief import (
    GENERATE_VISIT_BRIEF_TOOL,
    build_generate_visit_brief_handler,
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


class FailingRecordRepository:
    def list_timeline(self, profile_id):
        raise HealthRecordQueryError(f"record failure for {profile_id}")


class FailingKnowledgeRepository:
    def search(self, query, limit):
        raise HealthKnowledgeQueryError("synthetic knowledge failure")


class GenerateVisitBriefToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "visit-brief-tool.sqlite3"
        self.record_repository = SQLiteHealthRecordRepository(self.database_path)
        self.knowledge_repository = LocalHealthKnowledgeRepository()
        self.profile_id = "synthetic-visit-brief-tool-profile"
        for year in (2025, 2024):
            report_text = (
                FIXTURE_DIR / f"synthetic_checkup_{year}.txt"
            ).read_text(encoding="utf-8")
            result = confirm_health_records(
                {
                    "profile_id": self.profile_id,
                    "source_id": f"synthetic-visit-brief-{year}",
                    "report_text": report_text,
                    "expected_sha256": hashlib.sha256(
                        report_text.encode("utf-8")
                    ).hexdigest(),
                    "user_confirmed": True,
                },
                self.record_repository,
            ).to_dict()
            if result["status"] != "confirmed":
                raise AssertionError(
                    f"Synthetic setup confirmation failed: {result}"
                )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tool_definition_only_accepts_profile_id(self):
        self.assertEqual(
            "generate_visit_brief",
            GENERATE_VISIT_BRIEF_TOOL["name"],
        )
        schema = GENERATE_VISIT_BRIEF_TOOL["input_schema"]
        self.assertEqual("object", schema["type"])
        self.assertEqual({"profile_id"}, set(schema["properties"]))
        self.assertEqual(["profile_id"], schema["required"])
        self.assertEqual("string", schema["properties"]["profile_id"]["type"])
        for forbidden in (
            "query",
            "symptoms",
            "medications",
            "diagnosis",
            "threshold",
            "limit",
            "database_path",
            "markdown_template",
        ):
            self.assertNotIn(forbidden, schema["properties"])

    async def test_handler_json_matches_application_result(self):
        handler = build_generate_visit_brief_handler(
            self.record_repository,
            self.knowledge_repository,
        )
        tool_input = {"profile_id": self.profile_id}

        result_text = await handler(tool_input)

        self.assertEqual(
            generate_visit_brief(
                tool_input,
                self.record_repository,
                self.knowledge_repository,
            ).to_dict(),
            json.loads(result_text),
        )

    async def test_handler_propagates_both_repository_errors(self):
        record_handler = build_generate_visit_brief_handler(
            FailingRecordRepository(),
            self.knowledge_repository,
        )
        with self.assertRaises(HealthRecordQueryError):
            await record_handler({"profile_id": self.profile_id})

        knowledge_handler = build_generate_visit_brief_handler(
            self.record_repository,
            FailingKnowledgeRepository(),
        )
        with self.assertRaises(HealthKnowledgeQueryError):
            await knowledge_handler({"profile_id": self.profile_id})

    async def test_catalog_respects_tool_directory_dependencies(self):
        preview_only = build_healthvault_tool_registry()
        record_only = build_healthvault_tool_registry(
            record_repository=self.record_repository
        )
        knowledge_only = build_healthvault_tool_registry(
            knowledge_repository=self.knowledge_repository
        )
        both = build_healthvault_tool_registry(
            record_repository=self.record_repository,
            knowledge_repository=self.knowledge_repository,
        )
        second_both = build_healthvault_tool_registry(
            record_repository=self.record_repository,
            knowledge_repository=self.knowledge_repository,
        )

        for registry, dependencies in (
            (preview_only, frozenset()),
            (record_only, frozenset({"records"})),
            (knowledge_only, frozenset({"knowledge"})),
            (both, frozenset({"records", "knowledge"})),
        ):
            self.assertEqual(
                [spec.name for spec in TOOL_SPECS if spec.dependencies <= dependencies],
                [definition["name"] for definition in registry.definitions],
            )
        self.assertNotIn("generate_visit_brief", [item["name"] for item in record_only.definitions])
        self.assertIn("generate_visit_brief", [item["name"] for item in both.definitions])
        self.assertIsNot(both, second_both)

    async def test_real_agent_routes_brief_without_native_or_api_call(self):
        registry = build_healthvault_tool_registry(
            record_repository=self.record_repository,
            knowledge_repository=self.knowledge_repository,
        )
        adapter = BearCodeToolAdapter(registry)
        agent = make_agent(adapter)

        with patch(
            "healthvault.adapters.bearcode.tools.execute_tool",
            new=AsyncMock(return_value="unexpected native result"),
        ) as native_executor:
            result_text = await agent._execute_tool_call(
                "generate_visit_brief",
                {"profile_id": self.profile_id},
            )

        result = json.loads(result_text)
        tool_names = [definition["name"] for definition in agent.tools]
        self.assertIn("read_file", tool_names)
        self.assertIn("generate_visit_brief", tool_names)
        self.assertEqual("ok", result["status"])
        self.assertEqual(3, result["indicator_count"])
        self.assertEqual(4, len(result["questions"]))
        self.assertIn("# 就诊准备摘要", result["brief_markdown"])
        native_executor.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
