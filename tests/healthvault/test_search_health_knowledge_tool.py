import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent
from healthvault.adapters.bearcode.tools import BearCodeToolAdapter
from healthvault.application.search_health_knowledge import (
    search_health_knowledge,
)
from healthvault.infrastructure.local_health_knowledge import (
    LocalHealthKnowledgeRepository,
)
from healthvault.repositories.health_knowledge import HealthKnowledgeQueryError
from healthvault.tools.catalog import build_healthvault_tool_registry
from healthvault.tools.search_health_knowledge import (
    SEARCH_HEALTH_KNOWLEDGE_TOOL,
    build_search_health_knowledge_handler,
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


class FailingKnowledgeRepository:
    def search(self, query, limit):
        raise HealthKnowledgeQueryError(f"query failed for {query}")


class SearchHealthKnowledgeToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repository = LocalHealthKnowledgeRepository()

    def test_tool_definition_only_accepts_query_and_limit(self):
        self.assertEqual(
            "search_health_knowledge",
            SEARCH_HEALTH_KNOWLEDGE_TOOL["name"],
        )
        schema = SEARCH_HEALTH_KNOWLEDGE_TOOL["input_schema"]
        self.assertEqual("object", schema["type"])
        self.assertEqual({"query", "limit"}, set(schema["properties"]))
        self.assertEqual(["query"], schema["required"])
        self.assertEqual("string", schema["properties"]["query"]["type"])
        self.assertEqual("integer", schema["properties"]["limit"]["type"])
        self.assertEqual(1, schema["properties"]["limit"]["minimum"])
        self.assertEqual(5, schema["properties"]["limit"]["maximum"])
        for forbidden in (
            "profile_id",
            "indicator_value",
            "diagnosis",
            "threshold",
            "url",
            "file_path",
            "network",
        ):
            self.assertNotIn(forbidden, schema["properties"])

    async def test_handler_matches_application_and_propagates_query_error(self):
        handler = build_search_health_knowledge_handler(self.repository)
        tool_input = {"query": "空腹血糖", "limit": 1}

        result_text = await handler(tool_input)

        self.assertEqual(
            search_health_knowledge(tool_input, self.repository).to_dict(),
            json.loads(result_text),
        )
        failing_handler = build_search_health_knowledge_handler(
            FailingKnowledgeRepository()
        )
        with self.assertRaises(HealthKnowledgeQueryError):
            await failing_handler({"query": "a1c"})

    async def test_catalog_supports_all_repository_combinations_in_fixed_order(self):
        record_repository = MagicMock()
        preview_only = build_healthvault_tool_registry()
        records = build_healthvault_tool_registry(
            record_repository=record_repository
        )
        knowledge = build_healthvault_tool_registry(
            knowledge_repository=self.repository
        )
        both = build_healthvault_tool_registry(
            record_repository=record_repository,
            knowledge_repository=self.repository,
        )
        second_both = build_healthvault_tool_registry(
            record_repository=record_repository,
            knowledge_repository=self.repository,
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
            [definition["name"] for definition in records.definitions],
        )
        self.assertEqual(
            ["preview_health_report", "search_health_knowledge", "search_official_health_web"],
            [definition["name"] for definition in knowledge.definitions],
        )
        self.assertEqual(
            [
                "preview_health_report",
                "confirm_health_records",
                "query_health_timeline",
                "compare_health_indicators",
                "search_health_knowledge",
                "search_official_health_web",
                "generate_visit_brief",
            ],
            [definition["name"] for definition in both.definitions],
        )
        self.assertIsNot(both, second_both)

    async def test_real_agent_routes_search_without_native_or_api_call(self):
        registry = build_healthvault_tool_registry(
            knowledge_repository=self.repository
        )
        adapter = BearCodeToolAdapter(registry)
        agent = make_agent(adapter)

        with patch(
            "healthvault.adapters.bearcode.tools.execute_tool",
            new=AsyncMock(return_value="unexpected native result"),
        ) as native_executor:
            result_text = await agent._execute_tool_call(
                "search_health_knowledge",
                {"query": "就诊准备", "limit": 1},
            )

        result = json.loads(result_text)
        tool_names = [definition["name"] for definition in agent.tools]
        self.assertIn("read_file", tool_names)
        self.assertIn("preview_health_report", tool_names)
        self.assertIn("search_health_knowledge", tool_names)
        self.assertNotIn("confirm_health_records", tool_names)
        self.assertEqual("ok", result["status"])
        self.assertTrue(
            result["results"][0]["citation_id"].startswith(
                "HVK:r1-2026-08-24-v1:"
            )
        )
        self.assertIn("visit_preparation", result["results"][0]["topic_tags"])
        native_executor.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
