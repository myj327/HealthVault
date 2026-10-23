import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.bootstrap import (
    BLOCKED_BY,
    HealthVaultBootstrapConfig,
    HealthVaultBootstrapError,
    build_healthvault_components,
)
from healthvault.infrastructure.local_health_knowledge import (
    DEFAULT_HEALTH_KNOWLEDGE_PATH,
    LocalHealthKnowledgeRepository,
)
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.prompt import HEALTHVAULT_SYSTEM_PROMPT


READ_ONLY_TOOLS = [
    "preview_health_report",
    "query_health_timeline",
    "compare_health_indicators",
    "search_health_knowledge",
    "search_official_health_web",
    "generate_visit_brief",
]
DEVELOPMENT_TOOLS = [
    "preview_health_report",
    "confirm_health_records",
    "query_health_timeline",
    "compare_health_indicators",
    "search_health_knowledge",
    "search_official_health_web",
    "generate_visit_brief",
]


class HealthVaultBootstrapTests(unittest.TestCase):
    def build(self, directory, mode="read_only", **overrides):
        config = HealthVaultBootstrapConfig(
            mode=mode,
            database_path=Path(directory) / f"{mode}.sqlite3",
            **overrides,
        )
        return build_healthvault_components(config)

    def test_read_only_manifest_and_components_are_safe_and_exact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "agents.agent.Agent",
                side_effect=AssertionError("Agent must not be constructed"),
            ) as agent_constructor:
                components = self.build(
                    temp_dir,
                    database_mode="ephemeral",
                )

            self.assertIsInstance(
                components.record_repository,
                SQLiteHealthRecordRepository,
            )
            self.assertIsInstance(
                components.knowledge_repository,
                LocalHealthKnowledgeRepository,
            )
            self.assertEqual(HEALTHVAULT_SYSTEM_PROMPT, components.system_prompt)
            self.assertEqual(
                DEVELOPMENT_TOOLS,
                [
                    definition["name"]
                    for definition in components.registry.definitions
                ],
            )
            self.assertEqual(
                READ_ONLY_TOOLS,
                [
                    definition["name"]
                    for definition in components.adapter.combined_definitions
                ],
            )
            expected_manifest = {
                "status": "ok",
                "mode": "read_only",
                "tool_names": READ_ONLY_TOOLS,
                "native_tools_enabled": False,
                "confirmation_tool_enabled": False,
                "confirmation_boundary": "disabled",
                "live_chat_ready": False,
                "agent_constructed": False,
                "api_config_loaded": False,
                "blocked_by": list(BLOCKED_BY),
                "prompt_sha256": hashlib.sha256(
                    HEALTHVAULT_SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
                "knowledge_corpus": "r1-2026-08-24-v1",
                "knowledge_strategy": "hybrid_v1",
                "knowledge_engine": "sqlite_fts5_bm25_trigram",
                "knowledge_mode": "lexical",
                "dense_fallback_reason": None,
                "database_mode": "ephemeral",
            }
            self.assertEqual(expected_manifest, components.manifest)
            self.assertNotIn(str(Path(temp_dir).resolve()), str(components.manifest))
            agent_constructor.assert_not_called()

    def test_development_enables_only_legacy_confirmation_and_no_native_tools(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            components = self.build(temp_dir, mode="development")

            self.assertEqual(
                DEVELOPMENT_TOOLS,
                components.manifest["tool_names"],
            )
            self.assertTrue(components.manifest["confirmation_tool_enabled"])
            self.assertEqual(
                "stage1_legacy_unsafe",
                components.manifest["confirmation_boundary"],
            )
            self.assertFalse(components.manifest["native_tools_enabled"])
            self.assertFalse(components.manifest["live_chat_ready"])
            self.assertFalse(components.manifest["agent_constructed"])
            self.assertFalse(components.manifest["api_config_loaded"])
            self.assertEqual(list(BLOCKED_BY), components.manifest["blocked_by"])

    def test_custom_knowledge_and_repeated_builds_are_independent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_root = Path(temp_dir) / "custom-corpus"
            shutil.copytree(DEFAULT_HEALTH_KNOWLEDGE_PATH.parents[1], corpus_root)
            custom_knowledge = corpus_root / "chunks" / "hybrid_v1.json"
            first = build_healthvault_components(
                HealthVaultBootstrapConfig(
                    mode="read_only",
                    database_path=Path(temp_dir) / "first.sqlite3",
                    knowledge_path=custom_knowledge,
                )
            )
            second = build_healthvault_components(
                HealthVaultBootstrapConfig(
                    mode="read_only",
                    database_path=Path(temp_dir) / "second.sqlite3",
                    knowledge_path=custom_knowledge,
                )
            )

            self.assertEqual("r1-2026-08-24-v1", first.manifest["knowledge_corpus"])
            self.assertEqual(first.manifest, second.manifest)
            self.assertIsNot(first.registry, second.registry)
            self.assertIsNot(first.adapter, second.adapter)
            self.assertIsNot(first.record_repository, second.record_repository)
            self.assertIsNot(first.knowledge_repository, second.knowledge_repository)
            first.registry.register(
                {
                    "name": "synthetic_isolation_probe",
                    "description": "Test-only registry independence probe.",
                    "input_schema": {"type": "object", "properties": {}},
                },
                lambda value: value,
            )
            self.assertFalse(second.registry.contains("synthetic_isolation_probe"))

    def test_custom_natural_artifact_reports_actual_repository_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_root = Path(temp_dir) / "custom-natural-corpus"
            shutil.copytree(DEFAULT_HEALTH_KNOWLEDGE_PATH.parents[1], corpus_root)
            components = build_healthvault_components(
                HealthVaultBootstrapConfig(
                    mode="read_only",
                    database_path=Path(temp_dir) / "natural.sqlite3",
                    knowledge_path=corpus_root / "chunks" / "natural_section.json",
                )
            )

            metadata = components.knowledge_repository.index_metadata
            self.assertEqual("natural_section", metadata["strategy"])
            self.assertEqual(
                metadata["corpus_version"], components.manifest["knowledge_corpus"]
            )
            self.assertEqual(
                metadata["strategy"], components.manifest["knowledge_strategy"]
            )
            self.assertEqual(
                metadata["engine_id"], components.manifest["knowledge_engine"]
            )

    def test_invalid_modes_paths_and_database_mode_fail_without_agent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_knowledge = Path(temp_dir) / "missing.json"
            cases = (
                HealthVaultBootstrapConfig(
                    mode="invalid",
                    database_path=Path(temp_dir) / "invalid.sqlite3",
                ),
                HealthVaultBootstrapConfig(
                    mode="read_only",
                    database_path=Path(temp_dir) / "invalid-mode.sqlite3",
                    database_mode="unknown",
                ),
                HealthVaultBootstrapConfig(
                    mode="read_only",
                    database_path=Path(temp_dir) / "missing-knowledge.sqlite3",
                    knowledge_path=missing_knowledge,
                ),
                HealthVaultBootstrapConfig(
                    mode="read_only",
                    database_path=Path(temp_dir),
                ),
            )
            for config in cases:
                with self.subTest(config=config):
                    with self.assertRaises(HealthVaultBootstrapError):
                        build_healthvault_components(config)


if __name__ == "__main__":
    unittest.main()
