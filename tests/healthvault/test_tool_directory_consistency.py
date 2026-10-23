import sqlite3
import tempfile
import unittest
from pathlib import Path

from healthvault.adapters.bearcode.tools import BearCodeToolAdapter
from healthvault.evolution.skill_patch import ALLOWED_TOOL_PATH_VALUES, PATCH_SCHEMA
from healthvault.infrastructure.local_health_knowledge import LocalHealthKnowledgeRepository
from healthvault.infrastructure.sqlite_health_records import (
    ReadOnlySQLiteHealthRecordRepository,
    SQLiteHealthRecordRepository,
)
from healthvault.mcp.server import HealthVaultMCPServer, READ_ONLY_TOOLS, build_read_only_registry
from healthvault.runtime_context import RuntimeContext
from healthvault.tools.catalog import build_healthvault_tool_registry
from healthvault.tools.directory import (
    HARNESS_READ_ONLY_TOOL_NAMES,
    MCP_TOOL_NAMES,
    PROFILE_BOUND_TOOL_NAMES,
    SKILL_PATCH_TOOL_NAMES,
    TOOL_SPECS,
    tool_definition,
)


class ToolDirectoryConsistencyTests(unittest.TestCase):
    def test_all_surfaces_are_derived_from_one_directory(self):
        self.assertEqual(HARNESS_READ_ONLY_TOOL_NAMES, MCP_TOOL_NAMES)
        self.assertEqual(MCP_TOOL_NAMES, READ_ONLY_TOOLS)
        self.assertEqual(SKILL_PATCH_TOOL_NAMES, ALLOWED_TOOL_PATH_VALUES)
        self.assertEqual(
            list(SKILL_PATCH_TOOL_NAMES),
            PATCH_SCHEMA["properties"]["patches"]["items"]["properties"]
            ["tool_path"]["items"]["enum"],
        )
        self.assertEqual(len(TOOL_SPECS), len({spec.name for spec in TOOL_SPECS}))
        self.assertTrue(set(PROFILE_BOUND_TOOL_NAMES) <= set(MCP_TOOL_NAMES))
        for spec in TOOL_SPECS:
            self.assertEqual(spec.definition(), tool_definition(spec.name))

    def test_harness_and_mcp_advertise_the_same_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "records.sqlite3"
            SQLiteHealthRecordRepository(database)
            records = ReadOnlySQLiteHealthRecordRepository(database)
            knowledge = LocalHealthKnowledgeRepository()
            native_registry = build_healthvault_tool_registry(
                records, knowledge, allowed_names=HARNESS_READ_ONLY_TOOL_NAMES,
            )
            adapter = BearCodeToolAdapter(
                native_registry,
                include_native_tools=False,
                allowed_domain_tools=HARNESS_READ_ONLY_TOOL_NAMES,
                runtime_context=RuntimeContext("profile-a", "request-a"),
            )
            mcp_registry = build_read_only_registry(records, knowledge)
            mcp = HealthVaultMCPServer("profile-a", mcp_registry)

            native = {item["name"]: item for item in adapter.combined_definitions}
            advertised = {item["name"]: item for item in mcp.advertised_tools()}
            self.assertEqual(set(native), set(advertised))
            for name, definition in native.items():
                self.assertEqual(definition["description"], advertised[name]["description"])
                expected_schema = dict(definition["input_schema"])
                expected_schema["additionalProperties"] = False
                self.assertEqual(expected_schema, advertised[name]["inputSchema"])
                self.assertEqual({"readOnlyHint": True}, advertised[name]["annotations"])

    def test_agent_repository_uses_sqlite_enforced_read_only_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "records.sqlite3"
            SQLiteHealthRecordRepository(database)
            repository = ReadOnlySQLiteHealthRecordRepository(database)
            connection = repository._connect()
            try:
                self.assertEqual(1, connection.execute("PRAGMA query_only").fetchone()[0])
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute(
                        "INSERT INTO confirmed_reports "
                        "(idempotency_key, profile_id, source_id, source_sha256, report_id, "
                        "report_date, facility, raw_report_text, confirmed_at) "
                        "VALUES ('i','p','s','h','r','2026-01-01','f','raw','now')"
                    )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
