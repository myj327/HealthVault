import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from healthvault.agent_runtime import (
    HealthVaultAgentRuntimeConfig,
    build_healthvault_agent_runtime,
)
from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.bootstrap import READ_ONLY_TOOL_NAMES
from healthvault.tools.directory import HARNESS_LOCAL_READ_ONLY_TOOL_NAMES
from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext


FIXTURE_DIR = (
    Path(__file__).parents[1]
    / "fixtures"
    / "health_reports"
)


def seed_profile(
    repository,
    profile_id: str,
    filename: str = "synthetic_checkup_2024.txt",
    source_id: str | None = None,
) -> None:
    report_text = (FIXTURE_DIR / filename).read_text(encoding="utf-8")
    result = confirm_health_records(
        {
            "profile_id": profile_id,
            "source_id": source_id or f"source-{profile_id}",
            "report_text": report_text,
            "expected_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
            "user_confirmed": True,
        },
        repository,
    )
    if result.status != "confirmed":
        raise AssertionError(result)


class HealthVaultAgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_search_default_is_disabled_and_rejects_call(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            agent_module.openai, "AsyncOpenAI", return_value=MagicMock(),
        ):
            runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                context=RuntimeContext("profile", "request-offline"),
                database_path=Path(temp_dir) / "offline.sqlite3",
                model_config=HealthVaultModelConfig(
                    "openai-compatible", "synthetic-model",
                    "https://invalid.local/v1", "synthetic-test-key",
                ),
            ))
            self.assertNotIn("search_official_health_web", runtime.manifest["tool_names"])
            self.assertNotIn(
                "search_official_health_web",
                runtime.agent.runtime_policy.allowed_tool_names,
            )
            self.assertEqual(
                "Tool unavailable under Agent runtime policy: search_official_health_web",
                await runtime.agent._execute_tool_call("search_official_health_web", {}),
            )

    def test_runtime_context_requires_non_empty_application_keys(self):
        for profile_id, request_id in (
            ("", "request"),
            ("   ", "request"),
            ("profile", ""),
            ("profile", "   "),
        ):
            with self.subTest(
                profile_id=profile_id,
                request_id=request_id,
            ):
                with self.assertRaises(ValueError):
                    RuntimeContext(profile_id, request_id)

    async def test_openai_fake_tool_loop_uses_main_agent_and_isolated_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "synthetic.sqlite3"
            skill_root = Path(temp_dir) / ".bear" / "skills"
            allowed_dir = skill_root / "allowed"
            ignored_dir = skill_root / "ignored"
            allowed_dir.mkdir(parents=True)
            ignored_dir.mkdir(parents=True)
            (allowed_dir / "SKILL.md").write_text(
                "---\nname: allowed\n---\n\nALLOWED_SKILL_MARKER\n",
                encoding="utf-8",
            )
            (ignored_dir / "SKILL.md").write_text(
                "---\nname: ignored\n---\n\nIGNORED_SKILL_MARKER\n",
                encoding="utf-8",
            )
            writable_repository = SQLiteHealthRecordRepository(database_path)
            seed_profile(writable_repository, "profile-a")
            seed_profile(writable_repository, "profile-b")

            with patch.object(
                agent_module.openai,
                "AsyncOpenAI",
                return_value=MagicMock(),
            ), patch(
                "agents.skills.discover_skills",
                side_effect=AssertionError("global Skill discovery must not run"),
            ) as discover_skills:
                runtime = build_healthvault_agent_runtime(
                    HealthVaultAgentRuntimeConfig(
                        context=RuntimeContext(
                            profile_id="profile-a",
                            request_id="request-001",
                        ),
                        database_path=database_path,
                        model_config=HealthVaultModelConfig(
                            provider="openai-compatible",
                            model="synthetic-model",
                            api_base="https://invalid.local/v1",
                            api_key="synthetic-test-key",
                        ),
                        skill_root=skill_root,
                        skill_names=("allowed",),
                    )
                )

            context_bound_names = {
                "query_health_timeline",
                "compare_health_indicators",
                "generate_visit_brief",
            }
            for definition in runtime.agent.tools:
                if definition["name"] in context_bound_names:
                    self.assertNotIn(
                        "profile_id",
                        definition["input_schema"]["properties"],
                    )
            for definition in runtime.components.registry.definitions:
                if definition["name"] in context_bound_names:
                    self.assertIn(
                        "profile_id",
                        definition["input_schema"]["properties"],
                    )
            calls = 0

            async def fake_model_call():
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "query_health_timeline",
                                        "arguments": "{}",
                                    },
                                }],
                            }
                        }],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                    }
                tool_messages = [
                    message
                    for message in runtime.agent._openai_messages
                    if message.get("role") == "tool"
                ]
                self.assertEqual(1, len(tool_messages))
                self.assertIn("source-profile-a", tool_messages[0]["content"])
                self.assertNotIn("source-profile-b", tool_messages[0]["content"])
                source_id = json.loads(tool_messages[0]["content"])["available_citation_ids"][0]
                final_text = f"SYNTHETIC_FINAL_TEXT [{source_id}]"
                runtime.agent._emit_text(final_text)
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": final_text,
                            "tool_calls": None,
                        }
                    }],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }

            runtime.agent._call_openai_stream = fake_model_call
            with patch(
                "agents.agent.start_memory_prefetch",
                side_effect=AssertionError("Memory prefetch must be disabled"),
            ) as memory_prefetch, patch(
                "agents.agent.save_session",
                side_effect=AssertionError("Session save must be disabled"),
            ) as session_save, patch.object(
                runtime.agent._mcp_manager,
                "load_and_connect",
                new=AsyncMock(),
            ) as mcp_connect, patch.object(
                agent_module,
                "start_spinner",
            ), patch.object(
                agent_module,
                "stop_spinner",
            ), patch.object(
                agent_module,
                "print_tool_call",
            ), patch.object(
                agent_module,
                "print_tool_result",
            ), patch.object(
                agent_module,
                "print_cost",
            ), patch.object(
                agent_module,
                "print_divider",
            ):
                result = await runtime.run("列出合成档案时间线")

        self.assertFalse(runtime.agent.is_sub_agent)
        self.assertEqual(2, calls)
        self.assertTrue(result.text.startswith("SYNTHETIC_FINAL_TEXT [REC:"))
        self.assertEqual({"input": 17, "output": 5}, result.tokens)
        self.assertEqual(
            list(HARNESS_LOCAL_READ_ONLY_TOOL_NAMES),
            [definition["name"] for definition in runtime.agent.tools],
        )
        self.assertIn("ALLOWED_SKILL_MARKER", runtime.agent._system_prompt)
        self.assertNotIn("IGNORED_SKILL_MARKER", runtime.agent._system_prompt)
        self.assertEqual(["allowed"], [item["name"] for item in runtime.manifest["project_skills"]])
        self.assertNotIn(str(skill_root), str(runtime.manifest))
        memory_prefetch.assert_not_called()
        session_save.assert_not_called()
        mcp_connect.assert_not_awaited()
        discover_skills.assert_not_called()

    async def test_runtime_rejects_confirmation_native_mcp_skill_and_unknown(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            agent_module.openai,
            "AsyncOpenAI",
            return_value=MagicMock(),
        ):
            runtime = build_healthvault_agent_runtime(
                HealthVaultAgentRuntimeConfig(
                    context=RuntimeContext("profile", "request-rejection"),
                    database_path=Path(temp_dir) / "rejection.sqlite3",
                    model_config=HealthVaultModelConfig(
                        "openai-compatible",
                        "synthetic-model",
                        "https://invalid.local/v1",
                        "synthetic-test-key",
                    ),
                )
            )
            runtime.agent._mcp_manager.is_mcp_tool = MagicMock(return_value=True)
            runtime.agent._mcp_manager.call_tool = AsyncMock(
                return_value="unexpected MCP result"
            )
            with patch(
                "agents.agent.execute_tool",
                new=AsyncMock(return_value="unexpected native result"),
            ) as native_executor, patch(
                "agents.skills.execute_skill",
                side_effect=AssertionError("global Skill execution must be disabled"),
            ) as skill_executor:
                results = {
                    name: await runtime.agent._execute_tool_call(name, {})
                    for name in (
                        "confirm_health_records",
                        "read_file",
                        "mcp__synthetic__probe",
                        "skill",
                        "unknown_tool",
                    )
                }

        for name, result in results.items():
            self.assertEqual(
                f"Tool unavailable under Agent runtime policy: {name}",
                result,
            )
        native_executor.assert_not_awaited()
        skill_executor.assert_not_called()
        runtime.agent._mcp_manager.is_mcp_tool.assert_not_called()
        runtime.agent._mcp_manager.call_tool.assert_not_awaited()

    async def test_execution_allowlist_blocks_all_special_routes_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            agent_module.openai,
            "AsyncOpenAI",
            return_value=MagicMock(),
        ):
            runtime = build_healthvault_agent_runtime(
                HealthVaultAgentRuntimeConfig(
                    context=RuntimeContext("profile", "request-special-gate"),
                    allow_web_search=True,
                    database_path=Path(temp_dir) / "special-gate.sqlite3",
                    model_config=HealthVaultModelConfig(
                        "openai-compatible",
                        "synthetic-model",
                        "https://invalid.local/v1",
                        "synthetic-test-key",
                    ),
                )
            )
            runtime.agent._execute_agent_tool = AsyncMock(return_value="UNEXPECTED_AGENT")
            runtime.agent._execute_compact_context_tool = AsyncMock(
                return_value="UNEXPECTED_COMPACT"
            )
            runtime.agent._execute_plan_mode_tool = AsyncMock(
                return_value="UNEXPECTED_PLAN"
            )
            runtime.agent._tool_executor = AsyncMock(return_value="ALLOWED_TOOL_RESULT")
            runtime.agent._mcp_manager.is_mcp_tool = MagicMock(return_value=True)
            runtime.agent._mcp_manager.call_tool = AsyncMock(
                return_value="UNEXPECTED_MCP"
            )

            with patch.object(
                agent_module,
                "Agent",
                side_effect=AssertionError("sub-agent must not be constructed"),
            ) as sub_agent_constructor, patch.object(
                Path,
                "write_text",
                side_effect=AssertionError("special tool gate must not write files"),
            ) as write_text, patch(
                "agents.agent.execute_tool",
                new=AsyncMock(return_value="UNEXPECTED_NATIVE_EXECUTOR"),
            ) as native_executor:
                blocked = {
                    name: await runtime.agent._execute_tool_call(name, {})
                    for name in (
                        "agent",
                        "compact_context",
                        "enter_plan_mode",
                        "exit_plan_mode",
                    )
                }
                allowed = {
                    name: await runtime.agent._execute_tool_call(name, {})
                    for name in READ_ONLY_TOOL_NAMES
                }

        self.assertEqual(
            frozenset(READ_ONLY_TOOL_NAMES),
            runtime.agent.runtime_policy.allowed_tool_names,
        )
        for name, result in blocked.items():
            self.assertEqual(
                f"Tool unavailable under Agent runtime policy: {name}",
                result,
            )
        self.assertEqual(
            {name: "ALLOWED_TOOL_RESULT" for name in READ_ONLY_TOOL_NAMES},
            allowed,
        )
        runtime.agent._execute_agent_tool.assert_not_awaited()
        runtime.agent._execute_compact_context_tool.assert_not_awaited()
        runtime.agent._execute_plan_mode_tool.assert_not_awaited()
        sub_agent_constructor.assert_not_called()
        runtime.agent._mcp_manager.is_mcp_tool.assert_not_called()
        runtime.agent._mcp_manager.call_tool.assert_not_awaited()
        self.assertEqual(len(READ_ONLY_TOOL_NAMES), runtime.agent._tool_executor.await_count)
        native_executor.assert_not_awaited()
        write_text.assert_not_called()

    async def test_fake_model_cannot_invoke_unadvertised_agent_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            agent_module.openai,
            "AsyncOpenAI",
            return_value=MagicMock(),
        ) as model_client:
            runtime = build_healthvault_agent_runtime(
                HealthVaultAgentRuntimeConfig(
                    context=RuntimeContext("profile", "request-fake-special-gate"),
                    database_path=Path(temp_dir) / "fake-special-gate.sqlite3",
                    model_config=HealthVaultModelConfig(
                        "openai-compatible",
                        "synthetic-model",
                        "https://invalid.local/v1",
                        "synthetic-test-key",
                    ),
                )
            )
            model_client.reset_mock()
            runtime.agent._execute_agent_tool = AsyncMock(return_value="UNEXPECTED_AGENT")
            runtime.agent._execute_compact_context_tool = AsyncMock(
                return_value="UNEXPECTED_COMPACT"
            )
            runtime.agent._execute_plan_mode_tool = AsyncMock(
                return_value="UNEXPECTED_PLAN"
            )
            runtime.agent._tool_executor = AsyncMock(return_value="UNEXPECTED_EXECUTOR")
            runtime.agent._mcp_manager.is_mcp_tool = MagicMock(return_value=True)
            runtime.agent._mcp_manager.call_tool = AsyncMock(
                return_value="UNEXPECTED_MCP"
            )
            calls = 0

            async def fake_model_call():
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "blocked-agent-call",
                                    "type": "function",
                                    "function": {
                                        "name": "agent",
                                        "arguments": json.dumps({
                                            "type": "general",
                                            "prompt": "synthetic bypass probe",
                                        }),
                                    },
                                }],
                            }
                        }],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    }
                tool_messages = [
                    item
                    for item in runtime.agent._openai_messages
                    if item.get("role") == "tool"
                ]
                self.assertEqual(1, len(tool_messages))
                self.assertEqual(
                    "Tool unavailable under Agent runtime policy: agent",
                    tool_messages[0]["content"],
                )
                runtime.agent._emit_text("SPECIAL_GATE_OK")
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "SPECIAL_GATE_OK",
                            "tool_calls": None,
                        }
                    }],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                }

            runtime.agent._call_openai_stream = fake_model_call
            with patch.object(
                agent_module,
                "Agent",
                side_effect=AssertionError("sub-agent must not be constructed"),
            ) as sub_agent_constructor, patch.object(
                Path,
                "write_text",
                side_effect=AssertionError("special tool gate must not write files"),
            ) as write_text, patch(
                "agents.agent.execute_tool",
                new=AsyncMock(return_value="UNEXPECTED_NATIVE_EXECUTOR"),
            ) as native_executor, patch.object(
                agent_module,
                "start_spinner",
            ), patch.object(
                agent_module,
                "stop_spinner",
            ), patch.object(
                agent_module,
                "print_tool_call",
            ), patch.object(
                agent_module,
                "print_tool_result",
            ), patch.object(
                agent_module,
                "print_cost",
            ):
                with self.assertRaisesRegex(RuntimeError, "completion conditions remain unsatisfied"):
                    await runtime.run("synthetic special route probe")

        # An unavailable tool plus an ungrounded final is not a completed task.
        self.assertEqual(3, calls)
        runtime.agent._execute_agent_tool.assert_not_awaited()
        runtime.agent._execute_compact_context_tool.assert_not_awaited()
        runtime.agent._execute_plan_mode_tool.assert_not_awaited()
        sub_agent_constructor.assert_not_called()
        model_client.assert_not_called()
        runtime.agent._mcp_manager.is_mcp_tool.assert_not_called()
        runtime.agent._mcp_manager.call_tool.assert_not_awaited()
        runtime.agent._tool_executor.assert_not_awaited()
        native_executor.assert_not_awaited()
        write_text.assert_not_called()

    async def test_all_context_bound_tools_exclude_the_other_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            agent_module.openai,
            "AsyncOpenAI",
            return_value=MagicMock(),
        ):
            database_path = Path(temp_dir) / "isolation.sqlite3"
            writable_repository = SQLiteHealthRecordRepository(database_path)
            seed_profile(
                writable_repository,
                "profile-a",
                "synthetic_checkup_2024.txt",
                "source-a-2024",
            )
            seed_profile(
                writable_repository,
                "profile-a",
                "synthetic_checkup_2025.txt",
                "source-a-2025",
            )
            seed_profile(
                writable_repository,
                "profile-b",
                "synthetic_checkup_2024.txt",
                "source-b-2024",
            )
            runtime = build_healthvault_agent_runtime(
                HealthVaultAgentRuntimeConfig(
                    context=RuntimeContext("profile-a", "request-isolation"),
                    database_path=database_path,
                    model_config=HealthVaultModelConfig(
                        "openai-compatible",
                        "synthetic-model",
                        "https://invalid.local/v1",
                        "synthetic-test-key",
                    ),
                )
            )
            repository = runtime.components.record_repository

            results = {
                name: json.loads(await runtime.components.adapter.execute(name, {}))
                for name in (
                    "query_health_timeline",
                    "compare_health_indicators",
                    "generate_visit_brief",
                )
            }
            for name in results:
                rejected = await runtime.components.adapter.execute(
                    name,
                    {"profile_id": "profile-b"},
                )
                self.assertEqual(
                    "Tool input rejected: profile_id is supplied by RuntimeContext.",
                    rejected,
                )

        for result in results.values():
            self.assertEqual("profile-a", result["profile_id"])
            serialized = json.dumps(result, ensure_ascii=False)
            self.assertNotIn("profile-b", serialized)
            self.assertNotIn("source-b-2024", serialized)
        self.assertEqual(2, results["query_health_timeline"]["report_count"])
        self.assertEqual(2, results["compare_health_indicators"]["report_count"])
        self.assertEqual(2, results["generate_visit_brief"]["report_count"])

    def test_provider_selects_explicit_client_branch(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            agent_module.openai,
            "AsyncOpenAI",
            return_value=MagicMock(),
        ) as openai_client, patch.object(
            agent_module.anthropic,
            "AsyncAnthropic",
            return_value=MagicMock(),
        ) as anthropic_client:
            base = {
                "context": RuntimeContext("profile", "request"),
                "database_path": Path(temp_dir) / "provider.sqlite3",
            }
            openai_runtime = build_healthvault_agent_runtime(
                HealthVaultAgentRuntimeConfig(
                    **base,
                    model_config=HealthVaultModelConfig(
                        "openai-compatible",
                        "model-a",
                        "https://openai.invalid/v1",
                        "key-a",
                    ),
                )
            )
            anthropic_runtime = build_healthvault_agent_runtime(
                HealthVaultAgentRuntimeConfig(
                    **base,
                    model_config=HealthVaultModelConfig(
                        "anthropic",
                        "model-b",
                        "https://anthropic.invalid",
                        "key-b",
                    ),
                )
            )

        self.assertTrue(openai_runtime.agent.use_openai)
        self.assertFalse(anthropic_runtime.agent.use_openai)
        openai_client.assert_called_once_with(
            base_url="https://openai.invalid/v1",
            api_key="key-a",
        )
        anthropic_client.assert_called_once_with(
            api_key="key-b",
            base_url="https://anthropic.invalid",
        )


if __name__ == "__main__":
    unittest.main()
