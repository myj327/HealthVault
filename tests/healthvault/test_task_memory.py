import importlib.util
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext
from healthvault.task_memory import TaskMemoryConflict, TaskMemoryStore


class FakeRecords:
    def list_timeline(self, profile_id):
        return (SimpleNamespace(source_id=f"source-{profile_id}"),)


class TaskMemoryTests(unittest.TestCase):
    def test_cross_instance_restore_and_profile_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "task-memory.sqlite3"
            a = RuntimeContext("profile-a", "request-1")
            b = RuntimeContext("profile-b", "request-2")
            first = TaskMemoryStore(path, FakeRecords())
            saved = first.save(
                a, "task-1", expected_version=0,
                completed_steps=("query_health_timeline",),
                next_step="compare_health_indicators",
                source_refs=("source-profile-a",),
                artifact_refs=("artifact-1",),
                preferences={"language": "zh-CN"},
            )
            second = TaskMemoryStore(path, FakeRecords())
            restored = second.get(RuntimeContext("profile-a", "new-session"), "task-1")
            self.assertEqual(saved, restored)
            self.assertEqual(1, restored.version)
            self.assertIsNone(second.get(b, "task-1"))
            second.save(b, "task-1", expected_version=0, preferences={"language": "en-US"})
            self.assertEqual("zh-CN", second.get(a, "task-1").preferences["language"])
            self.assertEqual("en-US", second.get(b, "task-1").preferences["language"])

    def test_conflict_delete_and_expiry(self):
        with tempfile.TemporaryDirectory() as temp:
            store = TaskMemoryStore(Path(temp) / "memory.sqlite3", FakeRecords())
            context = RuntimeContext("profile-a", "request-1")
            store.save(context, "task-1", expected_version=0)
            with self.assertRaises(TaskMemoryConflict):
                store.save(context, "task-1", expected_version=0)
            completed = store.save(context, "task-1", expected_version=1,
                                   next_step="done", status="completed")
            self.assertEqual((2, "completed", "done"),
                             (completed.version, completed.status, completed.next_step))
            self.assertTrue(store.delete(context, "task-1"))
            self.assertIsNone(store.get(context, "task-1"))
            store.save(context, "task-1", expected_version=0, ttl=timedelta(microseconds=1))
            self.assertIsNone(store.get(context, "task-1"))
            self.assertFalse(store.delete(context, "task-1"))

    def test_schema_excludes_free_text_and_unconfirmed_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            store = TaskMemoryStore(Path(temp) / "memory.sqlite3", FakeRecords())
            context = RuntimeContext("profile-a", "request-1")
            invalid = (
                {"next_step": "diagnose diabetes"},
                {"preferences": {"diagnosis": "diabetes"}},
                {"artifact_refs": ("raw report text with spaces",)},
                {"source_refs": ("source-profile-b",)},
            )
            for fields in invalid:
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    store.save(context, "task-1", expected_version=0, **fields)
            with self.assertRaises(TypeError):
                store.save(context, "task-1", expected_version=0, report_text="raw report")
            self.assertIsNone(store.get(context, "task-1"))


def install_offline_model_stubs():
    for name, class_name in (("anthropic", "AsyncAnthropic"), ("openai", "AsyncOpenAI")):
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            module = types.ModuleType(name)
            setattr(module, class_name, MagicMock())
            sys.modules[name] = module


class RuntimeTaskMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_runtime_restores_previous_run_without_global_memory(self):
        install_offline_model_stubs()
        from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
        with tempfile.TemporaryDirectory() as temp, patch(
            "agents.agent.openai.AsyncOpenAI", return_value=MagicMock()
        ):
            root = Path(temp)
            config = dict(
                context=RuntimeContext("profile-a", "request-1"),
                database_path=root / "records.sqlite3",
                task_memory_path=root / "memory.sqlite3",
                task_id="task-1",
                model_config=HealthVaultModelConfig(
                    "openai-compatible", "synthetic-model", "https://invalid.local/v1", "fake-key"
                ),
            )
            first = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(**config))

            async def fake_run_once(_prompt):
                await first.agent._execute_tool_call("query_health_timeline", {})
                return {"text": "synthetic result", "tokens": {"input": 1, "output": 1}}

            first.agent.run_once = fake_run_once
            first_result = await first.run("Show my timeline")
            self.assertEqual("saved", first_result.task_memory_status)
            with patch.object(first.task_memory, "save", side_effect=TaskMemoryConflict("race")):
                conflict_result = await first.run("Continue my timeline")
            self.assertEqual("conflict", conflict_result.task_memory_status)
            self.assertTrue(first.manifest["task_memory_enabled"])
            self.assertFalse(first.manifest["memory_enabled"])
            self.assertFalse(first.manifest["mcp_enabled"])
            second = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                **{**config, "context": RuntimeContext("profile-a", "request-2")}
            ))
            self.assertTrue(second.manifest["task_memory_restored"])
            self.assertEqual(1, second.manifest["task_memory_version"])
            self.assertIn("query_health_timeline", second.agent._system_prompt)
            self.assertNotIn("synthetic result", second.agent._system_prompt)
            other = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                **{**config, "context": RuntimeContext("profile-b", "request-3")}
            ))
            self.assertFalse(other.manifest["task_memory_restored"])
            self.assertNotIn("# HealthVault task checkpoint", other.agent._system_prompt)

            def delete_during_save(*_args, **_kwargs):
                first.task_memory.delete(config["context"], "task-1")
                raise TaskMemoryConflict("deleted")

            with patch.object(first.task_memory, "save", side_effect=delete_during_save):
                deleted_result = await first.run("Continue after deletion")
            self.assertEqual("deleted_or_expired", deleted_result.task_memory_status)
            self.assertIsNone(first.task_memory.get(config["context"], "task-1"))

class CliTaskMemoryTests(unittest.TestCase):
    def test_chat_cli_enables_stable_task_memory_path(self):
        install_offline_model_stubs()
        from healthvault.agent_runtime import HealthVaultChatResult
        from healthvault.main import _default_task_memory_path, main
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "records.sqlite3"
            database.touch()
            fake_runtime = MagicMock()
            fake_runtime.run = unittest.mock.AsyncMock(return_value=HealthVaultChatResult(
                "ok", {"input": 0, "output": 0}, "request", [], [], "saved"
            ))
            args = ["chat", "--database", str(database), "--profile-id", "profile-a",
                    "--request-id", "request", "--task-id", "task-1",
                    "--provider", "openai-compatible", "--model", "fake",
                    "--api-base", "https://invalid.local/v1", "--prompt", "continue"]
            output = io.StringIO()
            with patch("healthvault.main.load_healthvault_model_config", return_value=MagicMock()), \
                 patch("healthvault.main.build_healthvault_agent_runtime", return_value=fake_runtime) as builder, \
                 redirect_stdout(output):
                code = main(args)
            self.assertEqual(0, code)
            runtime_config = builder.call_args.args[0]
            self.assertEqual("task-1", runtime_config.task_id)
            self.assertEqual(_default_task_memory_path(database), runtime_config.task_memory_path)
            self.assertIn('"task_memory_status": "saved"', output.getvalue())

    def test_chat_cli_attaches_explicit_mcp_for_one_run(self):
        install_offline_model_stubs()
        from healthvault.agent_runtime import HealthVaultChatResult
        from healthvault.main import main

        class Attachment:
            entered = False
            closed = False

            async def __aenter__(self):
                self.entered = True
                return self

            async def __aexit__(self, *_exc):
                self.closed = True

        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "records.sqlite3"
            database.touch()
            attachment = Attachment()
            fake_runtime = MagicMock()
            fake_runtime.run = unittest.mock.AsyncMock(return_value=HealthVaultChatResult(
                "ok", {"input": 0, "output": 0}, "request", [], [], "disabled"
            ))

            async def attach(runtime, *, config_path):
                self.assertIs(runtime, fake_runtime)
                self.assertEqual("approved.json", config_path)
                return attachment

            args = ["chat", "--database", str(database), "--profile-id", "profile-a",
                    "--request-id", "request", "--mcp-config", "approved.json",
                    "--provider", "openai-compatible", "--model", "fake",
                    "--api-base", "https://invalid.local/v1", "--prompt", "continue"]
            with patch("healthvault.main.load_healthvault_model_config", return_value=MagicMock()), \
                 patch("healthvault.main.build_healthvault_agent_runtime", return_value=fake_runtime), \
                 patch("healthvault.mcp.runtime_bridge.attach_approved_mcp", new=attach), \
                 redirect_stdout(io.StringIO()):
                code = main(args)
            self.assertEqual(0, code)
            self.assertTrue(attachment.entered)
            self.assertTrue(attachment.closed)
            fake_runtime.run.assert_awaited_once_with("continue")


if __name__ == "__main__":
    unittest.main()
