import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from healthvault.main import main
from healthvault.agent_runtime import HealthVaultChatResult
from healthvault.model_config import HealthVaultModelConfig
from healthvault.tools.directory import HARNESS_TOOL_NAMES, HARNESS_READ_ONLY_TOOL_NAMES


REAL_TEMPORARY_DIRECTORY = tempfile.TemporaryDirectory


class RecordingTemporaryDirectory:
    paths = []

    def __init__(self):
        self._inner = REAL_TEMPORARY_DIRECTORY()

    def __enter__(self):
        path = self._inner.__enter__()
        self.paths.append(Path(path))
        return path

    def __exit__(self, exc_type, exc_value, traceback):
        return self._inner.__exit__(exc_type, exc_value, traceback)


def run_main(arguments):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(arguments)
    return exit_code, stdout.getvalue(), stderr.getvalue()


class HealthVaultMainTests(unittest.TestCase):
    def test_help_does_not_build_components_or_expose_unsafe_parameters(self):
        with patch(
            "healthvault.main.build_healthvault_components",
            side_effect=AssertionError("Help must not build components"),
        ) as builder:
            exit_code, stdout, stderr = run_main(["--help"])

        self.assertEqual(0, exit_code)
        self.assertIn("check", stdout)
        self.assertIn("tools", stdout)
        self.assertIn("chat", stdout)
        for forbidden in (
            "--prompt",
            "--model",
            "--provider",
            "--api-key",
            "--profile",
            "--confirm",
        ):
            self.assertNotIn(forbidden, stdout)
        self.assertEqual("", stderr)
        builder.assert_not_called()

    def test_default_check_is_single_json_and_removes_ephemeral_database(self):
        RecordingTemporaryDirectory.paths = []
        with patch(
            "healthvault.main.tempfile.TemporaryDirectory",
            RecordingTemporaryDirectory,
        ):
            exit_code, stdout, stderr = run_main(["check"])

        manifest = json.loads(stdout)
        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr)
        self.assertEqual("read_only", manifest["mode"])
        self.assertEqual("ephemeral", manifest["database_mode"])
        self.assertFalse(manifest["native_tools_enabled"])
        self.assertFalse(manifest["confirmation_tool_enabled"])
        self.assertFalse(manifest["live_chat_ready"])
        self.assertEqual(1, stdout.count("\n"))
        self.assertEqual(1, len(RecordingTemporaryDirectory.paths))
        self.assertFalse(RecordingTemporaryDirectory.paths[0].exists())

    def test_tools_and_development_modes_list_without_executing_handlers(self):
        with patch(
            "healthvault.tools.registry.ToolRegistry.execute",
            new=AsyncMock(return_value="unexpected tool result"),
        ) as registry_execute, patch(
            "healthvault.adapters.bearcode.tools.execute_tool",
            new=AsyncMock(return_value="unexpected native result"),
        ) as native_execute, patch(
            "agents.agent.Agent",
            side_effect=AssertionError("Agent must not be constructed"),
        ) as agent_constructor, patch(
            "agents.main.load_dotenv",
            new=MagicMock(),
        ) as env_loader, patch(
            "agents.mcp_client.McpManager.load_and_connect",
            new=AsyncMock(),
        ) as mcp_connect:
            read_code, read_stdout, read_stderr = run_main(["tools"])
            dev_code, dev_stdout, dev_stderr = run_main(
                ["tools", "--mode", "development"]
            )

        read_tools = json.loads(read_stdout)
        dev_tools = json.loads(dev_stdout)
        self.assertEqual((0, 0), (read_code, dev_code))
        self.assertEqual(("", ""), (read_stderr, dev_stderr))
        self.assertEqual(set(HARNESS_READ_ONLY_TOOL_NAMES), set(read_tools["tool_names"]))
        self.assertNotIn("confirm_health_records", read_tools["tool_names"])
        self.assertEqual(set(HARNESS_TOOL_NAMES), set(dev_tools["tool_names"]))
        self.assertIn("confirm_health_records", dev_tools["tool_names"])
        self.assertEqual("disabled", read_tools["confirmation_boundary"])
        self.assertEqual(
            "stage1_legacy_unsafe",
            dev_tools["confirmation_boundary"],
        )
        for output in (read_tools, dev_tools):
            self.assertFalse(output["native_tools_enabled"])
            self.assertFalse(output["live_chat_ready"])
            self.assertFalse(output["agent_constructed"])
            self.assertFalse(output["api_config_loaded"])
        registry_execute.assert_not_awaited()
        native_execute.assert_not_awaited()
        agent_constructor.assert_not_called()
        env_loader.assert_not_called()
        mcp_connect.assert_not_awaited()

    def test_offline_commands_do_not_read_chat_model_configuration(self):
        with patch(
            "healthvault.main.load_healthvault_model_config",
            side_effect=AssertionError("offline commands must not read a key"),
        ) as config_loader:
            for arguments in (["--help"], ["check"], ["tools"]):
                with self.subTest(arguments=arguments):
                    exit_code, _stdout, _stderr = run_main(arguments)
                    self.assertEqual(0, exit_code)

        config_loader.assert_not_called()

    def test_chat_missing_configuration_fails_before_runtime_or_model(self):
        required = [
            "chat",
            "--database", "missing.sqlite3",
            "--profile-id", "synthetic-profile",
            "--request-id", "request-001",
            "--provider", "openai-compatible",
            "--model", "synthetic-model",
            "--api-base", "https://invalid.local/v1",
            "--prompt", "synthetic prompt",
        ]
        with patch(
            "healthvault.main.build_healthvault_agent_runtime",
            side_effect=AssertionError("runtime must not be constructed"),
        ) as runtime_builder:
            exit_code, stdout, stderr = run_main(required)

        self.assertNotEqual(0, exit_code)
        self.assertEqual("", stdout)
        self.assertIn("Error:", stderr)
        self.assertNotIn("Traceback", stderr)
        runtime_builder.assert_not_called()

    def test_development_chat_is_rejected_without_reading_api_key(self):
        arguments = [
            "chat",
            "--mode", "development",
            "--database", "synthetic.sqlite3",
            "--profile-id", "profile",
            "--request-id", "request",
            "--provider", "anthropic",
            "--model", "model",
            "--api-base", "https://invalid.local",
            "--prompt", "prompt",
        ]
        with patch(
            "healthvault.main.load_healthvault_model_config",
            side_effect=AssertionError("key must not be read"),
        ) as config_loader:
            exit_code, stdout, stderr = run_main(arguments)

        self.assertEqual(1, exit_code)
        self.assertEqual("", stdout)
        self.assertIn("read_only", stderr)
        config_loader.assert_not_called()

    def test_chat_requires_each_argument_and_provider_key_before_model_call(self):
        exit_code, stdout, stderr = run_main(["chat"])
        self.assertEqual(2, exit_code)
        self.assertEqual("", stdout)
        self.assertIn("required", stderr)

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "synthetic.sqlite3"
            database_path.touch()
            arguments = [
                "chat",
                "--database", str(database_path),
                "--profile-id", "profile",
                "--request-id", "request",
                "--provider", "openai-compatible",
                "--model", "model",
                "--api-base", "https://invalid.local/v1",
                "--prompt", "prompt",
            ]
            with patch.dict("os.environ", {}, clear=True), patch(
                "healthvault.main.build_healthvault_agent_runtime",
                side_effect=AssertionError("model runtime must not be built"),
            ) as runtime_builder:
                exit_code, stdout, stderr = run_main(arguments)

        self.assertEqual(1, exit_code)
        self.assertEqual("", stdout)
        self.assertIn("HEALTHVAULT_OPENAI_API_KEY", stderr)
        self.assertNotIn("Traceback", stderr)
        runtime_builder.assert_not_called()

    def test_fake_chat_outputs_final_text_tokens_and_safe_runtime_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "synthetic.sqlite3"
            database_path.touch()
            fake_runtime = MagicMock()
            fake_runtime.run = AsyncMock(
                return_value=HealthVaultChatResult(
                    text="SYNTHETIC_CHAT_OK",
                    tokens={"input": 12, "output": 4},
                    request_id="request-001",
                    tool_names=["query_health_timeline"],
                    project_skills=[],
                )
            )
            model_config = HealthVaultModelConfig(
                "openai-compatible",
                "synthetic-model",
                "https://invalid.local/v1",
                "synthetic-test-key",
            )
            arguments = [
                "chat",
                "--database", str(database_path),
                "--profile-id", "profile-001",
                "--request-id", "request-001",
                "--provider", "openai-compatible",
                "--model", "synthetic-model",
                "--api-base", "https://invalid.local/v1",
                "--prompt", "synthetic prompt",
            ]
            with patch(
                "healthvault.main.load_healthvault_model_config",
                return_value=model_config,
            ) as config_loader, patch(
                "healthvault.main.build_healthvault_agent_runtime",
                return_value=fake_runtime,
            ) as runtime_builder:
                exit_code, stdout, stderr = run_main(arguments)

        result = json.loads(stdout)
        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr)
        self.assertEqual("SYNTHETIC_CHAT_OK", result["text"])
        self.assertEqual({"input": 12, "output": 4}, result["tokens"])
        self.assertEqual("request-001", result["request_id"])
        self.assertNotIn("synthetic-test-key", stdout)
        config_loader.assert_called_once_with(
            provider="openai-compatible",
            model="synthetic-model",
            api_base="https://invalid.local/v1",
        )
        runtime_config = runtime_builder.call_args.args[0]
        self.assertEqual("profile-001", runtime_config.context.profile_id)
        self.assertEqual("request-001", runtime_config.context.request_id)
        fake_runtime.run.assert_awaited_once_with("synthetic prompt")

    def test_explicit_database_persists_but_path_is_not_disclosed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "explicit.sqlite3"
            exit_code, stdout, stderr = run_main(
                ["check", "--database", str(database_path)]
            )
            manifest = json.loads(stdout)

            self.assertEqual(0, exit_code)
            self.assertEqual("", stderr)
            self.assertTrue(database_path.exists())
            self.assertEqual("explicit", manifest["database_mode"])
            self.assertNotIn(str(database_path), stdout)

    def test_custom_knowledge_and_errors_are_short_without_traceback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_knowledge = Path(temp_dir) / "missing.json"
            unavailable_database = Path(temp_dir) / "database-directory"
            unavailable_database.mkdir()

            missing = run_main(
                ["check", "--knowledge", str(missing_knowledge)]
            )
            unavailable = run_main(
                ["check", "--database", str(unavailable_database)]
            )
            invalid_mode = run_main(["check", "--mode", "invalid"])

        for exit_code, stdout, stderr in (missing, unavailable):
            self.assertNotEqual(0, exit_code)
            self.assertEqual("", stdout)
            self.assertIn("Error:", stderr)
            self.assertNotIn("Traceback", stderr)
            self.assertNotIn("API_KEY", stderr)
        self.assertEqual(2, invalid_mode[0])
        self.assertEqual("", invalid_mode[1])
        self.assertIn("invalid choice", invalid_mode[2])
        self.assertNotIn("Traceback", invalid_mode[2])


if __name__ == "__main__":
    unittest.main()
