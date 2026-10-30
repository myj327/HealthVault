import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent


SYNTHETIC_TOOL = {
    "name": "synthetic_healthvault_probe",
    "description": "Return a synthetic probe result without accessing health data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "marker": {"type": "string"},
        },
        "required": ["marker"],
    },
}


def make_agent(**kwargs):
    with patch.object(agent_module.openai, "AsyncOpenAI", return_value=MagicMock()):
        return Agent(
            api_base="https://invalid.local/v1",
            api_key="synthetic-test-key",
            custom_system_prompt="Synthetic test prompt.",
            **kwargs,
        )


class ToolExecutorInjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_agent_has_no_filesystem_executor(self):
        agent = make_agent()
        self.assertEqual([], agent.tools)
        for name in ("read_file", "write_file", "run_shell", "agent", "skill"):
            self.assertIn("unavailable", await agent._execute_tool_call(name, {}))

    async def test_injected_executor_handles_custom_tool_call(self):
        injected_executor = AsyncMock(return_value="TASK003_INJECTED_EXECUTOR_OK")
        default_executor = AsyncMock(return_value="unexpected default result")
        tool_input = {"marker": "TASK003_SYNTHETIC_PROBE"}

        with patch.object(agent_module, "execute_tool", default_executor):
            agent = make_agent(
                custom_tools=[SYNTHETIC_TOOL],
                tool_executor=injected_executor,
            )
            result = await agent._execute_tool_call(
                "synthetic_healthvault_probe",
                tool_input,
            )

        injected_executor.assert_awaited_once_with(
            "synthetic_healthvault_probe",
            tool_input,
            agent._read_file_state,
        )
        self.assertIs(
            injected_executor.await_args.args[2],
            agent._read_file_state,
        )
        self.assertEqual("TASK003_INJECTED_EXECUTOR_OK", result)
        default_executor.assert_not_awaited()

    async def test_injected_executor_does_not_intercept_mcp_or_agent_special_routes(self):
        injected_executor = AsyncMock(return_value="unexpected injected result")
        agent = make_agent(
            custom_tools=[SYNTHETIC_TOOL],
            tool_executor=injected_executor,
        )
        mcp_manager = MagicMock()
        mcp_manager.is_mcp_tool.return_value = True
        mcp_manager.call_tool = AsyncMock(return_value="TASK003_MCP_ROUTE_OK")
        agent._mcp_manager = mcp_manager
        tool_input = {"marker": "TASK003_SYNTHETIC_MCP"}

        result = await agent._execute_tool_call(
            "mcp__synthetic__probe",
            tool_input,
        )

        self.assertEqual("TASK003_MCP_ROUTE_OK", result)
        mcp_manager.is_mcp_tool.assert_called_once_with("mcp__synthetic__probe")
        mcp_manager.call_tool.assert_awaited_once_with(
            "mcp__synthetic__probe",
            tool_input,
        )
        injected_executor.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
