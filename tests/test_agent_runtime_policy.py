import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent
from agents.runtime_policy import AgentRuntimePolicy


def make_agent(**kwargs):
    with patch.object(agent_module.openai, "AsyncOpenAI", return_value=MagicMock()):
        return Agent(
            api_base="https://invalid.local/v1",
            api_key="synthetic-test-key",
            custom_system_prompt="Synthetic policy test prompt.",
            **kwargs,
        )


class AgentRuntimePolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_default_policy_preserves_all_integrations(self):
        agent = make_agent()

        self.assertEqual(AgentRuntimePolicy(), agent.runtime_policy)
        self.assertTrue(agent.runtime_policy.mcp_enabled)
        self.assertTrue(agent.runtime_policy.memory_enabled)
        self.assertTrue(agent.runtime_policy.global_skills_enabled)
        self.assertTrue(agent.runtime_policy.session_persistence_enabled)
        self.assertIsNone(agent.runtime_policy.allowed_tool_names)

    async def test_default_policy_keeps_compaction_but_removes_coding_tools(self):
        agent = make_agent()
        agent._execute_compact_context_tool = AsyncMock(return_value="compact-routed")

        results = {
            name: await agent._execute_tool_call(name, {})
            for name in (
                "agent",
                "compact_context",
                "enter_plan_mode",
                "exit_plan_mode",
            )
        }

        self.assertIn("unavailable", results["agent"])
        self.assertEqual("compact-routed", results["compact_context"])
        self.assertIn("unavailable", results["enter_plan_mode"])
        self.assertIn("unavailable", results["exit_plan_mode"])
        agent._execute_compact_context_tool.assert_awaited_once_with({})

    async def test_disabled_policy_skips_mcp_global_skills_and_session_save(self):
        policy = AgentRuntimePolicy(
            mcp_enabled=False,
            memory_enabled=False,
            global_skills_enabled=False,
            session_persistence_enabled=False,
        )
        agent = make_agent(runtime_policy=policy)
        agent._chat_openai = AsyncMock()

        with patch.object(
            agent._mcp_manager,
            "load_and_connect",
            new=AsyncMock(),
        ) as mcp_connect, patch.object(
            agent,
            "_augment_user_message_with_skill_context",
            side_effect=AssertionError("global Skill retrieval must be disabled"),
        ) as retrieve_skills, patch.object(
            agent,
            "_schedule_background_skill_task",
            side_effect=AssertionError("global Skill tracking must be disabled"),
        ) as schedule_skill_work, patch.object(
            agent,
            "_auto_save",
            side_effect=AssertionError("Session persistence must be disabled"),
        ) as auto_save:
            await agent.chat("synthetic policy probe")

        self.assertFalse(agent.is_sub_agent)
        mcp_connect.assert_not_awaited()
        retrieve_skills.assert_not_called()
        schedule_skill_work.assert_not_called()
        auto_save.assert_not_called()

    def test_disabled_global_skills_does_not_read_evolution_environment(self):
        agent = make_agent(
            runtime_policy=AgentRuntimePolicy(global_skills_enabled=False)
        )

        with patch.object(
            agent_module.os.environ,
            "get",
            side_effect=AssertionError("Skill evolution config must not be read"),
        ) as env_get:
            self.assertFalse(agent._online_evolution_enabled())

        env_get.assert_not_called()

    def test_disabled_session_policy_keeps_fold_in_memory_without_saving(self):
        agent = make_agent(
            runtime_policy=AgentRuntimePolicy(
                session_persistence_enabled=False
            )
        )
        with patch.object(
            agent_module,
            "save_folded_session_memory",
        ) as save_folded:
            agent._record_folded_session_memory(
                "synthetic-trigger",
                {"summary": "synthetic folded memory"},
            )

        self.assertEqual(1, len(agent._folded_session_memories))
        save_folded.assert_not_called()


if __name__ == "__main__":
    unittest.main()
