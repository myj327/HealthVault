import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent
from agents.runtime_policy import AgentRuntimePolicy
from agents.ui import quiet_agent_ui
from healthvault.agent_runtime import (
    HealthVaultAgentRuntimeConfig,
    build_healthvault_agent_runtime,
    build_healthvault_model_only_runtime,
)
from healthvault.model_config import (
    HealthVaultModelConfig,
    HealthVaultModelConfigError,
    load_healthvault_model_config,
)
from healthvault.runtime_context import RuntimeContext
from healthvault.web import model_runtime


TOOL = {"name": "query_health_timeline", "description": "Read synthetic records.",
        "input_schema": {"type": "object", "properties": {}}}
POLICY = AgentRuntimePolicy(mcp_enabled=False, memory_enabled=False,
                           global_skills_enabled=False, session_persistence_enabled=False,
                           allowed_tool_names=frozenset({TOOL["name"]}))


def chunk(*, content=None, reasoning=None, tool_calls=None, finish=None, absent_reasoning=False):
    fields = {"content": content, "tool_calls": tool_calls}
    if not absent_reasoning:
        fields["reasoning_content"] = reasoning
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(
        delta=SimpleNamespace(**fields), finish_reason=finish)])


def function_delta(arguments, *, first=False):
    return SimpleNamespace(index=0, id="call-synthetic" if first else None,
                           function=SimpleNamespace(name=TOOL["name"] if first else None,
                                                    arguments=arguments))


def client_for(responses):
    requests = []
    remaining = iter(responses)

    async def create(**request):
        requests.append(copy.deepcopy(request))
        response = next(remaining)

        async def stream():
            for item in response:
                yield item
        return stream()

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create)
    return client, requests


def make_agent(client, **kwargs):
    with patch.object(agent_module.openai, "AsyncOpenAI", return_value=client):
        agent = Agent(api_base="https://invalid.local/v1", api_key="synthetic-key",
                      custom_system_prompt="Synthetic protocol test.", custom_tools=[TOOL],
                      runtime_policy=POLICY, permission_mode="dontAsk", max_turns=4, **kwargs)
    agent._run_compression_pipeline = MagicMock()
    agent._check_and_compact = AsyncMock()
    agent._emit_text = MagicMock()
    return agent


class DeepSeekStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_reasoning_survives_tool_result_and_next_request(self):
        client, requests = client_for([
            [chunk(reasoning="Need "), chunk(reasoning="records."),
             chunk(tool_calls=[function_delta("{", first=True)]),
             chunk(tool_calls=[function_delta("}")], finish="tool_calls")],
            [chunk(reasoning="Use the result."), chunk(content="Synthetic answer.", finish="stop")],
        ])
        executor = AsyncMock(return_value="Synthetic tool result.")
        agent = make_agent(client, tool_executor=executor, reasoning_effort="low")
        with quiet_agent_ui():
            await agent._chat_openai("Synthetic question.")
        executor.assert_awaited_once()
        self.assertEqual((TOOL["name"], {}), executor.await_args.args[:2])
        self.assertEqual(2, len(requests))
        self.assertTrue(all(request["reasoning_effort"] == "low" for request in requests))
        assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
        self.assertEqual("Need records.", assistant["reasoning_content"])
        self.assertEqual("{}", assistant["tool_calls"][0]["function"]["arguments"])
        self.assertEqual("call-synthetic", requests[1]["messages"][-1]["tool_call_id"])
        self.assertEqual("Synthetic answer.", agent._openai_messages[-1]["content"])
        self.assertEqual("Use the result.", agent._openai_messages[-1]["reasoning_content"])
        self.assertNotIn("Need records.", "".join(str(call.args[0]) for call in agent._emit_text.call_args_list))

    async def test_tool_enabled_followup_keeps_reasoning_of_prior_final_answer(self):
        client, requests = client_for([
            [chunk(reasoning="First reasoning."), chunk(content="First answer.", finish="stop")],
            [chunk(reasoning="Second reasoning."), chunk(content="Second answer.", finish="stop")],
        ])
        agent = make_agent(client, tool_executor=AsyncMock(), reasoning_effort="high")
        with quiet_agent_ui():
            await agent._chat_openai("First question.")
            await agent._chat_openai("Second question.")
        self.assertIn("tools", requests[1])
        prior = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
        self.assertEqual("First reasoning.", prior["reasoning_content"])
        agent._tool_executor.assert_not_awaited()

    async def test_legacy_provider_omits_optional_fields_and_preserves_text(self):
        client, requests = client_for([[chunk(content='{"ok":true}', absent_reasoning=True, finish="stop")]])
        agent = make_agent(client)
        with quiet_agent_ui():
            response = await agent._call_openai_stream()
        self.assertNotIn("reasoning_effort", requests[0])
        self.assertNotIn("reasoning_content", response["choices"][0]["message"])
        self.assertEqual('{"ok":true}', response["choices"][0]["message"]["content"])

    async def test_explicit_empty_reasoning_field_is_kept_without_changing_final(self):
        client, _ = client_for([[chunk(reasoning="", content="Synthetic final.", finish="stop")]])
        agent = make_agent(client)
        with quiet_agent_ui():
            response = await agent._call_openai_stream()
        self.assertEqual("", response["choices"][0]["message"]["reasoning_content"])
        self.assertEqual("Synthetic final.", response["choices"][0]["message"]["content"])

    async def test_model_only_stream_has_no_tools_and_does_not_return_reasoning_as_answer(self):
        client, requests = client_for([[chunk(reasoning="Private synthetic reasoning."),
                                       chunk(content='{"status":"ok"}', finish="stop")]])
        config = HealthVaultModelConfig("openai-compatible", "deepseek-v4-pro",
                                       "https://invalid.local", "synthetic-key", reasoning_effort="high")
        with patch.object(agent_module.openai, "AsyncOpenAI", return_value=client):
            runtime = build_healthvault_model_only_runtime(
                context=RuntimeContext("synthetic-profile", "synthetic-request"),
                model_config=config, system_prompt="Return JSON.")
        runtime.agent._emit_text = MagicMock()
        with quiet_agent_ui():
            response = await runtime.agent._call_openai_stream()
        self.assertNotIn("tools", requests[0])
        self.assertEqual("high", requests[0]["reasoning_effort"])
        self.assertEqual('{"status":"ok"}', response["choices"][0]["message"]["content"])
        self.assertEqual(1, runtime.agent.max_turns)
        self.assertEqual(frozenset(), runtime.agent.runtime_policy.allowed_tool_names)


class DeepSeekConfigTests(unittest.TestCase):
    def test_web_env_mapping_preserves_explicit_model_effort_and_secret_repr(self):
        fake_env = ("HEALTHVAULT_OPENAI_API_KEY=synthetic-key\n"
                    "HEALTHVAULT_OPENAI_BASE_URL=https://api.deepseek.com/\n"
                    "HEALTHVAULT_OPENAI_MODEL=deepseek-v4-pro\n"
                    "HEALTHVAULT_OPENAI_REASONING_EFFORT=low\n")
        with patch.dict(model_runtime.os.environ, {}, clear=True), \
                patch.object(model_runtime.Path, "is_file", return_value=True), \
                patch.object(model_runtime.Path, "read_text", return_value=fake_env):
            config = model_runtime.configured_web_model()
        self.assertEqual("https://api.deepseek.com", config.api_base)
        self.assertEqual("deepseek-v4-pro", config.model)
        self.assertEqual("low", config.reasoning_effort)
        self.assertNotIn("synthetic-key", repr(config))

    def test_invalid_effort_fails_before_client_construction_and_legacy_default_remains(self):
        for invalid in ("", "ultra", [], 3):
            with self.subTest(invalid=invalid), self.assertRaises(HealthVaultModelConfigError):
                HealthVaultModelConfig("openai-compatible", "fake", "https://invalid.local",
                                       "synthetic-key", reasoning_effort=invalid)
        with self.assertRaises(HealthVaultModelConfigError):
            HealthVaultModelConfig("anthropic", "fake", "https://invalid.local", "synthetic-key",
                                   reasoning_effort="high")
        settings = {"HEALTHVAULT_OPENAI_API_KEY": "synthetic-key",
                    "HEALTHVAULT_OPENAI_BASE_URL": "https://invalid.local",
                    "HEALTHVAULT_OPENAI_MODEL": "fake"}
        with patch.object(model_runtime, "_model_setting", side_effect=lambda name: settings.get(name, "")):
            self.assertIsNone(model_runtime.configured_web_model().reasoning_effort)
            settings["HEALTHVAULT_OPENAI_REASONING_EFFORT"] = "ultra"
            with self.assertRaises(model_runtime.ModelServiceError) as raised:
                model_runtime.configured_web_model()
        self.assertEqual(503, raised.exception.status_code)

    def test_runtime_builders_forward_effort_without_enabling_integrations(self):
        with patch.dict(model_runtime.os.environ, {"HEALTHVAULT_OPENAI_API_KEY": "synthetic-key"}):
            config = load_healthvault_model_config(provider="openai-compatible", model="deepseek-v4-pro",
                                                  api_base="https://invalid.local", reasoning_effort="low")
        components = SimpleNamespace(system_prompt="Synthetic prompt.", adapter=SimpleNamespace(
            execute=AsyncMock(), combined_definitions=[TOOL]))
        with patch("healthvault.agent_runtime.build_healthvault_components", return_value=components), \
                patch.object(agent_module.openai, "AsyncOpenAI", return_value=MagicMock()):
            runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                context=RuntimeContext("synthetic-profile", "synthetic-request"),
                database_path="unused-synthetic.sqlite", model_config=config))
            restricted = build_healthvault_model_only_runtime(
                context=runtime.context, model_config=config, system_prompt="Synthetic prompt.")
        for agent in (runtime.agent, restricted.agent):
            self.assertEqual("low", agent.reasoning_effort)
            self.assertFalse(agent.runtime_policy.mcp_enabled)
            self.assertFalse(agent.runtime_policy.memory_enabled)
            self.assertFalse(agent.runtime_policy.session_persistence_enabled)
        self.assertEqual(frozenset({TOOL["name"]}), runtime.agent.runtime_policy.allowed_tool_names)
        self.assertEqual([], restricted.agent.tools)


if __name__ == "__main__":
    unittest.main()
