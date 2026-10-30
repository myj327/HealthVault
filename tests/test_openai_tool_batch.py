import asyncio
import copy
import json
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent
from agents.runtime_policy import AgentRuntimePolicy
from agents.ui import quiet_agent_ui


POLICY = AgentRuntimePolicy(mcp_enabled=False, memory_enabled=False,
                           global_skills_enabled=False, session_persistence_enabled=False)


def make_agent(names, executor, **kwargs):
    requests = []

    async def create(**request):
        requests.append(copy.deepcopy(request))
        first = len(requests) == 1
        calls = [SimpleNamespace(index=index, id=f"call-{index}", function=SimpleNamespace(
            name=name, arguments=json.dumps({"marker": index}))) for index, name in enumerate(names)]

        async def stream():
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None if first else "Synthetic final.",
                                      tool_calls=calls if first else None),
                finish_reason="tool_calls" if first else "stop")])
        return stream()

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create)
    with patch.object(agent_module.openai, "AsyncOpenAI", return_value=client):
        agent = Agent(api_base="https://invalid.local/v1", api_key="synthetic-key",
                      custom_system_prompt="Synthetic batch test.", runtime_policy=POLICY,
                      custom_tools=[{"name": name, "description": "Synthetic probe.",
                                     "input_schema": {"type": "object", "properties": {}}}
                                    for name in dict.fromkeys(names)],
                      tool_executor=executor, **kwargs)
    agent._run_compression_pipeline = MagicMock()
    agent._check_and_compact = AsyncMock()
    return agent, requests


class OpenAIToolBatchTests(unittest.IsolatedAsyncioTestCase):
    def assert_paired(self, messages):
        calls = Counter(tc["id"] for message in messages if message["role"] == "assistant"
                        for tc in message.get("tool_calls") or [] if tc["type"] == "function")
        replies = Counter(message["tool_call_id"] for message in messages if message["role"] == "tool")
        self.assertEqual(calls, replies)
        self.assertTrue(all(count == 1 for count in replies.values()))
        return {message["tool_call_id"]: message["content"] for message in messages
                if message["role"] == "tool"}

    async def test_single_two_and_three_calls_execute_once_after_all_permissions(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                events = []

                def permission(name, inp, *_):
                    events.append(("permission", inp["marker"]))
                    return {"action": "allow"}

                async def execute(name, inp, state):
                    events.append(("execute", inp["marker"]))
                    return f"result-{inp['marker']}"

                agent, requests = make_agent(["synthetic_probe"] * count, execute)
                with quiet_agent_ui(), patch.object(agent_module, "check_permission", side_effect=permission):
                    result = await agent.run_once("Synthetic question.")
                self.assertEqual([("permission", i) for i in range(count)] +
                                 [("execute", i) for i in range(count)], events)
                self.assertEqual("Synthetic final.", result["text"].strip())
                self.assertEqual({f"call-{i}": f"result-{i}" for i in range(count)},
                                 self.assert_paired(requests[1]["messages"]))

    async def test_declared_safe_tools_still_run_concurrently(self):
        started = []
        all_started = asyncio.Event()

        async def execute(name, inp, state):
            started.append(name)
            if len(started) == 3:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=1)
            return name + " completed"

        names = ["safe_one", "safe_two", "safe_three"]
        agent, requests = make_agent(names, execute, permission_mode="bypassPermissions")
        with quiet_agent_ui(), patch.object(agent_module, "CONCURRENCY_SAFE_TOOLS", set(names)):
            await agent.run_once("Synthetic parallel question.")
        self.assertCountEqual(names, started)
        self.assertEqual(3, len(self.assert_paired(requests[1]["messages"])))
        self.assertTrue(all("completed" in value for value in self.assert_paired(agent._openai_messages).values()))

    async def test_mixed_and_all_denied_batches_reply_without_execution(self):
        for actions in (("deny", "allow", "confirm", "allow"), ("deny", "confirm")):
            with self.subTest(actions=actions):
                executor = AsyncMock(return_value="allowed result")
                confirm = AsyncMock(return_value=False)
                agent, requests = make_agent(["synthetic_probe"] * len(actions), executor, confirm_fn=confirm)
                with quiet_agent_ui(), patch.object(agent_module, "check_permission", side_effect=[
                        {"action": action, "message": f"synthetic restriction {index}"}
                        for index, action in enumerate(actions)]):
                    await agent.run_once("Synthetic permissions question.")
                self.assertEqual([i for i, action in enumerate(actions) if action == "allow"],
                                 [call.args[1]["marker"] for call in executor.await_args_list])
                replies = self.assert_paired(requests[1]["messages"])
                for index, action in enumerate(actions):
                    expected = {"allow": "allowed result", "deny": "Action denied", "confirm": "User denied"}[action]
                    self.assertIn(expected, replies[f"call-{index}"])
                confirm.assert_awaited_once()

    async def test_serial_and_concurrent_failure_do_not_drop_sibling_results(self):
        for names in (["synthetic_probe"] * 3, ["safe_one", "safe_two", "safe_three"]):
            with self.subTest(names=names):
                started = []

                async def execute(name, inp, state):
                    started.append(inp["marker"])
                    if inp["marker"] == 1:
                        raise RuntimeError("synthetic executor failure")
                    return f"result-{inp['marker']}"

                agent, requests = make_agent(names, execute, permission_mode="bypassPermissions")
                with quiet_agent_ui(), patch.object(agent_module, "CONCURRENCY_SAFE_TOOLS", {"safe_one", "safe_two", "safe_three"}):
                    await agent.run_once("Synthetic failure question.")
                self.assertCountEqual([0, 1, 2], started)
                replies = self.assert_paired(requests[1]["messages"])
                self.assertEqual("result-0", replies["call-0"])
                self.assertIn("Error", replies["call-1"])
                self.assertIn("synthetic executor failure", replies["call-1"])
                self.assertEqual("result-2", replies["call-2"])

    async def test_abort_during_serial_and_parallel_execution_keeps_completed_results(self):
        for names in (["synthetic_probe"] * 3, ["safe_one", "safe_two", "safe_three"]):
            with self.subTest(names=names):
                blocked = asyncio.Event()
                started = []
                cancelled = []

                async def execute(name, inp, state):
                    marker = inp["marker"]
                    started.append(marker)
                    if marker == 0:
                        return "completed result"
                    blocked.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.append(marker)
                        raise

                agent, requests = make_agent(names, execute, permission_mode="bypassPermissions")
                with quiet_agent_ui(), patch.object(agent_module, "CONCURRENCY_SAFE_TOOLS", {"safe_one", "safe_two", "safe_three"}):
                    running = asyncio.create_task(agent.run_once("Synthetic interrupted question."))
                    await asyncio.wait_for(blocked.wait(), timeout=1)
                    agent.abort()
                    await asyncio.wait_for(running, timeout=1)
                replies = self.assert_paired(agent._openai_messages)
                self.assertEqual("completed result", replies["call-0"])
                self.assertTrue(all("interrupt" in replies[f"call-{i}"].lower() for i in (1, 2)))
                self.assertCountEqual(started[1:], cancelled)
                self.assertEqual(1, len(requests))
                self.assertTrue(agent._aborted)
                if names[0] == "synthetic_probe":
                    self.assertEqual([0, 1], started)

    async def test_abort_while_confirming_executes_nothing_and_pairs_entire_batch(self):
        confirming = asyncio.Event()

        async def confirm(_):
            confirming.set()
            await asyncio.Event().wait()

        executor = AsyncMock(return_value="must not execute")
        agent, requests = make_agent(["synthetic_probe"] * 3, executor, confirm_fn=confirm)
        with quiet_agent_ui(), patch.object(agent_module, "check_permission", side_effect=[
                {"action": "allow"}, {"action": "confirm", "message": "synthetic approval"}]):
            running = asyncio.create_task(agent.run_once("Synthetic confirmation question."))
            await asyncio.wait_for(confirming.wait(), timeout=1)
            agent.abort()
            await asyncio.wait_for(running, timeout=1)
        executor.assert_not_awaited()
        self.assertEqual(1, len(requests))
        self.assertEqual(3, len(self.assert_paired(agent._openai_messages)))

    async def test_context_clear_stops_old_batch_without_orphan_replies(self):
        executor = AsyncMock(return_value="earlier result")
        agent, requests = make_agent(["synthetic_probe", "compact_context", "synthetic_probe"], executor)

        async def compact(**_):
            agent._openai_messages = [{"role": "system", "content": "Synthetic folded memory."}]
            return True

        agent._compact_conversation = AsyncMock(side_effect=compact)
        with quiet_agent_ui():
            await agent.run_once("Synthetic context compaction.")
        self.assertEqual([0], [call.args[1]["marker"] for call in executor.await_args_list])
        self.assertEqual({}, self.assert_paired(requests[1]["messages"]))
        self.assertTrue(any("Context compacted" in (message.get("content") or "")
                            for message in requests[1]["messages"]))
        self.assertFalse(agent._context_cleared)

    async def test_budget_stop_pairs_calls_before_return_or_completion_error(self):
        for checked in (False, True):
            with self.subTest(checked=checked):
                executor = AsyncMock(return_value="must not execute")
                agent, requests = make_agent(["synthetic_probe"] * 2, executor, max_turns=1,
                                             completion_check=(lambda _: None) if checked else None)
                with quiet_agent_ui():
                    if checked:
                        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                            await agent.run_once("Synthetic budget question.")
                    else:
                        await agent.run_once("Synthetic budget question.")
                executor.assert_not_awaited()
                self.assertEqual(1, len(requests))
                replies = self.assert_paired(agent._openai_messages)
                self.assertTrue(all("budget" in value.lower() for value in replies.values()))


if __name__ == "__main__":
    unittest.main()
