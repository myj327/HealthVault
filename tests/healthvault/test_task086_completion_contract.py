import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.agent import Agent
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime, MODEL_ONLY_RUNTIME_POLICY
from healthvault.application.workflow_contract import completion_feedback, workflow_intent
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext


def final(status='ok', citations=('REC:r:a',), refusal=False):
    return json.dumps({'final_status': status, 'citations': list(citations), 'refusal': refusal})


def tool(name, arguments=None):
    return {'role': 'assistant', 'content': '', 'reasoning_content': 'retained reasoning', 'tool_calls':
            [{'id': 'call-'+name, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments or {})}}]}


class CompletionContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2]/'.task_outputs/task-086-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        for name in ('start_spinner', 'stop_spinner', 'print_cost', 'print_divider', 'print_tool_call', 'print_tool_result'):
            self.stack.enter_context(patch.object(agent_module, name))
        self.stack.enter_context(patch.object(agent_module.openai, 'AsyncOpenAI', return_value=MagicMock()))
        self.stack.enter_context(patch.object(agent_module.anthropic, 'AsyncAnthropic', return_value=MagicMock()))

    def runtime(self, max_turns=6):
        runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
            context=RuntimeContext('synthetic', 'req'), database_path=self.root/'health.sqlite3',
            model_config=HealthVaultModelConfig('openai-compatible', 'fake', 'https://invalid.local/v1', 'synthetic-key'),
            max_turns=max_turns))
        runtime.agent._tool_executor = AsyncMock(return_value='{"status":"ok","available_citation_ids":["REC:r:a"]}')
        return runtime

    def script(self, agent, messages):
        remaining = iter(messages)
        async def response():
            message = next(remaining)
            agent._emit_text(message.get('content', ''))
            return {'choices': [{'message': message}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        agent._call_openai_stream = AsyncMock(side_effect=response)
        return agent._call_openai_stream

    async def test_premature_refusal_is_not_emitted_and_required_tool_is_checked_before_finish(self):
        runtime = self.runtime()
        early = final('insufficient_evidence', (), True)
        accepted = final()
        call = self.script(runtime.agent, [{'role': 'assistant', 'content': early}, tool('query_health_timeline'),
                                          {'role': 'assistant', 'content': accepted}])
        result = await runtime.run('查看历史时间线')
        self.assertEqual(json.loads(accepted), json.loads(result.text))
        self.assertEqual(accepted, runtime.agent._openai_messages[-1]['content'])
        self.assertEqual(3, call.await_count)
        self.assertEqual({'input': 30, 'output': 15}, result.tokens)
        self.assertEqual(early, runtime.agent._openai_messages[2]['content'])
        self.assertEqual('retained reasoning', runtime.agent._openai_messages[4]['reasoning_content'])
        runtime.agent._tool_executor.assert_awaited_once()
        self.assertIsNone(runtime.agent.completion_check)

    async def test_comparison_and_visit_require_their_own_tool_even_without_trace(self):
        for intent, necessary in [('对比历史变化', 'compare_health_indicators'), ('准备就诊摘要', 'generate_visit_brief')]:
            runtime = self.runtime()
            self.script(runtime.agent, [{'role': 'assistant', 'content': final('insufficient_evidence', (), True)},
                                        tool(necessary), {'role': 'assistant', 'content': final()}])
            await runtime.run(intent)
            self.assertEqual(necessary, runtime.agent._tool_executor.await_args.args[0])
            self.assertEqual('indicator_comparison' if '对比' in intent else 'visit_preparation', workflow_intent(intent))

    async def test_repeated_incomplete_completion_fails_without_normalizing_success(self):
        runtime = self.runtime()
        self.script(runtime.agent, [{'role': 'assistant', 'content': final('insufficient_evidence', (), True)}]*2)
        with self.assertRaisesRegex(RuntimeError, 'completion conditions'):
            await runtime.run('查看时间线')
        runtime.agent._tool_executor.assert_not_awaited()
        self.assertEqual(10, runtime.agent.total_output_tokens)
        self.assertIsNone(runtime.agent._output_buffer)
        self.assertIsNone(runtime.agent.completion_check)

    async def test_feedback_uses_existing_turn_budget(self):
        runtime = self.runtime(max_turns=1)
        call = self.script(runtime.agent, [{'role': 'assistant', 'content': '没有记录'}])
        with self.assertRaises(RuntimeError):
            await runtime.run('查看时间线')
        self.assertEqual(1, call.await_count)

    async def test_empty_result_needs_consistent_refusal_and_raw_json(self):
        runtime = self.runtime()
        runtime.agent._tool_executor = AsyncMock(return_value='{"status":"empty","available_citation_ids":[]}')
        self.script(runtime.agent, [tool('compare_health_indicators'),
                    {'role': 'assistant', 'content': final('insufficient_evidence', (), False)},
                    {'role': 'assistant', 'content': final('insufficient_evidence', (), True)}])
        self.assertTrue(json.loads((await runtime.run('比较变化')).text)['refusal'])
        observed = [{'name': 'query_health_timeline', 'result_status': 'ok', 'available_citation_ids': ['REC:r:a']}]
        for response in (final()+'说明', '{', '```json\n'+final()+'\n```'):
            self.assertIsNotNone(completion_feedback(response, observed, 'record_timeline'))
        self.assertIsNone(completion_feedback(final('rejected', (), True), observed, 'record_timeline'))
        rejected = observed + [{'name': 'generate_visit_brief', 'result_status': 'rejected',
                               'available_citation_ids': [], 'issue_codes': ['unsupported_diagnosis']}]
        self.assertIsNotNone(completion_feedback('检查结果正常。 [REC:r:a]', rejected,
                                                'visit_preparation', terminal=True))
        self.assertIsNotNone(completion_feedback(final(), rejected, 'visit_preparation', terminal=True))

    async def test_three_distinct_empty_queries_allowed_fourth_does_not_execute(self):
        runtime = self.runtime()
        runtime.agent._tool_executor = AsyncMock(return_value='{"status":"empty","available_citation_ids":[]}')
        call = self.script(runtime.agent, [tool('search_health_knowledge', {'query': query}) for query in ('a', 'b', 'c', 'd')])
        result = await runtime.run('查询知识资料')
        self.assertEqual(3, runtime.agent._tool_executor.await_count)
        self.assertEqual(4, call.await_count)
        self.assertEqual('rejected', json.loads(result.text)['final_status'])
        self.assertEqual(3, runtime.manifest['max_knowledge_queries'])

    async def test_default_agent_still_emits_unchecked_response(self):
        agent = Agent(api_base='https://invalid.local/v1', api_key='fake', custom_tools=[], runtime_policy=MODEL_ONLY_RUNTIME_POLICY)
        self.script(agent, [{'role': 'assistant', 'content': '{'}])
        self.assertEqual('{', (await agent.run_once('question'))['text'])
        self.assertEqual(0, agent._completion_rejections)

    async def test_anthropic_retains_rejected_message_and_accepts_only_final_text(self):
        agent = Agent(api_key='fake', custom_tools=[], runtime_policy=MODEL_ONLY_RUNTIME_POLICY,
                      completion_check=lambda text: 'retry' if text == 'wrong' else None)
        responses = iter(('wrong', 'right'))
        async def response(**_kwargs):
            text = next(responses)
            agent._emit_text(text)
            return SimpleNamespace(content=[SimpleNamespace(type='text', text=text)],
                                   usage=SimpleNamespace(input_tokens=7, output_tokens=2))
        agent._call_anthropic_stream = response
        result = await agent.run_once('question')
        self.assertEqual('right', result['text'])
        self.assertEqual({'input': 14, 'output': 4}, result['tokens'])
        self.assertEqual('wrong', agent._anthropic_messages[1]['content'][0]['text'])


if __name__ == '__main__':
    unittest.main()
