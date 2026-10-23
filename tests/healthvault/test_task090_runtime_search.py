import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.ui import quiet_agent_ui
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.application.workflow_contract import WORKFLOW_PROTOCOL
from healthvault.evolution.signal_extractor import SignalExtractor, observation_from_persisted_trace
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext
from tests.healthvault.test_skill_review_loop import ACTIVE_SKILL
from tests.healthvault.test_task086_completion_contract import final, tool


class RuntimeSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / '.task_outputs/task-090-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(quiet_agent_ui())
        self.stack.enter_context(patch.object(agent_module.openai, 'AsyncOpenAI', return_value=MagicMock()))
        self.runtime_count = 0

    def runtime(self, executor, *, owner='alice', tracing=False):
        self.runtime_count += 1
        root = self.root / str(self.runtime_count)
        root.mkdir()
        config = {}
        if tracing:
            active = root / '.bear/skills/HealthVault-Task-Execution/SKILL.md'
            active.parent.mkdir(parents=True)
            active.write_text(ACTIVE_SKILL, encoding='utf-8')
            config.update(skill_root=active.parents[1], skill_names=('HealthVault-Task-Execution',),
                          evolution_root=root / 'evolution', signal_context={
                              'intent': 'knowledge_lookup', 'preconditions': ['authenticated read-only task'],
                              'expected_outcome': 'validated workflow', 'validator_status': 'passed',
                              'contract_checks': {'tool': 'passed', 'status': 'passed', 'citation': 'passed'}})
        runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
            context=RuntimeContext(owner, f'request-{self.runtime_count}'), database_path=root / 'health.sqlite3',
            model_config=HealthVaultModelConfig('openai-compatible', 'fake', 'https://invalid.local/v1', 'synthetic-key'),
            **config))
        runtime.agent._tool_executor = runtime.trace_recorder.execute if tracing else executor
        if tracing:
            runtime.trace_recorder._executor = executor
        attempted = []
        original = runtime.agent._execute_tool_call
        async def observe(name, arguments):
            raw = await original(name, arguments)
            attempted.append({'name': name, 'arguments': arguments, 'result': raw})
            return raw
        runtime.agent._execute_tool_call = observe
        return runtime, attempted

    def script(self, runtime, messages):
        pending = iter(messages)
        async def response():
            message = next(pending)
            runtime.agent._emit_text(message.get('content', ''))
            return {'choices': [{'message': message}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        call = AsyncMock(side_effect=response)
        runtime.agent._call_openai_stream = call
        return call

    async def test_duplicate_expression_with_changed_limit_is_visible_and_stops_model(self):
        executor = AsyncMock(return_value='{"status":"empty","available_citation_ids":[]}')
        runtime, attempted = self.runtime(executor)
        call = self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C test', 'limit': 2}),
                                     tool('search_health_knowledge', {'query': '  a1c\t TEST ', 'limit': 5})])
        result = await runtime.run('查询知识资料')
        self.assertEqual(2, len(attempted))
        self.assertEqual('workflow_search_duplicate_query', json.loads(attempted[1]['result'])['issues'][0]['code'])
        executor.assert_awaited_once()
        self.assertEqual(2, call.await_count)
        self.assertEqual({'final_status': 'rejected', 'citations': [], 'refusal': True}, json.loads(result.text))
        self.assertEqual({'input': 20, 'output': 10}, result.tokens)
        self.assertEqual(0, runtime.manifest['read_only_cache_hits'])
        self.assertEqual(WORKFLOW_PROTOCOL, runtime.manifest['workflow_protocol'])
        self.assertIs(executor, runtime.agent._tool_executor)
        self.assertIsNone(runtime.agent.completion_check)

    async def test_nonempty_result_blocks_a_different_search(self):
        for status in ('ok', 'ready', 'partial'):
            with self.subTest(status=status):
                executor = AsyncMock(return_value=json.dumps({'status': status, 'available_citation_ids': ['HVK:synthetic:1']}))
                runtime, attempted = self.runtime(executor)
                call = self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C'}),
                                             tool('search_health_knowledge', {'query': 'HbA1c overview'})])
                result = await runtime.run('查询知识资料')
                self.assertEqual('workflow_search_after_nonempty_result', json.loads(attempted[1]['result'])['issues'][0]['code'])
                executor.assert_awaited_once()
                self.assertEqual(2, call.await_count)
                self.assertEqual('rejected', json.loads(result.text)['final_status'])

    async def test_two_empty_rewrites_can_recover_with_actual_citation(self):
        executor = AsyncMock(side_effect=['{"status":"empty"}', '{"status":"empty"}',
                                          '{"status":"ok","available_citation_ids":["HVK:synthetic:1"]}'])
        runtime, attempted = self.runtime(executor)
        accepted = final(citations=('HVK:synthetic:1',))
        call = self.script(runtime, [tool('search_health_knowledge', {'query': query}) for query in
                                    ('A1C context', 'HbA1c information', 'glycated hemoglobin')] +
                                   [{'role': 'assistant', 'content': accepted}])
        result = await runtime.run('查询知识资料')
        self.assertEqual(json.loads(accepted), json.loads(result.text))
        self.assertEqual(3, executor.await_count)
        self.assertEqual(3, len(attempted))
        self.assertEqual(4, call.await_count)

    async def test_fourth_distinct_query_remains_in_attempts_and_does_not_execute(self):
        executor = AsyncMock(return_value='{"status":"empty","available_citation_ids":[]}')
        runtime, attempted = self.runtime(executor)
        call = self.script(runtime, [tool('search_health_knowledge', {'query': query}) for query in ('a', 'b', 'c', 'd')])
        result = await runtime.run('查询知识资料')
        self.assertEqual(4, len(attempted))
        self.assertEqual('workflow_search_budget_exceeded', json.loads(attempted[3]['result'])['issues'][0]['code'])
        self.assertEqual(3, executor.await_count)
        self.assertEqual(4, call.await_count)
        self.assertEqual('rejected', json.loads(result.text)['final_status'])

    async def test_concurrent_queries_wait_for_the_preceding_actual_result(self):
        for scenario in ('ok', 'empty', 'empty_then_duplicate'):
            with self.subTest(scenario=scenario):
                first_status = 'ok' if scenario == 'ok' else 'empty'
                entered, release = asyncio.Event(), asyncio.Event()
                async def execute(_name, _arguments, _state):
                    if not entered.is_set():
                        entered.set()
                        await release.wait()
                        return json.dumps({'status': first_status, 'available_citation_ids': ['HVK:synthetic:1'] if first_status == 'ok' else []})
                    return '{"status":"ok","available_citation_ids":["HVK:synthetic:1"]}'
                executor = AsyncMock(side_effect=execute)
                runtime, attempted = self.runtime(executor)
                async def concurrent(_prompt):
                    queries = ('first expression', ' FIRST  EXPRESSION ', 'second expression') if scenario == 'empty_then_duplicate' else ('first expression', 'second expression')
                    tasks = [asyncio.create_task(runtime.agent._execute_tool_call('search_health_knowledge', {'query': query}))
                             for query in queries]
                    await asyncio.wait_for(entered.wait(), 2)
                    await asyncio.sleep(0)
                    self.assertEqual(1, executor.await_count)
                    release.set()
                    await asyncio.gather(*tasks)
                    return {'text': final(citations=('HVK:synthetic:1',)) if scenario == 'empty' else final('rejected', (), True),
                            'tokens': {'input': 0, 'output': 0}}
                runtime.agent.run_once = concurrent
                result = await runtime.run('查询知识资料')
                self.assertEqual(3 if scenario == 'empty_then_duplicate' else 2, len(attempted))
                self.assertEqual(2 if scenario == 'empty' else 1, executor.await_count)
                self.assertEqual('ok' if scenario == 'empty' else 'rejected', json.loads(result.text)['final_status'])
                if scenario == 'empty_then_duplicate':
                    self.assertEqual(['workflow_search_duplicate_query', 'workflow_search_after_nonempty_result'],
                                     [json.loads(item['result'])['issues'][0]['code'] for item in attempted[1:]])

    async def test_query_state_does_not_survive_a_new_request_or_other_owner(self):
        executor = AsyncMock(return_value='{"status":"empty","available_citation_ids":[]}')
        alice, attempted = self.runtime(executor)
        bob, bob_attempted = self.runtime(executor, owner='bob')
        for runtime in (alice, alice, bob):
            self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C'}),
                                  {'role': 'assistant', 'content': final('insufficient_evidence', (), True)}])
            self.assertEqual('insufficient_evidence', json.loads((await runtime.run('查询知识资料')).text)['final_status'])
        self.assertEqual(3, executor.await_count)
        self.assertEqual(2, len(attempted))
        self.assertEqual(1, len(bob_attempted))

    async def test_redacted_workflow_failure_is_not_success_or_safety_and_good_trace_still_qualifies(self):
        for mode in ('good', 'duplicate', 'write_and_duplicate'):
            with self.subTest(mode=mode):
                executor = AsyncMock(return_value=json.dumps({'status': 'ok' if mode == 'good' else 'empty',
                    'available_citation_ids': ['HVK:synthetic:1'] if mode == 'good' else []}))
                runtime, attempted = self.runtime(executor, tracing=True)
                private_query = 'synthetic-private-7.2'
                messages = [tool('confirm_health_records', {'report_text': private_query})] if mode == 'write_and_duplicate' else []
                messages.append(tool('search_health_knowledge', {'query': private_query}))
                messages.append({'role': 'assistant', 'content': final(citations=('HVK:synthetic:1',))} if mode == 'good'
                                else tool('search_health_knowledge', {'query': private_query.upper()}))
                self.script(runtime, messages)
                await runtime.run('查询知识资料')
                path = runtime.trace_recorder.trace_path
                text = path.read_text(encoding='utf-8')
                trace = json.loads(text)
                self.assertNotIn(private_query, text)
                self.assertNotIn(private_query.upper(), text)
                self.assertEqual(len(attempted), len(trace['tool_calls']))
                self.assertEqual('passed' if mode == 'good' else 'failed', trace['signal_context']['validator_status'])
                observation = observation_from_persisted_trace(path)
                self.assertEqual(mode == 'good', SignalExtractor._known_contract_passed(observation))
                self.assertEqual('succeeded' if mode == 'good' else 'failed', trace['result_status'])
                expected_failure = {'good': None, 'duplicate': 'workflow_search_duplicate_query',
                                    'write_and_duplicate': 'safety_write_violation'}[mode]
                self.assertEqual(expected_failure, trace['failure_code'])
                if mode == 'duplicate':
                    self.assertNotIn('safety', trace['signal_context']['contract_checks'])
                    self.assertEqual('workflow_protocol_violation', trace['tool_calls'][-1]['validator_code'])
                if mode == 'write_and_duplicate':
                    self.assertEqual('failed', trace['signal_context']['contract_checks']['safety'])


if __name__ == '__main__':
    unittest.main()
