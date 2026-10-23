import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.application.citation_policy import recommended_record_citation_ids
from healthvault.model_config import HealthVaultModelConfig
from healthvault.prompt import build_healthvault_system_prompt
from healthvault.runtime_context import RuntimeContext


class AgentWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / '.task_outputs/task-080-082-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        self.assertTrue(self.root.resolve().is_relative_to(scratch.resolve()))

    def tearDown(self):
        self.temp.cleanup()

    def runtime(self, profile='alice'):
        with patch.object(agent_module.openai, 'AsyncOpenAI', return_value=MagicMock()):
            return build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                context=RuntimeContext(profile, 'request-' + profile), database_path=self.root / (profile + '.sqlite3'),
                model_config=HealthVaultModelConfig('openai-compatible', 'fake', 'https://invalid.local/v1', 'synthetic-key')))

    async def scripted(self, runtime, calls, executor, question='检验知识'):
        runtime.agent._tool_executor = executor
        async def run_once(_prompt):
            for name, arguments in calls:
                await runtime.agent._execute_tool_call(name, arguments)
            return {'text': '本轮完成。', 'tokens': {'input': 0, 'output': 0}}
        runtime.agent.run_once = run_once
        return await runtime.run(question)

    async def test_identical_requests_reuse_result_but_preserve_model_attempts(self):
        runtime = self.runtime()
        executor = AsyncMock(return_value=json.dumps({'status': 'ok', 'available_citation_ids': ['REC:test:hba1c']}))
        attempted = []
        original = runtime.agent._execute_tool_call
        async def observe(name, arguments):
            attempted.append((name, arguments))
            return await original(name, arguments)
        runtime.agent._execute_tool_call = observe
        # Knowledge searches now reject duplicates; other read-only tools keep
        # the original per-request cache and all model attempts.
        calls = [('query_health_timeline', {'indicator_codes': ['hba1c'], 'limit': 2}),
                 ('query_health_timeline', {'limit': 2, 'indicator_codes': ['hba1c']})]
        await self.scripted(runtime, calls, executor, '查看历史时间线')
        self.assertEqual(1, executor.await_count)
        self.assertEqual(2, len(attempted))
        self.assertEqual(1, runtime.manifest['read_only_cache_hits'])
        self.assertIs(executor, runtime.agent._tool_executor)

    async def test_concurrent_identical_calls_share_one_execution(self):
        runtime = self.runtime()
        executor = AsyncMock(return_value='{"status":"ok"}')
        runtime.agent._tool_executor = executor
        async def run_once(_prompt):
            await asyncio.gather(*(runtime.agent._execute_tool_call('query_health_timeline', {}) for _ in range(2)))
            return {'text': '完成', 'tokens': {'input': 0, 'output': 0}}
        runtime.agent.run_once = run_once
        await runtime.run('列出时间线')
        executor.assert_awaited_once()

    async def test_empty_search_can_recover_after_two_distinct_rewrites(self):
        runtime = self.runtime()
        executor = AsyncMock(side_effect=['{"status":"empty"}', '{"status":"empty"}',
                                          '{"status":"ok","available_citation_ids":["HVK:test:1"]}'])
        calls = [('search_health_knowledge', {'query': query}) for query in
                 ('what affects lab results', 'lab result factors', 'understand laboratory results')]
        await self.scripted(runtime, calls, executor)
        self.assertEqual(3, executor.await_count)
        self.assertEqual(0, runtime.manifest['read_only_cache_hits'])

    async def test_failed_results_and_denied_tools_are_not_cached(self):
        runtime = self.runtime()
        executor = AsyncMock(side_effect=['{"status":"retryable_failure"}', '{"status":"ok"}'])
        calls = [('query_health_timeline', {})] * 2 + [('confirm_health_records', {})] * 2
        await self.scripted(runtime, calls, executor)
        self.assertEqual(2, executor.await_count)
        self.assertEqual(0, runtime.manifest['read_only_cache_hits'])

    async def test_cache_does_not_survive_new_requests_or_other_profiles(self):
        executor = AsyncMock(return_value='{"status":"empty"}')
        alice = self.runtime()
        await self.scripted(alice, [('search_health_knowledge', {'query': 'A1C'})] * 2, executor)
        await self.scripted(alice, [('search_health_knowledge', {'query': 'A1C'})], executor)
        await self.scripted(self.runtime('bob'), [('search_health_knowledge', {'query': 'A1C'})], executor)
        self.assertEqual(3, executor.await_count)


class CitationWorkflowTests(unittest.TestCase):
    def test_requested_comparison_and_visit_citations_use_actual_fields(self):
        available = [f'REC:{report}:{code}' for report in ('old', 'new') for code in ('fasting_glucose', 'hba1c')]
        comparisons = [{'code': code, 'earliest': {'report_id': 'old'}, 'latest': {'report_id': 'new'}}
                       for code in ('fasting_glucose', 'hba1c')]
        payload = {'available_citation_ids': list(available), 'comparisons': comparisons}
        self.assertEqual(['REC:old:hba1c', 'REC:new:hba1c'],
                         recommended_record_citation_ids('compare_health_indicators', payload, 'HbA1c 与上次相比'))
        payload['indicators'] = [{'code': 'hba1c', 'latest': {'record_citation_id': 'REC:new:hba1c'}}]
        payload['available_citation_ids'].append('HVK:general:1')
        self.assertEqual(['REC:new:hba1c'], recommended_record_citation_ids('generate_visit_brief', payload, '就诊前解释 HbA1c'))
        self.assertEqual(available + ['HVK:general:1'], payload['available_citation_ids'])

    def test_common_prompt_routes_tasks_and_preserves_empty_search_recovery(self):
        prompt = build_healthvault_system_prompt()
        self.assertIn('call compare_health_indicators directly', prompt)
        self.assertIn('call generate_visit_brief before deciding', prompt)
        self.assertIn('at most three distinct local searches', prompt)
        self.assertIn('date, side and negation', prompt)
        self.assertIn('both endpoint record citations', prompt)
        self.assertIn('even if the preceding knowledge search is empty', prompt)


if __name__ == '__main__':
    unittest.main()
