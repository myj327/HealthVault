import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.ui import quiet_agent_ui
from healthvault.evaluation import product_workflow_run as run
from healthvault.evaluation.skill.benchmark.task_025 import FinalParse, parse_final_v3
from healthvault.model_config import HealthVaultModelConfig


class ProductProtocolTests(unittest.TestCase):
    def setUp(self):
        parent = run.ROOT / '.task_outputs/task088-tests'
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = run.load_dataset(run.current.active_corpus_root() / 'chunks/hybrid_v1.json')
        self.cases = {case['case_id']: case for case in self.dataset['cases']}

    def call(self, query='celestial pollen', *, name='search_health_knowledge', status='empty'):
        return {'name': name, 'arguments': {'query': query} if name == 'search_health_knowledge' else {},
                'execution_status': 'ok', 'result_status': status, 'available_citation_ids': []}

    def attempt(self, case, calls, citations=None):
        expected = case['expected']
        refs = expected['allowed_citation_ids'][:expected['min_citations']] if citations is None else citations
        return {'calls': calls, 'final': FinalParse(expected['final_status'], tuple(refs), expected['refusal_required'], None),
                'observed_citation_ids': expected['allowed_citation_ids'], 'database_unchanged': True,
                'duplicate_write_detected': False, 'invalid_calls': 0, 'parser_error': None}

    def prepare(self):
        config = HealthVaultModelConfig('openai-compatible', 'deepseek-flash', 'https://example.invalid/v1', 'synthetic', reasoning_effort='high')
        with patch.object(run, 'configured_web_model', return_value=config):
            return run.prepare(self.root / 'preflight')

    def test_three_distinct_empty_searches_use_new_protocol_only(self):
        case = self.cases['R5E-020']
        original = copy.deepcopy(case)
        calls = [self.call(query) for query in ('celestial pollen', 'celestial pollen atlas', 'celestial pollen official')]
        attempt = self.attempt(case, calls)
        self.assertTrue(run.evaluate(case, attempt)['passed'])
        self.assertFalse(run.current.legacy.evaluate_three_layer_oracle(case, attempt)['passed'])
        self.assertEqual(original, case)
        self.assertEqual(20, len(self.dataset['cases']))
        self.assertFalse(self.dataset['untouched_held_out'])

    def test_successful_search_stops_recovery_and_keeps_final_outcome(self):
        case = copy.deepcopy(self.cases['R5E-020'])
        case['expected'].update(required_tool_result_statuses=['ok'], final_status='ok', refusal_required=False,
                                allowed_citation_ids=['HVK:synthetic-source'], min_citations=1, max_citations=1)
        recovered = [self.call(), self.call('celestial pollen atlas', status='ok')]
        self.assertTrue(run.evaluate(case, self.attempt(case, recovered))['passed'])
        redundant = [self.call(status='ok'), self.call('celestial pollen atlas', status='ok')]
        self.assertFalse(run.evaluate(case, self.attempt(case, redundant))['passed'])

    def test_duplicate_and_fourth_searches_fail_without_dropping_attempts(self):
        case = self.cases['R5E-020']
        for queries in (('celestial pollen', '  CELESTIAL   POLLEN '),
                        ('celestial pollen', 'celestial pollen atlas', 'celestial pollen official', 'celestial pollen overview')):
            calls = [self.call(query) for query in queries]
            result = run.evaluate(case, self.attempt(case, calls))
            self.assertFalse(result['passed'])
            self.assertEqual(len(calls), len(result['behavior']['actual_tools']))

    def test_required_tools_citation_count_and_raw_format_remain_strict(self):
        case = self.cases['R5E-013']
        good = self.attempt(case, [self.call(name='generate_visit_brief', status=case['expected']['required_tool_result_statuses'][0])])
        self.assertTrue(run.evaluate(case, good)['passed'])
        self.assertFalse(run.evaluate(case, {**good, 'calls': []})['passed'])
        extra = self.attempt(case, good['calls'], case['expected']['allowed_citation_ids'][:2])
        self.assertFalse(run.evaluate(case, extra)['response']['citation_correct'])
        malformed = parse_final_v3('{"final_status":"ok","citations":[],"refusal":false} explanation')
        self.assertIsNotNone(malformed.failure_code)
        self.assertFalse(run.evaluate(case, {**good, 'parser_error': malformed.failure_code})['passed'])

    def test_preparation_binds_new_model_and_source_without_gold_leak(self):
        plan = self.prepare()
        self.assertEqual('deepseek-flash', plan['model_config']['model'])
        self.assertEqual('high', plan['model_config']['reasoning_effort'])
        self.assertEqual(16384, plan['model_config']['max_output_tokens'])
        self.assertEqual(4, plan['limits']['model_turns_per_attempt'])
        self.assertEqual(0, plan['api_calls'])
        self.assertEqual(plan, run.validate(self.root / 'preflight/preflight.json'))
        prompt = run.task_prompt(self.cases['R5E-010'])
        self.assertNotIn('use 0 relevant', prompt)
        self.assertNotIn('expected', prompt)
        self.assertNotIn(self.cases['R5E-004']['expected']['allowed_citation_ids'][0], prompt)
        with patch.object(run, 'implementation_binding', return_value={'sha256': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'source or data'):
                run.validate(self.root / 'preflight/preflight.json')

    def test_deepseek_transport_uses_bound_output_and_keeps_raw_thinking(self):
        class Client:
            is_real = False
            requests = []
            def complete(self, payload):
                self.requests.append(payload)
                return {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'reasoning_content': 'synthetic-thought',
                    'content': '{"final_status":"insufficient_evidence","citations":[],"refusal":true}'}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        plan = self.prepare()
        output = self.root / 'transport'
        output.mkdir()
        client = Client()
        transport = run.current.E2ETransport(plan, output, client)
        payload = {'model': 'deepseek-flash', 'max_tokens': 16384, 'temperature': 0, 'stream': False,
                   'messages': [{'role': 'user', 'content': 'synthetic'}],
                   'tools': [{'function': {'name': name}} for name in plan['tool_names']]}
        response = asyncio.run(transport.post_json(plan['endpoint'], {}, payload, 60))
        self.assertEqual('synthetic-thought', response['choices'][0]['message']['reasoning_content'])
        self.assertNotIn('do_sample', client.requests[0])
        self.assertEqual(16384, client.requests[0]['max_tokens'])
        bad = {**payload, 'max_tokens': 2048}
        with self.assertRaisesRegex(ValueError, 'parameters changed'):
            asyncio.run(transport.post_json(plan['endpoint'], {}, bad, 60))
        self.assertEqual(1, len(client.requests))
        old_plan = copy.deepcopy(plan)
        old_plan['model_config'].update(model='glm-5.3-flash', max_output_tokens=2048, reasoning_effort='low')
        old_plan['limits']['maximum_calls'] = 1
        old_output = self.root / 'old-transport'
        old_output.mkdir()
        old_client = Client()
        old_client.requests = []
        old_transport = run.current.E2ETransport(old_plan, old_output, old_client)
        old_payload = {**payload, 'model': 'glm-5.3-flash', 'max_tokens': 2048}
        asyncio.run(old_transport.post_json(plan['endpoint'], {}, old_payload, 60))
        self.assertFalse(old_client.requests[0]['do_sample'])
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            asyncio.run(old_transport.post_json(plan['endpoint'], {}, old_payload, 60))
        self.assertEqual(1, len(old_client.requests))

    def test_invalid_approval_stops_before_provider_or_key_read(self):
        self.prepare()
        for name, content in (('approval.json', {'approved': True, 'authorization_source': 'old_budget', 'plan_sha256': 'wrong'}),
                              ('review.json', {'decision': 'PASS', 'plan_sha256': 'wrong'})):
            (self.root / name).write_text(json.dumps(content), encoding='utf-8')
        with patch.object(run, 'configured_web_model', side_effect=AssertionError('must not read provider')):
            with self.assertRaisesRegex(ValueError, 'new explicit paid approval'):
                asyncio.run(run.execute(self.root / 'preflight/preflight.json', self.root / 'real',
                    approval=self.root / 'approval.json', review=self.root / 'review.json'))
        self.assertFalse((self.root / 'real').exists())

    def test_actual_native_loop_cannot_finish_before_timeline_tool(self):
        case = self.cases['R5E-004']
        from healthvault.infrastructure import hybrid_health_knowledge
        class UnusedEmbedder:
            dimension = 1024
            def embed(self, _texts):
                raise AssertionError('A timeline task must not perform knowledge embedding.')
        class Client:
            is_real = False
            requests = []
            def complete(self, payload):
                self.requests.append(payload)
                tool_messages = [item for item in payload['messages'] if item['role'] == 'tool']
                if len(self.requests) == 1:
                    message = {'role': 'assistant', 'content': '{"final_status":"insufficient_evidence","citations":[],"refusal":true}'}
                    finish = 'stop'
                elif not tool_messages:
                    message = {'role': 'assistant', 'content': None, 'reasoning_content': 'synthetic-plan', 'tool_calls': [
                        {'id': 'timeline', 'type': 'function', 'function': {'name': 'query_health_timeline', 'arguments': '{}'}}]}
                    finish = 'tool_calls'
                else:
                    ids = json.loads(tool_messages[-1]['content'])['available_citation_ids']
                    message = {'role': 'assistant', 'content': json.dumps({'final_status': 'ok', 'citations': ids[:1], 'refusal': False})}
                    finish = 'stop'
                return {'choices': [{'finish_reason': finish, 'message': message}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        plan = self.prepare()
        output = self.root / 'loop'
        output.mkdir()
        client = Client()
        transport = run.current.E2ETransport(plan, output, client)
        transport.case_id, transport.arm_id = case['case_id'], 'project_baseline'
        with quiet_agent_ui(), patch.object(hybrid_health_knowledge, '_local_embedder', return_value=UnusedEmbedder()):
            result = asyncio.run(run.run_case(case, plan, transport, output / 'cases'))
        self.assertTrue(result['passed'], result)
        self.assertEqual(['query_health_timeline'], result['oracle']['behavior']['actual_tools'])
        self.assertEqual(3, len(client.requests))
        self.assertTrue(any(item.get('reasoning_content') == 'synthetic-plan' for item in client.requests[-1]['messages']))


if __name__ == '__main__':
    unittest.main()
