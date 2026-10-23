import contextlib
import copy
import gc
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.ui import quiet_agent_ui
from healthvault import agent_runtime as native
from healthvault.evaluation import entrypoint_skill_run as run
from healthvault.model_config import HealthVaultModelConfig
from tests.healthvault.test_task102_answer_flow import QUESTION, SOURCES, SHORT, review


def response(content, usage=None):
    message = content if isinstance(content, dict) else {'role': 'assistant', 'content': content}
    return {'choices': [{'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop', 'message': message}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5} if usage is None else usage}


def tool(name, arguments):
    return response({'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'call-1', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]})


class FakeClient:
    is_real = False

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def complete(self, payload):
        self.requests.append(copy.deepcopy(payload))
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result(payload) if callable(result) else result


def verification(payload):
    return response(json.dumps(review(json.loads(payload['messages'][-1]['content'])), ensure_ascii=False))


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        scratch = run.ROOT / '.task_outputs/task-114-offline-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(gc.collect)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(quiet_agent_ui())
        self.stack.enter_context(patch.object(agent_module.openai, 'AsyncOpenAI', return_value=MagicMock()))
        self.stack.enter_context(patch.object(agent_module.anthropic, 'AsyncAnthropic', return_value=MagicMock()))
        self.stack.enter_context(patch.dict('os.environ', {'HEALTHVAULT_UI_DB_PATH': str(self.root / 'unused.sqlite3'),
                                                        'HEALTHVAULT_KNOWLEDGE_MODE': 'lexical'}))
        self.model = HealthVaultModelConfig('openai-compatible', 'deepseek-flash', 'https://api.deepseek.com', 'fake', 'high')
        self.plan = {'model': run.public_model(self.model), 'skills': {}, 'limits': run.LIMITS,
                     'corpus_path': str(run.ROOT / 'healthvault/knowledge/r1-2026-08-24-v1/chunks/hybrid_v1.json'),
                     'hybrid_index_path': 'unused-offline-index', 'hybrid_model_path': 'unused-offline-model'}
        for arm in run.ARMS:
            path = self.root / arm / '.bear/skills' / run.TARGET_SKILL / 'SKILL.md'
            path.parent.mkdir(parents=True)
            text = run.PROJECT.read_text(encoding='utf-8')
            if arm == 'candidate':
                text += '\n<!--citation:v1-->\nTASK114 TEST PERSONAL: retain the requested direct citation.\n<!--/citation-->\n'
            path.write_text(text, encoding='utf-8')
            self.plan['skills'][arm] = {'root': str(path.parent.parent), 'sha256': run.digest(path)}
        self.index = 0

    def collect(self, case, replies, arm='baseline', result=None):
        self.index += 1
        output = self.root / str(self.index)
        output.mkdir()
        client = FakeClient(replies)
        calls = run.EntryCalls(self.plan, output, client)
        calls.case_id, calls.arm_id = case['case_id'], arm
        original = native.build_healthvault_agent_runtime
        self.options = []

        def offline_build(options):
            self.options.append(options)
            runtime = original(replace(options, knowledge_mode='lexical'))
            if result is not None:
                runtime.agent._tool_executor = AsyncMock(return_value=result if isinstance(result, str) else json.dumps(result))
            return runtime

        with patch.object(native, 'build_healthvault_agent_runtime', offline_build):
            row = run.run_case(case, arm, self.plan, calls, output / 'case', self.model)
        return row, calls, client

    def test_raw_http_entrypoint_does_not_leak_expected_or_force_final_json(self):
        case = copy.deepcopy(run.selected_cases()[0])
        case['expected'] = {'required_tools': ['EXPECTED_MUST_NOT_ENTER_RUNTIME']}
        case['product_contract'] = {'intent': 'EXPECTED_INTENT_MUST_NOT_ENTER_RUNTIME'}
        answer = '已确认记录包含空腹血糖。 [REC:R43-ORC-02:fasting_glucose]'
        row, calls, client = self.collect(case, [tool('query_health_timeline', {}), response(answer), verification])
        self.assertEqual(200, row['http_status'])
        self.assertEqual(case['prompt'], row['request']['prompt'])
        self.assertEqual(case['prompt'], client.requests[0]['messages'][-1]['content'])
        self.assertEqual(answer, row['response']['answer'])
        self.assertIsNone(row['task_success'])
        self.assertIsNone(row['complete_and_supported'])
        serialized = json.dumps(client.requests)
        self.assertNotIn('EXPECTED_MUST_NOT_ENTER_RUNTIME', serialized)
        self.assertNotIn('EXPECTED_INTENT_MUST_NOT_ENTER_RUNTIME', serialized)
        self.assertNotIn('This is a structured evaluation task', serialized)
        self.assertTrue(all('response_format' not in request for request in client.requests))
        self.assertNotEqual('EXPECTED_INTENT_MUST_NOT_ENTER_RUNTIME', row['manifest'].get('workflow_intent'))
        self.assertTrue(row['sources'])
        self.assertTrue(row['database_unchanged'])
        self.assertEqual('hybrid', self.options[0].knowledge_mode)
        self.assertEqual(['agent', 'agent', 'verification'], [call['phase'] for call in calls.rows])

    def test_all_agent_verifier_and_once_repair_usage_including_partial_unknown(self):
        case = {'case_id': 'offline-repair', 'profile_id': 'r5-orchid', 'prompt': QUESTION}
        evidence = {'status': 'ok', 'available_citation_ids': ['HVK:test:a'], 'results': [
            {'citation_id': source['citation_id'], 'title': source['title'], 'excerpt': source['detail']} for source in SOURCES]}
        draft = {'status': 'answered', 'claims': [
            {'text': 'A1C用于监测。', 'citation_id': 'HVK:test:a', 'point_ids': ['P1']},
            {'text': 'A1C反映三个月。', 'citation_id': 'HVK:test:a', 'point_ids': ['P2']}],
            'point_evidence': [{'point_id': 'P1', 'evidence_ids': ['S1-detail-1']},
                               {'point_id': 'P2', 'evidence_ids': ['S1-detail-2']}]}
        row, calls, _client = self.collect(case, [tool('search_health_knowledge', {'query': 'A1C'}), response(SHORT),
            verification, response(json.dumps(draft, ensure_ascii=False), {'prompt_tokens': 17}), verification], result=evidence)
        self.assertEqual(200, row['http_status'])
        self.assertEqual(['agent', 'agent', 'verification', 'repair', 'verification'], [item['phase'] for item in calls.rows])
        self.assertEqual(1, row['response']['measurement']['answer_repair_attempts'])
        self.assertEqual(1, len(row['tool_calls']))
        self.assertEqual(5, row['usage']['model_calls'])
        self.assertEqual(1, row['usage']['unknown_usage_calls'])
        self.assertEqual(57, row['usage']['observed_input_tokens'])
        self.assertEqual(20, row['usage']['observed_output_tokens'])
        self.assertIsNone(row['usage']['total_tokens'])
        self.assertFalse(calls.failed)

    def test_empty_and_error_returns_and_actual_scoped_personal_injection(self):
        case = copy.deepcopy(run.selected_cases()[0])
        row, _calls, client = self.collect(case, [tool('query_health_timeline', {}),
            response('已确认记录包含空腹血糖。 [REC:R43-ORC-02:fasting_glucose]'), verification], arm='candidate')
        self.assertTrue(row['injection_contract_passed'])
        self.assertTrue(row['manifest']['personal_skill_injected'])
        self.assertEqual('matched_single_source', row['manifest']['personal_skill_reason'])
        self.assertIn(self.plan['skills']['candidate']['sha256'], {item['sha256'] for item in row['manifest']['project_skills']})
        self.assertIn('TASK114 TEST PERSONAL', client.requests[0]['messages'][0]['content'])
        for status, normal in [('empty', 1), ('error', 0)]:
            with self.subTest(status=status):
                case = {'case_id': status, 'profile_id': 'r5-empty', 'prompt': 'Search knowledge about alpha.'}
                final = json.dumps({'final_status': 'insufficient_evidence', 'citations': [], 'refusal': True})
                row, _calls, _client = self.collect(case, [tool('search_health_knowledge', {'query': 'alpha'}), *[response(final) for _ in range(5)]],
                    arm='candidate', result={'status': status, 'available_citation_ids': [], 'results': []})
                self.assertEqual(normal, row['tool_metrics']['normal_tool_returns'])
                self.assertEqual(int(status == 'empty'), row['tool_metrics']['empty_searches'])
                self.assertFalse(row['manifest']['personal_skill_injected'])
                self.assertTrue(row['injection_contract_passed'])
                self.assertIsNot(row['task_success'], True)
        text = 'Tool unavailable under Agent runtime policy: query_health_timeline'
        case = {'case_id': 'text', 'profile_id': 'r5-empty', 'prompt': 'Read the stored timeline.'}
        final = json.dumps({'final_status': 'rejected', 'citations': [], 'refusal': True})
        row, _calls, client = self.collect(case, [tool('query_health_timeline', {}), *[response(final) for _ in range(5)]], result=text)
        self.assertEqual(text, row['tool_calls'][0]['raw_result'])
        self.assertEqual(text, next(message['content'] for message in client.requests[1]['messages'] if message['role'] == 'tool'))
        self.assertEqual(1, row['tool_metrics']['tool_attempts'])
        self.assertEqual(0, row['tool_metrics']['normal_tool_returns'])
        self.assertNotIn('error_type', row['tool_calls'][0])
        case['case_id'] = 'denied'
        row, _calls, _client = self.collect(case, [tool('skill_create', {'name': 'not-permitted'}), *[response(final) for _ in range(5)]])
        self.assertEqual(1, row['tool_metrics']['tool_attempts'])
        self.assertEqual(0, row['tool_metrics']['normal_tool_returns'])
        self.assertFalse(row['tool_calls'][0]['execution_started'])
        self.assertIn('Action denied', row['tool_calls'][0]['raw_result'])

    def test_frozen_binding_once_budget_protocol_stop_and_not_started_rows(self):
        output = self.root / 'budget'
        output.mkdir()
        calls = run.EntryCalls(self.plan, output, FakeClient([response('ok')] * 10))
        payload = {'messages': [], 'model': 'fake'}
        for _ in range(9):
            calls.request('agent', payload)
        with self.assertRaises(RuntimeError):
            calls.request('agent', payload)
        self.assertEqual(9, len(calls.client.requests))
        self.assertTrue(calls.failed)
        old = json.loads((run.SOURCE / 'preflight.json').read_text(encoding='utf-8'))
        candidate = Path(self.plan['skills']['candidate']['root']) / run.TARGET_SKILL / 'SKILL.md'
        with patch.object(run, 'configured_web_model', return_value=self.model), \
             patch.object(run, 'model_fingerprint', return_value=old['embedding_model_sha256']):
            plan = run.prepare(self.root / 'prepared', candidate)
            path = self.root / 'prepared/plan.json'
            run.validate(path)
            (self.root / 'prepared/dataset.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                run.validate(path)
        path = self.root / 'run-plan.json'
        dataset = self.root / 'dataset.json'
        run.save(path, {})
        run.save(dataset, {'cases': run.selected_cases()})
        plan = {**self.plan, 'dataset': str(dataset)}
        approval, checked = self.root / 'approval.json', self.root / 'review.json'
        run.save(approval, {'approved': True, 'authorization_source': 'human_user', 'plan_sha256': run.digest(path)})
        run.save(checked, {'decision': 'PASS', 'plan_sha256': run.digest(path)})
        fake = FakeClient([{'choices': [], 'usage': {'prompt_tokens': 19}}])
        def fail(case, arm, _plan, ledger, _root, _config):
            ledger.request('agent', payload)
        with patch.object(run, 'ROOT', self.root), patch.object(run, 'validate', return_value=(plan, self.model)), \
             patch.object(run, 'run_case', side_effect=fail):
            result = run.run(path, self.root / 'stopped', approval=approval, review=checked, client=fake)
            self.assertEqual('stopped_on_error', result['status'])
            self.assertEqual(24, result['planned_attempts'])
            self.assertEqual(23, result['not_started_attempts'])
            self.assertEqual(1, result['usage']['unknown_usage_calls'])
            self.assertEqual(19, result['usage']['observed_input_tokens'])
            with self.assertRaises(FileExistsError):
                run.run(path, self.root / 'second', approval=approval, review=checked, client=fake)
        self.assertEqual(1, len(fake.requests))


if __name__ == '__main__':
    unittest.main()
