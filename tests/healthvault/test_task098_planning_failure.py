import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from healthvault.evaluation.knowledge_quality_run import Calls, collect_questions, LIMITS, PHASE_CAPS
from healthvault.web.answers import answer_with_model
from healthvault.web.model_runtime import ModelServiceError
from healthvault.web.question_plan import create_question_plan, MAX_QUESTION_POINTS, rule_question_points


class PlanningFailureTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / '.task_outputs/task098-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assertTrue(self.root.resolve().is_relative_to(scratch.resolve()))
        fixture = Path(__file__).with_name('fixtures') / 'task098_original_outputs.json'
        self.originals = json.loads(fixture.read_text(encoding='utf-8'))['outputs']

    def test_original_overflow_plans_fall_back_without_keeping_a_prefix(self):
        for original in self.originals[:3]:
            seen = []
            def query(_prompt, data, **_kwargs):
                seen.append(data)
                self.assertEqual(4, data['max_additional_points'])
                self.assertEqual(MAX_QUESTION_POINTS, data['max_total_points'])
                return original['raw_content']
            plan = create_question_plan(original['question'], query=query)
            expected = rule_question_points(original['question'])
            self.assertEqual(expected, [{k: point[k] for k in ('question_quote', 'description')} for point in plan['points']])
            self.assertEqual('rule_fallback', plan['planning']['status'])
            self.assertEqual('additional_point_budget_exceeded', plan['planning']['reason'])
            self.assertEqual(len(json.loads(original['raw_content'])['additional_points']), plan['planning']['rejected_additional_points'])
            self.assertEqual(original['question'], plan['original_question'])
            self.assertEqual(1, len(seen))

    def test_explicit_requirements_and_remaining_slots_cannot_be_removed(self):
        question = '糖化血红蛋白和空腹血糖的用途、时间范围、准备、方法、机制、比较？'
        baseline = create_question_plan(question)
        self.assertEqual(12, len(baseline['points']))
        with patch('healthvault.web.question_plan.run_model_task', side_effect=AssertionError('No slots, no call.')):
            self.assertEqual(baseline, create_question_plan(question, use_model=True))
        with self.assertRaises(ModelServiceError) as failure:
            create_question_plan(question + '含义是什么？')
        self.assertEqual(422, failure.exception.status_code)
        question = '去年左侧检查的用途和时间范围？'
        minimum = rule_question_points(question)
        extra = {'question_quote': '左侧检查', 'description': '检查对象'}
        def planner(_prompt, data, **_kwargs):
            self.assertEqual(minimum, data['minimum_points'])
            return {'protocol': 'question_plan_v1', 'additional_points': [*minimum, extra, extra]}
        plan = create_question_plan(question, query=planner)
        self.assertEqual('model_accepted', plan['planning']['status'])
        self.assertEqual(minimum, [{k: p[k] for k in ('question_quote', 'description')} for p in plan['points']])
        self.assertEqual([extra], [{k: p[k] for k in ('question_quote', 'description')} for p in plan['optional_points']])
        self.assertEqual(question, plan['resolved_question'])

    def test_bad_original_json_or_invalid_last_point_cannot_use_budget_fallback(self):
        original = self.originals[0]
        broken = json.loads(original['raw_content'])
        broken['additional_points'][-1]['question_quote'] = 'not present in the question'
        for raw in ('{', original['raw_content'] + '说明', broken):
            with self.subTest(raw_type=type(raw).__name__), self.assertRaises(ModelServiceError) as failure:
                create_question_plan(original['question'], query=lambda *_a, **_k: raw)
            self.assertEqual(502, failure.exception.status_code)

    def test_web_exposes_fallback_and_generation_never_sees_rejected_suggestions(self):
        with patch.dict(os.environ, {'HEALTHVAULT_UI_DB_PATH': str(self.root / 'import.sqlite3'),
                                    'HEALTHVAULT_KNOWLEDGE_MODE': 'lexical',
                                    'HEALTHVAULT_KNOWLEDGE_RELEASE_ROOT': str(self.root / 'releases')}):
            from fastapi import Response
            from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
            from healthvault.web.api import AnswerInput, create_app
        original = self.originals[0]
        controller = SimpleNamespace(record_repository=SQLiteHealthRecordRepository(self.root / 'records.sqlite3'),
            knowledge_release_root=self.root / 'releases', search_knowledge=lambda *_a, **_k: {'results': [
                {'citation_id': 'HVK:test:definition', 'title': 'Diagnosis', 'excerpt': 'Tests for diagnosis.'}]})
        generated = []
        def generate(question, mode, evidence):
            generated.append(evidence)
            self.assertNotIn('additional_points', evidence['question_plan'])
            self.assertEqual(1, len(evidence['question_plan']['points']))
            return answer_with_model(question, mode, evidence, query=lambda *_a, **_k:
                                     json.dumps({'status': 'insufficient_evidence', 'claims': [],
                                                 'point_evidence': [{'point_id': p['point_id'], 'evidence_ids': []}
                                                                    for p in evidence['question_plan']['points']]}))
        app = create_app(controller, model_answerer=generate,
                         model_question_planner=lambda *_a, **_k: original['raw_content'],
                         skill_root=self.root / 'skills', skill_evolution_root=self.root / 'evolution')
        endpoint = next(r.endpoint for r in app.routes if r.path == '/api/answers/knowledge')
        request = SimpleNamespace(state=SimpleNamespace(auth=SimpleNamespace(user=SimpleNamespace(profile_id='alice'))))
        result = endpoint(AnswerInput(question=original['question'], model_consent=True), Response(), request)
        self.assertEqual('rule_fallback', result['question_plan']['planning']['status'])
        self.assertEqual('insufficient_evidence', result['status'])
        self.assertEqual(1, len(generated))
        self.assertEqual([], app.state.answer_history.list('alice'))

    def test_original_truncation_is_failed_and_preserved_even_for_complete_json(self):
        original = self.originals[3]
        for index, content in enumerate((original['raw_content'], '{"status":"insufficient_evidence","claims":[]}')):
            output = self.root / str(index)
            output.mkdir()
            response = {'choices': [{'finish_reason': original['finish_reason'], 'message': {'content': content}}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 16384}}
            sent = []
            def request(payload):
                sent.append(payload)
                return response
            calls = Calls({'limits': LIMITS, 'phase_caps': PHASE_CAPS, 'model': {'model': 'synthetic'}}, output, request)
            calls.case_id = original['case_id']
            with self.assertRaises(ValueError):
                calls.call('generation', 'JSON', {'question': original['question']})
            self.assertEqual('failed', calls.rows[0]['status'])
            self.assertEqual('output_token_limit', calls.rows[0]['failure_reason'])
            self.assertEqual('length', calls.rows[0]['finish_reason'])
            with self.assertRaises(RuntimeError):
                calls.call('generation', 'JSON', {})
            self.assertEqual(1, len(sent))
            saved = json.loads((output / 'responses/0001.json').read_text(encoding='utf-8'))
            self.assertEqual(response, saved['response'])
        with self.assertRaises(ModelServiceError):
            answer_with_model(original['question'], 'knowledge', {}, query=lambda *_a, **_k: original['raw_content'])

    def test_non_json_http_error_retains_status_body_latency_and_call_failure(self):
        class Client:
            count = 0
            def post(client, *_args, **_kwargs):
                client.count += 1
                return SimpleNamespace(status_code=500, text='Internal Server Error',
                                       json=lambda: (_ for _ in ()).throw(ValueError('not JSON')))
        client = Client()
        calls = Calls({}, self.root, None)
        calls.failed = True
        calls.rows = [{'sequence': 1, 'case_id': 'MQ-016', 'phase': 'generation', 'status': 'failed',
                       'error_type': 'ValueError', 'failure_reason': 'output_token_limit', 'finish_reason': 'length',
                       'usage': {'prompt_tokens': 10, 'completion_tokens': 16384}}]
        result = collect_questions(client, [{'query_id': 'MQ-016', 'query': 'What is cholesterol?'},
                                            {'query_id': 'remaining', 'query': 'Do not run.'}], calls, self.root, [])
        row = result['cases'][0]
        self.assertEqual('stopped_on_error', result['status'])
        self.assertEqual(1, client.count)
        self.assertEqual(1, result['http_error_cases'])
        self.assertEqual(500, row['http_status'])
        self.assertEqual('Internal Server Error', row['response_text'])
        self.assertGreaterEqual(row['latency_ms'], 0)
        self.assertEqual('output_token_limit', row['model_failures'][0]['failure_reason'])
        self.assertNotIn('result', row)
        self.assertFalse((self.root / 'remaining.json').exists())


if __name__ == '__main__':
    unittest.main()
