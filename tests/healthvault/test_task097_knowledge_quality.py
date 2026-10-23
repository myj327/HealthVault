import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from healthvault.evaluation.knowledge_quality_run import Calls, LIMITS, PHASE_CAPS, ROOT, collect_questions, digest, validate


class KnowledgeQualityTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / '.task_outputs/task097-test-scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assertTrue(self.root.resolve().is_relative_to(scratch.resolve()))
        self.plan = {'model': {'model': 'deepseek-flash'}}

    def calls(self, text='{}'):
        def request(payload):
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': text}}],
                    'usage': {'prompt_tokens': 12, 'completion_tokens': 4, 'total_tokens': 16}}
        calls = Calls(self.plan, self.root, request)
        calls.case_id = 'Q1'
        return calls

    def test_phase_and_question_budgets_apply_before_transmission(self):
        calls = self.calls()
        calls.call('planning', 'prompt', {})
        with self.assertRaises(RuntimeError):
            calls.call('planning', 'prompt', {})
        for phase in ('generation', 'verification'):
            calls.call(phase, 'prompt', {})
            calls.call(phase, 'prompt', {})
        self.assertEqual(5, len(calls.rows))
        with self.assertRaises(RuntimeError):
            calls.call('generation', 'prompt', {})
        calls.case_id = 'Q2'
        with self.assertRaises(ValueError):
            calls.call('generation', 'prompt', {'text': 'a' * LIMITS['max_input_bytes']})
        self.assertEqual(5, len(calls.rows))

    def test_raw_format_error_preserves_usage_and_blocks_further_calls(self):
        calls = self.calls('{}\n追加说明')
        with self.assertRaises(ValueError):
            calls.call('generation', 'prompt', {})
        self.assertTrue(calls.failed)
        self.assertEqual('failed', calls.rows[0]['status'])
        self.assertEqual(16, calls.rows[0]['usage']['total_tokens'])
        self.assertTrue((self.root / 'responses/0001.json').exists())
        with self.assertRaises(RuntimeError):
            calls.call('generation', 'prompt', {})
        self.assertEqual(1, len(calls.rows))

    def test_http_errors_stay_errors_and_reference_answers_are_not_posted(self):
        calls = self.calls()
        posted = []
        class Client:
            def post(self, url, json):
                posted.append((url, json))
                return SimpleNamespace(status_code=502, json=lambda: {'detail': 'invalid raw answer'})
        result = collect_questions(Client(), [{'query_id': 'Q1', 'query': 'A1C需要空腹吗？',
                                    'reference_claims': ['SECRET_GOLD']}], calls, self.root, [])
        self.assertEqual('completed_with_errors', result['status'])
        self.assertEqual(1, result['http_error_cases'])
        self.assertNotIn('SECRET_GOLD', json.dumps(posted))
        self.assertFalse(posted[0][1]['allow_web_search'])
        self.assertFalse(posted[0][1]['allow_query_rewrite'])

    def test_missing_or_drifted_approval_cannot_read_model_or_call_api(self):
        paths = [self.root / name for name in ('plan.json', 'approval.json', 'review.json')]
        for path, value in zip(paths, ({}, {'approved': True, 'authorization_source': 'human_user',
                                          'plan_sha256': 'wrong'}, {'decision': 'PASS', 'plan_sha256': 'wrong'})):
            path.write_text(json.dumps(value), encoding='utf-8')
        with patch('healthvault.web.model_runtime.configured_web_model', side_effect=AssertionError('must not load model')):
            with self.assertRaises(ValueError):
                validate(*paths)
            changed = {'protocol': 'current-knowledge-quality-v1', 'limits': LIMITS, 'phase_caps': PHASE_CAPS,
                       'source_sha256': 'changed'}
            paths[0].write_text(json.dumps(changed), encoding='utf-8')
            for path, value in ((paths[1], {'approved': True, 'authorization_source': 'human_user'}),
                                (paths[2], {'decision': 'PASS'})):
                path.write_text(json.dumps({**value, 'plan_sha256': digest(paths[0])}), encoding='utf-8')
            with self.assertRaises(ValueError):
                validate(*paths)

    def test_actual_web_entry_invokes_planner_generation_and_verifier(self):
        from fastapi.testclient import TestClient
        from healthvault.web.answer_coverage import SEMANTIC_DIMENSIONS
        from healthvault.evaluation.web_rag_run import FrozenController
        from healthvault.infrastructure.sqlite_fts_health_knowledge import SQLiteFTSHealthKnowledgeRepository
        from healthvault.web.answers import answer_with_model
        # This callback-contract test does not load an unused default BGE app.
        with patch.dict('os.environ', {'HEALTHVAULT_UI_DB_PATH': str(self.root / 'import.sqlite3'),
                                      'HEALTHVAULT_KNOWLEDGE_MODE': 'lexical'}):
            from healthvault.web.api import create_app
        corpus = SQLiteFTSHealthKnowledgeRepository()
        self.addCleanup(corpus.close)
        controller = FrozenController(self.root / 'records.sqlite3', corpus)
        controller.knowledge_release_root = self.root / 'releases'
        hit = corpus.hit_for_chunk_id('HVK:r1-2026-08-24-v1:niddk-diabetes-prediabetes-tests:003:ee350877b4db',
                                     rank=1, score=1.0, matched_terms=())
        controller.hits = (hit,)
        calls = self.calls()
        def request(payload):
            data = json.loads(payload['messages'][1]['content'])
            if 'minimum_points' in data:
                self.assertNotIn('reference_claims', data)
                raw = {'protocol': 'question_plan_v1', 'additional_points': []}
            elif 'evidence' in data:
                self.assertEqual('question_rules_with_model_suggestions', data['evidence']['question_plan']['method'])
                points = data['evidence']['question_plan']['points']
                ids = [span['evidence_id'] for span in data['evidence']['evidence_spans']]
                raw = {'status': 'answered', 'point_evidence': [{'point_id': p['point_id'], 'evidence_ids': ids} for p in points],
                       'claims': [{'text': 'A1C 无需空腹。', 'citation_id': hit.document.citation_id,
                                   'point_ids': [p['point_id'] for p in points]}]}
            else:
                ids = [span['evidence_id'] for span in data['evidence_spans'] if span['source_field'] == 'detail']
                raw = {'protocol': 'fact_span_v2', 'answer_supported': True, 'answers_question': True,
                       'claims': [{'claim_id': fact['claim_id'], 'supported': True, 'evidence_ids': ids,
                                   'checks': dict.fromkeys(SEMANTIC_DIMENSIONS, True)} for fact in data['facts']],
                       'question_coverage': [{**point, 'status': 'covered', 'evidence_ids': ids,
                                              'answer_claim_ids': [fact['claim_id'] for fact in data['facts']]}
                                             for point in data['required_points']]}
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(raw, ensure_ascii=False)}}],
                    'usage': {'prompt_tokens': 12, 'completion_tokens': 4, 'total_tokens': 16}}
        calls.request = request
        app = create_app(controller, model_answerer=lambda q, mode, evidence: answer_with_model(q, mode, evidence,
                           query=calls.callback('generation')), model_question_planner=calls.callback('planning'),
                           model_answer_verifier=calls.callback('verification'), skill_evolution_root=self.root / 'skills')
        client = TestClient(app)
        self.addCleanup(client.close)
        registered = client.post('/api/auth/register', json={'username': 'qa-evaluation', 'password': 'synthetic-password-123'})
        self.assertEqual(200, registered.status_code)
        client.headers['X-CSRF-Token'] = registered.json()['csrf_token']
        result = collect_questions(client, [{'query_id': 'Q1', 'query': 'A1C需要空腹吗？'}], calls, self.root, [])
        self.assertEqual(200, result['cases'][0]['http_status'], result)
        self.assertEqual('answered', result['cases'][0]['result']['status'])
        self.assertEqual(['planning', 'generation', 'verification'], [row['phase'] for row in calls.rows])


if __name__ == '__main__':
    unittest.main()
