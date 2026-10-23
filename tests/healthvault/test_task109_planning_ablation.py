import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from healthvault.evaluation import planning_ablation_run as ablation
from healthvault.evaluation import knowledge_quality_run as quality
from tests.healthvault.test_task102_answer_flow import QUESTION, SOURCES, review


class PlanningAblationTests(unittest.TestCase):
    def setUp(self):
        scratch = ablation.ROOT / '.task_outputs/task-109-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {'model': {'model': 'synthetic'}, 'conditions': ablation.CONDITIONS, 'scope': 'test'}
        self.questions = [{'query_id': 'Q1', 'query': QUESTION}]
        self.snapshot = {**self.questions[0], 'sources': copy.deepcopy(SOURCES), 'tool_result': {
            'status': 'ok', 'results': [{'citation_id': s['citation_id'], 'title': s['title'], 'excerpt': s['detail']}
                                      for s in SOURCES]}}

    def ledgers(self, request):
        calls = {}
        for arm, planning in ablation.ARMS.items():
            directory = self.root / arm
            directory.mkdir(exist_ok=True)
            calls[arm] = quality.Calls(self.plan, directory, lambda payload, arm=arm: request(arm, payload),
                limits={**quality.LIMITS, 'maximum_calls': 120 if planning else 96, 'per_question': 5 if planning else 4},
                phase_caps={**quality.PHASE_CAPS, 'planning': 24 if planning else 0})
        return calls

    @staticmethod
    def response(raw):
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(raw)}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}

    def test_question_extraction_and_evidence_freeze_never_read_gold_answers(self):
        gold = {'queries': [{**self.questions[0], 'reference_claims': ['SECRET_GOLD'],
                             'expected': {'required_tools': ['secret_tool']}}]}
        self.assertEqual(self.questions, ablation.questions_only(gold))
        repository = MagicMock(last_search_metadata={'dense_participated': True})
        received = []
        def search(tool_input, passed_repository):
            received.append(tool_input)
            self.assertIs(repository, passed_repository)
            return SimpleNamespace(to_dict=lambda: copy.deepcopy(self.snapshot['tool_result']))
        with patch('healthvault.infrastructure.sqlite_fts_health_knowledge.SQLiteFTSHealthKnowledgeRepository'), \
             patch('healthvault.infrastructure.hybrid_health_knowledge.HybridHealthKnowledgeRepository.from_local_artifacts',
                   return_value=repository), \
             patch('healthvault.application.search_health_knowledge.search_health_knowledge', side_effect=search):
            frozen = ablation.freeze_evidence({'corpus_path': str(self.root / 'chunks/hybrid_v1.json'),
                                               'hybrid_index_path': 'local', 'hybrid_model_path': 'local'},
                                              ablation.questions_only(gold))
        self.assertEqual(1, len(received))
        self.assertNotIn('SECRET_GOLD', json.dumps(frozen))
        self.assertNotIn('secret_tool', json.dumps(frozen))
        self.assertEqual(self.snapshot['tool_result'], frozen[0]['tool_result'])
        repository.close.assert_called_once()

    def test_each_arm_enforces_phase_and_question_limits_before_transport(self):
        transmitted = []
        calls = self.ledgers(lambda arm, payload: (transmitted.append(arm) or self.response({})))
        for arm, planning in ablation.ARMS.items():
            ledger = calls[arm]
            ledger.case_id = 'Q1'
            if planning:
                ledger.call('planning', 'prompt', {})
            else:
                with self.assertRaises(RuntimeError):
                    ledger.call('planning', 'prompt', {})
            for phase in ('generation', 'verification', 'generation', 'verification'):
                ledger.call(phase, 'prompt', {})
            with self.assertRaises(RuntimeError):
                ledger.call('generation', 'prompt', {})
            self.assertEqual(5 if planning else 4, len(ledger.rows))
        self.assertEqual(9, len(transmitted))
        self.assertEqual(120, quality.LIMITS['maximum_calls'])
        self.assertEqual(24, quality.PHASE_CAPS['planning'])

    def test_real_api_pair_only_changes_planning_and_keeps_repair_review_evidence(self):
        requests, generations = {arm: [] for arm in ablation.ARMS}, dict.fromkeys(ablation.ARMS, 0)
        def request(arm, payload):
            data = json.loads(payload['messages'][1]['content'])
            requests[arm].append(data)
            if 'minimum_points' in data:
                raw = {'protocol': 'question_plan_v1', 'additional_points': []}
            elif 'evidence' in data:
                generations[arm] += 1
                self.assertEqual(QUESTION, data['question'])
                raw = {'status': 'answered', 'point_evidence': [
                    {'point_id': 'P1', 'evidence_ids': ['S1-detail-1']},
                    {'point_id': 'P2', 'evidence_ids': ['S1-detail-2']}], 'claims': [
                    {'text': 'A1C用于监测。', 'citation_id': 'HVK:test:a', 'point_ids': ['P1']}]}
                if generations[arm] == 2:
                    raw['claims'].append({'text': 'A1C反映三个月。', 'citation_id': 'HVK:test:a', 'point_ids': ['P2']})
            else:
                raw = review(data)
            return self.response(raw)
        calls = self.ledgers(request)
        rows = ablation.pending_rows(self.questions)
        status = ablation.collect_pair(self.plan, self.questions, [self.snapshot], calls, self.root, rows)
        result = ablation.summarize(self.plan, rows, calls, status)
        self.assertEqual('completed', status)
        self.assertEqual([5, 4], [len(calls[arm].rows) for arm in ablation.ARMS])
        self.assertEqual(9, result['api_calls'])
        for row in rows:
            self.assertTrue(row['result']['verification']['passed'])
            self.assertEqual(1, len(row['result']['answer_repairs']))
            self.assertEqual(QUESTION, row['question'])
            self.assertFalse(row['application_error'])
            self.assertEqual('frozen_evidence_replay', row['searches'][0]['execution'])
            self.assertIsNone(row['quality_review']['complete_and_supported'])
        generated = [[r['evidence'] for r in requests[arm] if 'evidence' in r] for arm in ablation.ARMS]
        self.assertEqual(generated[0][0]['evidence_spans'], generated[1][0]['evidence_spans'])
        self.assertEqual(generated[0][0]['question_plan']['points'], generated[1][0]['question_plan']['points'])
        self.assertTrue(all(row['complete_and_supported_rate'] is None for row in result['arms'].values()))

    def test_application_error_and_unstarted_questions_are_preserved(self):
        calls = self.ledgers(lambda *_a: self.response({}))
        questions = self.questions + [{'query_id': 'Q2', 'query': '第二个问题'}]
        rows = ablation.pending_rows(questions)
        clients = []
        def make_client(_output, _calls, _controller, *, planning):
            client = MagicMock()
            clients.append(client)
            if planning:
                client.post.return_value = SimpleNamespace(status_code=502, json=lambda: {'detail': 'failed review'})
            else:
                client.post.side_effect = OSError('synthetic transport failure')
            return client
        with patch.object(ablation, 'make_client', side_effect=make_client):
            status = ablation.collect_pair(self.plan, questions, [self.snapshot] * 2, calls, self.root, rows)
        result = ablation.summarize(self.plan, rows, calls, status)
        self.assertEqual('stopped_on_error', status)
        self.assertEqual(['application_error'] * 2 + ['not_started'] * 2, [r['status'] for r in rows])
        self.assertEqual(4, len(result['cases']))
        for arm in result['arms'].values():
            self.assertEqual(1, arm['application_error_cases'])
            self.assertEqual(1, arm['uncompleted_cases'])
            self.assertIsNone(arm['complete_and_supported_rate'])
        for client in clients:
            client.close.assert_called_once()

    def test_offline_validation_detects_source_evidence_and_condition_drift(self):
        questions = [{'query_id': f'Q{i}', 'query': f'问题{i}'} for i in range(24)]
        paths = {name: self.root / f'{name}.json' for name in ('gold', 'questions', 'evidence', 'corpus_path')}
        for name, value in (('gold', {'queries': questions}), ('questions', questions), ('evidence', questions), ('corpus_path', {})):
            quality.save(paths[name], value)
        plan = {**self.plan, 'protocol': ablation.PROTOCOL, 'arms': ablation.ARMS, 'limits': ablation.LIMITS,
                'phase_caps': ablation.PHASE_CAPS, 'source_sha256': 'source', 'embedding_model_sha256': 'embed',
                'hybrid_model_path': 'synthetic', 'schedule': 'original_question_order_alternating_first_arm',
                'case_ids': [q['query_id'] for q in questions], **{key: str(path) for key, path in paths.items()},
                'bindings': {str(path.resolve()): quality.digest(path) for path in paths.values()}}
        plan_path = self.root / 'plan.json'
        quality.save(plan_path, plan)
        with patch.object(ablation, 'TASK100_GOLD', paths['gold']), \
             patch('healthvault.evaluation.current_delivery.implementation_binding', return_value={'sha256': 'source'}), \
             patch('healthvault.rag.vector_cache.model_fingerprint', return_value='embed'), \
             patch('healthvault.web.model_runtime.configured_web_model', return_value=object()), \
             patch.object(quality, 'public_model', return_value=self.plan['model']):
            self.assertEqual(ablation.PROTOCOL, ablation.validate(plan_path)[0]['protocol'])
            for field, value in (('source_sha256', 'drift'), ('conditions', {})):
                plan_path.write_text(json.dumps({**plan, field: value}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Frozen'):
                    ablation.validate(plan_path)
            plan_path.write_text(json.dumps(plan), encoding='utf-8')
            paths['evidence'].write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                ablation.validate(plan_path)

    def test_run_requires_new_bound_approval_before_model_or_transport(self):
        path = self.root / 'plan.json'
        quality.save(path, {})
        with patch('healthvault.web.model_runtime.configured_web_model', side_effect=AssertionError('no model')):
            with self.assertRaises(ValueError):
                ablation.run(path, None, None, self.root / 'run')
            approval, checked = self.root / 'approval.json', self.root / 'review.json'
            quality.save(approval, {'approved': True, 'authorization_source': 'human_user', 'plan_sha256': 'wrong'})
            quality.save(checked, {'decision': 'PASS', 'plan_sha256': 'wrong'})
            with self.assertRaises(ValueError):
                ablation.run(path, approval, checked, self.root / 'run')
        self.assertFalse((self.root / 'run').exists())


if __name__ == '__main__':
    unittest.main()
