import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from healthvault.evaluation.answer_quality import answer_quality_metrics
from healthvault.web.answer_validation import verify_answer
from healthvault.web.answers import answer_with_model, knowledge_evidence


class AnswerCompletenessTests(unittest.TestCase):
    def sources(self):
        return [{'citation_id': 'HVK:test:home', 'title': '家庭监测', 'detail': '家庭监测可使用血糖仪。'},
                {'citation_id': 'HVK:test:a1c', 'title': 'A1C', 'detail': 'A1C 反映过去三个月平均血糖。'}]

    def test_supported_but_incomplete_answer_is_distinguishable(self):
        sources = self.sources()
        def checker(_prompt, data, **_kwargs):
            self.assertEqual(sources, data['sources'])
            return {'protocol': 'fact_span_v1', 'answer_supported': True, 'answers_question': False,
                    'claims': [{'claim_id': 'F1', 'supported': True, 'evidence_ids': ['S1-detail-1']}]}
        checked = verify_answer('两种方法的用途和时间范围是什么？', '家庭监测可使用血糖仪。[HVK:test:home]', sources, query=checker)
        self.assertFalse(checked['passed'])
        self.assertTrue(checked['support_passed'])
        self.assertFalse(checked['completeness_passed'])
        self.assertEqual(1.0, checked['model_assessed_fact_support_rate'])
        self.assertIsNone(checked['answer'])

    def test_unreferenced_source_cannot_support_a_written_fact(self):
        sources = self.sources()
        def checker(_p, _data, **_k):
            return {'protocol': 'fact_span_v1', 'answer_supported': True, 'answers_question': True,
                    'claims': [{'claim_id': 'F1', 'supported': True, 'evidence_ids': ['S2-detail-1']}]}
        checked = verify_answer('A1C 是什么？', 'A1C 反映过去三个月平均血糖。[HVK:test:home]', sources, query=checker)
        self.assertFalse(checked['support_passed'])
        self.assertFalse(checked['passed'])

    def test_web_passes_unreferenced_sources_to_completeness_check(self):
        # Exercise the synchronous endpoint without an ASGI server/event loop.
        from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
        scratch = Path(__file__).resolve().parents[2] / '.task_outputs/task-080-082-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            self.assertTrue(root.resolve().is_relative_to(scratch.resolve()))
            with patch.dict(os.environ, {'HEALTHVAULT_UI_DB_PATH': str(root / 'import.sqlite3'),
                                        'HEALTHVAULT_KNOWLEDGE_MODE': 'lexical',
                                        'HEALTHVAULT_KNOWLEDGE_RELEASE_ROOT': str(root / 'releases')}):
                from fastapi import HTTPException, Response
                from healthvault.web.api import AnswerInput, create_app
            sources = self.sources()
            controller = SimpleNamespace(record_repository=SQLiteHealthRecordRepository(root / 'records.sqlite3'),
                knowledge_release_root=root / 'releases', search_knowledge=lambda *_a, **_k: {'results': [
                    {'citation_id': s['citation_id'], 'title': s['title'], 'excerpt': s['detail']} for s in sources]})
            captured = []
            def checker(_prompt, data, **_kwargs):
                captured.append(data)
                return {'protocol': 'fact_span_v1', 'answer_supported': True, 'answers_question': False,
                        'claims': [{'claim_id': 'F1', 'supported': True, 'evidence_ids': ['S1-detail-1']}]}
            app = create_app(controller, lambda *_a: '家庭监测可使用血糖仪。[HVK:test:home]',
                             skill_root=root / 'skills', skill_evolution_root=root / 'evolution', model_answer_verifier=checker)
            endpoint = next(route.endpoint for route in app.routes if route.path == '/api/answers/knowledge')
            request = SimpleNamespace(state=SimpleNamespace(auth=SimpleNamespace(user=SimpleNamespace(profile_id='alice'))))
            result = endpoint(AnswerInput(question='家庭监测与 A1C 有什么区别？', model_consent=True), Response(), request)
            self.assertEqual('insufficient_evidence', result['status'])
            self.assertEqual(2, len(captured[0]['sources']))
            self.assertEqual('HVK:test:a1c', captured[0]['sources'][1]['citation_id'])
            self.assertEqual(['S1-title-1', 'S1-detail-1'], captured[0]['facts'][0]['allowed_evidence_ids'])
            app.state.model_answerer = lambda *_a: '家庭监测可使用血糖仪。[HVK:test:home]\n' * 31
            with self.assertRaises(HTTPException) as error:
                endpoint(AnswerInput(question='家庭血糖监测是什么？', model_consent=True), Response(), request)
            self.assertEqual(502, error.exception.status_code)
            self.assertIn('超过 30 条引用事实', error.exception.detail)
            self.assertEqual(1, len(captured))

    def test_generation_receives_all_topics_without_inventing_missing_evidence(self):
        hits = [{'citation_id': s['citation_id'], 'title': s['title'], 'excerpt': s['detail']} for s in self.sources()]
        evidence, sources = knowledge_evidence({'results': hits})
        def generate(prompt, data, **_kwargs):
            self.assertIn('不能只因排序靠后就遗漏', prompt)
            self.assertIn('不能用常识补齐', prompt)
            self.assertEqual(sources, data['evidence']['sources'])
            return json.dumps({'status': 'answered', 'claims': [
                {'text': s['detail'], 'citation_id': s['citation_id']} for s in sources]}, ensure_ascii=False)
        answer = answer_with_model('两种方法是什么？', 'knowledge', evidence, query=generate)
        self.assertIn('HVK:test:home', answer)
        self.assertIn('HVK:test:a1c', answer)

    def test_reviewed_support_and_point_coverage_use_different_denominators(self):
        result = answer_quality_metrics([{'answerable': True, 'status': 'answered', 'reviewer_kind': 'assistant',
            'facts': [{'passed': True}] * 5, 'reference_points': [{'passed': False}, {'passed': False}, {'passed': True}]}])
        self.assertEqual(1.0, result['fact_support']['rate'])
        self.assertAlmostEqual(1 / 3, result['key_point_coverage']['rate'])
        self.assertFalse(result['human_confirmation_complete'])

    def test_missing_review_labels_do_not_disappear_or_become_success(self):
        result = answer_quality_metrics([{'answerable': True, 'status': 'answered',
            'facts': [{'passed': True}, {'passed': None}], 'reference_points': [{'passed': None}]}])
        self.assertIsNone(result['fact_support']['rate'])
        self.assertIsNone(result['key_point_coverage']['rate'])
        self.assertEqual(2, result['fact_support']['total'])
        self.assertEqual(1, result['fact_support']['reviewed'])
        absent = answer_quality_metrics([{'answerable': True, 'status': 'answered', 'facts': [{'passed': True}],
                                          'reference_points': [{'passed': True}]},
                                         {'answerable': True, 'status': 'answered'}])
        self.assertEqual(1, absent['fact_support']['unreviewed_cases'])
        self.assertIsNone(absent['fact_support']['rate'])
        self.assertIsNone(absent['key_point_coverage']['rate'])

    def test_false_refusal_and_false_answer_are_separate_from_errors(self):
        result = answer_quality_metrics([{'answerable': True, 'status': 'insufficient_evidence'},
                                         {'answerable': True, 'status': 'error'},
                                         {'answerable': False, 'status': 'answered'},
                                         {'answerable': False, 'status': 'insufficient_evidence'}])
        self.assertEqual(0.5, result['answerable_false_refusal_rate'])
        self.assertEqual(0.5, result['unanswerable_false_answer_rate'])
        self.assertEqual(1, result['errors'])


if __name__ == '__main__':
    unittest.main()
