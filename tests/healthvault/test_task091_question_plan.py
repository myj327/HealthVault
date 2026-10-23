import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from healthvault.web.answer_coverage import SEMANTIC_DIMENSIONS
from healthvault.web.answer_validation import verify_answer
from healthvault.web.answers import knowledge_evidence, recover_missing_knowledge
from healthvault.web.model_runtime import ModelServiceError
from healthvault.web.question_plan import create_question_plan


def checked(data, bindings, statuses=None):
    statuses = statuses or {}
    claims = [{'claim_id': fact['claim_id'], 'evidence_ids': bindings[fact['claim_id']], 'supported': True,
               'checks': dict.fromkeys(SEMANTIC_DIMENSIONS, True)} for fact in data['facts']]
    points = []
    for point in data['required_points']:
        status, ids, facts = statuses.get(point['point_id'], ('covered', bindings[data['facts'][0]['claim_id']], [data['facts'][0]['claim_id']]))
        points.append({**point, 'status': status, 'evidence_ids': ids, 'answer_claim_ids': facts})
    return {'protocol': 'fact_span_v2', 'answer_supported': True,
            'answers_question': all(point['status'] == 'covered' for point in points),
            'claims': claims, 'question_coverage': points}


class QuestionPlanTests(unittest.TestCase):
    def test_planner_only_sees_question_and_cannot_delete_explicit_requirements(self):
        def planner(_prompt, payload, **_kwargs):
            self.assertEqual({'original_question', 'resolved_question', 'minimum_points',
                              'max_additional_points', 'max_total_points'}, set(payload))
            return '{"protocol":"question_plan_v1","additional_points":[]}'
        plan = create_question_plan('A1C的用途和时间范围是什么？', query=planner)
        self.assertIn('用途', [point['description'] for point in plan['points']])
        self.assertIn('时间范围', [point['description'] for point in plan['points']])
        plan = create_question_plan('糖化血红蛋白和空腹血糖的用途？', query=planner)
        self.assertTrue(any('糖化血红蛋白' in point['description'] for point in plan['points']))
        self.assertTrue(any('空腹血糖' in point['description'] for point in plan['points']))
        for raw in ('{', '{"protocol":"question_plan_v1","additional_points":[]}说明',
                    {'protocol': 'question_plan_v1', 'additional_points': [{'question_quote': '不存在', 'description': '用途'}]}):
            with self.subTest(raw=raw), self.assertRaises(ModelServiceError):
                create_question_plan('A1C有什么用途？', query=lambda *_a, **_k: raw)

    def test_review_cannot_omit_or_replace_a_frozen_point(self):
        question = 'A1C用途和时间范围？'
        plan = create_question_plan(question)
        sources = [{'citation_id': 'HVK:test:a', 'title': 'A1C', 'detail': 'A1C monitoring. A1C reflects three months.'}]
        answer = 'A1C可以监测并反映三个月。 [HVK:test:a]'
        def reviewer(_prompt, data, **_kwargs):
            result = checked(data, {'F1': ['S1-detail-1']})
            if mode == 'omit':
                result['question_coverage'].pop()
            elif mode == 'replace':
                result['question_coverage'][0]['description'] = '另一项要求'
            elif mode == 'cross_citation':
                result['claims'][0]['evidence_ids'] = ['S2-detail-1']
            return result
        sources.append({'citation_id': 'HVK:test:b', 'title': 'Other', 'detail': 'A1C monitoring three months.'})
        for mode in ('valid', 'omit', 'replace', 'cross_citation'):
            result = verify_answer(question, answer, sources, query=reviewer, audit_coverage=True,
                                   required_points=plan['points'])
            self.assertEqual(mode == 'valid', result['passed'], mode)

    def test_recovery_is_bounded_and_preserves_explicit_constraints(self):
        question = '去年左侧检查的监测用途是什么？'
        points = [{'point_id': f'P{i}', 'question_quote': question, 'description': text}
                  for i, text in enumerate(('monitoring purpose', 'methods', 'time range'), 1)]
        plan = {'points': points}
        calls = []
        def search(expression, **_kwargs):
            calls.append(expression)
            return {'results': [{'citation_id': f'HVK:test:{len(calls)}', 'title': 'Source', 'excerpt': 'Evidence.'}]}
        review = {'support_passed': True, 'coverage_contract_passed': True, 'missing_evidence_points': ['P1', 'P2', 'P3']}
        merged, attempts = recover_missing_knowledge(search, question, plan, review, [])
        self.assertEqual(2, len(calls))
        for expression in calls:
            self.assertIn('去年', expression)
            self.assertIn('左侧', expression)
            self.assertNotIn('今年', expression)
            self.assertNotIn('右侧', expression)
        self.assertEqual(question, plan['points'][0]['question_quote'])
        self.assertEqual(2, len(merged))
        points[0]['description'] = '今年右侧检查'
        calls.clear()
        _, attempts = recover_missing_knowledge(search, question, plan, review, [])
        self.assertEqual('constraint_rejected', attempts[0]['status'])
        self.assertEqual(1, len(calls))

    def test_existing_source_conflict_and_no_added_source_are_not_recovery(self):
        question = 'A1C用途？'
        plan = create_question_plan(question)
        sources = [{'citation_id': 'HVK:test:a', 'title': 'A1C', 'detail': 'Original.'}]
        review = {'support_passed': True, 'coverage_contract_passed': True, 'missing_evidence_points': ['P1']}
        result = {'results': [{'citation_id': 'HVK:test:a', 'title': 'A1C', 'excerpt': 'Original.'}]}
        merged, attempts = recover_missing_knowledge(lambda *_a, **_k: result, question, plan, review, sources)
        self.assertEqual(sources, merged)
        self.assertEqual(0, attempts[0]['new_sources'])
        result['results'][0]['excerpt'] = 'Changed.'
        with self.assertRaises(ModelServiceError):
            recover_missing_knowledge(lambda *_a, **_k: result, question, plan, review, sources)

    def test_current_corpus_can_retrieve_a1c_monitoring_purpose(self):
        from healthvault.infrastructure.sqlite_fts_health_knowledge import SQLiteFTSHealthKnowledgeRepository
        from healthvault.application.search_health_knowledge import search_health_knowledge
        repository = SQLiteFTSHealthKnowledgeRepository()
        self.addCleanup(repository._connection.close)
        question = 'What is (are) Monitor Your Diabetes ?'
        plan = {'points': [{'point_id': 'P1', 'question_quote': question, 'description': 'A1C monitoring purpose'}]}
        review = {'support_passed': True, 'coverage_contract_passed': True, 'missing_evidence_points': ['P1']}
        sources, attempts = recover_missing_knowledge(lambda query, limit: search_health_knowledge(
            {'query': query, 'limit': limit}, repository).to_dict(), question, plan, review, [])
        matches = [source for source in sources if 'help monitor your condition' in source['detail']]
        self.assertTrue(matches, sources)
        self.assertTrue(matches[0]['citation_id'].startswith('HVK:r1-2026-08-24-v1:'))
        self.assertGreater(attempts[0]['new_sources'], 0)

    def test_chinese_missing_point_uses_same_current_corpus_without_dropping_original(self):
        from healthvault.infrastructure.sqlite_fts_health_knowledge import SQLiteFTSHealthKnowledgeRepository
        from healthvault.application.search_health_knowledge import search_health_knowledge
        repository = SQLiteFTSHealthKnowledgeRepository()
        self.addCleanup(repository._connection.close)
        question = '糖化血红蛋白的监测用途？'
        plan = create_question_plan(question)
        review = {'support_passed': True, 'coverage_contract_passed': True, 'missing_evidence_points': ['P1']}
        sources, _attempts = recover_missing_knowledge(lambda query, limit: search_health_knowledge(
            {'query': query, 'limit': limit}, repository).to_dict(), question, plan, review, [])
        self.assertTrue(any('help monitor your condition' in source['detail'] for source in sources))
        self.assertEqual(question, plan['original_question'])

    def endpoint(self, search, *, remains_missing=False, allow_web_search=False):
        from fastapi import Response
        from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
        from healthvault.web.api import create_app, AnswerInput
        scratch = Path(__file__).resolve().parents[2] / '.task_outputs/task-091-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        controller = SimpleNamespace(record_repository=SQLiteHealthRecordRepository(root / 'records.sqlite3'),
                                     knowledge_release_root=root / 'releases', search_knowledge=search)
        generated = []
        def generate(_question, _mode, evidence):
            generated.append(evidence)
            answer = f"A1C反映过去三个月。 [{evidence['sources'][0]['citation_id']}]"
            return answer + ('\nA1C可用于监测。 [HVK:test:purpose]' if len(evidence['sources']) > 1 else '')
        def review(_prompt, data, **_kwargs):
            bindings = {'F1': ['S1-detail-1']}
            missing = len(data['sources']) == 1 or remains_missing
            statuses = {'P1': ('missing_evidence', [], []) if missing else ('covered', ['S2-detail-1'], ['F2']),
                        'P2': ('covered', ['S1-detail-1'], ['F1'])}
            if len(data['sources']) > 1:
                bindings['F2'] = ['S2-detail-1']
            return checked(data, bindings, statuses)
        planned = []
        def planner(_prompt, data, **_kwargs):
            planned.append(data)
            return {'protocol': 'question_plan_v1', 'additional_points': []}
        app = create_app(controller, generate, model_answer_verifier=review, model_question_planner=planner,
                         skill_root=root / 'skills', skill_evolution_root=root / 'evolution')
        official_calls = []
        if allow_web_search:
            async def execute(name, arguments):
                if name == 'search_official_health_web':
                    official_calls.append(arguments)
                    return json.dumps({'results': [{'citation_id': 'HVW:test:time', 'title': 'A1C time',
                        'excerpt': 'A1C reflects three months.', 'source': {'url': 'https://medlineplus.gov/',
                        'publisher': 'NLM', 'retrieved_on': '2026-10-12', 'snapshot_sha256': 'synthetic'}}]})
                return json.dumps(search(arguments['query'], limit=arguments['limit']))
            app.state.read_only_tools = SimpleNamespace(execute=execute)
        app.state.fixture_official_calls = official_calls
        app.state.fixture_plans = planned
        endpoint = next(route.endpoint for route in app.routes if route.path == '/api/answers/knowledge')
        request = SimpleNamespace(state=SimpleNamespace(auth=SimpleNamespace(user=SimpleNamespace(profile_id='alice'))))
        result = endpoint(AnswerInput(question='A1C用途和时间范围？', model_consent=True,
                                      allow_web_search=allow_web_search), Response(), request)
        return result, generated, app

    def test_actual_web_answer_retrieves_missing_point_once_and_saves_only_final(self):
        calls = []
        def search(query, **_kwargs):
            calls.append(query)
            return {'results': [{'citation_id': 'HVK:test:time', 'title': 'A1C', 'excerpt': 'A1C reflects the past three months.'}]
                    if len(calls) == 1 else [{'citation_id': 'HVK:test:purpose', 'title': 'A1C uses', 'excerpt': 'A1C helps monitoring.'}]}
        result, generated, app = self.endpoint(search)
        self.assertEqual('answered', result['status'])
        self.assertEqual(2, len(generated))
        self.assertEqual(2, len(calls))
        self.assertEqual(1, len(app.state.answer_history.list('alice')))
        self.assertTrue(result['verification']['requirements_fixed'])

    def test_recovery_stops_without_new_sources_or_when_second_answer_is_still_missing(self):
        for extra, allow_web in ((False, False), (True, False), (False, True), (True, True)):
            calls = []
            def search(query, **_kwargs):
                calls.append(query)
                return {'results': [{'citation_id': 'HVK:test:time', 'title': 'A1C', 'excerpt': 'A1C reflects three months.'}]
                        if len(calls) == 1 or not extra else [{'citation_id': 'HVK:test:purpose', 'title': 'A1C uses', 'excerpt': 'A1C monitoring.'}]}
            result, generated, app = self.endpoint(search, remains_missing=True, allow_web_search=allow_web)
            self.assertEqual('insufficient_evidence', result['status'])
            self.assertEqual(2 if extra or allow_web else 1, len(generated))
            self.assertEqual(2, len(calls))
            self.assertEqual(1, len(app.state.fixture_plans))
            self.assertEqual(int(allow_web and not extra), len(app.state.fixture_official_calls))
            self.assertTrue(all(item['question_plan'] is generated[0]['question_plan'] for item in generated))
            self.assertEqual([], app.state.answer_history.list('alice'))


if __name__ == '__main__':
    unittest.main()
