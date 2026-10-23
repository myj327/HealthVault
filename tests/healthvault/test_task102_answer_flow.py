import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from healthvault.web.answer_coverage import SEMANTIC_DIMENSIONS
from healthvault.web.answer_validation import verify_answer
from healthvault.web.answers import answer_with_model
from healthvault.web.model_runtime import ModelServiceError
from healthvault.web.question_plan import create_question_plan


QUESTION = 'A1C的用途和时间范围？'
SOURCES = [{'citation_id': 'HVK:test:a', 'title': 'A1C',
            'detail': 'A1C helps monitoring.\nA1C reflects three months.'}]
SHORT = 'A1C用于监测。 [HVK:test:a]'
FULL = SHORT + '\nA1C反映三个月。 [HVK:test:a]'


def review(data, *, missing=False, unsupported=False):
    spans = [span for span in data['evidence_spans'] if span['source_field'] == 'detail']
    facts = data['facts']
    claims = [{'claim_id': f['claim_id'], 'supported': not unsupported,
               'evidence_ids': [s['evidence_id'] for s in spans if s['citation_id'] == f['citation_id']],
               'checks': dict.fromkeys(SEMANTIC_DIMENSIONS, not unsupported)} for f in facts]
    coverage = []
    for index, point in enumerate(data['required_points']):
        absent = index > 0 and len(facts) == 1
        coverage.append({**point, 'status': ('missing_evidence' if missing else 'omitted') if absent else 'covered',
                         'evidence_ids': [] if absent and missing else [s['evidence_id'] for s in spans],
                         'answer_claim_ids': [] if absent else [f['claim_id'] for f in facts]})
    return {'protocol': 'fact_span_v2', 'answer_supported': not unsupported,
            'answers_question': all(p['status'] == 'covered' for p in coverage),
            'claims': claims, 'question_coverage': coverage}


class AnswerFlowTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[2] / '.task_outputs/task-102-tests'
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = 0
        fixture = Path(__file__).with_name('fixtures') / 'task102_original_flow.json'
        self.originals = {r['case_id']: r for r in json.loads(fixture.read_text(encoding='utf-8'))['examples']}

    def endpoint(self, generate, verifier, *, sources=None, planner=None, **options):
        with patch.dict('os.environ', {'HEALTHVAULT_UI_DB_PATH': str(self.scratch / 'import.sqlite3'),
                                      'HEALTHVAULT_KNOWLEDGE_MODE': 'lexical'}):
            from healthvault.web.api import create_app, AnswerInput
        from fastapi import Response
        from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
        self.index += 1
        directory = self.root / str(self.index)
        directory.mkdir()
        sources = SOURCES if sources is None else sources
        searches = []
        def search(query, **_kwargs):
            searches.append(query)
            return {'results': [{'citation_id': s['citation_id'], 'title': s['title'], 'excerpt': s['detail']}
                                for s in sources]}
        controller = SimpleNamespace(record_repository=SQLiteHealthRecordRepository(directory / 'records.sqlite3'),
                                     knowledge_release_root=directory / 'releases', search_knowledge=search)
        app = create_app(controller, model_answerer=generate, model_answer_verifier=verifier,
                         model_question_planner=planner, skill_root=directory / 'skills',
                         skill_evolution_root=directory / 'evolution', **options)
        function = next(r.endpoint for r in app.routes if r.path == '/api/answers/knowledge')
        request = SimpleNamespace(state=SimpleNamespace(auth=SimpleNamespace(user=SimpleNamespace(profile_id='alice'))))
        return lambda question=QUESTION: function(AnswerInput(question=question, model_consent=True), Response(), request), app, searches

    def test_original_planner_expansion_is_optional_not_a_reason_to_refuse(self):
        original = self.originals['RAG-022']
        plan = create_question_plan(original['question'], query=lambda *_a, **_k: original['raw_content'])
        baseline = create_question_plan(original['question'])
        self.assertEqual(baseline['points'], plan['points'])
        self.assertEqual(1, len(plan['points']))
        self.assertIn('喝白水', plan['points'][0]['description'])
        self.assertEqual(4, len(plan['optional_points']))
        self.assertTrue(all(p['point_id'].startswith('O') for p in plan['optional_points']))
        self.assertIn('饮用量', str(plan['optional_points']))
        self.assertNotIn('饮用量', str(plan['points']))

    def test_definition_is_explicit_and_a_test_name_does_not_add_preparation(self):
        question = self.originals['MQ-016']['question']
        plan = create_question_plan(question)
        self.assertIn('定义', plan['points'][0]['description'])
        self.assertIn(question, plan['points'][0]['description'])
        self.assertIn('不能仅以作用或后果代替', plan['points'][0]['description'])
        plan = create_question_plan('空腹血糖是什么？')
        self.assertEqual(1, len(plan['points']))
        self.assertNotIn('检查准备', str(plan['points']))

    def draft(self):
        return {'status': 'answered',
                'point_evidence': [{'point_id': 'P1', 'evidence_ids': ['S1-detail-1']},
                                   {'point_id': 'P2', 'evidence_ids': ['S1-detail-2']}],
                'claims': [{'text': 'A1C用于监测。', 'citation_id': 'HVK:test:a', 'point_ids': ['P1']},
                           {'text': 'A1C反映三个月。', 'citation_id': 'HVK:test:a', 'point_ids': ['P2']}]}

    def test_generation_supplies_server_spans_and_records_point_ownership(self):
        evidence = {'sources': copy.deepcopy(SOURCES), 'question_plan': create_question_plan(QUESTION)}
        sent = []
        def generate(prompt, payload, **_kwargs):
            sent.append((prompt, payload))
            self.assertNotIn('detail', payload['evidence']['sources'][0])
            self.assertEqual('A1C helps monitoring.', payload['evidence']['evidence_spans'][0]['quote'])
            return json.dumps(self.draft())
        generated = answer_with_model(QUESTION, 'knowledge', evidence, query=generate)
        self.assertEqual(FULL.replace('\n', '\n\n'), generated)
        self.assertEqual(['HVK:test:a'], evidence['draft_grounding'][0]['citation_ids'])
        self.assertEqual(SOURCES, evidence['sources'])
        self.assertIn('point_evidence_v1', sent[0][0])

    def test_raw_json_missing_mapping_and_cross_source_assignments_fail(self):
        valid = self.draft()
        no_map = {k: v for k, v in valid.items() if k != 'point_evidence'}
        wrong_source = copy.deepcopy(valid)
        wrong_source['claims'][0]['citation_id'] = 'HVK:test:b'
        bad_span = copy.deepcopy(valid)
        bad_span['point_evidence'][0]['evidence_ids'] = ['S1-title-1']
        wrong_point = copy.deepcopy(valid)
        wrong_point['claims'][0]['point_ids'] = ['O1']
        for raw in ('{', json.dumps(valid) + '追加说明', json.dumps(no_map), json.dumps(wrong_source),
                    json.dumps(bad_span), json.dumps(wrong_point)):
            evidence = {'sources': [*copy.deepcopy(SOURCES), {'citation_id': 'HVK:test:b', 'title': 'Other',
                         'detail': 'Another source.'}], 'question_plan': create_question_plan(QUESTION)}
            with self.subTest(raw=raw[:60]), self.assertRaises(ModelServiceError):
                answer_with_model(QUESTION, 'knowledge', evidence, query=lambda *_a, **_k: raw)
            self.assertNotIn('draft_grounding', evidence)

    def test_web_repairs_existing_evidence_omission_once_and_saves_final_only(self):
        for initial in ('omitted', 'source_assignment'):
            drafts, seen = [], []
            def generate(q, mode, evidence):
                drafts.append(copy.deepcopy(evidence))
                raw = self.draft()
                if len(drafts) == 1:
                    if initial == 'omitted':
                        raw['claims'] = raw['claims'][:1]
                    else:
                        raw['claims'][0]['citation_id'] = 'HVK:test:b'
                return answer_with_model(q, mode, evidence, query=lambda *_a, **_kw: json.dumps(raw))
            def verifier(_p, data, **_kw):
                seen.append(data)
                return review(data)
            run, app, searches = self.endpoint(generate, verifier)
            result = run()
            self.assertEqual('answered', result['status'])
            self.assertEqual(2, len(drafts))
            self.assertEqual(2 if initial == 'omitted' else 1, len(seen))
            self.assertEqual(1, len(searches))
            self.assertEqual(drafts[0]['question_plan'], drafts[1]['question_plan'])
            if initial == 'omitted':
                self.assertEqual('P2', drafts[1]['answer_repair']['omitted_points'][0]['point_id'])
                self.assertEqual(SHORT, drafts[1]['answer_repair']['previous_answer'])
            else:
                self.assertEqual('claim_source_assignment', drafts[1]['answer_repair']['reason'])
            self.assertEqual(FULL.replace('\n', '\n\n'), result['answer'])
            self.assertEqual(2, len(result['point_evidence']))
            self.assertEqual(1, len(result['answer_repairs']))
            self.assertEqual(1, len(app.state.answer_history.list('alice')))

    def test_original_binding_errors_stay_rejected_then_get_one_fresh_review(self):
        for case_id in ('RAG-021', 'RAG-041'):
            original = self.originals[case_id]
            data = original['payload']
            first = verify_answer(data['question'], data['answer'], data['sources'], audit_coverage=True,
                                  required_points=data['required_points'], query=lambda *_a, **_k: original['raw_content'])
            self.assertTrue(first['support_passed'], case_id)
            self.assertFalse(first['coverage_contract_passed'], case_id)
            self.assertFalse(first['passed'], case_id)
            drafts, reviewed = [], []
            plan = {'original_question': data['question'], 'resolved_question': data['question'],
                    'points': data['required_points'], 'optional_points': []}
            def generate(_q, _m, evidence):
                drafts.append(copy.deepcopy(evidence))
                return data['answer']
            def verifier(_p, payload, **_kw):
                reviewed.append(payload)
                return original['raw_content'] if len(reviewed) == 1 else review(payload)
            run, app, _ = self.endpoint(generate, verifier, sources=data['sources'])
            with patch('healthvault.web.question_plan.create_question_plan', return_value=plan):
                result = run(data['question'])
            self.assertEqual('answered', result['status'], case_id)
            self.assertEqual(2, len(drafts), case_id)
            self.assertTrue(drafts[1]['answer_repair']['coverage_binding_error'])
            self.assertEqual(1, len(app.state.answer_history.list('alice')))

    def test_failed_repair_protocol_errors_and_real_missing_evidence_stop(self):
        from fastapi import HTTPException
        for initial in ('bad_citation', 'unsupported', 'bad_protocol', 'missing'):
            drafts, reviews = [], []
            def generate(_q, _m, evidence):
                drafts.append(copy.deepcopy(evidence))
                return '不受支持。 [HVK:missing]' if initial == 'bad_citation' and len(drafts) == 1 else SHORT
            def verifier(_p, data, **_kw):
                reviews.append(data)
                if initial == 'bad_protocol':
                    return {'protocol': 'fact_span_v1', 'claims': []}
                return review(data, missing=initial == 'missing', unsupported=initial != 'missing')
            sources = [{'citation_id': 'HVK:test:a', 'title': 'A1C', 'detail': 'A1C helps monitoring.'}]
            run, app, searches = self.endpoint(generate, verifier, sources=sources)
            if initial == 'missing':
                self.assertEqual('insufficient_evidence', run()['status'])
                self.assertEqual(1, len(drafts))
                self.assertEqual(2, len(searches))
            else:
                with self.assertRaises(HTTPException) as failure:
                    run()
                self.assertEqual(502, failure.exception.status_code)
                self.assertEqual(1 if initial == 'bad_protocol' else 2, len(drafts))
            self.assertEqual([], app.state.answer_history.list('alice'))

    def test_comparison_switches_change_one_step_and_keep_required_points_fixed(self):
        from fastapi import HTTPException
        observed = []
        for auto_plan, repair in ((True, False), (False, False), (False, True)):
            drafts, planned = [], []
            def planner(_p, payload, **_kw):
                planned.append(payload)
                return {'protocol': 'question_plan_v1', 'additional_points': [
                    {'question_quote': 'A1C', 'description': '可选补充说明'}]}
            def generate(_q, _m, evidence):
                drafts.append(copy.deepcopy(evidence))
                return SHORT if len(drafts) == 1 else FULL
            run, _app, _ = self.endpoint(generate, lambda _p, data, **_kw: review(data), planner=planner,
                                        auto_plan_suggestions=auto_plan, answer_repair_enabled=repair)
            if repair:
                self.assertEqual('answered', run()['status'])
            else:
                with self.assertRaises(HTTPException):
                    run()
            observed.append({'planning_calls': len(planned), 'generation_calls': len(drafts),
                             'points': drafts[0]['question_plan']['points'], 'sources': drafts[0]['sources']})
        self.assertEqual([1, 0, 0], [row['planning_calls'] for row in observed])
        self.assertEqual([1, 1, 2], [row['generation_calls'] for row in observed])
        self.assertTrue(all(row['points'] == observed[0]['points'] for row in observed))
        self.assertTrue(all(row['sources'] == observed[0]['sources'] for row in observed))


if __name__ == '__main__':
    unittest.main()
