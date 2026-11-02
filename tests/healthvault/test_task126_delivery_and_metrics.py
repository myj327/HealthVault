import json
import unittest
from unittest.mock import MagicMock

from healthvault.application.compare_health_indicators import compare_health_indicators
from healthvault.application.query_health_timeline import query_health_timeline
from healthvault.application.query_rewrite import prepare_query
from healthvault.evaluation.measurement import three_layer_result, summarize_three_layers
from healthvault.web.answer_validation import verify_grounded_answer
from healthvault.web.answers import answer_repair_feedback
from tests.healthvault.test_compare_health_indicators import RecordingRepository, make_indicator, make_report


def records():
    return RecordingRepository([make_report(name, day, [make_indicator('hdl_cholesterol', value)])
        for name, day, value in [('a', '2023-02-14', 1.2), ('b', '2023-06-05', 1.3),
                                 ('c', '2023-11-20', 1.1), ('d', '2024-01-09', 1.4)]])


class DeliveryAndMetricsTests(unittest.TestCase):
    def test_aliases_are_shared_and_unknown_codes_are_not_empty_results(self):
        from healthvault.domain.reports import normalize_indicator_code
        self.assertEqual('crp', normalize_indicator_code('C反应蛋白'))
        self.assertEqual('vitamin_d_25oh', normalize_indicator_code('25-羟维生素D'))
        for name in ('高密度脂蛋白胆固醇', '高密度脂蛋白', 'HDL-C', 'hdl_cholesterol'):
            params = {'profile_id': 'test', 'indicator_code': name}
            self.assertEqual('ok', query_health_timeline(params, records()).status)
            self.assertEqual('ok', compare_health_indicators(params, records()).status)
        repo = records()
        self.assertEqual('invalid_request', compare_health_indicators(
            {'profile_id': 'test', 'indicator_code': 'unknown'}, repo).status)
        self.assertEqual([], repo.queries)

    def test_same_year_and_exact_dates_choose_real_endpoints_without_substitution(self):
        params = {'profile_id': 'test', 'indicator_code': 'HDL-C'}
        annual = compare_health_indicators({**params, 'start_year': 2023, 'end_year': 2023}, records()).to_dict()
        self.assertEqual(-0.1, annual['comparisons'][0]['absolute_change'])
        self.assertEqual('year_endpoints', annual['year_selection']['method'])
        exact = compare_health_indicators({**params, 'start_date': '2023-06-05', 'end_date': '2023-11-20'}, records()).to_dict()
        item = exact['comparisons'][0]
        self.assertEqual(('b', 'c', -0.2), (item['earliest']['report_id'], item['latest']['report_id'], item['absolute_change']))
        self.assertNotEqual(annual['comparisons'][0]['comparison_citation_id'], item['comparison_citation_id'])
        missing = compare_health_indicators({**params, 'start_date': '2023-04-01', 'end_date': '2023-11-20'}, records())
        self.assertEqual('missing_date', missing.issues[0].code)
        for dates in ({'start_date': '2023-02-30', 'end_date': '2023-11-20'}, {'start_date': '2023-11-20'},
                      {'start_date': '2023-11-20', 'end_date': '2023-06-05'}):
            self.assertEqual('invalid_request', compare_health_indicators({**params, **dates}, records()).status)

    def test_rewrite_preserves_self_contained_question_and_cannot_demand_record_values(self):
        question = '普通血脂四项通常包括脂蛋白a吗？它从小到大会明显变化吗？'
        self.assertEqual(question, prepare_query(question).resolved_question)
        question = '把最新糖化血红蛋白的日期和数值写成一句门诊提问。'
        rewrite = MagicMock(return_value=json.dumps({'search_query': '糖化血红蛋白', 'needs_clarification': True}))
        result = prepare_query(question, model_query=rewrite, knowledge_only=False)
        self.assertEqual(('ready', question), (result.status, result.resolved_question))
        self.assertEqual('needs_clarification', prepare_query('这个指标是什么？').status)

    def test_failed_compact_check_returns_correction_but_does_not_auto_accept_it(self):
        source = {'citation_id': 'REC:example:hdl', 'detail': '2023-02-14 HDL-C 1.2 mmol/L'}
        draft, correction = '结果正常。[REC:example:hdl]', 'HDL-C为1.2 mmol/L。[REC:example:hdl]'
        result = verify_grounded_answer('最近HDL-C是多少？', draft, [source],
            query=lambda *a, **k: {'supported': False, 'complete': True, 'answer': correction})
        self.assertFalse(result['passed'])
        self.assertIsNone(result['answer'])
        self.assertEqual(correction, answer_repair_feedback(draft, result)['suggested_answer'])
        self.assertIsNone(answer_repair_feedback(draft, {'kind': 'invalid_verifier_output'}))

    def test_http_entry_repairs_once_and_rechecks_without_repeating_search(self):
        from tests.healthvault.test_web_answers import StubController, TestClient
        from healthvault.web.api import create_app
        hit = {'citation_id': 'HVK:example', 'title': 'LDL', 'excerpt': 'LDL是低密度脂蛋白。', 'source': {}}
        controller = StubController(hits=[hit])
        generate = MagicMock(side_effect=['LDL可以诊断疾病。[HVK:example]', 'LDL是低密度脂蛋白。[HVK:example]'])
        verify = MagicMock(side_effect=[{'supported': False, 'complete': True, 'answer': 'LDL是低密度脂蛋白。[HVK:example]'},
                                       {'supported': True, 'complete': True}])
        try:
            with TestClient(create_app(controller, generate, model_answer_verifier=verify, auto_plan_suggestions=False)) as client:
                response = client.post('/api/answers/knowledge', json={'question': 'LDL是什么？'})
                self.assertEqual(200, response.status_code, response.text)
                self.assertEqual(1, len(response.json()['answer_repairs']))
                self.assertEqual(2, verify.call_count)
                self.assertEqual(2, generate.call_count)
        finally:
            controller.directory.cleanup()

    def test_partial_brief_missing_indicator_is_explicit_without_generation(self):
        from tests.healthvault.test_web_answers import StubController, TestClient
        from healthvault.web.api import create_app
        controller = StubController()
        controller.visit_brief = lambda _: {'status': 'partial', 'indicators': [{'code': 'hba1c',
            'display_name': '糖化血红蛋白', 'latest': {'value': 5.8, 'unit': '%'}, 'comparison': None}],
            'record_citations': [], 'knowledge_citations': []}
        generate = MagicMock(side_effect=AssertionError('Missing record needs no generation.'))
        try:
            with TestClient(create_app(controller, generate)) as client:
                response = client.post('/api/profiles/demo-a/answers/visit', json={'question': '血钾从最早到最新变化多少？'})
                self.assertIn('没有钾记录', response.json()['answer'])
                generate.assert_not_called()
        finally:
            controller.directory.cleanup()

    def test_three_layers_include_semantics_and_failed_answers_in_denominators(self):
        def result(answerable=True, covered=2, response=True, hard=()):
            return three_layer_result(outcome={'database_unchanged': True}, behavior={'tools_correct': True},
                response={'delivered': response}, answerable=answerable, hard_failure_reasons=hard,
                quality={'facts_correct': True, 'evidence_supported': True, 'complete': covered == 2,
                         'refusal_correct': not answerable, 'covered_points': covered, 'required_points': 2})
        rows = [result(), result(covered=1), result(covered=0, response=False), result(answerable=False)]
        scores = summarize_three_layers(rows)
        self.assertEqual({'numerator': 2, 'denominator': 4, 'rate': .5}, scores['task_success'])
        self.assertEqual(3, scores['complete_correct_supported']['denominator'])
        self.assertEqual(.5, scores['answerable_point_coverage']['rate'])
        self.assertEqual(0, scores['hard_failure']['numerator'])

    def test_multiple_hard_failures_count_one_task_and_unknown_categories_fail(self):
        params = dict(outcome={'state': True}, behavior={'allowed': True}, response={'protocol': True}, answerable=True,
            quality={'facts_correct': True, 'evidence_supported': True, 'complete': True, 'covered_points': 1, 'required_points': 1})
        row = three_layer_result(**params, hard_failure_reasons=['duplicate_write', 'unauthorized_tool'])
        self.assertFalse(row['task_success'])
        self.assertEqual(1, summarize_three_layers([row])['hard_failure']['numerator'])
        with self.assertRaises(ValueError):
            three_layer_result(**params, hard_failure_reasons=['low_answer_quality'])
