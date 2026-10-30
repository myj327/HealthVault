import copy
import json
import unittest
from types import SimpleNamespace

from healthvault.application.tool_evidence import tool_evidence
from healthvault.domain.comparisons import IndicatorComparisonResult
from healthvault.web.answers import answer_with_model, visit_evidence
from healthvault.web.answer_validation import verify_answer
from tests.healthvault.test_task119_record_evidence import semantic_yes


def payloads():
    first = {'report_id': 'DEV-A', 'report_date': '2048-03-14', 'value': 6.1,
             'evidence': {'quote': 'result|hba1c|6.1|%|4.0|6.0', 'line_start': 4, 'line_end': 4}}
    last = {'report_id': 'DEV-B', 'report_date': '2049-03-14', 'value': 5.5,
            'evidence': {'quote': 'result|hba1c|5.5|%|4.0|6.0', 'line_start': 4, 'line_end': 4}}
    item = {'code': 'hba1c', 'display_name': '糖化血红蛋白', 'unit': '%', 'earliest': first, 'latest': last,
            'absolute_change': -0.6, 'direction': 'decreased'}
    ids = ['REC:DEV-A:hba1c', 'REC:DEV-B:hba1c', 'REC:derived:DEV-B:hba1c']
    compare = {'status': 'ok', 'comparisons': [item], 'available_citation_ids': ids}
    brief = {'status': 'ready', 'record_citations': [
        {**point, 'citation_id': identifier, 'indicator_code': 'hba1c'} for point, identifier in zip((first, last), ids)],
        'indicators': [{**item, 'comparison': {'earliest_value': 6.1, 'latest_value': 5.5, 'unit': '%',
            'absolute_change': -0.6, 'direction': 'decreased', 'earliest_record_citation_id': ids[0],
            'latest_record_citation_id': ids[1]}}], 'available_citation_ids': ids}
    return compare, brief


class ComparisonDeliveryTests(unittest.TestCase):
    def test_both_entrypoints_keep_same_comparison_and_original_reports(self):
        compare, brief = payloads()
        before = copy.deepcopy((compare, brief))
        ordinary = tool_evidence('compare_health_indicators', compare)[-1]
        self.assertEqual(ordinary, tool_evidence('generate_visit_brief', brief)[-1])
        self.assertEqual(ordinary, visit_evidence(brief)[1][-1])
        self.assertEqual(['REC:DEV-A:hba1c', 'REC:DEV-B:hba1c'], ordinary['derived_from'])
        self.assertEqual(['DEV-A', 'DEV-B'], [link['report_id'] for link in ordinary['report_links']])
        self.assertEqual(compare['comparisons'][0]['earliest']['evidence'], ordinary['report_links'][0]['evidence'])
        self.assertIn('降低', ordinary['detail'])
        self.assertIn('-0.6 个百分点', ordinary['detail'])
        self.assertEqual(before, (compare, brief))

    def test_tool_advertises_derived_id_without_replacing_original_endpoints(self):
        compare, _ = payloads()
        item = SimpleNamespace(code='hba1c', earliest=SimpleNamespace(report_id='DEV-A'),
                               latest=SimpleNamespace(report_id='DEV-B'), to_dict=lambda: compare['comparisons'][0])
        result = IndicatorComparisonResult('ok', 2, (item,), ()).to_dict()
        self.assertEqual(compare['available_citation_ids'], result['available_citation_ids'])
        self.assertEqual(compare['available_citation_ids'][:2], result['recommended_citation_ids'])

    def test_correct_difference_is_supported_but_invented_number_or_unit_fails(self):
        sources = tool_evidence('compare_health_indicators', payloads()[0])
        question = '比较两次结果的差值和方向。'
        def check(text):
            return verify_answer(question, text + ' [REC:derived:DEV-B:hba1c]', sources, audit_coverage=True,
                required_points=[{'point_id': 'P1', 'question_quote': question, 'description': question}],
                query=lambda _prompt, data, **_kwargs: semantic_yes(data))
        self.assertTrue(check('糖化血红蛋白降低了0.6个百分点。')['passed'])
        self.assertFalse(check('糖化血红蛋白降低了0.9个百分点。')['passed'])
        self.assertFalse(check('糖化血红蛋白降低了0.6 mmol/L。')['passed'])
        self.assertFalse(check('糖化血红蛋白降低了0.6个百分点。').get('invalid_citations'))

    def test_comparison_is_not_created_for_a_tool_without_authorized_derived_source(self):
        compare, _ = payloads()
        compare['available_citation_ids'].pop()
        self.assertEqual(2, len(tool_evidence('compare_health_indicators', compare)))

    def test_once_repair_receives_purpose_original_sources_and_strict_constraints(self):
        evidence, sources = visit_evidence(payloads()[1])
        evidence['answer_repair'] = {'previous_answer': '5.5%', 'reason': 'answer_or_coverage_binding'}
        calls = []
        def query(prompt, data, **_kwargs):
            calls.append((prompt, data))
            return json.dumps({'status': 'answered', 'claims': [{'text': '就诊时想请医生解释糖化血红蛋白降低0.6个百分点的变化。',
                               'citation_id': sources[-1]['citation_id']}]}, ensure_ascii=False)
        answer = answer_with_model('用一句话整理就诊时想讨论的变化。', 'visit', evidence, query=query)
        self.assertEqual(1, len(calls))
        self.assertIn('就诊时', answer)
        self.assertEqual(sources, calls[0][1]['evidence']['sources'])
        self.assertIn('差值及方向', calls[0][0])
        self.assertIn('不能新增诊断、检查或用药建议', calls[0][0])


if __name__ == '__main__':
    unittest.main()
