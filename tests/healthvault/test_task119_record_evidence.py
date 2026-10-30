import copy
import json
import unittest

from healthvault.application.tool_evidence import record_evidence, tool_evidence
from healthvault.web.answer_coverage import SEMANTIC_DIMENSIONS
from healthvault.web.answer_validation import verify_answer
from healthvault.web.answers import visit_evidence
from tests.healthvault import test_task114_entrypoint as entry


QUESTION = '核对报告 R-101 的 HbA1c 值及参考范围。'
CITATION = 'REC:R-101:hba1c'
EVIDENCE = {'quote': 'result|hba1c|5.1|%|4.0|6.0', 'source_id': 'synthetic',
            'line_start': 4, 'line_end': 4}


def source():
    return record_evidence(CITATION, '2048-03-14', 'hba1c', EVIDENCE, 'R-101')


def semantic_yes(data):
    claims = [{'claim_id': fact['claim_id'], 'supported': True,
               'checks': dict.fromkeys(SEMANTIC_DIMENSIONS, True),
               'evidence_ids': [s['evidence_id'] for s in data['evidence_spans']
                                if s['citation_id'] == fact['citation_id']]}
              for fact in data['facts']]
    return {'protocol': 'fact_span_v2', 'answer_supported': True, 'answers_question': True,
            'claims': claims, 'question_coverage': [{**point, 'status': 'covered',
                'evidence_ids': list(dict.fromkeys(i for c in claims for i in c['evidence_ids'])),
                'answer_claim_ids': [c['claim_id'] for c in claims]} for point in data['required_points']]}


class RecordEvidenceTests(unittest.TestCase):
    def verify(self, answer, *, sources=None, judge=semantic_yes):
        return verify_answer(QUESTION, answer, sources or [source()], audit_coverage=True,
                             required_points=[{'point_id': 'P1', 'question_quote': QUESTION, 'description': QUESTION}],
                             query=lambda _prompt, data, **_kwargs: judge(data))

    def test_same_original_record_for_timeline_comparison_brief_and_dedicated_entry(self):
        point = {'report_id': 'R-101', 'report_date': '2048-03-14', 'evidence': copy.deepcopy(EVIDENCE)}
        timeline = {'status': 'ok', 'available_citation_ids': [CITATION], 'timeline': [
            {**point, 'indicators': [{'code': 'hba1c', 'evidence': copy.deepcopy(EVIDENCE)}]}]}
        comparison = {'status': 'ok', 'available_citation_ids': [CITATION], 'comparisons': [
            {'code': 'hba1c', 'earliest': point, 'latest': point}]}
        brief = {'status': 'ready', 'available_citation_ids': [CITATION], 'record_citations': [
            {**point, 'citation_id': CITATION, 'indicator_code': 'hba1c'}], 'indicators': []}
        before = copy.deepcopy((timeline, comparison, brief))
        for name, payload in (('query_health_timeline', timeline), ('compare_health_indicators', comparison),
                              ('generate_visit_brief', brief)):
            self.assertEqual(source(), tool_evidence(name, payload)[0])
        self.assertEqual([source()], visit_evidence(brief)[1])
        self.assertEqual(before, (timeline, comparison, brief))
        self.assertEqual(EVIDENCE['quote'], source()['original_quote'])
        self.assertIn(EVIDENCE['quote'], source()['detail'])
        self.assertEqual({k: v for k, v in EVIDENCE.items() if k != 'quote'}, source()['evidence_locator'])

    def test_own_report_identifier_percentage_and_range_are_supported(self):
        answer = f'报告 R-101 中记录 HbA1c 为5.1%。 [{CITATION}]\n参考范围为4.0–6.0%。 [{CITATION}]'
        self.assertTrue(self.verify(answer)['passed'])

    def test_invented_value_unit_and_report_id_still_fail_even_if_model_accepts(self):
        for text in ('HbA1c为5.9%。', 'HbA1c为5.1 mmol/L。', '该报告编号为R-999。'):
            with self.subTest(text=text):
                self.assertFalse(self.verify(f'{text} [{CITATION}]')['support_passed'])

    def test_cross_source_and_title_only_cannot_replace_own_body_evidence(self):
        sources = [source(), {**source(), 'citation_id': 'REC:other:hba1c'}]
        for prefix in ('S2-', 'S1-title-'):
            def wrong(data):
                result = semantic_yes(data)
                result['claims'][0]['evidence_ids'] = [s['evidence_id'] for s in data['evidence_spans']
                                                       if s['evidence_id'].startswith(prefix)]
                return result
            with self.subTest(prefix=prefix):
                self.assertFalse(self.verify(f'HbA1c为5.1%。 [{CITATION}]', sources=sources, judge=wrong)['support_passed'])

    def test_free_text_is_unchanged_and_nonpercentage_values_are_not_reinterpreted(self):
        plain = record_evidence(CITATION, '2048-03-14', 'hba1c', {'quote': 'HbA1c 5.1%'}, 'R-101')
        self.assertEqual('HbA1c 5.1%', plain['original_quote'])
        self.assertIn('HbA1c 5.1%', plain['detail'])
        numeric = record_evidence('REC:R-101:glucose', '2048-03-14', 'glucose',
                                  {'quote': 'result|glucose|5.1|mmol/L|3.9|6.1'}, 'R-101')
        self.assertTrue(self.verify('记录值为5.1 mmol/L。 [REC:R-101:glucose]', sources=[numeric])['passed'])
        for text in ('记录值为5.1%。', '记录值为5 mmol/L。'):
            self.assertFalse(self.verify(text + ' [REC:R-101:glucose]', sources=[numeric])['support_passed'])

    def test_record_identity_date_and_value_are_supported_by_body_without_title(self):
        def body_only(data):
            result = semantic_yes(data)
            body_ids = {s['evidence_id'] for s in data['evidence_spans'] if s['source_field'] == 'detail'}
            for item in result['claims'] + result['question_coverage']:
                item['evidence_ids'] = [i for i in item['evidence_ids'] if i in body_ids]
            return result
        answer = (f'报告 R-101 已存在。 [{CITATION}]\n'
                  f'报告日期为2048-03-14。 [{CITATION}]\n'
                  f'HbA1c为5.1%。 [{CITATION}]')
        self.assertTrue(self.verify(answer, judge=body_only)['passed'])
        for text in ('报告日期为2049-03-14。', '报告编号为R-999。', 'HbA1c为5.9%。'):
            self.assertFalse(self.verify(f'{text} [{CITATION}]', judge=body_only)['support_passed'])

    def test_record_body_spans_keep_exact_rendered_offsets_and_original_locator(self):
        from healthvault.web.answer_validation import _spans
        original = {**EVIDENCE, 'quote': 'HbA1c 5.1%\n  原始备注保留  ', 'line_end': 5}
        card = record_evidence(CITATION, '2048-03-14', 'hba1c', original, 'R-101')
        self.assertEqual(original['quote'], card['original_quote'])
        self.assertIn(original['quote'], card['detail'])
        self.assertEqual({'source_id': 'synthetic', 'line_start': 4, 'line_end': 5}, card['evidence_locator'])
        for span in _spans([card]):
            self.assertEqual(span['quote'], card[span['source_field']][span['start']:span['end']])

    def test_knowledge_source_is_not_rendered_as_a_record(self):
        item = {'citation_id': 'HVK:example', 'title': '原始主题', 'excerpt': '原始知识正文。',
                'source': {'title': '原始主题', 'url': 'https://example.org/source'}}
        cards = tool_evidence('search_health_knowledge', {'status': 'ok', 'results': [item],
                              'available_citation_ids': ['HVK:example']})
        self.assertEqual([{'kind': 'knowledge', 'citation_id': 'HVK:example', 'title': '原始主题',
                           'detail': '原始知识正文。', 'url': 'https://example.org/source'}], cards)


class RecordEntryTests(unittest.TestCase):
    setUp = entry.EntryPointTests.setUp
    collect = entry.EntryPointTests.collect

    def test_actual_http_record_percentage_passes_full_verification_without_repair(self):
        answer = f'报告 R-101 中记录 HbA1c 为5.1%。 [{CITATION}]\n参考范围为4.0–6.0%。 [{CITATION}]'
        result = {'status': 'ok', 'available_citation_ids': [CITATION], 'timeline': [
            {'report_id': 'R-101', 'report_date': '2048-03-14',
             'indicators': [{'code': 'hba1c', 'evidence': EVIDENCE}]}]}
        def verify(payload):
            data = json.loads(payload['messages'][-1]['content'])
            return entry.response(json.dumps(semantic_yes(data), ensure_ascii=False))
        row, calls, _ = self.collect({'case_id': 'record-fields', 'profile_id': 'r5-orchid', 'prompt': QUESTION},
            [entry.tool('query_health_timeline', {}), entry.response(answer), verify], result=result)
        self.assertEqual(200, row['http_status'])
        self.assertEqual(['agent', 'agent', 'verification'], [c['phase'] for c in calls.rows])
        self.assertEqual(0, row['response']['measurement']['answer_repair_attempts'])
        self.assertEqual(1, len(row['tool_calls']))


if __name__ == '__main__':
    unittest.main()
