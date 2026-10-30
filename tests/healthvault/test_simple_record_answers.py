import unittest

from healthvault.application.query_health_timeline import query_health_timeline
from healthvault.web.answer_validation import verify_grounded_answer
from tests.healthvault.test_compare_health_indicators import RecordingRepository, make_indicator, make_report


class SimpleRecordAnswerTests(unittest.TestCase):
    def test_indicator_filter_keeps_only_matching_original_records_and_reports_missing(self):
        first = make_report('first', '2022-01-01', [make_indicator('hba1c', 5.7, unit='%')])
        last = make_report('last', '2024-01-01', [make_indicator('hba1c', 6.1, unit='%'),
                                               make_indicator('fasting_glucose', 5.2)])
        repo = RecordingRepository([first, last])
        result = query_health_timeline({'profile_id': 'alice', 'indicator_code': '空腹血糖'}, repo).to_dict()
        self.assertEqual('ok', result['status'])
        self.assertEqual(['REC:last:fasting_glucose'], result['available_citation_ids'])
        self.assertEqual(1, result['indicator_count'])
        self.assertEqual('fasting_glucose', result['indicator_code'])
        missing = query_health_timeline({'profile_id': 'alice', 'indicator_code': 'uric_acid'}, repo).to_dict()
        self.assertEqual('empty', missing['status'])
        self.assertEqual([], missing['timeline'])
        self.assertEqual([], missing['available_citation_ids'])
        self.assertEqual(['alice', 'alice'], repo.queries)
        self.assertEqual(2, len(last.indicators))

    def test_unknown_or_malformed_filter_is_not_misreported_as_missing_data(self):
        repo = RecordingRepository([])
        for code in ('not_a_supported_indicator', '', [], 42):
            self.assertEqual('invalid_request', query_health_timeline(
                {'profile_id': 'alice', 'indicator_code': code}, repo).status)
        self.assertEqual([], repo.queries)

    def test_record_correction_preserves_the_question_and_checks_citation_identity(self):
        question = '用2024年的糖化记录拟一句问医生的话，只引用一个原始来源。'
        sources = [{'citation_id': 'REC:last:hba1c', 'kind': 'record',
                    'detail': '2024-01-01 糖化血红蛋白 6.1 %'}]
        draft = '2024-01-01 糖化血红蛋白 6.1%。[REC:last:hba1c]'
        corrected = '医生，我2024-01-01的糖化血红蛋白为6.1%，这个结果怎样理解？[REC:last:hba1c]'
        calls = []
        def check(prompt, data, **kwargs):
            calls.append(data)
            return {'supported': True, 'complete': True, 'answer': corrected}
        result = verify_grounded_answer(question, draft, sources, query=check)
        self.assertTrue(result['passed'])
        self.assertEqual(corrected, result['answer'])
        self.assertEqual('whole_answer_record_check', result['kind'])
        self.assertEqual([{'question': question, 'answer': draft, 'sources': sources}], calls)
        for reply in ({'supported': False, 'complete': True},
                      {'supported': True, 'complete': False},
                      {'supported': True, 'complete': True, 'answer': corrected.replace('REC:last', 'REC:invented')}):
            self.assertFalse(verify_grounded_answer(question, draft, sources,
                query=lambda *args, **kwargs: reply)['passed'])
