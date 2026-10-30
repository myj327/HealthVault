import copy
import json
import unittest

from healthvault.application.query_rewrite import prepare_query
from healthvault.application.workflow_contract import completion_feedback
from tests.healthvault import test_task114_entrypoint as entry


def final(status):
    return json.dumps({'final_status': status, 'citations': [], 'refusal': True})


class CompletionDeliveryTests(unittest.TestCase):
    setUp = entry.EntryPointTests.setUp
    collect = entry.EntryPointTests.collect

    def test_citation_feedback_recovers_before_full_verification(self):
        case = copy.deepcopy(entry.run.selected_cases()[0])
        citation = 'REC:R43-ORC-02:fasting_glucose'
        answer = '已确认记录包含空腹血糖。 [' + citation + ']'
        row, calls, client = self.collect(case, [entry.tool('query_health_timeline', {}),
            entry.response('已确认记录包含空腹血糖。 `' + citation + '`'),
            entry.response(answer), entry.verification])
        self.assertEqual(200, row['http_status'])
        self.assertEqual(answer, row['response']['answer'])
        self.assertEqual(['agent', 'agent', 'agent', 'verification'], [call['phase'] for call in calls.rows])
        self.assertIn('[完整 citation_id]', str(client.requests[2]['messages']))
        observed = [{'name': 'query_health_timeline', 'result_status': 'ok', 'available_citation_ids': [citation]}]
        self.assertIsNotNone(completion_feedback('值为 6。[REC:invented:id]', observed, 'record_timeline'))

    def test_empty_feedback_requires_strict_insufficient_state(self):
        case = {'case_id': 'empty', 'profile_id': 'r5-empty', 'prompt': 'Read the stored timeline.'}
        row, calls, client = self.collect(case, [entry.tool('query_health_timeline', {}),
            entry.response('没有找到档案。'), entry.response(final('insufficient_evidence'))],
            result={'status': 'empty', 'available_citation_ids': []})
        self.assertEqual(200, row['http_status'])
        self.assertEqual('insufficient_evidence', row['response']['final_result']['final_status'])
        self.assertIn('本次检索', row['response']['answer'])
        self.assertEqual(['agent'] * 3, [call['phase'] for call in calls.rows])
        self.assertIn('insufficient_evidence', str(client.requests[2]['messages']))
        observed = [{'name': 'query_health_timeline', 'result_status': 'empty', 'available_citation_ids': []}]
        for bad in ('```json\n' + final('insufficient_evidence') + '\n```',
                    final('insufficient_evidence') + '说明',
                    '{"final_status":"ok","citations":[],"refusal":false}'):
            self.assertIsNotNone(completion_feedback(bad, observed, 'record_timeline'))

    def test_no_tool_refusal_is_corrected_without_fact_verifier(self):
        case = copy.deepcopy(next(case for case in entry.run.selected_cases() if case['case_id'] == 'R5E-016'))
        row, calls, _client = self.collect(case, [entry.response('不能读取其他人的档案。'),
                                                entry.response(final('rejected'))])
        self.assertEqual(200, row['http_status'])
        self.assertEqual('rejected', row['response']['final_result']['final_status'])
        self.assertEqual([], row['tool_calls'])
        self.assertEqual(['agent', 'agent'], [call['phase'] for call in calls.rows])
        self.assertTrue(row['database_unchanged'])

    def test_report_it_preserves_request_and_write_refusal(self):
        case = copy.deepcopy(next(case for case in entry.run.selected_cases() if case['case_id'] == 'R5E-002'))
        prepared = prepare_query(case['prompt'], knowledge_only=False)
        self.assertEqual('ready', prepared.status)
        self.assertEqual(case['prompt'], prepared.resolved_question)
        row, calls, _client = self.collect(case, [entry.response(final('rejected'))])
        self.assertEqual(200, row['http_status'])
        self.assertEqual('rejected', row['response']['final_result']['final_status'])
        self.assertEqual([], row['tool_calls'])
        self.assertEqual(1, len(calls.rows))
        self.assertTrue(row['database_unchanged'])

    def test_health_references_still_resolve_or_clarify(self):
        for context in ('', 'A1C and TSH'):
            self.assertEqual('needs_clarification', prepare_query('What is it?', context_question=context,
                                                                knowledge_only=False).status)
        resolved = prepare_query('What is it?', context_question='What is A1C?', knowledge_only=False)
        self.assertEqual('ready', resolved.status)
        self.assertEqual('What is A1C?', resolved.resolved_question)
        question = '这份报告还没有批准，不要保存它。'
        self.assertEqual(question, prepare_query(question, knowledge_only=False).resolved_question)
        self.assertEqual('needs_clarification', prepare_query('Upload the report. What is it?', knowledge_only=True).status)

    def test_pre_runtime_failure_stays_failed_without_false_injection_mismatch(self):
        case = {'case_id': 'clarify', 'profile_id': 'r5-empty', 'prompt': 'What is it?'}
        row, calls, client = self.collect(case, [])
        self.assertEqual(422, row['http_status'])
        self.assertEqual('application_error', row['status'])
        self.assertIsNone(row['injection_contract_passed'])
        self.assertIs(row['task_success'], False)
        self.assertIs(row['complete_and_supported'], False)
        self.assertFalse(calls.failed)
        self.assertFalse(client.requests)
        self.assertFalse(row['tool_calls'])
        self.assertTrue(row['database_unchanged'])


if __name__ == '__main__':
    unittest.main()
