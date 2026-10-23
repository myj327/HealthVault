import json
from pathlib import Path
import unittest

from healthvault.application.workflow_contract import completion_feedback


def final(status='ok', citations=('REC:test:a',), refusal=False):
    return json.dumps({'final_status': status, 'citations': list(citations), 'refusal': refusal})


def call(name='query_health_timeline', status='ok', citations=('REC:test:a',)):
    return {'name': name, 'result_status': status, 'available_citation_ids': list(citations)}


class RefusalCompletionTests(unittest.TestCase):
    def test_original_refusals_can_finish_without_rewriting_old_scores(self):
        path = Path(__file__).with_name('fixtures') / 'task099_refusal_original.json'
        for original in json.loads(path.read_text(encoding='utf-8'))['cases']:
            with self.subTest(case=original['case_id']):
                self.assertEqual('rejected', json.loads(original['raw_final'])['final_status'])
                self.assertIsNone(completion_feedback(original['raw_final'], original['calls'], original['intent']))

    def test_success_still_requires_actual_evidence_and_comparison_allows_two(self):
        self.assertIsNone(completion_feedback(final(), [call()], 'record_timeline'))
        comparison = call('compare_health_indicators', citations=('REC:test:a', 'REC:test:b'))
        self.assertIsNone(completion_feedback(final(citations=comparison['available_citation_ids']),
                                             [comparison], 'indicator_comparison'))
        for text in (final(citations=()), final(refusal=True), final('insufficient_evidence', (), True)):
            self.assertIsNotNone(completion_feedback(text, [call()], 'record_timeline'))
        self.assertIsNotNone(completion_feedback(final(citations=()), [call(citations=())], 'record_timeline'))
        self.assertIsNotNone(completion_feedback(final(), [], 'general_read_only_workflow'))
        self.assertIsNotNone(completion_feedback(final(citations=()), [], 'general_read_only_workflow'))
        self.assertIsNone(completion_feedback(final('rejected', (), True), [], 'general_read_only_workflow'))
        self.assertIsNone(completion_feedback('请通过报告确认页面保存。', [], 'general_read_only_workflow'))

    def test_refusal_cannot_bypass_required_lookup_but_terminal_safety_can_stop(self):
        for intent, necessary in (('record_timeline', 'query_health_timeline'),
                                   ('indicator_comparison', 'compare_health_indicators'),
                                   ('visit_preparation', 'generate_visit_brief')):
            text = final('rejected', (), True)
            self.assertIn(necessary, completion_feedback(text, [], intent))
            self.assertIsNone(completion_feedback(text, [call(necessary)], intent))
        rejected = [call('generate_visit_brief', 'rejected', ())]
        self.assertIsNone(completion_feedback(final('rejected', (), True), rejected,
                                             'visit_preparation', terminal=True))
        self.assertIsNotNone(completion_feedback(final(), rejected, 'visit_preparation', terminal=True))

    def test_empty_lookup_allows_safe_refusal_but_not_positive_answer(self):
        calls = [call('search_health_knowledge', 'empty', ())]
        for status in ('rejected', 'insufficient_evidence'):
            self.assertIsNone(completion_feedback(final(status, (), True), calls, 'knowledge_lookup'))
            self.assertIsNotNone(completion_feedback(final(status, (), False), calls, 'knowledge_lookup'))
        self.assertIsNotNone(completion_feedback(final(citations=()), calls, 'knowledge_lookup'))

    def test_invalid_or_excess_citations_cannot_be_normalized_into_success(self):
        calls = [call(citations=('REC:test:a', 'REC:test:b'))]
        for text in (final(citations=('REC:test:invented',)), final(citations=('REC:test:a', 'REC:test:b')),
                     final(citations=('REC:test:a', 'REC:test:a')), final('rejected', ('REC:test:a',), True)):
            with self.subTest(text=text):
                self.assertIsNotNone(completion_feedback(text, calls, 'record_timeline'))
        self.assertIsNone(completion_feedback(final('rejected', (), True), calls, 'record_timeline'))

    def test_raw_json_format_and_schema_are_checked_before_completion(self):
        for text in ('{', final() + '说明', '```json\n' + final() + '\n```', '[]',
                     '{"final_status":"rejected","citations":[],"refusal":"true"}',
                     '{"final_status":"rejected","citations":[],"refusal":true,"extra":1}',
                     '{"final_status":"unknown","citations":[],"refusal":true}'):
            with self.subTest(text=text):
                self.assertIsNotNone(completion_feedback(text, [call()], 'record_timeline'))
                self.assertIsNotNone(completion_feedback(text, [], 'general_read_only_workflow'))


if __name__ == '__main__':
    unittest.main()
