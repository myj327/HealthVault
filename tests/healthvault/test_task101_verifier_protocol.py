import copy
import json
from pathlib import Path
import unittest

from healthvault.web.answer_coverage import SEMANTIC_DIMENSIONS
from healthvault.web.answer_validation import prepare_verification, verify_answer


class VerifierProtocolTests(unittest.TestCase):
    def setUp(self):
        self.question = 'A1C有什么用途？'
        self.sources = [{'citation_id': 'HVK:test:a', 'title': 'A1C', 'detail': 'A1C helps monitoring.'}]
        self.answer = 'A1C可用于监测。 [HVK:test:a]'
        self.points = [{'point_id': 'P1', 'question_quote': '用途', 'description': '监测用途'}]

    def valid(self):
        return {'protocol': 'fact_span_v2', 'answer_supported': True, 'answers_question': True,
                'claims': [{'claim_id': 'F1', 'evidence_ids': ['S1-detail-1'], 'supported': True,
                            'checks': dict.fromkeys(SEMANTIC_DIMENSIONS, True)}],
                'question_coverage': [{**self.points[0], 'status': 'covered',
                                       'evidence_ids': ['S1-detail-1'], 'answer_claim_ids': ['F1']}]}

    def check(self, result, **kwargs):
        return verify_answer(self.question, self.answer, self.sources,
                             query=lambda *_a, **_k: result, audit_coverage=True,
                             required_points=self.points, **kwargs)

    def test_audited_prompt_has_one_version_and_one_consistent_claim_schema(self):
        prompt, payload = prepare_verification(self.question, self.answer, self.sources,
                                               audit_coverage=True, required_points=self.points)
        self.assertEqual(1, prompt.count('fact_span_v2'))
        self.assertNotIn('fact_span_v1', prompt)
        self.assertIn('supported(boolean),checks(', prompt)
        self.assertNotIn('本次协议改为', prompt)
        self.assertIn('必须每个事实编号恰好出现一次', prompt)
        self.assertIn('不得选择列表外的片段', prompt)
        self.assertIn('原样保留 point_id、question_quote、description', prompt)
        self.assertEqual(self.points, payload['required_points'])
        self.assertEqual(self.sources, payload['sources'])

    def test_original_v1_outputs_still_fail_online_without_version_rewriting(self):
        fixture = Path(__file__).with_name('fixtures') / 'task101_protocol_original.json'
        for original in json.loads(fixture.read_text(encoding='utf-8'))['cases']:
            self.assertEqual('fact_span_v1', json.loads(original['raw_response'])['protocol'])
            seen = []
            def query(prompt, _payload, **_kwargs):
                seen.append(prompt)
                self.assertNotIn('fact_span_v1', prompt)
                return original['raw_response']
            result = verify_answer(original['question'], original['answer'], original['sources'], query=query,
                                   audit_coverage=True, required_points=original['required_points'])
            self.assertFalse(result['passed'])
            self.assertEqual('invalid_verifier_output', result['kind'])
            self.assertEqual(1, len(seen))

    def test_valid_v2_and_explicit_legacy_v1_both_keep_their_own_contract(self):
        self.assertTrue(self.check(self.valid())['passed'])
        legacy = self.valid()
        legacy['protocol'] = 'fact_span_v1'
        legacy.pop('question_coverage')
        legacy['claims'][0].pop('checks')
        def query(prompt, payload, **_kwargs):
            self.assertEqual(1, prompt.count('fact_span_v1'))
            self.assertNotIn('fact_span_v2', prompt)
            self.assertNotIn('checks(', prompt)
            self.assertNotIn('question_coverage', prompt)
            self.assertNotIn('required_points', payload)
            return legacy
        result = verify_answer(self.question, self.answer, self.sources, query=query)
        self.assertTrue(result['passed'])

    def test_format_missing_facts_and_cross_citation_stay_rejected(self):
        for raw in ('{', json.dumps(self.valid()) + '说明'):
            self.assertFalse(self.check(raw)['passed'])
        self.sources.append({'citation_id': 'HVK:test:b', 'title': 'Other', 'detail': 'A1C helps monitoring.'})
        crossed = self.valid()
        crossed['claims'][0]['evidence_ids'] = ['S2-detail-1']
        self.assertFalse(self.check(crossed)['passed'])
        incomplete = copy.deepcopy(self.valid())
        incomplete['claims'][0]['checks'].pop('conditions')
        self.assertFalse(self.check(incomplete)['passed'])
        self.answer += '\nA1C监测有帮助。 [HVK:test:a]'
        self.assertFalse(self.check(self.valid())['passed'])


if __name__ == '__main__':
    unittest.main()
