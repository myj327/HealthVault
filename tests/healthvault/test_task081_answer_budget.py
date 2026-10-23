import json
import unittest
from unittest.mock import Mock

from healthvault.web.answer_validation import MAX_CITED_CLAIMS, MAX_VERIFIED_CLAUSES, verify_answer
from healthvault.web.answers import answer_with_model
from healthvault.web.model_runtime import ModelServiceError


class AnswerBudgetTests(unittest.TestCase):
    def data(self, count, text='尺寸（长径 8 mm，短径 6 mm）。'):
        claims = [{'text': text, 'citation_id': 'REC:report:size'} for _ in range(count)]
        answer = '\n\n'.join(item['text'] + ' [' + item['citation_id'] + ']' for item in claims)
        sources = [{'citation_id': 'REC:report:size', 'title': '原始尺寸', 'detail': text}]
        return claims, answer, sources

    def checker(self, _prompt, data, **_kwargs):
        self.checked_count = len(data['facts'])
        return {'protocol': 'fact_span_v1', 'answer_supported': True, 'answers_question': True,
                'claims': [{'claim_id': item['claim_id'], 'evidence_ids': ['S1-detail-1'], 'supported': True}
                           for item in data['facts']]}

    def generate(self, claims):
        return answer_with_model('这些尺寸是多少？', 'visit', {},
                                 query=lambda *_a, **_k: json.dumps({'status': 'answered', 'claims': claims}, ensure_ascii=False))

    def test_original_sixteen_claims_keep_all_thirty_two_clauses(self):
        claims, answer, sources = self.data(16)
        self.assertEqual(answer, self.generate(claims))
        result = verify_answer('核对尺寸', answer, sources, query=self.checker)
        self.assertTrue(result['passed'], result)
        self.assertEqual(32, self.checked_count)
        self.assertEqual(answer, result['answer'])

    def test_generation_and_verification_accept_same_boundary(self):
        claims, answer, sources = self.data(MAX_CITED_CLAIMS)
        self.assertEqual(answer, self.generate(claims))
        result = verify_answer('核对全部尺寸', answer, sources, query=self.checker)
        self.assertTrue(result['passed'], result)
        self.assertEqual(MAX_VERIFIED_CLAUSES, self.checked_count)

    def test_cited_claim_overflow_is_explicit_and_skips_verifier(self):
        claims, answer, sources = self.data(MAX_CITED_CLAIMS + 1, '尺寸为 8 mm。')
        with self.assertRaisesRegex(ModelServiceError, '超过 30 条引用事实'):
            self.generate(claims)
        query = Mock(side_effect=AssertionError('must not call a model'))
        result = verify_answer('核对尺寸', answer, sources, query=query)
        self.assertFalse(result['passed'])
        self.assertEqual('answer_budget_exceeded', result['kind'])
        self.assertIn('30 条', result['message'])
        query.assert_not_called()

    def test_clause_overflow_keeps_count_and_skips_verifier(self):
        claims, answer, sources = self.data(1, '尺寸为 8 mm；' * (MAX_VERIFIED_CLAUSES + 1))
        with self.assertRaisesRegex(ModelServiceError, '超过 60 个待核对分句'):
            self.generate(claims)
        query = Mock()
        result = verify_answer('核对尺寸', answer, sources, query=query)
        self.assertEqual(MAX_VERIFIED_CLAUSES + 1, result['claim_count'])
        self.assertFalse(result['passed'])
        self.assertIsNone(result['answer'])
        query.assert_not_called()

    def test_missing_clause_in_long_answer_is_not_accepted(self):
        _, answer, sources = self.data(16)
        def missing(prompt, data, **kwargs):
            verdict = self.checker(prompt, data, **kwargs)
            verdict['claims'].pop()
            return verdict
        result = verify_answer('核对尺寸', answer, sources, query=missing)
        self.assertFalse(result['passed'])
        self.assertFalse(result['coverage_passed'])

    def test_uncited_tail_fails_before_model_request(self):
        _, answer, sources = self.data(1)
        query = Mock()
        result = verify_answer('核对尺寸', answer + '\n另有右侧结节。', sources, query=query)
        self.assertFalse(result['passed'])
        self.assertEqual('invalid_cited_answer', result['kind'])
        query.assert_not_called()

    def test_legacy_verifier_checks_all_thirty_two_clauses(self):
        _, answer, sources = self.data(16)
        def legacy(_prompt, data, **_kwargs):
            return {'answer_supported': True, 'answers_question': True,
                    'claims': [{'text': item['text'], 'citation_id': item['citation_id'],
                                'quote': sources[0]['detail'], 'supported': True} for item in data['facts']]}
        result = verify_answer('核对尺寸', answer, sources, query=legacy)
        self.assertTrue(result['passed'], result)
        self.assertEqual(32, result['claim_count'])


if __name__ == '__main__':
    unittest.main()
