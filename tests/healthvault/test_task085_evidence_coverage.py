import json
from pathlib import Path
import unittest

from healthvault.web.answer_validation import verify_answer, prepare_verification, valid_verifier_output
from healthvault.web.answer_coverage import SEMANTIC_DIMENSIONS


class EvidenceCoverageTests(unittest.TestCase):
    def setUp(self):
        self.question = 'A1C有什么用途和时间范围？'
        self.sources = [{'citation_id': 'HVK:test:a', 'title': 'A1C',
                         'detail': 'A1C is used for monitoring.\nA1C reflects the past three months.'}]
        self.answer = 'A1C可用于监测。 [HVK:test:a]'

    def review(self, *, status='covered', point_span='S1-detail-1'):
        return {'protocol': 'fact_span_v2', 'answer_supported': True, 'answers_question': status == 'covered',
                'claims': [{'claim_id': 'F1', 'evidence_ids': ['S1-detail-1'], 'supported': True,
                            'checks': dict.fromkeys(SEMANTIC_DIMENSIONS, True)}],
                'question_coverage': [{'point_id': 'P1', 'question_quote': 'A1C有什么用途和时间范围',
                                      'description': '监测用途', 'status': status,
                                      'evidence_ids': [] if status == 'missing_evidence' else [point_span],
                                      'answer_claim_ids': ['F1'] if status == 'covered' else []}]}

    def verify(self, result, **kwargs):
        return verify_answer(self.question, self.answer, self.sources, query=lambda *_a, **_k: json.dumps(result),
                             audit_coverage=True, **kwargs)

    def test_original_mq001_response_is_rejected_by_own_source_qualifier(self):
        case = json.loads((Path(__file__).parent/'fixtures/task085_mq001_original.json').read_text(encoding='utf-8'))
        result = verify_answer(case['question'], case['answer'], case['sources'], query=lambda *_a, **_k: case['verifier_raw'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['support_passed'])
        self.assertTrue(result['coverage_passed'])
        self.assertNotIn('F1', [item['claim_id'] for item in result['checked_claims']])

    def test_legal_cross_language_qualifier_and_complete_report_pass(self):
        self.answer = '左叶结节通过血液检测评估。 [HVK:test:a]'
        self.sources[0]['detail'] = 'The left lobe nodule was assessed with a blood test.'
        self.assertTrue(self.verify(self.review())['passed'])

    def test_other_citation_cannot_supply_missing_sample_or_side(self):
        self.answer = '右侧结节通过尿液检测评估。 [HVK:test:a]'
        self.sources[0]['detail'] = 'The left lobe nodule was assessed with a blood test.'
        self.sources.append({'citation_id': 'HVK:test:b', 'title': 'Right', 'detail': 'Right nodule urine test.'})
        result = self.review()
        self.assertFalse(self.verify(result)['passed'])
        result['claims'][0]['evidence_ids'] = ['S2-detail-1']
        self.assertFalse(self.verify(result)['passed'])

    def test_omitted_and_missing_evidence_are_separate(self):
        omitted = self.verify(self.review(status='omitted', point_span='S1-detail-2'))
        self.assertFalse(omitted['passed'])
        self.assertTrue(omitted['support_passed'])
        self.assertEqual(['P1'], omitted['omitted_points'])
        self.assertEqual([], omitted['missing_evidence_points'])
        missing = self.verify(self.review(status='missing_evidence'))
        self.assertTrue(missing['coverage_contract_passed'])
        self.assertEqual(['P1'], missing['missing_evidence_points'])

    def test_bare_boolean_or_partial_semantic_checks_cannot_pass_v2(self):
        result = self.review()
        result.pop('question_coverage')
        self.assertFalse(self.verify(result)['passed'])
        result = self.review()
        result['claims'][0]['checks'].pop('polarity')
        self.assertFalse(valid_verifier_output(result))
        self.assertFalse(self.verify(result)['passed'])
        for dimension in ('polarity', 'conditions', 'relation', 'qualifiers'):
            result = self.review()
            result['claims'][0]['checks'][dimension] = False
            self.assertFalse(self.verify(result)['passed'])

    def test_point_bindings_and_boolean_must_agree(self):
        for change in ({'answer_claim_ids': ['F99']}, {'evidence_ids': ['S99']},
                       {'evidence_ids': ['S1-detail-2']}, {'question_quote': '不存在的问题'},
                       {'status': 'omitted', 'answer_claim_ids': []}):
            result = self.review()
            result['question_coverage'][0].update(change)
            self.assertFalse(self.verify(result)['passed'], change)

    def test_whole_source_set_is_available_for_detecting_omissions(self):
        prompt, data = prepare_verification(self.question, self.answer, self.sources, audit_coverage=True)
        self.assertIn('fact_span_v2', prompt)
        self.assertEqual(['S1-title-1', 'S1-detail-1', 'S1-detail-2'], [s['evidence_id'] for s in data['evidence_spans']])
        result = self.review(status='omitted', point_span='S1-detail-2')
        checked = self.verify(result)
        self.assertIn('three months', checked['question_points'][0]['quotes'][0])
        self.assertFalse(checked['human_confirmed'])

    def test_numerical_invention_and_legacy_response_remain_rejected_online(self):
        self.answer = 'A1C反映过去4个月。 [HVK:test:a]'
        result = self.review()
        result['claims'][0]['evidence_ids'] = ['S1-detail-2']
        result['question_coverage'][0]['evidence_ids'] = ['S1-detail-2']
        self.assertFalse(self.verify(result)['passed'])
        result['protocol'] = 'fact_span_v1'
        self.assertFalse(self.verify(result)['passed'])


if __name__ == '__main__':
    unittest.main()
