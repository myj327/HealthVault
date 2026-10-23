import copy
import unittest

from healthvault.application.tool_evidence import tool_evidence
from healthvault.web.answer_coverage import SEMANTIC_DIMENSIONS
from healthvault.web.answer_validation import verify_answer
from healthvault.web.answers import answer_repair_feedback, knowledge_evidence


class RepairEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.question = '血液检查需要准备多久？'
        self.point = {'point_id': 'P1', 'question_quote': '准备多久', 'description': '检查准备时间'}
        self.hit = {'citation_id': 'HVK:test:a', 'title': 'How long?',
                    'source': {'title': 'Blood tests', 'url': 'https://example.invalid/blood'},
                    'excerpt': 'Prepare for 8 hours.'}
        self.sources = knowledge_evidence({'results': [self.hit]})[1]
        self.answer = '血液检查准备8小时。 [HVK:test:a]'
        self.raw = {'protocol': 'fact_span_v2', 'answer_supported': True, 'answers_question': True,
                    'claims': [{'claim_id': 'F1', 'evidence_ids': ['S1-title-1', 'S1-detail-1'],
                                'supported': True, 'checks': dict.fromkeys(SEMANTIC_DIMENSIONS, True)}],
                    'question_coverage': [{**self.point, 'evidence_ids': ['S1-detail-1'],
                                           'answer_claim_ids': ['F1'], 'status': 'covered'}]}

    def verify(self):
        return verify_answer(self.question, self.answer, self.sources, audit_coverage=True,
                             required_points=[self.point], query=lambda *_a, **_k: self.raw)

    def test_parent_context_is_shared_by_both_entries_and_body_still_required(self):
        payload = {'status': 'ok', 'results': [self.hit], 'available_citation_ids': ['HVK:test:a']}
        self.assertEqual(self.sources, tool_evidence('search_health_knowledge', payload))
        self.assertEqual('Blood tests / How long?', self.sources[0]['title'])
        self.assertTrue(self.verify()['passed'])
        self.raw['claims'][0]['evidence_ids'] = ['S1-title-1']
        failed = self.verify()
        self.assertFalse(failed['passed'])
        self.assertIn('missing_body_evidence', failed['semantic_checks'][0]['failure_reasons'])

    def test_another_source_title_cannot_supply_the_qualifier(self):
        self.sources[0]['title'] = 'How long?'
        self.sources.append({'citation_id': 'HVK:test:b', 'title': 'Blood tests', 'detail': 'Prepare for 8 hours.'})
        failed = self.verify()
        self.assertFalse(failed['passed'])
        self.assertIn('missing_qualifier:blood_sample', failed['semantic_checks'][0]['failure_reasons'])
        self.raw['claims'][0]['evidence_ids'] = ['S2-title-1', 'S1-detail-1']
        self.assertFalse(self.verify()['support_passed'])

    def test_numeric_unit_and_semantic_failure_are_located(self):
        self.answer = '血液指标为9 mg/dL。 [HVK:test:a]'
        self.raw['claims'][0]['checks']['relation'] = False
        failed = self.verify()
        reasons = failed['semantic_checks'][0]['failure_reasons']
        self.assertTrue({'unsupported_number', 'unsupported_unit', 'semantic_check_failed'} <= set(reasons))
        feedback = answer_repair_feedback(self.answer, failed)
        self.assertEqual('F1', feedback['failed_claims'][0]['claim_id'])
        self.assertEqual('HVK:test:a', feedback['failed_claims'][0]['citation_id'])
        self.assertEqual('血液指标为9 mg/dL。', feedback['failed_claims'][0]['text'])

    def test_coverage_feedback_retains_exact_requirement_and_supported_facts(self):
        self.raw['question_coverage'][0]['question_quote'] = self.question
        failed = self.verify()
        self.assertTrue(failed['support_passed'])
        self.assertFalse(failed['coverage_contract_passed'])
        feedback = answer_repair_feedback(self.answer, failed)
        self.assertEqual(self.point, feedback['coverage_errors'][0]['expected_point'])
        self.assertIn('fixed_requirement_changed', feedback['coverage_errors'][0]['failure_reasons'])
        self.assertEqual('血液检查准备8小时。', feedback['supported_claims'][0]['text'])
        self.assertEqual(['S1-title-1', 'S1-detail-1'], feedback['supported_claims'][0]['evidence_ids'])

    def test_omitted_binding_is_repaired_without_losing_already_supported_part(self):
        self.sources[0]['detail'] += '\nFollow the provider instructions.'
        self.raw['question_coverage'][0].update(status='omitted', evidence_ids=['S1-detail-2'], answer_claim_ids=[])
        self.raw['answers_question'] = False
        failed = self.verify()
        feedback = answer_repair_feedback(self.answer, failed)
        self.assertTrue(failed['coverage_contract_passed'])
        self.assertFalse(failed['passed'])
        self.assertEqual('P1', feedback['omitted_points'][0]['point_id'])
        self.assertEqual('F1', feedback['supported_claims'][0]['claim_id'])
        # A new draft must still go through all checks; old support is not a pass token.
        self.answer = '血液检查准备9小时。 [HVK:test:a]'
        self.assertFalse(self.verify()['support_passed'])

    def test_missing_evidence_and_protocol_errors_do_not_trigger_rewrite(self):
        self.raw['question_coverage'][0].update(status='missing_evidence', evidence_ids=[], answer_claim_ids=[])
        self.raw['answers_question'] = False
        self.assertIsNone(answer_repair_feedback(self.answer, self.verify()))
        self.raw = copy.deepcopy(self.raw)
        self.raw['protocol'] = 'invalid'
        self.assertIsNone(answer_repair_feedback(self.answer, self.verify()))


if __name__ == '__main__':
    unittest.main()
