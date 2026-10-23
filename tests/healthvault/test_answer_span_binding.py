import copy
import unittest

from healthvault.web.answer_validation import verify_answer, valid_verifier_output
from healthvault.web.answers import knowledge_evidence


SOURCE = {'citation_id': 'HVK:synthetic', 'title': 'Fasting blood test',
          'detail': 'Fast at least eight hours.\nBut you can drink plain water.\nDo not chew gum.\nDo not smoke.'}


class AnswerSpanBindingTests(unittest.TestCase):
    def check(self, text, make_result=None, sources=None):
        captured = {}
        def verifier(_prompt, payload, **_kwargs):
            captured.update(payload)
            result = {'protocol': 'fact_span_v1', 'answer_supported': True, 'answers_question': True,
                      'claims': [{'claim_id': fact['claim_id'], 'evidence_ids': [span['evidence_id'] for span in payload['evidence_spans']
                                  if span['citation_id'] == fact['citation_id'] and span['source_field'] == 'detail'], 'supported': True}
                                 for fact in payload['facts']]}
            return make_result(result) if make_result else result
        return verify_answer('空腹抽血有哪些要求？', text, sources or [SOURCE], query=verifier), captured

    def test_english_count_and_multiline_evidence_bind_without_recopying(self):
        answer = '至少禁食8小时，并且不能嚼口香糖、吸烟。 [HVK:synthetic]'
        result, payload = self.check(answer)
        self.assertTrue(result['passed'], result)
        self.assertEqual(answer, result['answer'])
        for span in payload['evidence_spans']:
            self.assertEqual(span['quote'], SOURCE[span['source_field']][span['start']:span['end']])
        self.assertEqual(4, len(result['checked_claims'][0]['quotes']))

    def test_whole_original_fact_including_prefix_and_parenthesis_is_preserved(self):
        answer = '是的，空腹抽血时可以喝白水（plain water）。 [HVK:synthetic]'
        result, payload = self.check(answer)
        self.assertTrue(result['passed'])
        self.assertEqual(answer, result['answer'])
        self.assertEqual('是的，空腹抽血时可以喝白水（plain water）。', payload['facts'][0]['text'])

    def test_missing_duplicate_or_uncited_facts_cannot_pass(self):
        answer = '可以喝白水。 [HVK:synthetic]\n\n不要吸烟。 [HVK:synthetic]'
        for alter in (lambda value: {**value, 'claims': value['claims'][:1]},
                      lambda value: {**value, 'claims': [value['claims'][0], copy.deepcopy(value['claims'][0])]}):
            self.assertFalse(self.check(answer, alter)[0]['passed'])
        self.assertFalse(self.check(answer + ' 另有一个右侧结节。')[0]['passed'])

    def test_unknown_or_other_source_span_cannot_bind(self):
        other = {**SOURCE, 'citation_id': 'HVK:other'}
        for identifier in ('missing', 'S2-detail-1'):
            def alter(value):
                value['claims'][0]['evidence_ids'] = [identifier]
                return value
            self.assertFalse(self.check('禁食8小时。 [HVK:synthetic]', alter, [SOURCE, other])[0]['passed'])

    def test_forged_measurement_and_percent_are_rejected(self):
        for answer in ('需要禁食80小时。 [HVK:synthetic]', '数值为8%。 [HVK:synthetic]'):
            self.assertFalse(self.check(answer)[0]['passed'])
        source = {**SOURCE, 'detail': 'Left nodule 6 mm.'}
        self.assertFalse(self.check('左侧结节60 mm。 [HVK:synthetic]', sources=[source])[0]['passed'])

    def test_semantic_failure_and_malformed_protocol_fail_closed(self):
        for alter in (lambda value: {**value, 'answers_question': False},
                      lambda value: {**value, 'protocol': 'unknown'},
                      lambda value: {**value, 'claims': [{'claim_id': 'F1', 'evidence_ids': [], 'supported': False}]}):
            self.assertFalse(self.check('可以喝白水。 [HVK:synthetic]', alter)[0]['passed'])
        self.assertFalse(valid_verifier_output({'answer_supported': True, 'answers_question': True,
                                               'protocol': 'fact_span_v1', 'claims': [{'claim_id': 'F1', 'supported': True}]}))

    def test_retrieved_fragment_is_not_silently_cut_at_1200_characters(self):
        text = 'Synthetic introductory text. ' * 55 + 'But you can drink plain water.'
        evidence, sources = knowledge_evidence({'results': [{'citation_id': 'HVK:synthetic', 'title': 'Test', 'excerpt': text}]})
        self.assertGreater(len(text), 1200)
        self.assertEqual(text, evidence['sources'][0]['detail'])
        self.assertEqual(sources, evidence['sources'])


if __name__ == '__main__':
    unittest.main()
