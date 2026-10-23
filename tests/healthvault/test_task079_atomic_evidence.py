import copy
import json
import unittest
from pathlib import Path

from healthvault.web.answer_validation import verify_answer
from healthvault.web.answers import answer_with_model


FIXTURE = Path(__file__).parent / 'fixtures/task079_atomic_replays.json'


class AtomicEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURE.read_text(encoding='utf-8'))['cases']

    def replay(self, case, answer=None, verdict=None, mutate=None):
        self.payload = None
        def checker(_prompt, data, **_kwargs):
            self.payload = data
            result = copy.deepcopy(verdict) if verdict else {
                'protocol': 'fact_span_v1', 'answer_supported': True, 'answers_question': True,
                'claims': [{'claim_id': fact['claim_id'], 'supported': True,
                            'evidence_ids': [item for item in fact['allowed_evidence_ids'] if '-detail-' in item]}
                           for fact in data['facts']]}
            if mutate:
                mutate(result)
            return result
        return verify_answer(case['question'], answer or case['answer'], case['sources'], query=checker)

    def test_unmodified_real_generation_and_verifier_no_longer_pass_mq001(self):
        case = self.cases[0]
        generated = answer_with_model(case['question'], 'knowledge', {'sources': case['sources']},
                                      query=lambda *_args, **_kwargs: case['raw_generation'])
        self.assertEqual(case['answer'], generated)
        result = self.replay(case, verdict=json.loads(case['raw_verification']))
        self.assertFalse(result['passed'])
        self.assertFalse(result['coverage_passed'])
        self.assertIsNone(result['answer'])
        self.assertEqual(['F1.1', 'F1.2'], [item['claim_id'] for item in self.payload['facts'][:2]])
        self.assertIn('血液检测', self.payload['facts'][0]['text'])
        self.assertIn('医生会', self.payload['facts'][1]['text'])
        self.assertEqual(self.payload['facts'][0]['citation_id'], self.payload['facts'][1]['citation_id'])
        self.assertNotIn('blood', case['sources'][2]['detail'].casefold())

    def test_correctly_split_mq001_uses_its_own_000_and_002_sources(self):
        case = self.cases[0]
        raw = json.loads(case['raw_generation'])
        raw['claims'][:1] = [
            {'text': '2 型糖尿病的确诊通过血液检测完成。', 'citation_id': case['sources'][0]['citation_id']},
            {'text': '医生会根据有无症状或是否怀孕推荐不同的检测方法。', 'citation_id': case['sources'][2]['citation_id']}]
        answer = answer_with_model(case['question'], 'knowledge', {'sources': case['sources']},
                                   query=lambda *_args, **_kwargs: json.dumps(raw, ensure_ascii=False))
        checked = self.replay(case, answer=answer)
        self.assertTrue(checked['passed'], checked)
        self.assertEqual(answer, checked['answer'])
        for fact in self.payload['facts']:
            spans = [span for span in self.payload['evidence_spans'] if span['evidence_id'] in fact['allowed_evidence_ids']]
            self.assertTrue(spans)
            self.assertTrue(all(span['citation_id'] == fact['citation_id'] for span in spans))
        self.assertEqual(case['sources'][0]['citation_id'], self.payload['facts'][0]['citation_id'])
        self.assertEqual(case['sources'][2]['citation_id'], self.payload['facts'][1]['citation_id'])

    def test_original_legal_rag027_and_rag042_answers_still_pass_per_clause(self):
        for case in self.cases[1:]:
            with self.subTest(query=case['query_id'], arm=case['arm']):
                original = json.loads(case['raw_verification'])
                by_parent = {item['claim_id']: item for item in original['claims']}
                def checker(_prompt, data, **_kwargs):
                    return {**original, 'claims': [{**by_parent[fact['claim_id'].split('.')[0]],
                                                    'claim_id': fact['claim_id']} for fact in data['facts']]}
                checked = verify_answer(case['question'], case['answer'], case['sources'], query=checker)
                self.assertTrue(checked['passed'], checked)
                self.assertEqual(case['answer'], checked['answer'])
                self.assertGreater(checked['claim_count'], len(original['claims']))

    def test_omitted_or_duplicate_subclaim_cannot_cover_the_compound_sentence(self):
        case = self.cases[0]
        first = case['answer'].split('\n\n')[0]
        text, citation = first.rsplit(' [', 1)
        for statement in (text, text.replace('，', '；', 1), '（' + text + '）'):
            for mutate in (lambda result: result['claims'].pop(0),
                           lambda result: result['claims'].__setitem__(0, copy.deepcopy(result['claims'][1]))):
                checked = self.replay(case, answer=statement + ' [' + citation, mutate=mutate)
                self.assertGreaterEqual(len(self.payload['facts']), 2)
                self.assertFalse(checked['passed'])
                self.assertFalse(checked['coverage_passed'])

    def test_other_citation_cannot_supply_the_missing_blood_test_subclaim(self):
        case = self.cases[0]
        first = case['answer'].split('\n\n')[0]
        checked = self.replay(case, answer=first,
                              mutate=lambda result: result['claims'][0].update(evidence_ids=['S1-detail-1']))
        self.assertFalse(checked['passed'])
        self.assertIsNone(checked['answer'])
        self.assertEqual('F1.1', self.payload['facts'][0]['claim_id'])
        self.assertNotIn('S1-detail-1', self.payload['facts'][0]['allowed_evidence_ids'])

    def test_invented_subclaim_measurements_and_uncited_tail_still_fail(self):
        case = {'question': '报告记录了什么？', 'sources': [
            {'citation_id': 'REC:synthetic', 'detail': '左叶 6 mm。右叶 8 mm。比例 7%。'}]}
        for text in ('左叶 6 mm，右叶 60 mm。', '左叶 6 mm，右叶 8 cm。',
                     '左叶 6 mm，比例 7。', '左叶 6 mm，比例 8%。'):
            with self.subTest(text=text):
                self.assertFalse(self.replay(case, answer=text + ' [REC:synthetic]')['passed'])
        self.assertFalse(self.replay(case, answer='左叶 6 mm。 [REC:synthetic] 另有一个结节。')['passed'])

    def test_original_context_preserves_conditions_negation_and_decimal_dimensions(self):
        case = {'question': '核对明示内容', 'sources': [
            {'citation_id': 'REC:synthetic', 'detail': '报告尺寸为 5.2×3.4 mm。医生要求空腹时遵医嘱，不要自行决定。'}]}
        text = '如果医生要求空腹，则应遵医嘱，不要自行决定是否空腹。'
        self.assertTrue(self.replay(case, answer=text + ' [REC:synthetic]')['passed'])
        facts = self.payload['facts']
        self.assertEqual(['如果医生要求空腹，则应遵医嘱，', '不要自行决定是否空腹。'], [item['text'] for item in facts])
        self.assertTrue(all(item['context'] == text for item in facts))
        legacy = {'answer_supported': True, 'answers_question': True, 'claims': [
            {'text': fact['text'], 'citation_id': fact['citation_id'],
             'quote': case['sources'][0]['detail'], 'supported': True} for fact in reversed(facts)]}
        checked = self.replay(case, answer=text + ' [REC:synthetic]', verdict=legacy)
        self.assertTrue(checked['passed'], checked)
        self.assertEqual(text + ' [REC:synthetic]', checked['answer'])
        text = '是的，报告尺寸为 5.2×3.4 mm（长径、短径）。'
        self.assertTrue(self.replay(case, answer=text + ' [REC:synthetic]')['passed'])
        self.assertEqual([text], [item['text'] for item in self.payload['facts']])

    def test_legacy_whole_sentence_or_cross_citation_verdict_cannot_bypass_subclaims(self):
        case = self.cases[0]
        first = case['answer'].split('\n\n')[0]
        text = first[:first.rfind(' [')]
        for citation, quote in ((case['sources'][2]['citation_id'], case['sources'][2]['detail']),
                                (case['sources'][0]['citation_id'], case['sources'][0]['detail'])):
            verdict = {'answer_supported': True, 'answers_question': True, 'claims': [
                {'text': text, 'citation_id': citation, 'quote': quote, 'supported': True}]}
            checked = self.replay(case, answer=first, verdict=verdict)
            self.assertFalse(checked['passed'])
            self.assertIsNone(checked['answer'])


if __name__ == '__main__':
    unittest.main()
