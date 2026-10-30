import json
import unittest

from healthvault.web.answer_validation import verify_knowledge_answer
from healthvault.web.answers import answer_with_model


SOURCES = [{'citation_id': 'HVK:test:ldl', 'title': 'Cholesterol',
            'detail': 'LDL means low-density lipoprotein.'}]
ANSWER = 'LDL 指低密度脂蛋白。[HVK:test:ldl] 以上为一般科普，不构成个人医疗建议。'


class SimpleKnowledgeTests(unittest.TestCase):
    def test_whole_answer_including_notice_reaches_one_semantic_check(self):
        calls = []
        def check(prompt, payload, **kwargs):
            calls.append(payload)
            return {'supported': True, 'complete': True}
        result = verify_knowledge_answer('LDL是什么', ANSWER, SOURCES, query=check)
        self.assertTrue(result['passed'])
        self.assertEqual(ANSWER, result['answer'])
        self.assertEqual([{'question': 'LDL是什么', 'answer': ANSWER, 'sources': SOURCES}], calls)
        corrected = 'LDL 是低密度脂蛋白。[HVK:test:ldl]'
        result = verify_knowledge_answer('LDL是什么', ANSWER, SOURCES,
            query=lambda *args, **kwargs: {'supported': True, 'complete': True, 'answer': corrected})
        self.assertEqual(corrected, result['answer'])

    def test_wrong_sources_unsupported_facts_omissions_and_bad_protocol_still_fail(self):
        def never(*args, **kwargs):
            self.fail('Invented citations must be rejected before a model call.')
        self.assertFalse(verify_knowledge_answer('LDL是什么', ANSWER.replace('HVK:test:ldl', 'HVK:fake'), SOURCES, query=never)['passed'])
        for reply in ({'supported': False, 'complete': True}, {'supported': True, 'complete': False},
                      {'supported': 'true', 'complete': True}, '{}',
                      {'supported': True, 'complete': True, 'answer': '错误来源。[HVK:fake]'}):
            with self.subTest(reply=reply):
                self.assertFalse(verify_knowledge_answer('LDL是什么', ANSWER, SOURCES,
                                                       query=lambda *args, **kwargs: reply)['passed'])

    def test_knowledge_generation_needs_no_point_or_span_numbering(self):
        calls = []
        def generate(prompt, payload, **kwargs):
            calls.append(payload)
            return json.dumps({'status': 'answered', 'claims': [
                {'text': 'LDL 指低密度脂蛋白。', 'citation_id': 'HVK:test:ldl'}]})
        evidence = {'sources': SOURCES, 'question_plan': {'answer_contract': 'point_evidence_v1', 'points': []}}
        answer = answer_with_model('LDL是什么', 'knowledge', evidence, query=generate)
        self.assertIn('[HVK:test:ldl]', answer)
        self.assertNotIn('evidence_spans', calls[0]['evidence'])
        self.assertNotIn('draft_grounding', evidence)


if __name__ == '__main__':
    unittest.main()
