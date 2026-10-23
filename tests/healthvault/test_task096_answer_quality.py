import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from healthvault.evaluation.answer_quality import load_saved_review, reviewed_answer_quality


def reviewed(query_id='Q1', facts=(True,), points=(True,), *, answerable=True, status='answered', refusal=None):
    return {'query_id': query_id, 'answerable': answerable, 'status': status, 'correct_refusal': refusal,
            'facts': [{'passed': value} for value in facts],
            'reference_points': [{'passed': value} for value in points], 'reviewer_kind': 'assistant'}


class AnswerQualityTests(unittest.TestCase):
    def test_whole_question_requires_support_and_coverage(self):
        result = reviewed_answer_quality([reviewed(), reviewed('Q2', facts=(), points=(),
                                            answerable=False, status='insufficient_evidence', refusal=True)])
        self.assertEqual(2, result['whole_question']['passed'])
        self.assertEqual(1.0, result['whole_question']['rate'])
        self.assertEqual(1, result['fact_support']['total'])
        self.assertFalse(result['human_confirmation_complete'])

    def test_partial_support_or_missing_key_point_fails_whole_answer(self):
        result = reviewed_answer_quality([reviewed('MQ-001', facts=(True, False)),
                                          reviewed('MQ-011', points=(True, False, False))])
        self.assertEqual(2, result['whole_question']['failed'])
        self.assertEqual(0.0, result['whole_question']['rate'])
        self.assertIn('fact_support_failed', result['decisions'][0]['reasons'])
        self.assertIn('reference_coverage_failed', result['decisions'][1]['reasons'])

    def test_unknown_and_missing_labels_keep_denominators(self):
        result = reviewed_answer_quality([reviewed('good'), reviewed('ambiguous', facts=(True, None)),
                                          reviewed('bad', facts=(False,))])
        self.assertIsNone(result['whole_question']['rate'])
        self.assertEqual(1, result['whole_question']['pending'])
        self.assertAlmostEqual(1 / 3, result['whole_question']['lower_bound'])
        self.assertAlmostEqual(2 / 3, result['whole_question']['upper_bound'])
        self.assertEqual(4, result['fact_support']['total'])
        self.assertEqual(0.5, result['fact_support']['lower_bound'])
        missing = reviewed_answer_quality([reviewed(facts=(), points=())])
        self.assertIsNone(missing['whole_question']['rate'])
        self.assertIsNone(missing['fact_support']['upper_bound'])

    def test_refusal_errors_and_missing_refusal_judgment_are_distinct(self):
        rows = [reviewed('refused', facts=(), status='insufficient_evidence'),
                reviewed('error', facts=(), status='error'),
                reviewed('false_answer', answerable=False),
                reviewed('unreviewed_refusal', answerable=False, facts=(), points=(), status='insufficient_evidence')]
        result = reviewed_answer_quality(rows)
        self.assertEqual(0.5, result['answerable_false_refusal_rate'])
        self.assertEqual(0.5, result['unanswerable_false_answer_rate'])
        self.assertEqual(1, result['errors'])
        self.assertEqual(3, result['whole_question']['failed'])
        self.assertEqual(1, result['whole_question']['pending'])
        with self.assertRaises(ValueError):
            reviewed_answer_quality([reviewed(facts=(1,))])

    def saved_fixture(self, root):
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
            return hashlib.sha256(path.read_bytes()).hexdigest()
        quote = 'What if I ate during my fast?'
        claim = {'claim_id': 'F1', 'text': '告知医生。', 'context': '告知医生。',
                 'citation_id': 'HVK:test:1', 'quotes': [quote], 'evidence_ids': ['S1-title-1']}
        answer = {'question': '误食后怎么办？', 'answer': '告知医生。 [HVK:test:1]', 'status': 'answered',
                  'sources': [{'citation_id': 'HVK:test:1', 'title': quote, 'detail': 'Tell your provider.'}],
                  'verification': {'checked_claims': [claim]}}
        original = {'query_id': 'Q1', 'answerable': True, 'hybrid': answer}
        point = {'claim_id': 'RC1', 'text': 'Tell your provider.'}
        case_hash = save(root / 'Q1.json', original)
        results_hash = save(root / 'results.json', {'rows': [original]})
        gold_hash = save(root / 'gold.json', {'queries': [{'query_id': 'Q1', 'query': answer['question'],
                                                         'answerable': True, 'reference_claims': [point]}]})
        contexts_hash = save(root / 'contexts.json', {'queries': [{'query_id': 'Q1', 'query': answer['question']}]})
        raw = json.dumps({'status': 'answered', 'claims': [{'text': '告知医生。', 'citation_id': 'HVK:test:1'}]})
        save(root / 'responses/0001.json', {'case_id': 'Q1', 'content': raw})
        event = {'sequence': 1, 'phase': 'generation', 'status': 'completed', 'case_id': 'Q1',
                 'response_sha256': hashlib.sha256(raw.encode()).hexdigest()}
        (root / 'calls.jsonl').write_text(json.dumps(event) + '\n', encoding='utf-8')
        label = {'query_id': 'Q1', 'arm': 'hybrid', 'case_file_sha256': case_hash, 'answer': answer['answer'],
                 'program_status': 'answered', 'answerable': True, 'question': answer['question'],
                 'correct_refusal': False, 'facts': [{**claim, 'fact_id': 'F1', 'supported': True}],
                 'reference_claims': [{**point, 'covered': True}]}
        save(root / 'review.json', {'rag_results_sha256': results_hash, 'gold_sha256': gold_hash,
             'contexts_sha256': contexts_hash, 'implementation_sha256': 'old', 'review_date': '2026-10-10', 'reviews': [label]})
        return [root / name for name in ('results.json', 'gold.json', 'contexts.json', 'review.json')]

    def test_saved_review_accepts_original_title_evidence_and_raw_json(self):
        with tempfile.TemporaryDirectory() as scratch:
            paths = self.saved_fixture(Path(scratch))
            rows, binding = load_saved_review(*paths)
            self.assertEqual(1.0, reviewed_answer_quality(rows)['whole_question']['rate'])
            self.assertEqual('pending', binding['human_confirmation'])
            self.assertEqual(1, len(binding['raw_generation_responses']))

    def test_drift_missing_reviews_and_trailing_explanation_are_rejected(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            paths = self.saved_fixture(root)
            originals = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
            for kind in ('gold_drift', 'missing_review', 'raw_suffix'):
                with self.subTest(kind=kind):
                    for path, value in originals.items():
                        path.write_bytes(value)
                    if kind == 'gold_drift':
                        paths[1].write_text('{}', encoding='utf-8')
                    elif kind == 'missing_review':
                        data = json.loads(paths[3].read_text(encoding='utf-8'))
                        data['reviews'] = []
                        paths[3].write_text(json.dumps(data), encoding='utf-8')
                    else:
                        path = root / 'responses/0001.json'
                        data = json.loads(path.read_text(encoding='utf-8'))
                        data['content'] += '\n补充说明'
                        path.write_text(json.dumps(data), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_saved_review(*paths)


if __name__ == '__main__':
    unittest.main()
