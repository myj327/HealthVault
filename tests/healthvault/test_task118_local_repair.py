import json
import unittest

from healthvault.web.answer_validation import _facts, verify_answer
from healthvault.web.answers import answer_repair_feedback
from healthvault.web.question_plan import create_question_plan
from tests.healthvault import test_task114_entrypoint as entry
from tests.healthvault.test_task102_answer_flow import QUESTION, SOURCES, SHORT, review


class LocalRepairTests(unittest.TestCase):
    setUp = entry.EntryPointTests.setUp
    collect = entry.EntryPointTests.collect

    def evidence(self):
        return {'status': 'ok', 'available_citation_ids': ['HVK:test:a'], 'results': [
            {'citation_id': s['citation_id'], 'title': s['title'], 'excerpt': s['detail']} for s in SOURCES]}

    def draft(self):
        return {'status': 'answered', 'claims': [
            {'text': 'A1C用于监测。', 'citation_id': 'HVK:test:a', 'point_ids': ['P1']},
            {'text': 'A1C反映三个月。', 'citation_id': 'HVK:test:a', 'point_ids': ['P2']}],
            'point_evidence': [{'point_id': 'P1', 'evidence_ids': ['S1-detail-1']},
                               {'point_id': 'P2', 'evidence_ids': ['S1-detail-2']}]}

    def case(self, suffix):
        return {'case_id': 'local-' + suffix, 'profile_id': 'r5-orchid', 'prompt': QUESTION}

    def test_terminal_punctuation_keeps_facts_and_full_semantic_check(self):
        calls = []
        def check(_prompt, payload, **_kwargs):
            calls.append(payload)
            return review(payload)
        result = verify_answer(QUESTION, SHORT + '。', SOURCES, query=check, audit_coverage=True,
                               required_points=create_question_plan(QUESTION, resolved_question=QUESTION)['points'])
        self.assertEqual(_facts(SHORT), _facts(SHORT + '。'))
        self.assertEqual(1, len(calls))
        self.assertFalse(result['passed'])  # The omitted point still fails, despite valid punctuation.

    def test_unbound_words_numbers_and_adjacent_citations_are_not_discarded(self):
        for tail in (' 6.2', '。无需就医。', ' [HVK:test:a]', '\nClinical claim.'):
            with self.subTest(tail=tail):
                self.assertEqual([], _facts(SHORT + tail))
        self.assertEqual([], _facts('。'))

    def test_local_failure_is_repairable_but_verifier_protocol_is_not(self):
        def prohibited(*_args, **_kwargs):
            self.fail('Invalid locally bound prose must not call the verifier yet.')
        draft = SHORT + '\n一段未绑定说明。'
        checked = verify_answer(QUESTION, draft, SOURCES, query=prohibited, audit_coverage=True)
        self.assertEqual('invalid_cited_answer', checked['kind'])
        feedback = answer_repair_feedback(draft, checked)
        self.assertEqual('unbound_answer_content', feedback['reason'])
        self.assertEqual(draft, feedback['previous_answer'])
        self.assertIsNone(answer_repair_feedback(draft, {'kind': 'invalid_verifier_output'}))
        self.assertIsNone(answer_repair_feedback(draft, {'kind': 'answer_budget_exceeded'}))

    def test_http_repairs_once_then_verifies_without_repeating_tool(self):
        initial = SHORT + '\n未绑定说明。'
        row, calls, client = self.collect(self.case('repair'), [entry.tool('search_health_knowledge', {'query': 'A1C'}),
            entry.response(initial), entry.response(json.dumps(self.draft(), ensure_ascii=False)), entry.verification], result=self.evidence())
        self.assertEqual(200, row['http_status'])
        self.assertEqual(['agent', 'agent', 'repair', 'verification'], [c['phase'] for c in calls.rows])
        self.assertEqual(1, row['response']['measurement']['answer_repair_attempts'])
        self.assertEqual(1, len(row['tool_calls']))
        self.assertNotIn('未绑定说明', row['response']['answer'])
        self.assertIn('unbound_answer_content', str(client.requests[2]))

    def test_bad_repair_json_is_not_rescued_with_another_generation(self):
        row, calls, _client = self.collect(self.case('bad-json'), [entry.tool('search_health_knowledge', {'query': 'A1C'}),
            entry.response(SHORT + '\n未绑定说明。'), entry.response('{')], result=self.evidence())
        self.assertEqual(502, row['http_status'])
        self.assertEqual(['agent', 'agent', 'repair'], [c['phase'] for c in calls.rows])
        self.assertTrue(calls.failed)
        self.assertEqual(1, len(row['tool_calls']))

    def test_repaired_but_unsupported_answer_still_fails_and_has_no_third_draft(self):
        def reject(payload):
            return entry.response(json.dumps(review(json.loads(payload['messages'][-1]['content']), unsupported=True), ensure_ascii=False))
        row, calls, _client = self.collect(self.case('unsupported'), [entry.tool('search_health_knowledge', {'query': 'A1C'}),
            entry.response(SHORT + '\n未绑定说明。'), entry.response(json.dumps(self.draft(), ensure_ascii=False)), reject], result=self.evidence())
        self.assertEqual(502, row['http_status'])
        self.assertEqual(['agent', 'agent', 'repair', 'verification'], [c['phase'] for c in calls.rows])
        self.assertEqual(1, len(row['tool_calls']))
        self.assertFalse(calls.failed)  # Product quality failure, not transport/JSON failure.


if __name__ == '__main__':
    unittest.main()
