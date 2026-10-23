import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.ui import quiet_agent_ui
from healthvault.evaluation import personal_skill_run as run
from healthvault.evaluation.current_delivery import digest, implementation_binding, save
from healthvault.evaluation.measurement import paired_skill_metrics
from healthvault.model_config import HealthVaultModelConfig


@unittest.skipUnless(
    (run.ROOT / '.task_outputs/task-088-product-preflight-v5/preflight.json').is_file()
    and (run.ROOT / '.task_outputs/task-088-product-preflight-v5/dataset.json').is_file(),
    'Requires the local frozen product-evaluation inputs, which are not published.',
)
class PersonalPairTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        parent = run.ROOT / '.task_outputs/task-105-tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        frozen = run.ROOT / '.task_outputs/task-088-product-preflight-v5'
        self.base = json.loads((frozen / 'preflight.json').read_text(encoding='utf-8'))
        self.dataset = json.loads((frozen / 'dataset.json').read_text(encoding='utf-8'))
        self.base['implementation_sha256'] = implementation_binding()['sha256']
        self.base['bindings'] = {name: digest(Path(name)) for name in self.base['bindings']}
        self.base['protocol'] = run.workflow.PROTOCOL
        self.config = HealthVaultModelConfig('openai-compatible', 'deepseek-flash',
            self.base['model_config']['api_base'], 'synthetic', reasoning_effort='high')

    def base_prepare(self, output):
        output.mkdir(parents=True)
        save(output / 'preflight.json', self.base)
        save(output / 'dataset.json', self.dataset)
        return copy.deepcopy(self.base)

    async def prepare(self):
        output = self.root / 'preflight'
        with patch.object(run.workflow, 'prepare', side_effect=self.base_prepare):
            plan = await run.prepare(output)
        return plan, output / 'preflight.json'

    async def test_prepare_binds_current_private_candidate_and_rejects_drift(self):
        project = run.proposal.PROJECT / run.proposal.TARGET_SKILL / 'SKILL.md'
        original = digest(project)
        plan, path = await self.prepare()
        self.assertEqual(160, plan['limits']['maximum_calls'])
        self.assertEqual(0, plan['api_calls'])
        self.assertEqual(20, len(plan['case_order']))
        self.assertFalse(plan['publishes_skill'])
        self.assertEqual('not_measured', plan['fresh_task_benefit'])
        self.assertNotEqual(plan['arms']['baseline']['skill_sha256'], plan['arms']['candidate']['skill_sha256'])
        # The baseline loader has its own tests; here its frozen inputs are reused to isolate the new wrapper.
        with patch.object(run.workflow, 'validate', return_value=self.base):
            self.assertEqual(plan, run.validate(path))
            candidate = Path(plan['arms']['candidate']['root']) / run.proposal.TARGET_SKILL / 'SKILL.md'
            candidate.write_bytes(candidate.read_bytes() + b'\nchanged')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                run.validate(path)
        self.assertEqual(original, digest(project))

    async def test_legacy_loop_without_original_question_cannot_claim_scoped_injection(self):
        plan, path = await self.prepare()
        first = next(case for case in self.dataset['cases'] if case['case_id'] == 'R5E-004')
        plan['case_order'] = [first['case_id']] + [key for key in plan['case_order'] if key != first['case_id']]
        class UnusedEmbedder:
            dimension = 1024
            def embed(self, _texts):
                raise AssertionError('Timeline task cannot require semantic retrieval.')
        class Client:
            is_real = False
            requests = []
            def complete(self, payload):
                self.requests.append(copy.deepcopy(payload))
                messages = payload['messages']
                tool = next((m for m in reversed(messages) if m['role'] == 'tool'), None)
                if tool is None:
                    message = {'role': 'assistant', 'content': None, 'reasoning_content': 'synthetic-plan', 'tool_calls': [
                        {'id': 'timeline', 'type': 'function', 'function': {'name': 'query_health_timeline', 'arguments': '{}'}}]}
                    finish = 'tool_calls'
                else:
                    ids = json.loads(tool['content'])['available_citation_ids']
                    text = json.dumps({'final_status': 'ok', 'citations': ids[:1], 'refusal': False})
                    if run.proposal.CONTENT in messages[0]['content']:
                        text += ' extra'
                    message, finish = {'role': 'assistant', 'content': text}, 'stop'
                return {'choices': [{'finish_reason': finish, 'message': message}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        auth, review = self.root / 'approval.json', self.root / 'review.json'
        save(auth, {'approved': True, 'authorization_source': 'human_user', 'plan_sha256': digest(path)})
        save(review, {'decision': 'PASS', 'plan_sha256': digest(path)})
        from healthvault.infrastructure import hybrid_health_knowledge
        client = Client()
        with quiet_agent_ui(), patch.object(run, 'validate', return_value=plan), patch.object(run, 'ROOT', self.root), \
             patch.object(run.workflow, 'configured_web_model', return_value=self.config), \
             patch.object(run, 'ModelHTTPClient', return_value=client), \
             patch.object(hybrid_health_knowledge, '_local_embedder', return_value=UnusedEmbedder()):
            result = await run.execute(path, self.root / 'result', approval=auth, review=review)
        self.assertEqual('stopped_on_error', result['status'])
        self.assertEqual(2, result['completed_attempts'])
        baseline, candidate = result['results']
        self.assertTrue(baseline['passed'], baseline)
        self.assertFalse(candidate['passed'])
        self.assertFalse(baseline['measurement']['personal_skill_injected'])
        self.assertFalse(candidate['measurement']['personal_skill_injected'])
        self.assertFalse(candidate['injection_contract_passed'])
        self.assertEqual('passed', candidate['attempt']['raw_final_validation'])
        self.assertEqual(30, baseline['attempt']['total_tokens'])
        self.assertEqual(30, candidate['attempt']['total_tokens'])
        self.assertEqual(60, result['observed_tokens'])
        self.assertEqual(0, result['api_calls'])
        self.assertEqual([first['case_id']], result['paired']['regressed_cases'])
        self.assertEqual([], result['paired']['hard_failure_regressions'])
        self.assertEqual([], result['paired']['safety_regressions'])
        self.assertIsNone(result['whole_product_task_success_rate'])
        with patch.object(run, 'validate', return_value=plan), patch.object(run, 'ROOT', self.root), \
             patch.object(run.workflow, 'configured_web_model', return_value=self.config):
            with self.assertRaises(FileExistsError):
                await run.execute(path, self.root / 'second-result', approval=auth, review=review)

    async def test_wrong_authorization_and_actual_transport_budget_stop_before_provider(self):
        plan, path = await self.prepare()
        save(self.root / 'approval.json', {'approved': True, 'authorization_source': 'old_budget', 'plan_sha256': digest(path)})
        save(self.root / 'review.json', {'decision': 'PASS', 'plan_sha256': digest(path)})
        with patch.object(run, 'validate', return_value=plan), \
             patch.object(run.workflow, 'configured_web_model', side_effect=AssertionError('No provider read before approval')):
            with self.assertRaisesRegex(ValueError, 'new explicit paid approval'):
                await run.execute(path, self.root / 'result', approval=self.root / 'approval.json', review=self.root / 'review.json')
        class Client:
            is_real = False
            calls = 0
            def complete(self, _payload):
                self.calls += 1
                return {'choices': [{'finish_reason': 'stop', 'message': {'content':
                    '{"final_status":"insufficient_evidence","citations":[],"refusal":true}'}}], 'usage': {}}
        client = Client()
        bounded = copy.deepcopy(plan)
        bounded['limits'].update(maximum_calls=2, model_turns_per_attempt=1)
        transport = run.CurrentTransport(bounded, self.root, client)
        payload = {'model': 'deepseek-flash', 'max_tokens': 16384, 'temperature': 0, 'stream': False,
                   'messages': [{'role': 'user', 'content': 'synthetic'}],
                   'tools': [{'function': {'name': name}} for name in plan['tool_names']]}
        for arm in run.ARMS:
            transport.arm_id = arm
            await transport.post_json(plan['endpoint'], {}, payload, 60)
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            await transport.post_json(plan['endpoint'], {}, payload, 60)
        self.assertEqual(2, client.calls)

    async def test_pair_costs_keep_failures_unknown_usage_and_unnecessary_calls(self):
        row = {'case_id': 'case', 'arm_id': 'baseline', 'passed': False, 'hard_failure': False,
               'oracle': {'hard_failure_reasons': []},
               'product_contract': {'required_tools': ['query_health_timeline']},
               'attempt': {'calls': [{'name': 'query_health_timeline'}, {'name': 'generate_visit_brief'}],
                           'total_tokens': 17, 'usage_unknown_calls': 1, 'latency_ms': 20}}
        baseline = run.paired_row(row)
        self.assertIsNone(baseline['total_tokens'])
        self.assertEqual(17, baseline['observed_tokens'])
        self.assertEqual(1, baseline['unnecessary_calls'])
        candidate = {**baseline, 'arm_id': 'candidate', 'passed': True, 'total_tokens': 30}
        paired = paired_skill_metrics([baseline, candidate])
        self.assertEqual(['case'], paired['improved_cases'])
        self.assertEqual([], paired['safety_regressions'])
        self.assertIsNone(paired['total_tokens_delta'])
        candidate.update(passed=False, hard_failure=True)
        paired = paired_skill_metrics([baseline, candidate])
        self.assertEqual(['case'], paired['hard_failure_regressions'])
        self.assertEqual([], paired['safety_regressions'])
        row['oracle']['hard_failure_reasons'] = ['illegal_final']
        self.assertFalse(run.paired_row(row)['safety_failure'])
        row['oracle']['hard_failure_reasons'] = ['unauthorized_tool']
        self.assertTrue(run.paired_row(row)['safety_failure'])
        del candidate['safety_failure']
        self.assertIsNone(paired_skill_metrics([baseline, candidate])['safety_regressions'])
        with self.assertRaises(ValueError):
            paired_skill_metrics([baseline])


if __name__ == '__main__':
    unittest.main()
