import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from healthvault.evaluation import deepseek_verifier_run as runner


class VerifierPreflightTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT/'.task_outputs/task-087-local-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        self.assertTrue(self.root.resolve().is_relative_to(scratch.resolve()))
        self.addCleanup(self.temp.cleanup)
        config = SimpleNamespace(provider='openai-compatible', api_base='https://api.deepseek.com',
                                 model='deepseek-flash', reasoning_effort='high', api_key='synthetic-test-secret')
        self.config_patch = patch.object(runner, 'configured_web_model', return_value=config)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def prepared(self):
        preflight = self.root/'preflight'
        runner.prepare(preflight)
        approval = self.root/'synthetic-approval.json'
        runner.save(approval, {'approved': True, 'authorization_source': 'human_user',
                               'plan_sha256': runner.digest(preflight/'plan.json')})
        return preflight, approval

    def test_preparation_binds_twelve_requests_without_secrets_or_network(self):
        with patch.object(runner, 'request_once') as network:
            preflight, approval = self.prepared()
            plan, requests, _config = runner.validate(preflight, approval)
        network.assert_not_called()
        self.assertEqual(12, len(requests))
        self.assertEqual(0, plan['api_calls_during_preparation'])
        self.assertEqual('pending', plan['human_confirmation'])
        self.assertEqual(0, plan['retries'])
        for item in requests:
            self.assertEqual(16384, item['payload']['max_tokens'])
            self.assertEqual('high', item['payload']['reasoning_effort'])
            self.assertNotIn('tools', item['payload'])
            self.assertLessEqual(len(json.dumps(item['payload'], ensure_ascii=False).encode()), 80000)
            data = json.loads(item['payload']['messages'][1]['content'])
            self.assertTrue(data['required_points'])
        self.assertEqual('question_rules', plan['question_plan_method'])
        self.assertFalse(plan['online_question_planner_evaluated'])
        self.assertNotIn('synthetic-test-secret', (preflight/'plan.json').read_text(encoding='utf-8')+(preflight/'requests.json').read_text(encoding='utf-8'))

    def test_wrong_approval_or_changed_requests_stop_before_network(self):
        preflight, approval = self.prepared()
        with patch.object(runner, 'request_once') as network:
            runner.save(self.root/'wrong.json', {'approved': True, 'authorization_source': 'human_user', 'plan_sha256': 'old'})
            with self.assertRaises(ValueError):
                runner.run(preflight, self.root/'wrong.json', self.root/'wrong-result')
            with (preflight/'requests.json').open('a') as handle:
                handle.write('\n')
            with self.assertRaises(ValueError):
                runner.run(preflight, approval, self.root/'changed-result')
        network.assert_not_called()
        self.assertFalse((self.root/'changed-result').exists())

    def test_malformed_raw_answer_is_failed_even_for_expected_rejection(self):
        preflight, approval = self.prepared()
        response = {'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': '{'}}]}
        with patch.object(runner, 'request_once', return_value=response) as network:
            result = runner.run(preflight, approval, self.root/'result')
        network.assert_called_once()
        self.assertEqual(1, result['failed_calls'])
        self.assertEqual(11, result['not_executed'])
        self.assertEqual(15, result['known_tokens']['total_tokens'])
        self.assertNotIn('gate_support_match', result['cases'][0])
        self.assertEqual('{', json.loads((self.root/'result/response-01.json').read_text(encoding='utf-8'))['choices'][0]['message']['content'])
        with self.assertRaises(FileExistsError):
            runner.run(preflight, approval, self.root/'result')


if __name__ == '__main__':
    unittest.main()
