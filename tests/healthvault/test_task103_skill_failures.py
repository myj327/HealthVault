import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.ui import quiet_agent_ui
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.evolution.failures import classify_trace_failure
from healthvault.evolution.personal import PersonalSkillManager
from healthvault.evolution.trace import TaskTraceRecorder
from healthvault.evolution.verifier import CandidateVerifier
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext
from tests.healthvault.test_skill_review_loop import ACTIVE_SKILL, signal_context


class SkillFailureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / '.task_outputs/task-103-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / '.bear' / 'skills'
        active = self.project / 'HealthVault-Task-Execution/SKILL.md'
        active.parent.mkdir(parents=True)
        active.write_text(ACTIVE_SKILL, encoding='utf-8')
        self.manager = PersonalSkillManager(self.project, self.root / 'evolution', None)
        self.service = self.manager.service('alice')
        self.baseline = hashlib.sha256(self.service.store.active_path.read_bytes()).hexdigest()
        candidate = ACTIVE_SKILL.replace('1. Identify', '1a. Verify the requested workflow.\n1. Identify')
        verified = CandidateVerifier('HealthVault-Task-Execution').verify_content(candidate, action='evolve', active_content=ACTIVE_SKILL)
        release = self.service.store.publisher.publish_content(candidate, action='evolve', verification=verified, approved=True)
        self.assertEqual('published', release.status)
        self.version = release.active_sha256

    async def trace(self, task, code=None, *, success=False, intent=None):
        async def execute(*_args):
            return '{"status":"ok"}'
        context = signal_context('passed' if success else 'failed')
        if intent:
            context['intent'] = intent
        recorder = TaskTraceRecorder(self.service.root, request_id=task + str(len(list((self.service.root / 'traces').glob('*')))),
            task_id=task, skill={'name': 'HealthVault-Task-Execution', 'sha256': self.version}, executor=execute,
            signal_context=context)
        await recorder.execute('query_health_timeline', {'untrusted': 'synthetic-private-original-99'})
        return recorder.finish('succeeded' if success else 'failed', failure_code=code, tokens={'input': 11, 'output': 5})

    async def test_original_three_failure_causes_have_distinct_categories(self):
        for code, expected in (('answer_verification_failed', 'answer_quality'), ('workflow_completion_failed', 'workflow'),
                               ('APIConnectionError', 'external_service'), ('unclassified_fault', 'unknown')):
            trace = json.loads((await self.trace(code, code)).read_text(encoding='utf-8'))
            self.assertEqual(expected, trace['failure_category'])
            self.assertEqual(expected, classify_trace_failure(trace))
        self.assertEqual('tool', classify_trace_failure({'failure_code': 'APIConnectionError', 'result_status': 'failed',
            'tool_calls': [{'result_status': 'failed'}]}))

    async def test_quality_workflow_and_unknown_failures_enter_observation(self):
        for code in ('answer_verification_failed', 'workflow_completion_failed', 'unclassified_fault'):
            result = self.manager.record_usage('alice', await self.trace(code, code), injected=True)
            self.assertEqual('recorded', result['status'])
        result = self.manager.record_usage('alice', await self.trace('outage', 'APIConnectionError'), injected=True)
        self.assertEqual('excluded_runtime_failure', result['status'])
        evaluation = result['evaluation']
        self.assertEqual(3, evaluation['effective_samples'])
        self.assertEqual(4, evaluation['observed_tasks'])
        self.assertEqual(4, evaluation['task_failures'])
        self.assertEqual(1, evaluation['answer_quality_failures'])
        self.assertEqual(1, evaluation['workflow_failures'])
        self.assertEqual(1, evaluation['unknown_failures'])
        self.assertEqual(1, evaluation['external_service_failures'])
        self.assertEqual(64, evaluation['total_tokens'])

    async def test_quality_failures_reach_existing_rollback_threshold_and_restart(self):
        for index in range(10):
            result = self.manager.record_usage('alice', await self.trace(str(index), 'answer_verification_failed'), injected=True)
        self.assertEqual('disable', result['evaluation']['decision'])
        self.assertEqual(['disable', 'rollback', 'archive'], result['lifecycle']['actions'])
        self.assertEqual(self.baseline, self.manager.usage('alice')['active_sha256'])
        restarted = PersonalSkillManager(self.project, self.root / 'evolution', None)
        self.assertFalse(restarted.lifecycle('alice').is_injectable(self.version))
        self.assertEqual(ACTIVE_SKILL, (self.project / 'HealthVault-Task-Execution/SKILL.md').read_text(encoding='utf-8'))

    async def test_service_failures_are_saved_and_do_not_displace_attributable_window(self):
        for index in range(4):
            self.manager.record_usage('alice', await self.trace('before-' + str(index), 'answer_verification_failed'), injected=True)
        for index in range(12):
            result = self.manager.record_usage('alice', await self.trace('outage-' + str(index), 'APIConnectionError'), injected=True)
            self.assertEqual('watch', result['evaluation']['decision'])
        for index in range(6):
            result = self.manager.record_usage('alice', await self.trace('after-' + str(index), 'workflow_completion_failed'), injected=True)
        self.assertEqual(10, result['evaluation']['effective_samples'])
        self.assertEqual(22, result['evaluation']['observed_tasks'])
        self.assertEqual(12, result['evaluation']['external_service_failures'])
        self.assertEqual('disable', result['evaluation']['decision'])

    async def test_late_quality_failure_upgrades_success_and_safety_cannot_be_downgraded(self):
        self.manager.record_usage('alice', await self.trace('same', success=True), injected=True)
        bad = await self.trace('same', 'answer_verification_failed')
        result = self.manager.record_usage('alice', bad, injected=True)
        self.assertEqual('recorded', result['status'])
        self.assertEqual(1, result['evaluation']['effective_samples'])
        self.assertEqual(0, result['evaluation']['success_count'])
        safety = self.manager.record_usage('alice', await self.trace('same', 'safety_write_violation'), injected=True)
        self.assertEqual(1, safety['evaluation']['safety_hard_failures'])
        self.assertEqual('already_recorded', self.manager.record_usage('alice', bad, injected=True)['status'])

    def runtime(self, *, verifier=None):
        from tests.healthvault.test_healthvault_agent_runtime import seed_profile
        from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
        database = self.root / 'records.sqlite3'
        seed_profile(SQLiteHealthRecordRepository(database), 'alice')
        return build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
            context=RuntimeContext('alice', 'request'), database_path=database,
            model_config=HealthVaultModelConfig('openai-compatible', 'synthetic', 'https://invalid.local/v1', 'synthetic'),
            skill_root=self.project, skill_names=('HealthVault-Task-Execution',),
            personal_skill_root=self.manager.injectable_root('alice'), evolution_root=self.service.root,
            task_id='real-runtime', signal_context=signal_context('passed'), answer_verifier=verifier))

    async def test_real_completion_failure_has_canonical_code_and_consumed_tokens(self):
        with patch.object(agent_module.openai, 'AsyncOpenAI', return_value=MagicMock()), quiet_agent_ui():
            runtime = self.runtime()
            async def model():
                runtime.agent.total_input_tokens += 7
                runtime.agent.total_output_tokens += 5
                text = '{"final_status":"insufficient_evidence","citations":[],"refusal":true}'
                runtime.agent._emit_text(text)
                return {'choices': [{'message': {'role': 'assistant', 'content': text}}]}
            runtime.agent._call_openai_stream = model
            with self.assertRaises(RuntimeError):
                await runtime.run('查询报告时间线')
        trace = json.loads(runtime.trace_recorder.trace_path.read_text(encoding='utf-8'))
        self.assertEqual('workflow_completion_failed', trace['failure_code'])
        self.assertEqual('workflow', trace['failure_category'])
        self.assertEqual({'input': 14, 'output': 10}, trace['tokens'])

    async def test_real_answer_verification_failure_is_counted(self):
        with patch.object(agent_module.openai, 'AsyncOpenAI', return_value=MagicMock()), quiet_agent_ui():
            runtime = self.runtime(verifier=AsyncMock(return_value={'passed': False}))
            calls = 0
            async def model():
                nonlocal calls
                calls += 1
                if calls == 1:
                    message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-1', 'type': 'function',
                        'function': {'name': 'query_health_timeline', 'arguments': '{}'}}]}
                else:
                    result = json.loads(next(m['content'] for m in reversed(runtime.agent._openai_messages) if m['role'] == 'tool'))
                    text = '已查询时间线。 [' + result['available_citation_ids'][0] + ']'
                    runtime.agent._emit_text(text)
                    message = {'role': 'assistant', 'content': text}
                return {'choices': [{'message': message}]}
            runtime.agent._call_openai_stream = model
            with self.assertRaisesRegex(RuntimeError, '复核'):
                await runtime.run('查询报告时间线')
        result = self.manager.record_usage('alice', runtime.trace_recorder.trace_path, injected=True)
        self.assertEqual('answer_quality', result['failure_category'])
        self.assertEqual(1, result['evaluation']['validator_failures'])
        self.assertEqual(1, result['evaluation']['task_failures'])

    async def test_observation_is_private_redacted_and_restart_deduplicated(self):
        path = await self.trace('one', 'answer_verification_failed')
        self.manager.record_usage('alice', path, injected=True)
        restarted = PersonalSkillManager(self.project, self.root / 'evolution', None)
        self.assertEqual('already_recorded', restarted.record_usage('alice', path, injected=True)['status'])
        self.assertEqual([], restarted.usage('bob')['evaluations'])
        with self.assertRaises(ValueError):
            restarted.record_usage('bob', path, injected=True)
        for file in (self.service.root / 'web-usage').rglob('*.json'):
            self.assertNotIn('synthetic-private-original-99', file.read_text(encoding='utf-8'))
        directory = self.service.root / 'web-usage/events'
        # Simulate an existing pre-upgrade event; no migration may double-count it.
        old_path = next(directory.glob('*.json'))
        old = json.loads(old_path.read_text(encoding='utf-8'))
        legacy_id = hashlib.sha256(f"one\0{self.version}".encode()).hexdigest()[:20]
        old['run_id'] = legacy_id
        legacy_path = directory / (legacy_id + '.json')
        legacy_path.write_text(json.dumps(old), encoding='utf-8')
        old_path.unlink()
        self.assertEqual('already_recorded', restarted.record_usage('alice', path, injected=True)['status'])
        other_type = await self.trace('one', 'answer_verification_failed', intent='knowledge_lookup')
        self.assertEqual('recorded', restarted.record_usage('alice', other_type, injected=True)['status'])
        events = [json.loads(p.read_text(encoding='utf-8')) for p in directory.glob('*.json')]
        self.assertEqual(2, len(events))
        self.assertEqual(2, len({e['task_type'] for e in events}))
        self.assertEqual(2, len(restarted.usage('alice')['evaluations']))
        self.assertEqual('already_recorded', restarted.record_usage('alice', other_type, injected=True)['status'])
        safety = await self.trace('one', 'safety_write_violation')
        self.assertEqual(1, restarted.record_usage('alice', safety, injected=True)['evaluation']['safety_hard_failures'])
        self.assertEqual(2, len(list(directory.glob('*.json'))))


if __name__ == '__main__':
    unittest.main()
