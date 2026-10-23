import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from healthvault.evolution.personal import PersonalSkillManager
from healthvault.evolution.workflow_eval import CASES, evaluate_workflow_probes
from healthvault.evolution.trace import TaskTraceRecorder
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.runtime_context import RuntimeContext
from healthvault.model_config import HealthVaultModelConfig
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app
from tests.healthvault import test_skill_review_loop as fixtures

ACTIVE_SKILL = fixtures.ACTIVE_SKILL
signal_context = fixtures.signal_context


class PersonalSkillTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / '.bear' / 'skills'
        self.active = self.project / 'HealthVault-Task-Execution' / 'SKILL.md'
        self.active.parent.mkdir(parents=True)
        self.active.write_text(ACTIVE_SKILL, encoding='utf-8')
        self.calls = []
        async def evaluate(active, candidate):
            async def model(prompt, payload):
                if payload.get('phase') == 'execution_tool':
                    return {'tool_name': 'compare_health_indicators', 'arguments': {}} if payload['case_id'] == 'exec-compare' else {
                        'tool_name': 'search_health_knowledge', 'arguments': {'query': 'A1C'}}
                if payload.get('phase') == 'execution_final':
                    citations = payload['tool_result'].get('available_citation_ids', [])
                    return {'final_status': 'ok', 'citations': citations if payload['case_id'] == 'exec-compare' else citations[:1], 'refusal': False}
                return {'plans':[{'case_id':case[0],'tool_path':[case[3]] if case[3] else [],'refuse':case[4]} for case in CASES]}
            return await evaluate_workflow_probes(active,candidate,model)
        self.manager = PersonalSkillManager(self.project, self.root / 'evolution', self.fake_patch,evaluator=evaluate)

    def tearDown(self):
        self.temp.cleanup()

    async def fake_patch(self, prompt, schema):
        return await fixtures.SkillReviewLoopTests.fake_patch(self, prompt, schema)

    async def traces(self, profile):
        service = self.manager.service(profile)
        async def tool(*args):
            return json.dumps({'status': 'ok'})
        for index in range(3):
            recorder = TaskTraceRecorder(service.root, request_id=f'request-{index}', task_id=f'task-{index}',
                skill={'name': 'HealthVault-Task-Execution', 'sha256': hashlib.sha256(service.store.active_path.read_bytes()).hexdigest()},
                executor=tool, signal_context=signal_context())
            await recorder.execute('query_health_timeline', {})
            recorder.finish('succeeded')

    async def test_default_auto_release_is_private_and_survives_restart(self):
        await self.traces('alice')
        record = await self.manager.consider('alice')
        self.assertEqual('published', record['status'])
        self.assertEqual('personal_skill', record['scope'])
        self.assertEqual(ACTIVE_SKILL, self.active.read_text(encoding='utf-8'))
        self.assertEqual(ACTIVE_SKILL, self.manager.service('bob').store.active_path.read_text(encoding='utf-8'))
        restarted = PersonalSkillManager(self.project, self.root / 'evolution', self.fake_patch)
        self.assertIn('1a.', restarted.service('alice').store.active_path.read_text(encoding='utf-8'))
        result = restarted.service('alice').store.rollback('alice')
        self.assertEqual('rolled_back', result['status'])
        self.assertEqual(ACTIVE_SKILL, restarted.service('alice').store.active_path.read_text(encoding='utf-8'))

    async def test_optional_manual_gate_and_failed_gate_never_auto_publish(self):
        self.manager.set_manual_review('alice', True)
        await self.traces('alice')
        record = await self.manager.consider('alice')
        self.assertEqual('pending_review', record['status'])
        self.assertEqual(ACTIVE_SKILL, self.manager.service('alice').store.active_path.read_text(encoding='utf-8'))
        self.manager.service('alice').store.decide(record['candidate_id'], 'approve', 'alice')
        async def invalid(*args):
            return {'patches': []}
        self.manager.side_query = invalid
        await self.traces('bob')
        rejected = await self.manager.consider('bob')
        self.assertEqual('gate_rejected', rejected['status'])
        self.assertEqual(ACTIVE_SKILL, self.manager.service('bob').store.active_path.read_text(encoding='utf-8'))

    async def test_failed_behavior_or_unavailable_evaluator_cannot_publish(self):
        await self.traces('alice')
        self.manager.evaluator = None
        pending = await self.manager.consider('alice')
        self.assertEqual('pending_review',pending['status'])
        with self.assertRaises(ValueError):
            self.manager.service('alice').store.decide(pending['candidate_id'],'approve','alice')
        async def failed(active,candidate):
            return {'passed':False,'regression_passed':False,'held_out_passed':False}
        self.manager.evaluator = failed
        rejected = await self.manager.consider('alice')
        self.assertEqual('gate_rejected',rejected['status'])
        self.assertEqual(ACTIVE_SKILL,self.manager.service('alice').store.active_path.read_text(encoding='utf-8'))

    async def test_shared_proposal_requires_admin_review_without_project_mutation(self):
        await self.traces('alice')
        record = await self.manager.consider('alice')
        proposal = self.manager.share('alice', record['candidate_id'])
        self.assertEqual('pending_review', proposal['status'])
        self.assertNotIn('candidate_content', proposal)
        approved = self.manager.decide_proposal(proposal['proposal_id'], 'approve', 'admin')
        self.assertEqual('approved_for_developer', approved['status'])
        self.assertEqual(ACTIVE_SKILL, self.active.read_text(encoding='utf-8'))
        with self.assertRaises(ValueError):
            self.manager.share('bob', record['candidate_id'])

    async def test_runtime_keeps_unscoped_personal_version_out_of_prompt(self):
        await self.traces('alice')
        await self.manager.consider('alice')
        for user in ('alice', 'bob'):
            with patch('healthvault.agent_runtime.Agent') as agent:
                runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                    context=RuntimeContext(user, 'request'), database_path=self.root / 'records.sqlite3',
                    model_config=HealthVaultModelConfig('openai-compatible', 'fake', 'https://invalid.local/v1', 'synthetic'),
                    skill_root=self.project, skill_names=('HealthVault-Task-Execution',),
                    personal_skill_root=self.manager.service(user).store.skill_root,
                    evolution_root=self.manager.service(user).root))
                prompt = agent.call_args.kwargs['custom_system_prompt']
                self.assertNotIn('1a.', prompt)
                self.assertEqual(1, len(runtime.manifest['project_skills']))
                self.assertFalse(runtime.manifest['personal_skill_injected'])
                self.assertEqual('unknown_question' if user == 'alice' else 'same_as_project_skill',
                                 runtime.manifest['personal_skill_reason'])

    async def test_api_isolation_and_shared_review_permissions(self):
        app = create_app(HealthVaultUIController(self.root / 'web.sqlite3'), skill_root=self.project,
                         skill_evolution_root=self.root / 'api-evolution', skill_patch_side_query=AsyncMock())
        with TestClient(app) as client:
            first = client.post('/api/auth/register', json={'username':'alice', 'password':'synthetic-password-123'}).json()
            settings = client.get('/api/skills').json()
            self.assertFalse(settings['settings']['manual_review'])
            client.put('/api/skills/settings', json={'manual_review': True}, headers={'x-csrf-token': first['csrf_token']})
            second = client.post('/api/auth/register', json={'username':'bobby', 'password':'synthetic-password-123'}).json()
            self.assertFalse(client.get('/api/skills').json()['settings']['manual_review'])
            response = client.post('/api/skills/candidates/' + 'a'*20 + '/decision', json={'decision':'approve'}, headers={'x-csrf-token': second['csrf_token']})
            self.assertEqual(404, response.status_code)
            self.assertEqual(403, client.get('/api/admin/skills/candidates').status_code)

if __name__ == '__main__':
    unittest.main()
