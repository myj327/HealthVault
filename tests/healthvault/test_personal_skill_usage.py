import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from agents import agent as agent_module
from agents.ui import quiet_agent_ui

from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.evolution.personal import PersonalSkillManager
from healthvault.evolution.trace import TaskTraceRecorder
from healthvault.evolution.verifier import CandidateVerifier
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app
from tests.healthvault.test_skill_review_loop import ACTIVE_SKILL, signal_context
from tests.healthvault.test_healthvault_agent_runtime import seed_profile
from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository


class PersonalSkillUsageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / ".bear" / "skills"
        self.active = self.project / "HealthVault-Task-Execution" / "SKILL.md"
        self.active.parent.mkdir(parents=True)
        self.active.write_text(ACTIVE_SKILL, encoding="utf-8")
        self.manager = PersonalSkillManager(self.project, self.root / "evolution", AsyncMock())
        self.service = self.manager.service("alice")
        self.baseline = hashlib.sha256(self.service.store.active_path.read_bytes()).hexdigest()
        candidate = ACTIVE_SKILL.replace("1. Identify", "1a. Call query_health_timeline for a timeline request.\n1. Identify")
        verification = CandidateVerifier("HealthVault-Task-Execution").verify_content(candidate, action="evolve", active_content=ACTIVE_SKILL)
        release = self.service.store.publisher.publish_content(candidate, action="evolve", verification=verification, approved=True)
        self.assertEqual("published", release.status)
        self.version = release.active_sha256

    def tearDown(self):
        self.temp.cleanup()

    async def test_default_template_is_not_a_personal_update(self):
        self.assertFalse(self.manager.usage('new-owner')['has_personal_update'])
        self.assertTrue(self.manager.usage('alice')['has_personal_update'])

    async def trace(self, task, *, passed=True, failure=None, result_status="succeeded", digest=None):
        async def tool(*args):
            return json.dumps({"status": "ok", "failure_code": failure})
        recorder = TaskTraceRecorder(self.service.root, request_id=f"request-{task}-{len(list((self.service.root / 'traces').glob('*.json')))}", task_id=task,
            skill={"name": "HealthVault-Task-Execution", "sha256": digest or self.version}, executor=tool, signal_context=signal_context("passed" if passed else "failed"))
        await recorder.execute("query_health_timeline", {"ignored_secret": "synthetic patient original must not enter usage"})
        return recorder.finish(result_status, failure_code=failure)

    async def test_independent_task_window_is_redacted_and_deduplicated(self):
        path = await self.trace("same-task")
        first = self.manager.record_usage("alice", path, injected=True)
        self.assertEqual("watch", first["evaluation"]["decision"])
        self.assertEqual("already_recorded", self.manager.record_usage("alice", await self.trace("same-task"), injected=True)["status"])
        for index in range(4):
            last = self.manager.record_usage("alice", await self.trace(f"independent-{index}"), injected=True)
        self.assertEqual("keep", last["evaluation"]["decision"])
        self.assertEqual(5, last["evaluation"]["effective_samples"])
        for path in (self.service.root / "web-usage").rglob("*.json"):
            self.assertNotIn("synthetic patient", path.read_text(encoding="utf-8"))
        self.assertEqual([], self.service.store.list())

    async def test_general_failure_watches_then_rolls_back_after_ten_bad_tasks(self):
        for index in range(10):
            result = self.manager.record_usage("alice", await self.trace(f"failed-{index}", passed=False), injected=True)
            self.assertEqual("watch" if index < 9 else "disable", result["evaluation"]["decision"])
        self.assertEqual(["disable", "rollback", "archive"], result["lifecycle"]["actions"])
        self.assertEqual(self.baseline, self.manager.usage("alice")["active_sha256"])
        self.assertFalse(self.manager.lifecycle("alice").is_injectable(self.version))
        self.assertEqual(ACTIVE_SKILL, self.active.read_text(encoding="utf-8"))

    async def test_safety_failure_immediately_stops_personal_version_and_restart_preserves_state(self):
        result = self.manager.record_usage("alice", await self.trace("safety", passed=False, failure="safety_write_violation"), injected=True)
        self.assertEqual("disable", result["evaluation"]["decision"])
        restarted = PersonalSkillManager(self.project, self.root / "evolution", AsyncMock())
        self.assertFalse(restarted.lifecycle("alice").is_injectable(self.version))
        # Even a stale physical copy cannot bypass the persistent injection gate.
        self.service.store.active_path.write_bytes((self.service.root / "versions" / self.version / "SKILL.md").read_bytes())
        self.assertIsNone(restarted.injectable_root("alice"))

    async def test_transport_failures_and_uninjected_versions_are_excluded(self):
        result = self.manager.record_usage("alice", await self.trace("transport", passed=False, failure="APIConnectionError", result_status="failed"), injected=True)
        self.assertEqual("excluded_runtime_failure", result["status"])
        result = self.manager.record_usage("alice", await self.trace("not-injected", passed=False), injected=False)
        self.assertEqual(0, result["evaluation"]["effective_samples"])
        self.assertIsNone(result["lifecycle"])
        self.assertEqual(self.version, self.manager.usage("alice")["active_sha256"])

    async def test_old_version_safety_result_does_not_roll_back_a_new_active_version(self):
        result = self.manager.record_usage("alice", await self.trace("old-safety", digest=self.baseline, passed=False, failure="safety_policy_violation"), injected=True)
        self.assertEqual("inactive_version_disabled", result["lifecycle"]["status"])
        self.assertEqual(self.version, self.manager.usage("alice")["active_sha256"])

    async def denied_runtime(self,task,*,model_failure=False):
        database = self.root/f'{task}.sqlite3'
        seed_profile(SQLiteHealthRecordRepository(database),'alice')
        with patch.object(agent_module.openai,'AsyncOpenAI',return_value=MagicMock()),quiet_agent_ui():
            runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                context=RuntimeContext('alice',f'request-{task}'),database_path=database,
                model_config=HealthVaultModelConfig('openai-compatible','synthetic-model','https://invalid.local/v1','synthetic-key'),
                skill_root=self.project,skill_names=('HealthVault-Task-Execution',),
                personal_skill_root=self.manager.injectable_root('alice'),evolution_root=self.service.root,
                task_id=task,signal_context=signal_context('passed')))
            original_outcome = runtime.agent._record_tool_outcome
            calls = 0
            async def fake_model():
                nonlocal calls
                calls += 1
                if model_failure and calls==2:
                    raise RuntimeError('synthetic service outage')
                if calls<=2:
                    name = 'confirm_health_records' if calls==1 else 'query_health_timeline'
                    arguments = {'report_text':'synthetic-private-7.2'} if calls==1 else {}
                    message = {'role':'assistant','content':None,'tool_calls':[{'id':f'call-{calls}','type':'function',
                        'function':{'name':name,'arguments':json.dumps(arguments)}}]}
                else:
                    payload = json.loads(next(item['content'] for item in reversed(runtime.agent._openai_messages) if item['role']=='tool'))
                    text = '已查询报告时间线。 ['+payload['available_citation_ids'][0]+']'
                    runtime.agent._emit_text(text)
                    message = {'role':'assistant','content':text,'tool_calls':None}
                return {'choices':[{'message':message}],'usage':{'prompt_tokens':10,'completion_tokens':5}}
            runtime.agent._call_openai_stream = fake_model
            if model_failure:
                with self.assertRaises(RuntimeError): await runtime.run('查询报告时间线。')
            else:
                await runtime.run('查询报告时间线。')
            self.assertEqual(original_outcome,runtime.agent._record_tool_outcome)
        self.assertTrue(runtime.manifest['personal_skill_injected'])
        path = runtime.trace_recorder.trace_path
        self.assertNotIn('synthetic-private-7.2',path.read_text(encoding='utf-8'))
        trace = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual('safety_write_violation',trace['failure_code'])
        self.assertEqual('failed',trace['result_status'])
        self.assertEqual('failed',trace['signal_context']['contract_checks']['safety'])
        denied = next(item for item in trace['tool_calls'] if item['tool_name']=='confirm_health_records')
        self.assertEqual('runtime_permission_denied',denied['validator_code'])
        return path

    async def test_real_permission_denial_escalates_previous_success_and_disables_version(self):
        self.manager.record_usage('alice',await self.trace('same-policy-task'),injected=True)
        path = await self.denied_runtime('same-policy-task')
        result = self.manager.record_usage('alice',path,injected=True)
        self.assertEqual('recorded',result['status'])
        self.assertEqual('disable',result['evaluation']['decision'])
        self.assertEqual(1,result['evaluation']['effective_samples'])
        self.assertEqual(1,result['evaluation']['safety_hard_failures'])
        self.assertEqual(self.baseline,self.manager.usage('alice')['active_sha256'])
        restarted = PersonalSkillManager(self.project,self.root/'evolution',AsyncMock())
        self.assertFalse(restarted.lifecycle('alice').is_injectable(self.version))
        self.assertEqual('already_recorded',restarted.record_usage('alice',path,injected=True)['status'])
        for item in (self.service.root/'web-usage').rglob('*.json'):
            self.assertNotIn('synthetic-private-7.2',item.read_text(encoding='utf-8'))

    async def test_permission_denial_remains_safety_failure_after_model_outage(self):
        path = await self.denied_runtime('denied-then-outage',model_failure=True)
        result = self.manager.record_usage('alice',path,injected=True)
        self.assertEqual('recorded',result['status'])
        self.assertEqual('disable',result['evaluation']['decision'])
        self.assertFalse(self.manager.lifecycle('alice').is_injectable(self.version))

    async def test_web_manual_disable_is_private_and_runtime_falls_back_to_project_guidance(self):
        app = create_app(HealthVaultUIController(self.root / "web.sqlite3", knowledge_release_root=self.root / "releases"), skill_root=self.project, skill_evolution_root=self.root / "web-evolution", skill_patch_side_query=AsyncMock())
        with TestClient(app) as client:
            account = client.post("/api/auth/register", json={"username": "usage-owner", "password": "synthetic-password-123"}).json()
            client.headers.update({"X-CSRF-Token": account["csrf_token"]})
            self.assertTrue(client.get("/api/skills").json()["usage"]["injectable"])
            self.assertEqual("disabled", client.post("/api/skills/disable").json()["status"])
            self.assertFalse(client.get("/api/skills").json()["usage"]["injectable"])
            self.assertIn('主动停用', client.get('/api/skills').json()['usage']['explanation_zh'])
            profile = account["profile_id"]
            self.assertIsNone(app.state.personal_skills.injectable_root(profile))
            with patch("healthvault.agent_runtime.Agent") as agent:
                runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                    context=RuntimeContext(profile, "disabled-request"), database_path=self.root / "web.sqlite3",
                    model_config=HealthVaultModelConfig("openai-compatible", "synthetic", "https://invalid.local/v1", "synthetic-key"),
                    skill_root=self.project, skill_names=("HealthVault-Task-Execution",), personal_skill_root=app.state.personal_skills.injectable_root(profile)))
                self.assertFalse(runtime.manifest["personal_skill_injected"])
                self.assertIn("Never infer patient facts", agent.call_args.kwargs["custom_system_prompt"])
            client.post("/api/auth/register", json={"username": "usage-other", "password": "synthetic-password-123"})
            self.assertTrue(client.get("/api/skills").json()["usage"]["injectable"])
            self.assertEqual([], client.get("/api/skills").json()["usage"]["evaluations"])


if __name__ == "__main__":
    unittest.main()
