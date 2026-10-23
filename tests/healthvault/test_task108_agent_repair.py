import contextlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.ui import quiet_agent_ui
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, HealthVaultAgentRuntimeError, build_healthvault_agent_runtime
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext
from healthvault.web.answers import answer_with_model
from tests.healthvault.test_task086_completion_contract import tool
from tests.healthvault.test_task102_answer_flow import QUESTION, SOURCES, SHORT, FULL, review


class AgentRepairTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / '.task_outputs/task-108-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(quiet_agent_ui())
        self.stack.enter_context(patch.object(agent_module.openai, 'AsyncOpenAI', return_value=MagicMock()))
        self.stack.enter_context(patch.object(agent_module.anthropic, 'AsyncAnthropic', return_value=MagicMock()))
        self.stack.enter_context(patch.dict('os.environ', {
            'HEALTHVAULT_UI_DB_PATH': str(self.root / 'import.sqlite3'), 'HEALTHVAULT_KNOWLEDGE_MODE': 'lexical'}))
        self.model = HealthVaultModelConfig('openai-compatible', 'fake', 'https://invalid.local/v1', 'fake')

    def script(self, runtime, messages):
        pending = iter(messages)
        async def response():
            message = next(pending)
            runtime.agent._emit_text(message.get('content', ''))
            return {'choices': [{'message': message}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        runtime.agent._call_openai_stream = AsyncMock(side_effect=response)

    def endpoint(self, *, initial=SHORT, repair='full', failure=None, enabled=True):
        from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
        from healthvault.web.api import AgentMessageInput, create_app
        controller = SimpleNamespace(record_repository=SQLiteHealthRecordRepository(self.root / 'web.sqlite3'),
                                     knowledge_release_root=self.root / 'releases')
        self.drafts, self.reviews, self.runtimes = [], [], []

        def generate(q, mode, evidence):
            self.drafts.append(copy.deepcopy(evidence))
            claims = [{'text': 'A1C用于监测。', 'citation_id': 'HVK:test:a', 'point_ids': ['P1']}]
            if repair == 'full':
                claims.append({'text': 'A1C反映三个月。', 'citation_id': 'HVK:test:a', 'point_ids': ['P2']})
            draft = {'status': 'answered', 'claims': claims, 'point_evidence': [
                {'point_id': 'P1', 'evidence_ids': ['S1-detail-1']},
                {'point_id': 'P2', 'evidence_ids': ['S1-detail-2']}]}
            return answer_with_model(q, mode, evidence, query=lambda *_a, **_k: json.dumps(draft))

        def verify(_prompt, data, **_kwargs):
            self.reviews.append(copy.deepcopy(data))
            checked = review(data, missing=failure in {'missing', 'mixed'},
                             unsupported=failure == 'mixed' or failure == 'unsupported' and len(self.reviews) == 1)
            if failure == 'protocol':
                checked['protocol'] = 'fact_span_v1'
            return checked

        app = create_app(controller, model_answerer=generate, model_answer_verifier=verify,
                         auto_plan_suggestions=False, answer_repair_enabled=enabled,
                         skill_root=self.root / 'skills', skill_evolution_root=self.root / 'evolution')
        endpoint = next(r.endpoint for r in app.routes if r.path.endswith('/agent/tasks/{task_id}/messages'))
        payload = {'status': 'ok', 'available_citation_ids': ['HVK:test:a'], 'results': [
            {'citation_id': s['citation_id'], 'title': s['title'], 'excerpt': s['detail']} for s in SOURCES]}

        def build(config):
            runtime = build_healthvault_agent_runtime(config)
            self.runtimes.append(runtime)
            runtime.agent._tool_executor = AsyncMock(return_value=json.dumps(payload))
            self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C'}),
                                  {'role': 'assistant', 'content': initial}])
            return runtime

        self.stack.enter_context(patch('healthvault.web.api.configured_web_model', return_value=self.model))
        self.stack.enter_context(patch('healthvault.agent_runtime.build_healthvault_agent_runtime', side_effect=build))
        return app, lambda: endpoint('alice', 'repair', AgentMessageInput(
            prompt=QUESTION, expected_version=0, model_consent=True))

    async def test_web_omission_repair_uses_current_evidence_and_saves_only_verified_answer(self):
        app, run = self.endpoint()
        reply = await run()
        expected = FULL.replace('\n', '\n\n')
        self.assertEqual(expected, reply['answer'])
        self.assertTrue(reply['verification']['passed'])
        self.assertEqual(1, len(self.drafts))
        self.assertEqual(2, len(self.reviews))
        feedback = self.drafts[0]['answer_repair']
        self.assertEqual(SHORT, feedback['previous_answer'])
        self.assertEqual('P2', feedback['omitted_points'][0]['point_id'])
        self.assertEqual(QUESTION, self.drafts[0]['question_plan']['original_question'])
        self.assertEqual(SOURCES[0]['detail'], self.drafts[0]['sources'][0]['detail'])
        self.assertEqual(self.reviews[0]['required_points'], self.reviews[1]['required_points'])
        runtime = self.runtimes[0]
        runtime.agent._tool_executor.assert_awaited_once()
        self.assertEqual(2, runtime.agent._call_openai_stream.await_count)
        saved = app.state.agent_tasks.get('alice', 'repair')
        self.assertEqual(expected, saved['dialogue'][-1]['text'])
        self.assertEqual(expected, saved['messages'][-1]['content'])
        self.assertNotIn(SHORT, [m.get('content') for m in saved['messages']])
        self.assertTrue(reply['measurement']['features']['answer_repair'])
        self.assertEqual(1, reply['measurement']['answer_repair_attempts'])
        self.assertIn('repair', reply['measurement']['usage_scope'])

    async def test_unsupported_claim_gets_same_single_repair_and_full_review(self):
        _app, run = self.endpoint(initial=FULL, failure='unsupported')
        reply = await run()
        self.assertTrue(reply['verification']['passed'])
        self.assertEqual(['F1', 'F2'], self.drafts[0]['answer_repair']['unsupported_claim_ids'])
        self.assertEqual(2, len(self.reviews))
        self.runtimes[0].agent._tool_executor.assert_awaited_once()

    async def assert_failed_without_history(self, run, app):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as failure:
            await run()
        self.assertEqual(502, failure.exception.status_code)
        self.assertIsNone(app.state.agent_tasks.get('alice', 'repair'))
        self.runtimes[0].agent._tool_executor.assert_awaited_once()

    async def test_failed_second_answer_stops_without_third_generation_or_saved_draft(self):
        app, run = self.endpoint(repair='short')
        await self.assert_failed_without_history(run, app)
        self.assertEqual(1, len(self.drafts))
        self.assertEqual(2, len(self.reviews))

    async def test_invalid_verifier_protocol_does_not_trigger_generation(self):
        app, run = self.endpoint(failure='protocol')
        await self.assert_failed_without_history(run, app)
        self.assertEqual([], self.drafts)
        self.assertEqual(1, len(self.reviews))

    async def test_missing_evidence_even_with_other_errors_cannot_be_fixed_by_rewriting(self):
        for missing in ('missing', 'mixed'):
            with self.subTest(missing=missing):
                app, run = self.endpoint(failure=missing)
                await self.assert_failed_without_history(run, app)
                self.assertEqual([], self.drafts)
                self.assertEqual(1, len(self.reviews))

    async def test_disabled_repair_keeps_verification_and_fails_without_saving(self):
        app, run = self.endpoint(enabled=False)
        await self.assert_failed_without_history(run, app)
        self.assertEqual([], self.drafts)
        self.assertEqual(1, len(self.reviews))
        self.assertFalse(self.runtimes[0].manifest['answer_repair_enabled'])
        self.assertTrue(self.runtimes[0].manifest['answer_verification_enabled'])

    async def test_no_current_tool_evidence_never_reaches_repair_or_semantic_service(self):
        verifier, repairer = AsyncMock(), AsyncMock()
        runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
            context=RuntimeContext('alice', 'req'), database_path=self.root / 'records.sqlite3',
            model_config=self.model, answer_verifier=verifier, answer_repairer=repairer))
        self.script(runtime, [{'role': 'assistant', 'content': SHORT}])
        with self.assertRaises(HealthVaultAgentRuntimeError):
            await runtime.run('你好')
        verifier.assert_not_awaited()
        repairer.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
