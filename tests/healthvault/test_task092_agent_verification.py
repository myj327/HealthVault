import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.ui import quiet_agent_ui
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, HealthVaultAgentRuntimeError, build_healthvault_agent_runtime
from healthvault.application.tool_evidence import merge_tool_evidence, tool_evidence
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext
from healthvault.web.answer_validation import verify_answer
from healthvault.web.question_plan import create_question_plan
from tests.healthvault.test_skill_review_loop import ACTIVE_SKILL
from tests.healthvault.test_task086_completion_contract import final, tool
from tests.healthvault.test_task091_question_plan import checked


CITATION = 'HVK:synthetic:purpose'
PAYLOAD = {'status': 'ok', 'available_citation_ids': [CITATION], 'results': [
    {'citation_id': CITATION, 'title': 'A1C', 'excerpt': 'A1C helps monitoring.'}]}


class AgentVerificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / '.task_outputs/task-092-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(quiet_agent_ui())
        self.stack.enter_context(patch.object(agent_module.openai, 'AsyncOpenAI', return_value=MagicMock()))
        self.stack.enter_context(patch.object(agent_module.anthropic, 'AsyncAnthropic', return_value=MagicMock()))
        self.question = 'A1C用途？'
        self.plan = create_question_plan(self.question)

    def runtime(self, *, tracing=False, verifier=None):
        extra = {}
        if tracing:
            active = self.root / '.bear/skills/HealthVault-Task-Execution/SKILL.md'
            active.parent.mkdir(parents=True, exist_ok=True)
            active.write_text(ACTIVE_SKILL, encoding='utf-8')
            extra.update(skill_root=active.parents[1], skill_names=('HealthVault-Task-Execution',),
                         evolution_root=self.root / 'evolution', signal_context={
                             'intent': 'knowledge_lookup', 'preconditions': ['read-only'],
                             'expected_outcome': 'validated answer', 'validator_status': 'failed',
                             'contract_checks': {'status': 'failed'}})
        async def assess(text, sources):
            return verify_answer(self.question, text, sources, query=lambda _p, data, **_k: checked(
                data, {fact['claim_id']: ['S1-detail-1'] for fact in data['facts']}),
                                 audit_coverage=True, required_points=self.plan['points'])
        runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
            context=RuntimeContext('alice', 'req'), database_path=self.root / 'records.sqlite3',
            model_config=HealthVaultModelConfig('openai-compatible', 'fake', 'https://invalid.local/v1', 'fake'),
            answer_verifier=verifier or assess, question_plan=self.plan, **extra))
        executor = AsyncMock(return_value=json.dumps(PAYLOAD))
        if tracing:
            runtime.trace_recorder._executor = executor
        else:
            runtime.agent._tool_executor = executor
        return runtime, executor

    def script(self, runtime, messages):
        pending = iter(messages)
        async def response():
            message = next(pending)
            runtime.agent._emit_text(message.get('content', ''))
            return {'choices': [{'message': message}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        call = AsyncMock(side_effect=response)
        runtime.agent._call_openai_stream = call
        return call

    async def test_supported_natural_answer_is_verified_before_success_trace(self):
        runtime, executor = self.runtime(tracing=True)
        answer = f'A1C可用于监测。 [{CITATION}]'
        self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C'}), {'role': 'assistant', 'content': answer}])
        reply = await runtime.run(self.question)
        self.assertEqual(answer, reply.text)
        self.assertTrue(reply.verification['passed'])
        trace = json.loads(runtime.trace_recorder.trace_path.read_text(encoding='utf-8'))
        self.assertEqual('passed', trace['signal_context']['validator_status'])
        executor.assert_awaited_once()

    async def test_partial_semantic_support_fails_and_trace_contains_no_answer_or_patient_values(self):
        runtime, _executor = self.runtime(tracing=True)
        answer = f'A1C通过血液检测监测，结果42.7。 [{CITATION}]'
        original_anthropic = runtime.agent._call_anthropic_stream
        self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C'}), {'role': 'assistant', 'content': answer}])
        with self.assertRaises(HealthVaultAgentRuntimeError):
            await runtime.run(self.question)
        text = runtime.trace_recorder.trace_path.read_text(encoding='utf-8')
        trace = json.loads(text)
        self.assertEqual('failed', trace['result_status'])
        self.assertEqual('answer_verification_failed', trace['failure_code'])
        self.assertEqual({'input': 20, 'output': 10}, trace['tokens'])
        self.assertNotIn('42.7', text)
        self.assertNotIn(answer, text)
        self.assertEqual(original_anthropic, runtime.agent._call_anthropic_stream)

    async def test_missing_or_omitted_point_and_invalid_verifier_cannot_return_answer(self):
        for response in ({'passed': False, 'omitted_points': ['P1']},
                         {'passed': False, 'missing_evidence_points': ['P1']},
                         {'passed': False, 'kind': 'invalid_verifier_output'}):
            with self.subTest(response=response):
                runtime, _executor = self.runtime(verifier=AsyncMock(return_value=response))
                self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C'}),
                                      {'role': 'assistant', 'content': f'A1C可用于监测。 [{CITATION}]'}])
                with self.assertRaises(HealthVaultAgentRuntimeError):
                    await runtime.run(self.question)

    async def test_previous_request_sources_cannot_support_current_reply(self):
        reviewer = AsyncMock(return_value={'passed': True})
        runtime, _executor = self.runtime(verifier=reviewer)
        answer = f'A1C可用于监测。 [{CITATION}]'
        self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C'}), {'role': 'assistant', 'content': answer}])
        await runtime.run(self.question)
        self.script(runtime, [{'role': 'assistant', 'content': answer}])
        with self.assertRaises(HealthVaultAgentRuntimeError):
            await runtime.run(self.question)
        self.assertEqual(1, reviewer.await_count)

    async def test_metadata_only_final_uses_raw_contract_without_semantic_request(self):
        reviewer = AsyncMock(side_effect=AssertionError('No factual answer to review'))
        runtime, _executor = self.runtime(verifier=reviewer)
        accepted = final(citations=(CITATION,))
        self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C'}), {'role': 'assistant', 'content': accepted}])
        reply = await runtime.run(self.question)
        self.assertEqual(json.loads(accepted), json.loads(reply.text))
        reviewer.assert_not_awaited()
        self.assertIsNone(reply.verification)

    async def test_source_adapters_and_conflict_keep_original_source_identity(self):
        evidence = {'quote': '血糖 5.2 mmol/L', 'line_start': 1, 'line_end': 1}
        timeline = {'status': 'ok', 'available_citation_ids': ['REC:r:glucose'], 'timeline': [
            {'report_id': 'r', 'report_date': '2026-10-01', 'indicators': [{'code': 'glucose', 'evidence': evidence}]}]}
        point = {'report_id': 'r', 'report_date': '2026-10-01', 'evidence': evidence}
        comparison = {'status': 'ok', 'available_citation_ids': ['REC:r:glucose'], 'comparisons': [
            {'code': 'glucose', 'earliest': point, 'latest': point}]}
        brief = {'status': 'ready', 'available_citation_ids': ['REC:r:glucose'], 'record_citations': [
            {'citation_id': 'REC:r:glucose', 'report_id': 'r', 'report_date': '2026-10-01', 'indicator_code': 'glucose', 'evidence': evidence}]}
        sources = tool_evidence('query_health_timeline', timeline)
        self.assertEqual(sources[0], tool_evidence('compare_health_indicators', comparison)[0])
        self.assertEqual(sources, tool_evidence('generate_visit_brief', brief))
        with self.assertRaises(ValueError):
            merge_tool_evidence(sources, [{**sources[0], 'detail': 'Another report.'}])
        runtime, executor = self.runtime()
        executor.side_effect = [json.dumps(PAYLOAD), json.dumps({'status': 'ok', 'available_citation_ids': [CITATION],
            'knowledge_citations': [{**PAYLOAD['results'][0], 'excerpt': 'Changed.'}]})]
        call = self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C'}), tool('generate_visit_brief', {'profile_id': 'alice'})])
        self.assertEqual('rejected', json.loads((await runtime.run(self.question)).text)['final_status'])
        self.assertEqual(2, call.await_count)

    async def test_web_wires_actual_runtime_and_does_not_save_failed_factual_reply(self):
        from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
        from healthvault.web.api import AgentMessageInput, create_app
        controller = SimpleNamespace(record_repository=SQLiteHealthRecordRepository(self.root / 'web.sqlite3'),
                                     knowledge_release_root=self.root / 'releases')
        for supported in (True, False):
            def review(_prompt, data, **_kwargs):
                result = checked(data, {'F1': ['S1-detail-1']})
                result['claims'][0]['checks']['relation'] = supported
                return result
            app = create_app(controller, lambda *_a: '', model_answer_verifier=review,
                             model_question_planner=lambda *_a, **_k: {'protocol': 'question_plan_v1', 'additional_points': []},
                             skill_root=self.root / 'webskills', skill_evolution_root=self.root / 'webevolution')
            endpoint = next(route.endpoint for route in app.routes if route.path.endswith('/agent/tasks/{task_id}/messages'))
            runtimes = []
            def build(config):
                runtime = build_healthvault_agent_runtime(config)
                runtimes.append(runtime)
                runtime.agent._tool_executor = AsyncMock(return_value=json.dumps(PAYLOAD))
                self.script(runtime, [tool('search_health_knowledge', {'query': 'A1C'}),
                                      {'role': 'assistant', 'content': f'A1C可用于监测。 [{CITATION}]'}])
                return runtime
            with patch('healthvault.web.api.configured_web_model', return_value=HealthVaultModelConfig(
                    'openai-compatible', 'fake', 'https://invalid.local/v1', 'fake')), \
                 patch('healthvault.agent_runtime.build_healthvault_agent_runtime', side_effect=build):
                if supported:
                    reply = await endpoint('alice', 'accepted', AgentMessageInput(prompt=self.question, expected_version=0, model_consent=True))
                    self.assertTrue(reply['verification']['passed'])
                else:
                    from fastapi import HTTPException
                    with self.assertRaises(HTTPException):
                        await endpoint('alice', 'rejected', AgentMessageInput(prompt=self.question, expected_version=0, model_consent=True))
                    self.assertIsNone(app.state.agent_tasks.get('alice', 'rejected'))
            self.assertTrue(runtimes[0].manifest['answer_verification_enabled'])


if __name__ == '__main__':
    unittest.main()
