import copy
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from healthvault.agent_runtime import HealthVaultModelOnlyRuntime
from healthvault.evaluation.measurement import paired_skill_metrics, repair_metrics, tool_execution_metrics
from healthvault.evolution.execution_probe import evaluate_execution, valid_execution_gate
from healthvault.evolution.workflow_eval import CASES, evaluate_workflow_plans, evaluate_workflow_probes, score_plans
from healthvault.runtime_context import RuntimeContext
from tests.healthvault import test_personal_skills as personal_fixtures
from tests.healthvault import test_task102_answer_flow as answer_fixtures


def good_model(_prompt, data):
    if data.get('phase') == 'execution_tool':
        return {'tool_name': 'compare_health_indicators', 'arguments': {}} if data['case_id'] == 'exec-compare' else {
            'tool_name': 'search_health_knowledge', 'arguments': {'query': 'A1C'}}
    if data.get('phase') == 'execution_final':
        citations = data['tool_result'].get('available_citation_ids', [])
        return {'final_status': 'ok', 'citations': citations[:2] if data['case_id'] == 'exec-compare' else citations[:1], 'refusal': False}
    return {'plans': [{'case_id': case[0], 'tool_path': [case[3]] if case[3] else [], 'refuse': case[4]} for case in CASES]}


class SkillExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_indicator_filter_is_valid_but_unrequested_year_filter_is_not(self):
        for arguments, expected in (({'indicator_code': 'fasting_glucose'}, True),
                                    ({'start_year': 2024, 'end_year': 2025}, False)):
            def model(prompt, data):
                result = good_model(prompt, data)
                if data.get('phase') == 'execution_tool' and data['case_id'] == 'exec-compare':
                    result['arguments'] = arguments
                return result
            result = await evaluate_execution('baseline', 'candidate', model)
            self.assertEqual(result['passed'], expected)

    async def test_derived_reference_does_not_replace_requested_original_endpoints(self):
        active = personal_fixtures.ACTIVE_SKILL
        valid = await evaluate_execution(active, active, good_model)
        self.assertTrue(valid['passed'])
        def extra(prompt, data):
            result = good_model(prompt, data)
            if data.get('phase') == 'execution_final' and data['case_id'] == 'exec-compare':
                result['citations'] = data['tool_result']['available_citation_ids']
            return result
        self.assertFalse((await evaluate_execution(active, active, extra))['passed'])

    async def test_all_tool_plan_and_duplicate_tool_are_not_valid_routes(self):
        plan = good_model('', {})
        all_tools = [c[3] for c in CASES if c[3]]
        for item in plan['plans']:
            if not item['refuse']:
                item['tool_path'] = all_tools
        self.assertEqual(2, score_plans(plan)['passed'])
        duplicate = good_model('', {})
        duplicate['plans'][0]['tool_path'] *= 2
        self.assertEqual(5, score_plans(duplicate)['passed'])

    async def test_two_arms_really_execute_native_tools_with_equal_fixtures(self):
        requests = []
        def model(prompt, data):
            requests.append(copy.deepcopy(data))
            return good_model(prompt, data)
        result = await evaluate_workflow_probes('baseline', 'candidate', model)
        self.assertTrue(result['passed'], result)
        self.assertEqual(10, result['logical_model_calls'])
        self.assertEqual(10, len(requests))
        self.assertFalse(result['untouched_held_out'])
        self.assertTrue(valid_execution_gate(result, hashlib.sha256(b'baseline').hexdigest(), hashlib.sha256(b'candidate').hexdigest()))
        execution = result['execution_regression']
        for arm in ('baseline', 'candidate'):
            self.assertEqual(2, execution[arm]['passed'])
            for row in execution[arm]['cases']:
                self.assertTrue(row['executed'])
                self.assertTrue(row['database_unchanged'])
                self.assertEqual('ok', row['execution_status'])
                self.assertFalse(row['personal_skill_injected'])
        finals = [r for r in requests if r.get('phase') == 'execution_final']
        for index in range(2):
            self.assertEqual(finals[index]['tool_result'], finals[index + 2]['tool_result'])

    async def test_wrong_parameters_and_empty_retrieval_fail_without_inventing_success(self):
        for bad_case, arguments in (('exec-compare', {'profile_id': 'other'}),
                                    ('exec-knowledge', {'query': 'unfindable-zzzxqv'})):
            def model(prompt, data):
                result = good_model(prompt, data)
                if data.get('phase') == 'execution_tool' and data['case_id'] == bad_case:
                    result['arguments'] = arguments
                return result
            result = await evaluate_execution('baseline', 'candidate', model)
            self.assertFalse(result['passed'])
            row = next(r for r in result['candidate']['cases'] if r['case_id'] == bad_case)
            self.assertTrue(row['executed'])
            if bad_case == 'exec-knowledge':
                self.assertEqual('ok', row['execution_status'])
                self.assertTrue(row['empty_search'])
                self.assertFalse(row['arguments_passed'])
            else:
                self.assertEqual('failed', row['execution_status'])

    async def test_missing_fake_and_trailing_json_finals_are_kept_as_failures(self):
        for bad_final in ({'final_status': 'ok', 'citations': [], 'refusal': False},
                          {'final_status': 'ok', 'citations': ['HVK:invented'], 'refusal': False},
                          '{"final_status":"ok","citations":[],"refusal":false} extra'):
            def model(prompt, data):
                return bad_final if data.get('phase') == 'execution_final' else good_model(prompt, data)
            result = await evaluate_execution('baseline', 'candidate', model)
            self.assertFalse(result['passed'])
            self.assertEqual(0, result['candidate']['passed'])

    async def test_plan_only_gate_cannot_publish_a_personal_candidate(self):
        fixture = personal_fixtures.PersonalSkillTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.manager.evaluator = lambda active, candidate: evaluate_workflow_plans(active, candidate, good_model)
        await fixture.traces('alice')
        record = await fixture.manager.consider('alice')
        self.assertEqual('gate_rejected', record['status'])
        self.assertTrue(record['evaluation']['workflow_probe']['passed'])
        with self.assertRaises(ValueError):
            fixture.manager.service('alice').store.decide(record['candidate_id'], 'approve', 'test')
        self.assertEqual(personal_fixtures.ACTIVE_SKILL, fixture.manager.service('alice').store.active_path.read_text(encoding='utf-8'))

    async def test_publication_rejects_stale_version_or_execution_bindings(self):
        result = await evaluate_workflow_probes('baseline', 'candidate', good_model)
        active_sha, candidate_sha = result['active_sha256'], result['candidate_sha256']
        for target, key in ((result, 'candidate_sha256'), (result['execution_regression'], 'fixture_sha256'),
                            (result['execution_regression'], 'protocol')):
            previous = target[key]
            target[key] = 'stale'
            self.assertFalse(valid_execution_gate(result, active_sha, candidate_sha))
            target[key] = previous
        result['execution_regression']['candidate']['cases'][0]['executed'] = False
        self.assertFalse(valid_execution_gate(result, active_sha, candidate_sha))

    async def test_model_only_dedicated_qa_and_native_entries_record_real_features(self):
        runtime = HealthVaultModelOnlyRuntime(RuntimeContext('alice', 'test'), SimpleNamespace(
            run_once=AsyncMock(return_value={'text': 'synthetic', 'tokens': {'input': 2, 'output': 3}})))
        measurement = (await runtime.run('question')).to_dict()['measurement']
        self.assertEqual('model_only_task', measurement['entrypoint'])
        self.assertFalse(measurement['personal_skill_injected'])
        self.assertEqual([], measurement['skill_versions'])
        fixture = answer_fixtures.AnswerFlowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        run, _app, _ = fixture.endpoint(lambda *_a: answer_fixtures.FULL,
            lambda _p, data, **_kw: answer_fixtures.review(data), auto_plan_suggestions=False, answer_repair_enabled=False)
        measurement = run()['measurement']
        self.assertEqual('knowledge_qa', measurement['entrypoint'])
        self.assertFalse(measurement['personal_skill_injected'])
        self.assertTrue(measurement['features']['answer_verification'])
        self.assertFalse(measurement['features']['answer_repair'])
        self.assertEqual(1, measurement['generation_attempts'])
        # The actual native entry is covered by the existing private-injection runtime fixture.
        from tests.healthvault import test_task103_skill_failures as failure_fixtures
        native = failure_fixtures.SkillFailureTests()
        native.setUp()
        self.addCleanup(native.temp.cleanup)
        from agents import agent as agent_module
        from unittest.mock import MagicMock
        with patch.object(agent_module.openai, 'AsyncOpenAI', return_value=MagicMock()):
            actual = native.runtime()
            actual.manifest['workflow_intent'] = 'general_read_only_workflow'
            actual.agent.run_once = AsyncMock(return_value={'text': 'synthetic', 'tokens': {'input': 2, 'output': 3}})
            output = await actual.run('synthetic plain request')
        self.assertTrue(output.measurement['personal_skill_injected'])
        self.assertEqual('native_agent', output.measurement['entrypoint'])
        self.assertFalse(output.measurement['features']['answer_repair'])
        self.assertIn(native.version, [v['sha256'] for v in output.measurement['skill_versions']])

    async def test_per_call_goal_repair_and_paired_benefit_are_distinct(self):
        tools = tool_execution_metrics([{'name': 'search_health_knowledge', 'execution_status': 'ok', 'result_status': 'empty'},
                                        {'name': 'query_health_timeline', 'execution_status': 'failed'}])
        self.assertEqual(0.5, tools['tool_execution_success_rate'])
        self.assertEqual(1, tools['empty_searches'])
        repair = repair_metrics([{'passed': True, 'generation_attempts': 1}, {'passed': True, 'generation_attempts': 2},
                                 {'passed': False, 'generation_attempts': 2}, {'passed': False}])
        self.assertEqual((1, 1, 2, 1), (repair['first_pass_count'], repair['repair_rescued_count'], repair['repair_attempts'], repair['unknown_cases']))
        rows = [{'case_id': str(i), 'arm_id': arm, 'passed': passed, 'hard_failure': False, 'safety_failure': False, 'total_tokens': tokens}
                for i, values in enumerate(((False, True), (True, False)))
                for arm, passed, tokens in (('baseline', values[0], 5), ('candidate', values[1], 8))]
        paired = paired_skill_metrics(rows)
        self.assertEqual(['0'], paired['improved_cases'])
        self.assertEqual(['1'], paired['regressed_cases'])
        self.assertEqual([], paired['safety_regressions'])
        self.assertEqual(6, paired['total_tokens_delta'])
        self.assertIsNone(paired['latency_ms_delta'])
        self.assertFalse(paired['stable_benefit_proven'])
        with self.assertRaises(ValueError):
            paired_skill_metrics(rows[:-1])


if __name__ == '__main__':
    unittest.main()
