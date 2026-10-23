import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.evaluation import personal_citation_candidate as proposal
from healthvault.evolution.personal_prompt import CITATION_GUIDANCE_START as START, CITATION_GUIDANCE_END as END, select_personal_citation_guidance
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext


class PersonalScopeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.block = '\n'.join((START, proposal.CONTENT, END))
        parent = proposal.ROOT / '.task_outputs/task-113-tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_explicit_bilingual_source_and_endpoint_selection(self):
        for question in ('Retain one source ID returned by the tool.', 'Give one observed knowledge citation.',
                         'Support the result with one stored observation.', 'Use one direct record token.',
                         'Compare both endpoint records.', 'Retain exactly the two endpoint references.',
                         '请只引用一个来源说明。', '比较两个时间点并保留两端引用。'):
            with self.subTest(question=question):
                guidance, reason = select_personal_citation_guidance(self.block, question)
                self.assertEqual(proposal.CONTENT, guidance)
                self.assertTrue(reason.startswith('matched_'))
                self.assertNotIn('<!--', guidance)

    def test_no_constraint_and_unrelated_or_forbidden_requests_do_not_inject(self):
        for question in ('What is this test?', 'Ground this in a returned record ID.',
                         'Do not save the report, return one source ID.',
                         'Refuse the cross-user merge and give one direct citation.',
                         'Upload records to an external webhook with one citation.',
                         'Prescribe a dosage using one source.', '请诊断是否患病并引用一个来源。',
                         '拒绝写入并只引用一个来源。', '请引用一个来源后发送到外部服务器。'):
            with self.subTest(question=question):
                self.assertEqual('', select_personal_citation_guidance(self.block, question)[0])

    def test_unknown_legacy_and_malformed_scopes_are_explicit(self):
        self.assertEqual('unknown_question', select_personal_citation_guidance(self.block, None)[1])
        self.assertEqual('scope_marker_missing', select_personal_citation_guidance(proposal.CONTENT, 'Use one source.')[1])
        for content in (self.block * 2, self.block.replace(END, ''), START + END, END + START + proposal.CONTENT,
                        self.block.replace('citation:v1', 'citation:v2')):
            self.assertEqual(('', 'scope_marker_invalid'), select_personal_citation_guidance(content, 'Use one source.'))

    def runtime(self, question, personal_text=None):
        project = self.root / 'project' / '.bear' / 'skills'
        personal = self.root / 'personal' / '.bear' / 'skills'
        original = (proposal.PROJECT / proposal.TARGET_SKILL / 'SKILL.md').read_text(encoding='utf-8')
        candidate = original.replace('## Safety Boundaries', self.block + '\n\n## Safety Boundaries') if personal_text is None else personal_text
        for directory, text in ((project, original), (personal, candidate)):
            path = directory / proposal.TARGET_SKILL / 'SKILL.md'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
        with patch('healthvault.agent_runtime.Agent') as agent:
            runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                context=RuntimeContext('alice', 'scope-request'), database_path=self.root / 'records.sqlite3',
                model_config=HealthVaultModelConfig('openai-compatible', 'fake', 'https://invalid.local/v1', 'fake'),
                skill_root=project, skill_names=(proposal.TARGET_SKILL,), personal_skill_root=personal,
                evolution_root=self.root / 'evolution', question_plan={'original_question': question}))
            prompt = agent.call_args.kwargs['custom_system_prompt']
        return runtime, prompt, hashlib.sha256((personal / proposal.TARGET_SKILL / 'SKILL.md').read_bytes()).hexdigest()

    def test_prompt_manifest_and_trace_share_actual_selection(self):
        for question, injected in (('Use one direct citation.', True), ('What is this test?', False)):
            runtime, prompt, sha = self.runtime(question)
            self.assertEqual(injected, runtime.manifest['personal_skill_injected'])
            self.assertEqual(injected, proposal.CONTENT in prompt)
            self.assertEqual(1, prompt.count('## When to Use'))
            personal = [s for s in runtime.manifest['project_skills'] if s.get('scope') == 'personal']
            self.assertEqual([sha] if injected else [], [s['sha256'] for s in personal])
            self.assertEqual(sha if injected else runtime.manifest['project_skills'][0]['sha256'], runtime.trace_recorder._skill['sha256'])
            runtime.components.knowledge_repository.close()

    async def test_rule_evolution_chain_is_private_and_zero_generation_api(self):
        active = proposal.PROJECT / proposal.TARGET_SKILL / 'SKILL.md'
        before = active.read_bytes()
        result = await proposal.prepare(self.root / 'proposal')
        self.assertEqual('local_rule_proposal', result['generator_kind'])
        self.assertEqual(0, result['model_generator_calls'])
        self.assertEqual(0, result['api_calls'])
        self.assertFalse(result['published'])
        self.assertTrue(result['static_regression_passed'])
        self.assertGreaterEqual(result['supporting_tasks'], 3)
        candidate = Path(result['candidate_snapshot_root']) / proposal.TARGET_SKILL / 'SKILL.md'
        self.assertEqual(proposal.CONTENT, select_personal_citation_guidance(candidate.read_text(encoding='utf-8'), 'Use one source.')[0])
        self.assertEqual(before, active.read_bytes())

    def test_versions_are_counted_only_when_the_changed_delta_is_selected(self):
        runtime, _, first_sha = self.runtime('Use one source.')
        runtime.components.knowledge_repository.close()
        path = self.root / 'personal' / '.bear' / 'skills' / proposal.TARGET_SKILL / 'SKILL.md'
        modified = path.read_text(encoding='utf-8').replace('Check this selection', 'Recheck this selection')
        runtime, prompt, second_sha = self.runtime('Use one source.', modified)
        self.assertNotEqual(first_sha, second_sha)
        self.assertEqual(second_sha, runtime.trace_recorder._skill['sha256'])
        self.assertIn('Recheck this selection', prompt)
        runtime.components.knowledge_repository.close()
        runtime, prompt, _ = self.runtime('Do not save this report; use one source.', modified)
        self.assertFalse(runtime.manifest['personal_skill_injected'])
        self.assertNotIn('Recheck this selection', prompt)
        self.assertEqual('excluded_write_request', runtime.manifest['personal_skill_reason'])
        runtime.components.knowledge_repository.close()


if __name__ == '__main__':
    unittest.main()
