import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.evaluation import personal_citation_candidate as run
from healthvault.evolution.personal import PersonalSkillManager
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext


class PersonalCitationCandidateTests(unittest.TestCase):
    def setUp(self):
        parent = run.ROOT / '.task_outputs/task089-tests'
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.active_path = run.PROJECT / run.TARGET_SKILL / 'SKILL.md'
        self.sha = run.digest(self.active_path)
        self.source = json.loads(run.SOURCE.read_text(encoding='utf-8'))

    def test_failure_sources_are_version_matched_deduplicated_and_redacted(self):
        rows = run.observations(self.source, self.sha)
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(len(rows), len({row['task_id'] for row in rows}))
        duplicate = copy.deepcopy(self.source)
        duplicate['attempts'] *= 2
        self.assertEqual(rows, run.observations(duplicate, self.sha))
        self.assertEqual([], run.observations(self.source, 'another-version'))
        self.assertNotIn('REC:', json.dumps(rows))
        self.assertNotIn('HVK:', json.dumps(rows))
        self.assertTrue(all(not row['effect_verified'] for row in rows))
        card = run.evidence_card(rows, self.sha)
        support = [row for row in rows if row['task_id'] in card.task_ids]
        self.assertGreaterEqual(len(support), 3)
        self.assertEqual(len(support), card.evidence_count)
        self.assertTrue(all(tuple(call['name'] + ':' + call['status'] for call in row['tools']) == card.tool_path
                            for row in support))
        mixed = copy.deepcopy(rows[:3])
        for index, row in enumerate(mixed):
            row['tools'] = [{'name': 'different-route-' + str(index), 'status': 'ok'}]
        with self.assertRaisesRegex(ValueError, 'same tool path'):
            run.evidence_card(mixed, self.sha)

    def test_local_preview_preserves_safety_and_contains_no_case_or_patient_facts(self):
        active = self.active_path.read_bytes().decode('utf-8')
        card = run.evidence_card(run.observations(self.source, self.sha), self.sha)
        preview = asyncio.run(run.make_preview(active, card))
        self.assertEqual('pending_manual_review', preview.status, preview.reasons)
        self.assertEqual(active.split('## Safety Boundaries')[1].replace('\r\n', '\n'),
                         preview.candidate_content.split('## Safety Boundaries')[1])
        self.assertIn(run.CONTENT, preview.candidate_content)
        self.assertNotIn('R3H-', preview.candidate_content)
        self.assertNotIn('REC:', preview.candidate_content)
        self.assertEqual('Verification', preview.patches[0].target_section)
        self.assertFalse(preview.published)

    def test_real_private_store_isolated_and_cannot_publish_without_model_probe(self):
        output = self.root / 'proposal'
        result = asyncio.run(run.prepare(output))
        self.assertFalse(result['published'])
        self.assertEqual(0, result['api_calls'])
        self.assertEqual('not_run', result['independent_task_benefit'])
        manager = PersonalSkillManager(run.PROJECT, output / 'isolated-evolution', None)
        own = manager.service(run.OWNER)
        record = own.store.get(result['candidate_id'])
        self.assertEqual('personal_skill', record['scope'])
        self.assertEqual('pending_review', record['status'])
        self.assertIsNone(manager.service('task089-other-owner').store.get(result['candidate_id']))
        with self.assertRaisesRegex(ValueError, 'workflow regression'):
            own.store.decide(result['candidate_id'], 'approve', 'synthetic-owner')
        self.assertEqual(self.sha, run.digest(own.store.active_path))
        self.assertEqual(self.sha, run.digest(self.active_path))
        with self.assertRaises(FileExistsError):
            asyncio.run(run.prepare(output))

    def test_evaluation_snapshot_uses_actual_personal_runtime_injection(self):
        result = asyncio.run(run.prepare(self.root / 'proposal'))
        with patch('healthvault.agent_runtime.Agent') as agent:
            runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                context=RuntimeContext('synthetic-owner', 'synthetic-request'), database_path=self.root / 'records.sqlite3',
                model_config=HealthVaultModelConfig('openai-compatible', 'synthetic', 'https://example.invalid', 'synthetic'),
                skill_root=run.PROJECT, skill_names=(run.TARGET_SKILL,),
                personal_skill_root=Path(result['candidate_snapshot_root'])))
            self.assertIn(run.CONTENT, agent.call_args.kwargs['custom_system_prompt'])
            self.assertEqual(2, len(runtime.manifest['project_skills']))
        self.assertEqual(self.sha, run.digest(self.active_path))


if __name__ == '__main__':
    unittest.main()
