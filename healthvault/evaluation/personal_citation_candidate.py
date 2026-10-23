"""Prepare a private, unpublished citation-selection proposal from seen failures."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from healthvault.evaluation.current_delivery import ROOT, digest, save
from healthvault.evolution.personal import PersonalSkillManager
from healthvault.evolution.personal_prompt import CITATION_GUIDANCE_END, CITATION_GUIDANCE_START
from healthvault.evolution.review import TARGET_SKILL, evaluate_candidate
from healthvault.evolution.signal_extractor import ExperienceCard
from healthvault.evolution.skill_patch import PatchGenerator, SECTION_END, build_patch_preview, diff_artifact
from healthvault.evolution.verifier import CandidateVerifier

SOURCE = ROOT / '.task_outputs/task-083-skill-real-v1/results.json'
PROJECT = ROOT / '.bear/skills'
OWNER = 'task089-evaluation-owner'
CONTENT = ('Select citations for the requested object and time window; do not copy the full returned pool. '
           'Honor an explicit one-source limit; keep both endpoints for a comparison. '
           'Check this selection before composing the final response.')


def observations(value: dict, skill_sha: str) -> list[dict]:
    found = {}
    for row in value.get('attempts', []):
        checks = row.get('validation', {})
        if (row.get('arm_id') != 'active_v1' or row.get('skill_sha256') != skill_sha
                or checks.get('citation_correct') is not False
                or not all(checks.get(key) is True for key in ('tool_policy_correct', 'status_correct', 'refusal_correct'))
                or row.get('failure_code') or row.get('final_status') != 'ok'):
            continue
        identifier = str(row['case_id'])
        found.setdefault(identifier, {'trace_id': 'TASK083-active-' + identifier, 'task_id': 'TASK083-' + identifier,
            'skill_sha256': skill_sha, 'source_kind': 'redacted_evaluation_attempt',
            'tools': [{'name': call['name'], 'status': call['status']} for call in row['tool_calls']],
            'finding': 'citation_constraint_failed_after_valid_tool_flow', 'effect_verified': False})
    return [found[key] for key in sorted(found)]


def evidence_card(rows: list[dict], skill_sha: str) -> ExperienceCard:
    groups = {}
    for row in rows:
        path = tuple(call['name'] + ':' + call['status'] for call in row['tools'])
        groups.setdefault(path, {})[row['task_id']] = row
    path, tasks = max(groups.items(), key=lambda item: (len(item[1]), item[0]), default=((), {}))
    if len(tasks) < 3:
        raise ValueError('Three distinct tasks with the same tool path and citation failures are required.')
    support = [tasks[key] for key in sorted(tasks)]
    return ExperienceCard(card_id='citation-selection-observation-v1', intent='citation_selection',
        preconditions=('Several references returned by read-only tools', 'The request narrows reference selection'),
        lesson='Proposed citation selection correction; its task benefit remains unverified.',
        tool_path=path,
        expected_outcome='citation_constraint_compliance', when_not_to_use=('No reference selection requirement',),
        trace_ids=tuple(row['trace_id'] for row in support), task_ids=tuple(row['task_id'] for row in support),
        skill_sha256=skill_sha, evidence_count=len(support), verification_source='deterministic',
        verification_strength='medium', reusable=True)


async def make_preview(active: str, card: ExperienceCard):
    def local_proposal(_prompt, _schema):
        return {'patches': [{'patch_id': 'citation-selection-v1', 'operation': 'add',
            'source_trace_ids': list(card.trace_ids), 'verification_source': card.verification_source,
            'verification_strength': card.verification_strength, 'target_section': 'Verification',
            'target_paragraph': SECTION_END, 'applicability_conditions': list(card.preconditions),
            'when_not_to_use': list(card.when_not_to_use),
            'tool_path': list(dict.fromkeys(item.partition(':')[0] for item in card.tool_path)),
            'stop_conditions': ['Reference selection has been checked'],
            'content': '\n'.join((CITATION_GUIDANCE_START, CONTENT, CITATION_GUIDANCE_END)),
            'rationale': 'Repeated deterministic reference-selection failures; no task benefit is claimed.'}]}

    def regression(candidate):
        checked = evaluate_candidate(active, candidate)
        return checked['passed'], tuple(item['case_id'] for group in ('legacy_regression', 'contract_probes')
                                      for item in checked[group] if not item['passed'])

    return await build_patch_preview((card,), active_content=active, generator=PatchGenerator(local_proposal),
        verifier=CandidateVerifier(TARGET_SKILL), regression_check=regression)


async def prepare(output: Path, *, source: Path = SOURCE, project: Path = PROJECT) -> dict:
    if output.exists():
        raise FileExistsError('Use a new private proposal directory.')
    active_path = project / TARGET_SKILL / 'SKILL.md'
    active_sha = digest(active_path)
    active = active_path.read_bytes().decode('utf-8')
    source_sha = digest(source)
    rows = observations(json.loads(source.read_text(encoding='utf-8')), active_sha)
    card = evidence_card(rows, active_sha)
    preview = await make_preview(active, card)
    if preview.status != 'pending_manual_review' or preview.candidate_content is None:
        raise ValueError('Local proposal failed existing gates: ' + ', '.join(preview.reasons))
    output.mkdir(parents=True, exist_ok=False)
    manager = PersonalSkillManager(project, output / 'isolated-evolution', None)
    service = manager.service(OWNER)
    evaluation = evaluate_candidate(active, preview.candidate_content)
    evaluation['workflow_probe'] = {'passed': False, 'status': 'not_run',
        'reason': 'Needs separately authorized real workflow verification; static checks do not prove benefit.'}
    record = service.store.create(preview, (card,), evidence_fingerprint=source_sha, evaluation=evaluation)
    snapshot = output / 'candidate-snapshot/.bear/skills' / TARGET_SKILL / 'SKILL.md'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(preview.candidate_content, encoding='utf-8', newline='\n')
    other_sha = digest(manager.service('task089-other-owner').store.active_path)
    if digest(active_path) != active_sha or digest(service.store.active_path) != active_sha or other_sha != active_sha:
        raise RuntimeError('Proposal preparation changed active or another user Skill.')
    save(output / 'sources.json', {'source_path': str(source.resolve()), 'source_sha256': source_sha,
        'observations': rows, 'effect_verified': False, 'classification': 'seen_synthetic_development_evidence'})
    save(output / 'patch.json', diff_artifact(preview))
    result = {'task': 'TASK-089', 'status': 'private_unpublished_proposal', 'candidate_id': record['candidate_id'],
        'owner_ref': manager.owner_ref(OWNER), 'source_sha256': source_sha, 'source_tasks': len(rows),
        'supporting_tasks': card.evidence_count,
        'active_skill_sha256': active_sha, 'candidate_skill_sha256': digest(snapshot),
        'candidate_snapshot_root': str(snapshot.parent.parent.resolve()), 'snapshot_role': 'evaluation_only',
        'static_regression_passed': evaluation['passed'], 'workflow_verification': 'not_run',
        'independent_task_benefit': 'not_run', 'untouched_held_out': False,
        'human_gold': False, 'published': False, 'generator_kind': 'local_rule_proposal',
        'model_generator_calls': 0, 'api_calls': 0}
    save(output / 'summary.json', result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(prepare(args.output))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
