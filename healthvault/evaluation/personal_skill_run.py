"""Versioned actual personal-injection A/B on seen read-only workflow cases."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from healthvault.evaluation import personal_citation_candidate as proposal
from healthvault.evaluation import product_workflow_run as workflow
from healthvault.evaluation.current_delivery import ROOT, digest, implementation_binding, save
from healthvault.evaluation.current_skill_run import CurrentTransport
from healthvault.evaluation.measurement import paired_skill_metrics
from healthvault.evaluation.web_rag_run import ModelHTTPClient, value_hash

PROTOCOL = 'healthvault-personal-skill-paired-2026-10-18-v1'
MAX_CALLS = 160
ARMS = ('baseline', 'candidate')


async def prepare(output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new evaluation directory.')
    snapshot = await proposal.prepare(output / 'proposal')
    base = workflow.prepare(output / 'workflow')
    baseline = output / 'baseline/.bear/skills' / proposal.TARGET_SKILL / 'SKILL.md'
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes((proposal.PROJECT / proposal.TARGET_SKILL / 'SKILL.md').read_bytes())
    candidate_root = Path(snapshot['candidate_snapshot_root'])
    extra = [output / 'workflow/preflight.json', output / 'proposal/summary.json', output / 'proposal/patch.json',
             output / 'proposal/sources.json', proposal.SOURCE, baseline, candidate_root / proposal.TARGET_SKILL / 'SKILL.md']
    plan = {**base, 'protocol': PROTOCOL, 'limits': {**base['limits'], 'maximum_calls': MAX_CALLS},
            'base_preflight': str((output / 'workflow/preflight.json').resolve()),
            'arms': {'baseline': {'root': str(baseline.parent.parent.resolve()),
                                'skill_sha256': snapshot['active_skill_sha256'], 'personal_skill_injected': False},
                     'candidate': {'root': str(candidate_root), 'skill_sha256': snapshot['candidate_skill_sha256'],
                                   'personal_skill_injected': True}},
            'scope': 'seen_read_only_workflow_paired_regression_with_actual_personal_injection',
            'candidate_source': 'TASK089 unpublished local proposal from TASK083 failures',
            'publishes_skill': False, 'human_gold': False, 'fresh_task_benefit': 'not_measured',
            'natural_answer_quality': 'not_measured', 'whole_product_task_success_rate': None,
            'comparison_note': 'Same current runtime and budget, only personal Skill content changes; no old-score improvement claim.'}
    plan['bindings'] = {**base['bindings'], **{str(path.resolve()): digest(path) for path in extra}}
    plan.pop('sha256', None)
    plan['sha256'] = value_hash(plan)
    save(output / 'preflight.json', plan)
    return plan


def validate(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan.get('protocol') != PROTOCOL or plan.get('sha256') != value_hash({k: v for k, v in plan.items() if k != 'sha256'}):
        raise ValueError('Personal evaluation integrity failed.')
    base = workflow.validate(Path(plan['base_preflight']))
    if plan['implementation_sha256'] != implementation_binding()['sha256'] or any(digest(Path(p)) != sha for p, sha in plan['bindings'].items()):
        raise ValueError('Bound implementation or inputs changed.')
    if (plan['limits'] != {**base['limits'], 'maximum_calls': MAX_CALLS} or plan['model_config'] != base['model_config']
            or plan['case_order'] != base['case_order'] or plan['dataset_semantic_sha256'] != base['dataset_semantic_sha256']
            or len(plan['case_order']) != 20 or set(plan['arms']) != set(ARMS)
            or any(plan[key] != base[key] for key in ('endpoint', 'tool_names', 'structured_output', 'corpus_path',
                       'hybrid_model_path', 'hybrid_index_path', 'embedding_model_sha256'))):
        raise ValueError('Frozen paired conditions changed.')
    for arm in ARMS:
        spec = plan['arms'][arm]
        skill = Path(spec['root']) / proposal.TARGET_SKILL / 'SKILL.md'
        if (not skill.resolve().is_relative_to(path.parent.resolve()) or spec['skill_sha256'] != digest(skill)
                or plan['bindings'].get(str(skill.resolve())) != spec['skill_sha256']
                or spec['personal_skill_injected'] is not (arm == 'candidate')):
            raise ValueError('Private arm binding changed.')
    if (plan['arms']['baseline']['skill_sha256'] != digest(proposal.PROJECT / proposal.TARGET_SKILL / 'SKILL.md')
            or plan['arms']['baseline']['skill_sha256'] == plan['arms']['candidate']['skill_sha256']):
        raise ValueError('Baseline or candidate version changed.')
    return plan


def paired_row(row: dict) -> dict:
    attempt = row['attempt']
    calls = attempt.get('calls', [])
    required = row['product_contract']['required_tools']
    unnecessary = 0
    position, queries, previous_status = 0, set(), None
    for call in calls:
        name = call['name']
        if required == ['search_health_knowledge']:
            query = call.get('arguments', {}).get('query', '')
            issue = workflow.knowledge_search_issue(query, queries, previous_status)
            unnecessary += int(name != required[0] or issue is not None or position >= workflow.MAX_KNOWLEDGE_QUERIES)
            queries.add(workflow.knowledge_query(query))
            previous_status = call.get('result_status')
            position += 1
        elif position < len(required) and name == required[position]:
            position += 1
        else:
            unnecessary += 1
    reasons = row.get('oracle', {}).get('hard_failure_reasons')
    safety = (bool(set(reasons) & {'unauthorized_tool', 'destructive_data_change', 'duplicate_write'})
              if isinstance(reasons, list) and all(isinstance(reason, str) for reason in reasons) else None)
    return {'case_id': row['case_id'], 'arm_id': row['arm_id'], 'passed': row['passed'],
            'hard_failure': row['hard_failure'], 'safety_failure': safety,
            'total_tokens': attempt['total_tokens'] if not attempt['usage_unknown_calls'] else None,
            'observed_tokens': attempt['total_tokens'], 'latency_ms': attempt['latency_ms'],
            'tool_attempts': len(calls), 'unnecessary_calls': unnecessary}


async def execute(path: Path, output: Path, *, approval: Path, review: Path) -> dict:
    plan = validate(path)
    plan_sha = digest(path)
    auth, checked = [json.loads(p.read_text(encoding='utf-8')) for p in (approval, review)]
    if (auth.get('approved') is not True or auth.get('authorization_source') != 'human_user'
            or auth.get('plan_sha256') != plan_sha or checked.get('decision') != 'PASS' or checked.get('plan_sha256') != plan_sha):
        raise ValueError('This paired version needs independent review and new explicit paid approval.')
    if output.exists() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Use a new workspace output directory.')
    config = workflow.configured_web_model()
    if config.model != plan['model_config']['model'] or config.api_base.rstrip('/') != plan['model_config']['api_base'] or config.reasoning_effort != 'high':
        raise ValueError('Configured model changed after approval.')
    claim = ROOT / '.task_outputs/task-105-approvals' / (plan_sha + '.json')
    claim.parent.mkdir(parents=True, exist_ok=True)
    with claim.open('x', encoding='utf-8') as stream:
        json.dump({'plan_sha256': plan_sha, 'output': str(output.resolve())}, stream)
    output.mkdir(parents=True, exist_ok=False)
    transport = CurrentTransport(plan, output, ModelHTTPClient(plan['endpoint'], config.api_key))
    cases = {case['case_id']: case for case in json.loads((Path(plan['base_preflight']).parent / 'dataset.json').read_text(encoding='utf-8'))['cases']}
    rows, status, aborted = [], 'completed', None
    for index, identifier in enumerate(plan['case_order']):
        for arm in ARMS[::1 if index % 2 == 0 else -1]:
            transport.case_id, transport.arm_id = identifier, arm
            try:
                row = await workflow.run_case(cases[identifier], plan, transport, output / 'cases' / arm,
                    api_key=config.api_key, personal_root=Path(plan['arms'][arm]['root']))
            except Exception as exc:
                status, aborted = 'aborted_exception', {'case_id': identifier, 'arm_id': arm, 'error_type': type(exc).__name__}
                break
            row['arm_id'] = arm
            actual = row['measurement']
            spec = plan['arms'][arm]
            injection_ok = (actual['personal_skill_injected'] is spec['personal_skill_injected']
                and spec['skill_sha256'] in {v['sha256'] for v in actual['skill_versions']}
                and digest(proposal.PROJECT / proposal.TARGET_SKILL / 'SKILL.md') == plan['arms']['baseline']['skill_sha256'])
            row['injection_contract_passed'] = injection_ok
            row['passed'] = row['passed'] and injection_ok
            rows.append(row)
            save(output / (identifier + '-' + arm + '.json'), row)
            print(f"Personal Skill {len(rows)}/40: {identifier}/{arm}, passed={row['passed']}", flush=True)
            if transport.failed or not injection_ok:
                status = 'stopped_on_error'
                break
        if status != 'completed':
            break
    paired = [paired_row(row) for row in rows]
    counts = {identifier: sum(row['case_id'] == identifier for row in paired) for identifier in plan['case_order']}
    complete = [row for row in paired if counts[row['case_id']] == 2]
    known_usage = [row['usage'] for row in transport.calls if row.get('usage')
                   and all(key in row['usage'] for key in ('prompt_tokens', 'completion_tokens'))]
    result = {'protocol': PROTOCOL, 'status': status, 'aborted_attempt': aborted, 'results': rows, 'completed_attempts': len(rows), 'planned_attempts': 40,
              'paired': paired_skill_metrics(complete), 'unpaired_cases': [key for key, count in counts.items() if count == 1],
              'api_calls': sum(row['transport_kind'] == 'real' for row in transport.calls),
              'observed_tokens': sum(item['prompt_tokens'] + item['completion_tokens'] for item in known_usage),
              'usage_unknown_calls': len(transport.calls) - len(known_usage),
              'metric_scope': 'fixed_read_only_workflow_contract', 'natural_answer_quality': 'not_measured',
              'whole_product_task_success_rate': None, 'untouched_held_out': False,
              'human_review_status': 'pending', 'fresh_task_benefit': 'not_measured', 'publishes_skill': False}
    save(output / 'results.json', result)
    save(output / 'human-review.json', {'status': 'pending', 'minimum_cases': 10,
        'cases': [{'case_id': key, 'question': cases[key]['prompt'], 'expected': cases[key]['expected'],
                   'attempts': [row for row in rows if row['case_id'] == key], 'human_verdict': None}
                  for key, count in counts.items() if count == 2]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        asyncio.run(prepare(args.output))
        print(json.dumps({'status': 'prepared', 'api_calls': 0, 'maximum_calls': MAX_CALLS}))
    else:
        if not all((args.preflight, args.approval, args.review)):
            parser.error('Run requires a bound preflight, independent review and explicit human approval.')
        asyncio.run(execute(args.preflight, args.output, approval=args.approval, review=args.review))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
