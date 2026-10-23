"""Versioned product-contract regression; old Oracles and scores stay unchanged."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from healthvault.application.workflow_contract import MAX_KNOWLEDGE_QUERIES, REQUIRED_TOOLS, WORKFLOW_PROTOCOL, knowledge_query, knowledge_search_issue
from healthvault.evaluation import current_e2e_run as current
from healthvault.evaluation.current_delivery import ROOT, digest, implementation_binding, save
from healthvault.evaluation.skill.benchmark.task_025 import FinalParse
from healthvault.evaluation.web_rag_run import ModelHTTPClient, value_hash
from healthvault.web.model_runtime import configured_web_model

PROTOCOL = 'healthvault-product-e2e-2026-10-18-v3'
MAX_CALLS, MAX_OUTPUT = 80, 16384
INTENTS = {tool: intent for intent, tool in REQUIRED_TOOLS.items()}


def load_dataset(corpus_path: Path) -> dict:
    dataset = copy.deepcopy(current.load_current_dataset(corpus_path))
    dataset.update(dataset_version='product-e2e-r5-regression-2026-10-11-v2', held_out=False,
                   untouched_held_out=False, source_dataset_version=dataset['dataset_version'])
    for case in dataset['cases']:
        tools = case['expected']['required_tools']
        case['product_contract'] = {
            'protocol': WORKFLOW_PROTOCOL,
            'intent': INTENTS.get(tools[0], 'general_read_only_workflow') if tools else 'general_read_only_workflow',
            'required_tools': tools,
            'max_calls': MAX_KNOWLEDGE_QUERIES if tools == ['search_health_knowledge'] else len(tools),
            'search_recovery': 'distinct_queries_after_empty_only',
            'min_citations': case['expected']['min_citations'],
            'max_citations': case['expected']['max_citations'],
        }
    return dataset


def task_prompt(case: dict) -> str:
    contract = case['product_contract']
    return (case['prompt'] + '\nProduct workflow contract: check the required read-only tool before finishing. '
            'An empty knowledge search may be retried with a distinct expression, up to '
            f'{contract["max_calls"]} calls. Stop searching after a nonempty result. '
            'Use only references relevant to the original request. Honor an explicit one-source limit '
            'and retain both endpoints for comparisons; refusals use none. '
            'Return only the strict final JSON object, with the actual tool outcome.')


def evaluate(case: dict, attempt: dict) -> dict:
    """Keep all actual calls; expand a declared empty-search recovery rule only."""
    expected = copy.deepcopy(case['expected'])
    calls = attempt.get('calls', [])
    search_case = expected['required_tools'] == ['search_health_knowledge']
    recovery_ok = True
    if search_case and calls:
        count = len(calls)
        previous_queries, previous_status = set(), None
        for call in calls:
            query = call.get('arguments', {}).get('query', '')
            recovery_ok = recovery_ok and bool(knowledge_query(query)) and knowledge_search_issue(query, previous_queries, previous_status) is None
            previous_queries.add(knowledge_query(query))
            previous_status = call.get('result_status')
        expected.update(required_tools=['search_health_knowledge'] * count,
                        required_tool_statuses=['ok'] * count,
                        required_tool_result_statuses=['empty'] * (count - 1) + expected['required_tool_result_statuses'],
                        required_argument_rules=expected['required_argument_rules'] * count,
                        max_calls=MAX_KNOWLEDGE_QUERIES)
    oracle = current.legacy.evaluate_three_layer_oracle({**case, 'expected': expected}, attempt)
    oracle['behavior']['search_recovery_passed'] = bool(recovery_ok)
    oracle['behavior']['passed'] = oracle['behavior']['passed'] and bool(recovery_ok)
    oracle['passed'] = oracle['outcome']['passed'] and oracle['behavior']['passed'] and oracle['response']['passed']
    oracle['protocol'] = PROTOCOL
    return oracle


async def run_case(case: dict, plan: dict, transport, root: Path, *, api_key: str = 'fake', personal_root: Path | None = None) -> dict:
    import healthvault.agent_runtime as native
    original_build = native.build_healthvault_agent_runtime
    actual_runtimes = []

    def build(config):
        config = replace(config, personal_skill_root=personal_root) if personal_root else config
        runtime = original_build(config)
        # Entry metadata is task intent, not expected records, citations or final state.
        runtime.manifest['workflow_intent'] = case['product_contract']['intent']
        actual_runtimes.append(runtime)
        return runtime

    with patch.object(native, 'build_healthvault_agent_runtime', build):
        result = await current.run_case({**case, 'prompt': task_prompt(case)}, plan, transport, root, api_key=api_key)
    attempt = result['attempt']
    final = attempt.get('final')
    parsed = FinalParse(final['final_status'], tuple(final['citations']), final['refusal'], None) if final else None
    result['legacy_oracle'] = result['oracle']
    result['oracle'] = evaluate(case, {**attempt, 'final': parsed})
    result['oracle']['passed'] = result['oracle']['passed'] and attempt['hybrid_path_passed']
    result.update(passed=result['oracle']['passed'], hard_failure=result['oracle']['hard_failure'],
                  product_contract=case['product_contract'], original_question=case['prompt'])
    from healthvault.evaluation.measurement import tool_execution_metrics
    manifest = actual_runtimes[-1].manifest if actual_runtimes else {}
    result['measurement'] = {'entrypoint': 'native_agent_structured_workflow_evaluation',
        'personal_skill_injected': manifest.get('personal_skill_injected'),
        'skill_versions': manifest.get('project_skills', []),
        'features': {'answer_verification': manifest.get('answer_verification_enabled', False), 'answer_repair': False},
        'scope': 'fixed_read_only_workflow_contract', 'natural_answer_quality': 'not_measured',
        'upload_confirmation_calendar_coverage': False, 'usage_scope': 'all_agent_requests_in_transport_ledger'}
    result['tool_metrics'] = {**tool_execution_metrics(attempt.get('calls', [])),
                              'tool_trajectory_passed': result['oracle']['behavior']['passed']}
    return result


def prepare(output: Path) -> dict:
    config = configured_web_model()
    if config.model != 'deepseek-flash' or config.reasoning_effort != 'high':
        raise ValueError('This regression is frozen to the configured DeepSeek Flash/high.')
    output.mkdir(parents=True, exist_ok=False)
    baseline = current.prepare_run(output / 'legacy-inputs', api_base=config.api_base)
    dataset = load_dataset(Path(baseline['corpus_path']))
    save(output / 'dataset.json', dataset)
    matrix = [{'case_id': case['case_id'], 'prompt': task_prompt(case),
               'original_question': case['prompt'], 'product_contract': case['product_contract'],
               'expected': case['expected']} for case in dataset['cases']]
    save(output / 'case-matrix.json', matrix)
    plan = {key: value for key, value in baseline.items() if key != 'sha256'}
    plan.update(protocol=PROTOCOL, source_dataset_version=dataset['dataset_version'],
                dataset_semantic_sha256=value_hash(dataset), workflow_protocol=WORKFLOW_PROTOCOL,
                model_config={'model': config.model, 'api_base': config.api_base.rstrip('/'),
                              'thinking': {'type': 'enabled'}, 'reasoning_effort': config.reasoning_effort,
                              'max_output_tokens': MAX_OUTPUT},
                scope='product_contract_regression_on_seen_r5_cases', approval_required=True, api_calls=0,
                comparison_note='New search recovery contract; old scores are not recomputed or compared as improvement.')
    plan['bindings'].update({str(path.resolve()): digest(path) for path in (output / 'dataset.json', output / 'case-matrix.json')})
    plan['sha256'] = value_hash(plan)
    save(output / 'preflight.json', plan)
    return plan


def validate(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan['protocol'] != PROTOCOL or plan['sha256'] != value_hash({k: v for k, v in plan.items() if k != 'sha256'}):
        raise ValueError('Product preflight integrity failed.')
    if plan['implementation_sha256'] != implementation_binding()['sha256'] or any(digest(Path(p)) != sha for p, sha in plan['bindings'].items()):
        raise ValueError('Bound source or data changed.')
    if (plan['limits'] != {'maximum_calls': MAX_CALLS, 'model_turns_per_attempt': 4, 'transport_retries': 0,
                          'input_bytes_per_call': 160_000, 'timeout_seconds': 60}
            or plan['workflow_protocol'] != WORKFLOW_PROTOCOL
            or plan['model_config'] != {'model': 'deepseek-flash', 'api_base': plan['model_config']['api_base'],
                                       'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high', 'max_output_tokens': MAX_OUTPUT}
            or plan['tool_names'] != list(current.HARNESS_LOCAL_READ_ONLY_TOOL_NAMES)
            or plan['structured_output'] != current.STRUCTURED_OUTPUT
            or plan['endpoint'] != plan['model_config']['api_base'].rstrip('/') + '/chat/completions'):
        raise ValueError('Product runtime or budget changed.')
    dataset = load_dataset(Path(plan['corpus_path']))
    if (len(dataset['cases']) != 20 or plan['dataset_semantic_sha256'] != value_hash(dataset)
            or plan['case_order'] != [case['case_id'] for case in dataset['cases']]
            or current.model_fingerprint(Path(plan['hybrid_model_path'])) != plan['embedding_model_sha256']):
        raise ValueError('Frozen cases or embedding changed.')
    return plan


async def execute(path: Path, output: Path, *, approval: Path, review: Path) -> dict:
    plan = validate(path)
    plan_sha = digest(path)
    auth, checked = [json.loads(p.read_text(encoding='utf-8')) for p in (approval, review)]
    if (auth.get('approved') is not True or auth.get('authorization_source') != 'human_user'
            or auth.get('plan_sha256') != plan_sha or checked.get('decision') != 'PASS'
            or checked.get('plan_sha256') != plan_sha):
        raise ValueError('This version needs independent review and new explicit paid approval.')
    if output.exists() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Use a new output directory inside the workspace.')
    config = configured_web_model()
    if (config.model != plan['model_config']['model'] or config.reasoning_effort != 'high'
            or config.api_base.rstrip('/') != plan['model_config']['api_base']):
        raise ValueError('Configured model changed after approval.')
    save(ROOT / '.task_outputs/task-088-approvals' / (plan_sha + '.json'),
         {'plan_sha256': plan_sha, 'authorization_source': 'human_user', 'output': str(output.resolve())})
    output.mkdir(parents=True, exist_ok=False)
    transport = current.E2ETransport(plan, output, ModelHTTPClient(plan['endpoint'], config.api_key))
    transport.arm_id = 'project_baseline'
    rows, status = [], 'completed'
    for case in load_dataset(Path(plan['corpus_path']))['cases']:
        transport.case_id = case['case_id']
        row = await run_case(case, plan, transport, output / 'cases', api_key=config.api_key)
        rows.append(row)
        save(output / (case['case_id'] + '.json'), row)
        print(f"Product E2E {len(rows)}/20: {case['case_id']}, passed={row['passed']}", flush=True)
        if transport.failed:
            status = 'stopped_on_error'
            break
    value = {'protocol': PROTOCOL, 'status': status, 'results': rows, 'completed_cases': len(rows), 'planned_cases': 20,
             'task_success_count': sum(row['passed'] for row in rows),
             'hard_failure_count': sum(row['hard_failure'] for row in rows),
             'api_calls': len(transport.calls), 'tokens': {key: sum(row['attempt'][key] for row in rows)
                for key in ('input_tokens', 'output_tokens', 'total_tokens', 'usage_unknown_calls')},
             'untouched_held_out': False, 'human_review_status': 'pending', 'publishes_skill': False}
    value.update(metric_scope='fixed_read_only_workflow_contract', workflow_case_pass_count=value['task_success_count'],
                 legacy_metric_aliases={'task_success_count': 'workflow_case_pass_count'},
                 natural_answer_quality='not_measured', whole_product_task_success_rate=None)
    save(output / 'results.json', value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
        print(json.dumps({'status': 'prepared', 'api_calls': 0, 'maximum_calls': MAX_CALLS}))
    else:
        if not all((args.preflight, args.approval, args.review)):
            parser.error('Run requires preflight, independent review and human approval files.')
        asyncio.run(execute(args.preflight, args.output, approval=args.approval, review=args.review))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
