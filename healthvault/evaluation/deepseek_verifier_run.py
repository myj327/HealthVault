"""Small, version-bound verifier challenge. Preparation never calls an API."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import time
import urllib.request
from urllib.parse import urlparse

from healthvault.web.answer_validation import prepare_verification, verify_answer, valid_verifier_output
from healthvault.web.model_runtime import configured_web_model
from healthvault.web.question_plan import create_question_plan

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'healthvault/evaluation/semantic_challenge_v1.json'
PROTOCOL = 'deepseek-verifier-challenge-v1'
MAX_CALLS, MAX_OUTPUT_TOKENS, MAX_INPUT_BYTES, TIMEOUT_SECONDS = 12, 16384, 80000, 120
SOURCE_FILES = ('agents/agent.py', 'healthvault/agent_runtime.py', 'healthvault/model_config.py',
                'healthvault/application/workflow_contract.py', 'healthvault/web/api.py',
                'healthvault/web/model_runtime.py', 'healthvault/web/answer_validation.py',
                'healthvault/web/answer_coverage.py', 'healthvault/evaluation/deepseek_verifier_run.py',
                'healthvault/web/question_plan.py', 'healthvault/web/answers.py',
                'healthvault/application/tool_evidence.py',
                'tests/healthvault/test_task085_evidence_coverage.py',
                'tests/healthvault/test_task086_completion_contract.py', 'tests/healthvault/test_task087_preflight.py',
                'tests/healthvault/test_task091_question_plan.py', 'tests/healthvault/test_task092_agent_verification.py',
                'tests/healthvault/test_task093_anthropic_terminal.py')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def settings(config) -> dict:
    if (urlparse(config.api_base).hostname != 'api.deepseek.com' or config.provider != 'openai-compatible'
            or config.model != 'deepseek-flash' or config.reasoning_effort != 'high'):
        raise ValueError('This challenge requires the approved DeepSeek Flash/high configuration.')
    return {'provider': config.provider, 'api_base': config.api_base, 'model': config.model,
            'reasoning_effort': config.reasoning_effort}


def prepare(output: Path) -> dict:
    model = settings(configured_web_model())
    dataset = json.loads(DATA.read_text(encoding='utf-8'))
    requests = []
    for case in dataset['cases']:
        requirements = create_question_plan(case['question'])['points']
        system, data = prepare_verification(case['question'], case['answer'], case['sources'], audit_coverage=True,
                                             required_points=requirements)
        payload = {'model': model['model'], 'reasoning_effort': model['reasoning_effort'], 'stream': False,
                   'max_tokens': MAX_OUTPUT_TOKENS,
                   'messages': [{'role': 'system', 'content': system},
                                {'role': 'user', 'content': json.dumps(data, ensure_ascii=False)}]}
        if len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_INPUT_BYTES:
            raise ValueError('Challenge request exceeds its input budget.')
        requests.append({'case_id': case['case_id'], 'payload': payload})
    if len(requests) != MAX_CALLS:
        raise ValueError('Frozen challenge must contain exactly twelve cases.')
    output.mkdir(parents=True, exist_ok=False)
    save(output/'requests.json', requests)
    plan = {'protocol': PROTOCOL, 'nonce': secrets.token_hex(12), 'model': model,
            'source_sha256': {name: digest(ROOT/name) for name in SOURCE_FILES},
            'data_sha256': digest(DATA), 'requests_sha256': digest(output/'requests.json'),
            'max_calls': MAX_CALLS, 'max_output_tokens': MAX_OUTPUT_TOKENS,
            'max_input_bytes': MAX_INPUT_BYTES, 'timeout_seconds': TIMEOUT_SECONDS, 'retries': 0,
            'label_author': dataset['label_author'], 'human_confirmation': dataset['human_confirmation'],
            'question_plan_method': 'question_rules', 'online_question_planner_evaluated': False,
            'case_ids': [item['case_id'] for item in requests], 'api_calls_during_preparation': 0}
    save(output/'plan.json', plan)
    return plan


def validate(preflight: Path, approval: Path) -> tuple[dict, list, object]:
    plan = json.loads((preflight/'plan.json').read_text(encoding='utf-8'))
    authorized = json.loads(approval.read_text(encoding='utf-8'))
    if (authorized.get('approved') is not True or authorized.get('authorization_source') != 'human_user'
            or authorized.get('plan_sha256') != digest(preflight/'plan.json')):
        raise ValueError('New human approval bound to this exact plan is required.')
    if (plan.get('protocol') != PROTOCOL or plan.get('max_calls') != MAX_CALLS
            or plan.get('max_output_tokens') != MAX_OUTPUT_TOKENS or plan.get('retries') != 0
            or plan.get('max_input_bytes') != MAX_INPUT_BYTES or plan.get('timeout_seconds') != TIMEOUT_SECONDS
            or plan.get('source_sha256') != {name: digest(ROOT/name) for name in SOURCE_FILES}
            or plan.get('data_sha256') != digest(DATA)
            or plan.get('requests_sha256') != digest(preflight/'requests.json')):
        raise ValueError('Source, data, requests or bounds changed; prepare and approve a new plan.')
    config = configured_web_model()
    if plan['model'] != settings(config):
        raise ValueError('Model configuration changed after preparation.')
    requests = json.loads((preflight/'requests.json').read_text(encoding='utf-8'))
    if len(requests) != MAX_CALLS or [item['case_id'] for item in requests] != plan['case_ids']:
        raise ValueError('Challenge case sequence changed.')
    return plan, requests, config


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def request_once(config, payload: dict) -> dict:
    request = urllib.request.Request(config.api_base.rstrip('/')+'/chat/completions',
        data=json.dumps(payload, ensure_ascii=False).encode(), method='POST',
        headers={'Authorization': 'Bearer '+config.api_key, 'Content-Type': 'application/json'})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode('utf-8'))


def run(preflight: Path, approval: Path, output: Path) -> dict:
    plan, requests, config = validate(preflight, approval)
    cases = {case['case_id']: case for case in json.loads(DATA.read_text(encoding='utf-8'))['cases']}
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    def append(row):
        with (output/'calls.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    for item in requests:
        case = cases[item['case_id']]
        row = {'case_id': item['case_id'], 'sequence': len(rows)+1, 'status': 'started', 'usage': None}
        append(row)
        started = time.perf_counter()
        try:
            response = request_once(config, item['payload'])
            save(output/f"response-{row['sequence']:02d}.json", response)
            row['usage'] = response.get('usage')
            choice = response['choices'][0]
            raw = choice['message']['content']
            result = json.loads(raw)
            if (choice.get('finish_reason') != 'stop' or choice['message'].get('tool_calls')
                    or result.get('protocol') != 'fact_span_v2' or not valid_verifier_output(result)):
                raise ValueError('Invalid raw verifier result; cannot count as a correct refusal.')
            verification = verify_answer(case['question'], case['answer'], case['sources'],
                                          query=lambda *_args, **_kwargs: raw, audit_coverage=True,
                                          required_points=json.loads(item['payload']['messages'][1]['content'])['required_points'])
            row.update(status='completed', verification=verification,
                       expected_support=case['expected_support'], expected_complete=case['expected_complete'])
            row['model_answer_supported'] = result['answer_supported']
            row['model_answers_question'] = result['answers_question']
            row['model_support_match'] = result['answer_supported'] is case['expected_support']
            row['gate_support_match'] = verification['support_passed'] is case['expected_support']
            row['coverage_binding_passed'] = verification['coverage_contract_passed']
            row['completeness_match'] = (None if case['expected_complete'] is None
                                        else verification['completeness_passed'] is case['expected_complete'])
        except Exception as exc:
            row.update(status='failed', error_type=type(exc).__name__)
        finally:
            row['latency_ms'] = round((time.perf_counter()-started)*1000, 3)
            append(row)
            rows.append(row)
        if row['status'] == 'failed':
            break
    summary = {'protocol': PROTOCOL, 'plan_sha256': digest(preflight/'plan.json'), 'model': plan['model'],
               'api_calls': len(rows), 'planned_cases': len(requests), 'not_executed': len(requests)-len(rows),
               'failed_calls': sum(row['status'] == 'failed' for row in rows),
               'usage_unknown_calls': sum(not isinstance(row['usage'], dict) or
                    not all(type(row['usage'].get(key)) is int for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')) for row in rows),
               'known_tokens': {key: sum(row['usage'].get(key, 0) for row in rows
                    if isinstance(row['usage'], dict) and type(row['usage'].get(key)) is int)
                    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
               'human_confirmation': 'pending', 'scope': 'Diagnostic challenge, not overall accuracy or model causal A/B',
               'cases': rows}
    save(output/'results.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    preparation = sub.add_parser('prepare'); preparation.add_argument('--output', type=Path, required=True)
    execution = sub.add_parser('run')
    execution.add_argument('--preflight', type=Path, required=True)
    execution.add_argument('--approval', type=Path, required=True)
    execution.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare(args.output)
        print(json.dumps({'plan': str(args.output/'plan.json'), 'sha256': digest(args.output/'plan.json'),
                          'model': plan['model']['model'], 'max_calls': plan['max_calls'], 'api_calls': 0}))
    else:
        result = run(args.preflight, args.approval, args.output)
        print(json.dumps({key: result[key] for key in ('api_calls', 'failed_calls', 'usage_unknown_calls', 'known_tokens')}))


if __name__ == '__main__':
    main()
