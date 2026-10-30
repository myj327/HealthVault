"""TASK-114: paired, unprompted ordinary Agent HTTP entrypoint collection."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import secrets
import sqlite3
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from healthvault.evaluation.current_delivery import ROOT, digest, implementation_binding, save
from healthvault.evaluation.e2e import task_043 as fixture
from healthvault.evaluation.measurement import tool_execution_metrics
from healthvault.evaluation.web_rag_run import ModelHTTPClient, value_hash
from healthvault.evolution.review import TARGET_SKILL
from healthvault.rag.vector_cache import model_fingerprint
from healthvault.web.model_runtime import configured_web_model

PROTOCOL = 'task114-ordinary-agent-paired-v1'
SOURCE = ROOT / '.task_outputs/task-105-personal-preflight-v2/workflow'
PROJECT = ROOT / '.bear/skills' / TARGET_SKILL / 'SKILL.md'
CASE_IDS = [f'R5E-{number:03d}' for number in (1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 16, 20)]
ARMS = ('baseline', 'candidate')
LIMITS = {'maximum_calls': 216, 'calls_per_attempt': 9, 'max_output_tokens': 16384,
          'input_bytes_per_call': 160000, 'transport_retries': 0, 'timeout_seconds': 60}
CONDITIONS = {'entrypoint': '/api/profiles/{profile_id}/agent/tasks/{task_id}/messages',
              'auto_plan_suggestions': False, 'answer_repair_enabled': True,
              'answer_verification': True, 'allow_web_search': False,
              'automatic_candidate_generation': False, 'knowledge_mode': 'hybrid',
              'final_output': 'ordinary_product_response', 'runtime_expected_data': False}
ORIGINAL_WORLD_STATE = fixture.world_state


def public_model(config) -> dict:
    return {'model': config.model, 'api_base': config.api_base.rstrip('/'),
            'reasoning_effort': config.reasoning_effort, 'thinking': {'type': 'enabled'}}


def selected_cases() -> list[dict]:
    cases = {case['case_id']: case for case in json.loads((SOURCE / 'dataset.json').read_text(encoding='utf-8'))['cases']}
    return [cases[identifier] for identifier in CASE_IDS]


def prepare(output: Path, candidate: Path) -> dict:
    if output.exists() or not output.resolve().is_relative_to(ROOT):
        raise ValueError('Use a new workspace output directory.')
    config = configured_web_model()
    old = json.loads((SOURCE / 'preflight.json').read_text(encoding='utf-8'))
    if public_model(config) != {key: old['model_config'][key] for key in public_model(config)} or config.model != 'deepseek-flash' or config.reasoning_effort != 'high':
        raise ValueError('Keep the frozen DeepSeek Flash/high model and endpoint.')
    corpus, index = Path(old['corpus_path']), Path(old['hybrid_index_path'])
    paths = [SOURCE / 'dataset.json', SOURCE / 'preflight.json', fixture.FIXTURE_PATH, PROJECT, candidate,
             corpus, corpus.parent.parent / 'aliases/aliases.json',
             *(index / name for name in ('manifest.json', 'index.faiss', 'chunk_ids.json'))]
    if any(str(path.resolve()) in old['bindings'] and digest(path) != old['bindings'][str(path.resolve())] for path in paths):
        raise ValueError('Original fixture, corpus, index or project Skill changed.')
    if digest(candidate) == digest(PROJECT) or model_fingerprint(Path(old['hybrid_model_path'])) != old['embedding_model_sha256']:
        raise ValueError('A distinct candidate and the frozen embedding model are required.')
    save(output / 'dataset.json', {'cases': selected_cases()})
    skills = {}
    for arm, source in zip(ARMS, (PROJECT, candidate)):
        target = output / 'skills' / arm / '.bear/skills' / TARGET_SKILL / 'SKILL.md'
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        skills[arm] = {'root': str(target.parent.parent.resolve()), 'sha256': digest(target)}
        paths.append(target)
    paths.append(output / 'dataset.json')
    plan = {'protocol': PROTOCOL, 'source_sha256': implementation_binding()['sha256'],
            'bindings': {str(path.resolve()): digest(path) for path in paths},
            'dataset': str((output / 'dataset.json').resolve()), 'case_ids': CASE_IDS,
            'skills': skills, 'model': public_model(config), 'limits': LIMITS, 'conditions': CONDITIONS,
            **{key: old[key] for key in ('corpus_path', 'hybrid_index_path', 'hybrid_model_path', 'embedding_model_sha256')},
            'scope': '12_seen_R5_scenarios_paired_ordinary_HTTP_entrypoint',
            'candidate_source': 'TASK-113 scoped rule candidate; zero generation API calls; unpublished',
            'schedule': 'original_case_order_alternating_first_arm', 'human_gold': False,
            'untouched_held_out': False, 'publishes_skill': False, 'api_calls_during_preparation': 0}
    plan['sha256'] = value_hash(plan)
    save(output / 'plan.json', plan)
    return plan


def validate(path: Path) -> tuple[dict, object]:
    plan = json.loads(path.read_text(encoding='utf-8'))
    config = configured_web_model()
    if (plan.get('protocol') != PROTOCOL or plan.get('sha256') != value_hash({key: value for key, value in plan.items() if key != 'sha256'})
            or plan.get('source_sha256') != implementation_binding()['sha256']
            or plan.get('limits') != LIMITS or plan.get('conditions') != CONDITIONS
            or plan.get('case_ids') != CASE_IDS or set(plan.get('skills', {})) != set(ARMS)
            or plan.get('model') != public_model(config)
            or any(digest(Path(name)) != sha for name, sha in plan['bindings'].items())
            or model_fingerprint(Path(plan['hybrid_model_path'])) != plan['embedding_model_sha256']):
        raise ValueError('Frozen source, inputs, model, candidate or limits changed.')
    dataset = Path(plan['dataset'])
    required = [dataset, PROJECT, fixture.FIXTURE_PATH, SOURCE / 'dataset.json', SOURCE / 'preflight.json',
                Path(plan['corpus_path']), Path(plan['corpus_path']).parent.parent / 'aliases/aliases.json',
                *(Path(plan['hybrid_index_path']) / name for name in ('manifest.json', 'index.faiss', 'chunk_ids.json'))]
    for arm in ARMS:
        skill = Path(plan['skills'][arm]['root']) / TARGET_SKILL / 'SKILL.md'
        if not skill.resolve().is_relative_to(path.parent.resolve()) or digest(skill) != plan['skills'][arm]['sha256']:
            raise ValueError('Skill snapshot binding changed.')
        required.append(skill)
    if (any(str(item.resolve()) not in plan['bindings'] for item in required)
            or json.loads(dataset.read_text(encoding='utf-8'))['cases'] != selected_cases()
            or plan['skills']['baseline']['sha256'] != digest(PROJECT)
            or plan['skills']['baseline']['sha256'] == plan['skills']['candidate']['sha256']):
        raise ValueError('Original sample or Skill coverage changed.')
    return plan, config


def usage(rows: list[dict]) -> dict:
    def known(row, key):
        return isinstance(row.get('usage'), dict) and type(row['usage'].get(key)) is int and row['usage'][key] >= 0
    unknown = sum(not all(known(row, key) for key in ('prompt_tokens', 'completion_tokens')) for row in rows)
    inputs = sum(row['usage']['prompt_tokens'] for row in rows if known(row, 'prompt_tokens'))
    outputs = sum(row['usage']['completion_tokens'] for row in rows if known(row, 'completion_tokens'))
    return {'model_calls': len(rows), 'api_calls': sum(row.get('transport_kind') == 'real' for row in rows),
            'unknown_usage_calls': unknown, 'observed_input_tokens': inputs, 'observed_output_tokens': outputs,
            'observed_total_tokens': inputs + outputs, 'total_tokens': None if unknown else inputs + outputs,
            'latency_ms': round(sum(row.get('latency_ms', 0) for row in rows), 3)}


def world_state(path: Path) -> dict:
    connections = []
    def connect(*args, **kwargs):
        connection = sqlite3.connect(*args, **kwargs)
        connections.append(connection)
        return connection
    try:
        with patch.object(fixture, 'sqlite3', SimpleNamespace(connect=connect, Row=sqlite3.Row)):
            return ORIGINAL_WORLD_STATE(path)
    finally:
        for connection in connections:
            connection.close()


class EntryCalls:
    """One ledger for Agent, verifier and repair, without final-answer constraints."""
    def __init__(self, plan: dict, output: Path, client):
        self.plan, self.output, self.client = plan, output, client
        self.rows, self.counts, self.failed = [], Counter(), False
        self.case_id = self.arm_id = ''

    def append(self, row):
        with (self.output / 'calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    def request(self, phase: str, payload: dict) -> dict:
        key = (self.case_id, self.arm_id)
        if self.failed or len(self.rows) >= LIMITS['maximum_calls'] or self.counts[key] >= LIMITS['calls_per_attempt']:
            self.failed = True
            raise RuntimeError('Call budget exhausted or previous transport/protocol failed.')
        if len(json.dumps(payload, ensure_ascii=False).encode('utf-8')) > LIMITS['input_bytes_per_call']:
            self.failed = True
            raise ValueError('Model input byte limit exceeded.')
        row = {'sequence': len(self.rows) + 1, 'case_id': self.case_id, 'arm_id': self.arm_id, 'phase': phase,
               'status': 'started', 'usage': None, 'request_sha256': value_hash(payload),
               'transport_kind': 'real' if self.client.is_real else 'fake'}
        self.rows.append(row)
        self.counts[key] += 1
        self.append(dict(row))
        started = time.perf_counter()
        try:
            response = self.client.complete(payload)
            row['usage'] = response.get('usage') if isinstance(response, dict) else None
            save(self.output / 'responses' / f"{row['sequence']:04d}.json", {'request': payload, 'response': response})
            choice = response['choices'][0]
            message = choice['message']
            tools, text = message.get('tool_calls'), message.get('content')
            if choice.get('finish_reason') not in {'stop', 'tool_calls'} or not isinstance(message, dict):
                raise ValueError('Incomplete original model response.')
            if tools:
                if phase != 'agent' or not isinstance(tools, list):
                    raise ValueError('Unexpected tool calls.')
                for tool in tools:
                    if (tool.get('type') != 'function' or not tool.get('id') or not tool['function'].get('name')
                            or not isinstance(json.loads(tool['function']['arguments']), dict)):
                        raise ValueError('Invalid original tool-call protocol.')
            elif choice.get('finish_reason') != 'stop' or not isinstance(text, str) or not text.strip():
                raise ValueError('Empty or incomplete original response.')
            if phase != 'agent':
                parsed = json.loads(text)  # Product side tasks require complete JSON; no extraction or repair.
                if phase == 'verification':
                    from healthvault.web.answer_validation import AUDITED_VERIFICATION_PROTOCOL, valid_verifier_output
                    if not valid_verifier_output(parsed) or parsed.get('protocol') != AUDITED_VERIFICATION_PROTOCOL:
                        raise ValueError('Invalid original verifier protocol.')
            row['status'] = 'completed'
            return response
        except Exception as exc:
            self.failed = True
            row.update(status='failed', error_type=type(exc).__name__,
                       failure_category='format_protocol' if isinstance(exc, (ValueError, KeyError, TypeError, IndexError)) else 'transport')
            raise
        finally:
            row['latency_ms'] = round((time.perf_counter() - started) * 1000, 3)
            self.append(dict(row))

    async def agent_call(self, agent, phase='agent'):
        model = self.plan['model']
        payload = {'model': model['model'], 'messages': copy.deepcopy(agent._openai_messages),
                   'temperature': 0, 'thinking': model['thinking'], 'reasoning_effort': model['reasoning_effort'],
                   'max_tokens': LIMITS['max_output_tokens'], 'stream': False}
        if agent.tools:
            payload['tools'] = [{'type': 'function', 'function': {'name': item['name'], 'description': item.get('description', ''),
                                'parameters': item.get('input_schema', {'type': 'object'})}} for item in agent.tools]
        response = await asyncio.to_thread(self.request, phase, payload)
        message = response['choices'][0]['message']
        if message.get('content') and not message.get('tool_calls'):
            agent._emit_text(message['content'])
        # The Harness assumes complete usage when present. Keep partial raw usage
        # in the outer ledger and omit it from this compatibility view, never fill zeros.
        if usage([{'usage': response.get('usage')}])['unknown_usage_calls']:
            response = {key: value for key, value in response.items() if key != 'usage'}
        return response

    def callback(self, phase):
        def query(system, data, **kwargs):
            from healthvault.web import model_runtime
            original = model_runtime.build_healthvault_model_only_runtime
            def build(**options):
                runtime = original(**options)
                runtime.agent._call_openai_stream = lambda: self.agent_call(runtime.agent, phase)
                return runtime
            with patch.object(model_runtime, 'build_healthvault_model_only_runtime', build):
                return model_runtime.run_model_task(system, data, **kwargs)
        return query


def pending_rows(cases: list[dict]) -> list[dict]:
    return [{'case_id': case['case_id'], 'question': case['prompt'], 'arm_id': arm, 'status': 'not_started',
             'task_success': None, 'complete_and_supported': None, 'quality_review': 'pending_independent_AI',
             'tool_calls': [], 'model_calls': [], 'usage': usage([])} for case in cases for arm in ARMS]


def run_case(case: dict, arm: str, plan: dict, calls: EntryCalls, root: Path, config) -> dict:
    from fastapi.testclient import TestClient
    from healthvault import agent_runtime as native
    from healthvault.application.tool_evidence import tool_evidence
    from healthvault.evolution.personal_prompt import select_personal_citation_guidance
    from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
    with patch.dict(os.environ, {'HEALTHVAULT_UI_DB_PATH': str(root / 'unused.sqlite3'), 'HEALTHVAULT_KNOWLEDGE_MODE': 'lexical'}):
        from healthvault.web.api import create_app
    from healthvault.web.answers import answer_with_model

    root.mkdir(parents=True, exist_ok=True)
    database = root / 'fixture.sqlite3'
    with patch.object(fixture, 'world_state', world_state):
        before = fixture.materialize_fixture(database)['world_state']
    controller = SimpleNamespace(record_repository=SQLiteHealthRecordRepository(database), knowledge_release_root=root / 'releases',
        knowledge_path=plan['corpus_path'], knowledge_mode='hybrid', hybrid_index_path=plan['hybrid_index_path'], hybrid_model_path=plan['hybrid_model_path'])
    app = create_app(controller, skill_root=plan['skills']['baseline']['root'], skill_evolution_root=root / 'evolution',
        auto_plan_suggestions=False, answer_repair_enabled=True, model_answer_verifier=calls.callback('verification'),
        model_answerer=lambda question, mode, evidence: answer_with_model(question, mode, evidence, query=calls.callback('repair')))
    async def frozen_consider(*_args, **_kwargs):
        return None
    app.state.personal_skills.consider = frozen_consider
    profile = case['profile_id']
    user = app.state.auth_store.register('entrypoint-evaluation', secrets.token_urlsafe(24), user_id=profile)
    token, csrf = app.state.auth_store.create_session(user)
    personal = app.state.personal_skills.service(profile)
    skill_path = Path(plan['skills'][arm]['root']) / TARGET_SKILL / 'SKILL.md'
    personal.store.active_path.write_bytes(skill_path.read_bytes())
    runtimes, tools, sources, searches = [], [], [], []
    original_build = native.build_healthvault_agent_runtime

    def build(options):
        runtime = original_build(options)
        runtimes.append(runtime)
        async def model_call():
            result = await calls.agent_call(runtime.agent)
            for call in result['choices'][0]['message'].get('tool_calls') or []:
                tools.append({'tool_call_id': call['id'], 'name': call['function']['name'],
                              'arguments': json.loads(call['function']['arguments']),
                              'execution_started': False, 'execution_status': 'not_executed'})
            return result
        runtime.agent._call_openai_stream = model_call
        execute = runtime.agent._execute_tool_call
        async def traced(name, arguments):
            record = next(item for item in tools if item['name'] == name and item['arguments'] == arguments
                          and not item['execution_started'])
            record.update(execution_started=True, execution_status='error')
            try:
                raw = await execute(name, arguments)
                record['raw_result'] = raw
                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    payload = None
                if isinstance(payload, dict):
                    record.update(result=payload, result_status=payload.get('status'),
                                  execution_status='ok' if payload.get('status') in {'ok', 'ready', 'empty', 'partial', 'confirmation_required'} else 'error')
                    sources.extend(tool_evidence(name, payload))
                else:
                    record.update(result_format='plain_text_or_non_object', result_status='unstructured_return')
                return raw
            except Exception as exc:
                record['error_type'] = type(exc).__name__
                raise
        runtime.agent._execute_tool_call = traced
        repository = runtime.components.knowledge_repository
        search = repository.search
        def traced_search(query, limit):
            hits = search(query, limit)
            searches.append({'query': query, 'limit': limit, 'metadata': copy.deepcopy(getattr(repository, 'last_search_metadata', {}))})
            return hits
        repository.search = traced_search
        return runtime

    request = {'prompt': case['prompt'], 'expected_version': 0, 'model_consent': True, 'allow_web_search': False}
    started = time.perf_counter()
    row = {'case_id': case['case_id'], 'question': case['prompt'], 'arm_id': arm, 'request': request,
           'task_success': None, 'complete_and_supported': None, 'quality_review': 'pending_independent_AI'}
    client = TestClient(app, raise_server_exceptions=False)
    client.cookies.set('hv_session', token)
    client.headers['X-CSRF-Token'] = csrf
    try:
        with patch.object(native, 'build_healthvault_agent_runtime', build), \
             patch('healthvault.web.api.configured_web_model', return_value=config), \
             patch('healthvault.web.model_runtime.configured_web_model', return_value=config), \
             patch.dict(os.environ, {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}):
            response = client.post(CONDITIONS['entrypoint'].format(profile_id=profile, task_id='task114'), json=request)
        row.update(http_status=response.status_code, response=response.json(),
                   status='collected' if response.status_code == 200 else 'application_error')
        if response.status_code != 200:
            row.update(task_success=False, complete_and_supported=False)
    finally:
        client.close()
        for runtime in runtimes:
            runtime.components.knowledge_repository.close()
    runtime = runtimes[-1] if runtimes else None
    if runtime:
        returned = {message['tool_call_id']: message.get('content') for message in runtime.agent._openai_messages if message.get('role') == 'tool'}
        for tool in tools:
            if 'raw_result' not in tool:
                tool['raw_result'] = returned.get(tool['tool_call_id'])
            if not tool['execution_started']:
                tool.update(execution_status='error', result_status='rejected_or_skipped')
    manifest = copy.deepcopy(runtime.manifest) if runtime else {}
    expected_injection = arm == 'candidate' and bool(select_personal_citation_guidance(skill_path.read_text(encoding='utf-8'), case['prompt'])[0])
    actual_versions = {item['sha256'] for item in manifest.get('project_skills', [])}
    row.update(tool_calls=tools, sources=sources, searches=searches, manifest=manifest,
               injection_contract_passed=bool(runtime) and manifest.get('personal_skill_injected') is expected_injection
                   and plan['skills']['baseline']['sha256'] in actual_versions
                   and (not expected_injection or plan['skills'][arm]['sha256'] in actual_versions),
               expected_personal_injection=expected_injection,
               database_unchanged=world_state(database)['sha256'] == before['sha256'],
               initial_world_state_sha256=before['sha256'], latency_ms=round((time.perf_counter() - started) * 1000, 3),
               tool_metrics=tool_execution_metrics(tools))
    if runtime and runtime.trace_recorder and runtime.trace_recorder.trace_path:
        row['trace'] = json.loads(runtime.trace_recorder.trace_path.read_text(encoding='utf-8'))
    row['model_calls'] = copy.deepcopy([item for item in calls.rows if (item['case_id'], item['arm_id']) == (case['case_id'], arm)])
    if (not runtime and row['http_status'] != 200 and not row['model_calls'] and not tools
            and row['database_unchanged']):
        row['injection_contract_passed'] = None
    if row['injection_contract_passed'] is False or not row['database_unchanged']:
        calls.failed = True
        row.update(status='condition_failure', task_success=False, complete_and_supported=False)
    row['usage'] = usage(row['model_calls'])
    return row


def summarize(rows: list[dict], calls: EntryCalls, status: str) -> dict:
    return {'protocol': PROTOCOL, 'status': status, 'planned_attempts': len(rows),
            'collected_attempts': sum(row['status'] == 'collected' for row in rows),
            'not_started_attempts': sum(row['status'] == 'not_started' for row in rows),
            'cases': rows, 'calls': calls.rows, 'usage': usage(calls.rows),
            'tool_metrics': tool_execution_metrics([tool for row in rows for tool in row['tool_calls']]),
            'task_success_rate': None, 'complete_and_supported_rate': None, 'tokens_per_success': None,
            'quality_review': 'pending_independent_AI', 'human_confirmation': 'pending',
            'paired_improvements': None, 'paired_regressions': None,
            'scope': 'seen_synthetic_R5_tasks; no_new_task_generalization; no_human_gold', 'publishes_skill': False}


def run(path: Path, output: Path, *, approval: Path, review: Path, client=None) -> dict:
    plan, config = validate(path)
    sha = digest(path)
    auth, checked = [json.loads(item.read_text(encoding='utf-8')) for item in (approval, review)]
    if (auth.get('approved') is not True or auth.get('authorization_source') != 'human_user' or auth.get('plan_sha256') != sha
            or checked.get('decision') != 'PASS' or checked.get('plan_sha256') != sha):
        raise ValueError('This exact plan needs the current human authorization and independent review.')
    if output.exists() or not output.resolve().is_relative_to(ROOT) or client is not None and client.is_real:
        raise ValueError('Use a new workspace output; injected clients must be offline.')
    save(ROOT / '.task_outputs/task-114-skill-consumed' / (sha + '.json'), {'plan_sha256': sha, 'output': str(output.resolve())})
    output.mkdir(parents=True)
    calls = EntryCalls(plan, output, client or ModelHTTPClient(plan['model']['api_base'] + '/chat/completions', config.api_key))
    cases = json.loads(Path(plan['dataset']).read_text(encoding='utf-8'))['cases']
    rows, status = pending_rows(cases), 'completed'
    save(output / 'planned-attempts.json', rows)
    for index, case in enumerate(cases):
        for arm in ARMS[::1 if index % 2 == 0 else -1]:
            calls.case_id, calls.arm_id = case['case_id'], arm
            row = next(row for row in rows if row['case_id'] == case['case_id'] and row['arm_id'] == arm)
            row['status'] = 'started'
            try:
                row.update(run_case(case, arm, plan, calls, output / 'cases' / case['case_id'] / arm, config))
            except (Exception, KeyboardInterrupt) as exc:
                row.update(status='incomplete', error_type=type(exc).__name__, task_success=False)
                calls.failed = True
            row['model_calls'] = copy.deepcopy([item for item in calls.rows if (item['case_id'], item['arm_id']) == (case['case_id'], arm)])
            row['usage'] = usage(row['model_calls'])
            save(output / (case['case_id'] + '-' + arm + '.json'), row)
            print(f"TASK114 Skill: {case['case_id']}/{arm} {row['status']}", flush=True)
            if calls.failed:
                status = 'stopped_on_error'
                break
        if status != 'completed':
            break
    if status == 'completed' and any(row['status'] != 'collected' for row in rows):
        status = 'completed_with_errors'
    result = summarize(rows, calls, status)
    save(output / 'results.json', result)
    save(output / 'independent-review.json', {'status': 'pending', 'paid_judge_calls': 0,
         'cases': [{'case_id': case['case_id'], 'question': case['prompt'], 'external_expected': case['expected'],
                    'attempts': [row for row in rows if row['case_id'] == case['case_id']]} for case in cases]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'validate', 'run'))
    for name in ('output', 'candidate', 'plan', 'approval', 'review'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not args.output or not args.candidate:
            parser.error('prepare requires --output and --candidate')
        result = prepare(args.output, args.candidate)
        print(json.dumps({'status': 'prepared', 'sha256': result['sha256'], 'api_calls': 0}))
    elif args.mode == 'validate':
        validate(args.plan)
        print(json.dumps({'status': 'validated', 'api_calls': 0}))
    else:
        if not all((args.plan, args.output, args.approval, args.review)):
            parser.error('run requires --plan, --output, --approval and --review')
        result = run(args.plan, args.output, approval=args.approval, review=args.review)
        print(json.dumps({key: result[key] for key in ('status', 'planned_attempts', 'collected_attempts', 'usage')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
