"""Compare planning suggestions through the knowledge API with identical frozen evidence."""
from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from healthvault.evaluation import knowledge_quality_run as quality

ROOT = quality.ROOT
PROTOCOL = 'knowledge-planning-ablation-v1'
TASK100_GOLD = ROOT / '.task_outputs/task-100-knowledge-preflight-v2/gold.json'
ARMS = {'planning_on': True, 'planning_off': False}
LIMITS = {**quality.LIMITS, 'maximum_calls': 216, 'per_question': {'planning_on': 5, 'planning_off': 4}}
PHASE_CAPS = {'planning': 24, 'generation': 96, 'verification': 96}
CONDITIONS = {'entrypoint': '/api/answers/knowledge', 'variable': 'auto_plan_suggestions',
              'answer_repair_enabled': True, 'answer_verification': True, 'personal_skill_injected': False,
              'allow_web_search': False, 'allow_query_rewrite': False,
              'evidence_policy': 'same_frozen_initial_pool_including_supplemental_searches',
              'online_retrieval_evaluated': False, 'autonomous_agent_routing_evaluated': False}


def questions_only(gold: dict) -> list[dict]:
    return [{key: question[key] for key in ('query_id', 'query')} for question in gold['queries']]


def freeze_evidence(plan: dict, questions: list[dict]) -> list[dict]:
    from healthvault.application.query_rewrite import prepare_query
    from healthvault.application.search_health_knowledge import search_health_knowledge
    from healthvault.infrastructure.hybrid_health_knowledge import HybridHealthKnowledgeRepository
    from healthvault.infrastructure.sqlite_fts_health_knowledge import SQLiteFTSHealthKnowledgeRepository
    from healthvault.web.answers import knowledge_evidence, retrieve_question_knowledge
    corpus = Path(plan['corpus_path'])
    lexical = SQLiteFTSHealthKnowledgeRepository(corpus, corpus.parent.parent / 'aliases/aliases.json')
    with patch.dict(os.environ, {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}):
        repository = HybridHealthKnowledgeRepository.from_local_artifacts(
            lexical, artifact_path=corpus, index_directory=plan['hybrid_index_path'], model_path=plan['hybrid_model_path'])
        rows = []
        try:
            for question in questions:
                searches = []
                def search(query, limit):
                    result = search_health_knowledge({'query': query, 'limit': limit}, repository).to_dict()
                    searches.append({'query': query, 'limit': limit, 'result': copy.deepcopy(result),
                                     'retrieval': copy.deepcopy(repository.last_search_metadata)})
                    return result
                prepared = prepare_query(question['query'])
                result = retrieve_question_knowledge(search, prepared.resolved_question, search_query=prepared.search_query)
                rows.append({**question, 'resolved_question': prepared.resolved_question, 'searches': searches,
                             'tool_result': result, 'sources': knowledge_evidence(result)[1]})
            return rows
        finally:
            repository.close()


def prepare(output: Path, gold_path: Path = TASK100_GOLD) -> dict:
    gold = json.loads(gold_path.read_text(encoding='utf-8'))
    questions = questions_only(gold)
    if questions != questions_only(json.loads(TASK100_GOLD.read_text(encoding='utf-8'))) or len(questions) != 24:
        raise ValueError('Use exactly the 24 original TASK-100 questions in their original order.')
    if output.exists() or not output.resolve().is_relative_to(ROOT):
        raise ValueError('Use a fresh output directory inside the workspace.')
    # Keep the existing input/source freeze as an input receipt, not a second run authorization.
    base = quality.prepare(output / 'inputs', gold_path)
    evidence = freeze_evidence(base, questions)
    quality.save(output / 'questions.json', questions)
    quality.save(output / 'evidence.json', evidence)
    paths = [output / 'inputs/plan.json', output / 'questions.json', output / 'evidence.json', TASK100_GOLD]
    plan = {**base, 'protocol': PROTOCOL, 'arms': ARMS, 'limits': LIMITS, 'phase_caps': PHASE_CAPS,
            'conditions': CONDITIONS, 'questions': str((output / 'questions.json').resolve()),
            'evidence': str((output / 'evidence.json').resolve()),
            'bindings': {**base['bindings'], **{str(path.resolve()): quality.digest(path) for path in paths}},
            'scope': '24_seen_knowledge_questions_paired_frozen_evidence_product_api_regression',
            'schedule': 'original_question_order_alternating_first_arm', 'api_calls_during_preparation': 0}
    quality.save(output / 'plan.json', plan)
    return plan


def validate(plan_path: Path, approval: Path | None = None, review: Path | None = None):
    """Validate offline; execution additionally requires both bound authorization files."""
    from healthvault.evaluation.current_delivery import implementation_binding
    from healthvault.rag.vector_cache import model_fingerprint
    from healthvault.web.model_runtime import configured_web_model
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if approval is not None or review is not None:
        if approval is None or review is None:
            raise ValueError('Both human approval and independent review are required for execution.')
        auth, checked = [json.loads(path.read_text(encoding='utf-8')) for path in (approval, review)]
        sha = quality.digest(plan_path)
        if (auth.get('approved') is not True or auth.get('authorization_source') != 'human_user'
                or auth.get('plan_sha256') != sha or checked.get('decision') != 'PASS'
                or checked.get('plan_sha256') != sha):
            raise ValueError('New human approval and independent review must bind this exact plan.')
    if (plan.get('protocol') != PROTOCOL or plan.get('arms') != ARMS or plan.get('limits') != LIMITS
            or plan.get('phase_caps') != PHASE_CAPS or plan.get('conditions') != CONDITIONS
            or plan.get('schedule') != 'original_question_order_alternating_first_arm'
            or plan.get('source_sha256') != implementation_binding()['sha256']
            or any(quality.digest(Path(name)) != sha for name, sha in plan['bindings'].items())
            or plan.get('embedding_model_sha256') != model_fingerprint(Path(plan['hybrid_model_path']))):
        raise ValueError('Frozen source, input, evidence, model or experiment conditions changed.')
    bound_paths = {str(Path(name).resolve()) for name in plan['bindings']}
    if not {str(Path(plan[key]).resolve()) for key in ('gold', 'questions', 'evidence', 'corpus_path')} <= bound_paths:
        raise ValueError('Question, Gold and evidence files must all be hash-bound.')
    questions = json.loads(Path(plan['questions']).read_text(encoding='utf-8'))
    evidence = json.loads(Path(plan['evidence']).read_text(encoding='utf-8'))
    if (questions != questions_only(json.loads(TASK100_GOLD.read_text(encoding='utf-8')))
            or len(questions) != 24 or [q['query_id'] for q in questions] != plan['case_ids']
            or [{key: row[key] for key in ('query_id', 'query')} for row in evidence] != questions):
        raise ValueError('Frozen original question coverage changed.')
    config = configured_web_model()
    if quality.public_model(config) != plan['model']:
        raise ValueError('Configured model changed.')
    return plan, config


class FrozenEvidenceController:
    """Supply the same original tool result on initial and supplemental API searches."""
    def __init__(self, output: Path):
        self.record_repository = SimpleNamespace(database_path=output / 'records.sqlite3')
        self.knowledge_release_root = output / 'releases'
        self.snapshot, self.searches = {}, []

    def search_knowledge(self, query: str, *, limit: int = 5) -> dict:
        result = copy.deepcopy(self.snapshot['tool_result'])
        # The fixed evidence pool is intentional; no additional source is available to either arm.
        self.searches.append({'query': query, 'limit': limit, 'execution': 'frozen_evidence_replay',
                              'result': copy.deepcopy(result)})
        return result


def make_client(output: Path, calls: quality.Calls, controller: FrozenEvidenceController, *, planning: bool):
    from fastapi.testclient import TestClient
    # Importing api creates a default app; keep that unused app local and free of model downloads.
    with patch.dict(os.environ, {'HEALTHVAULT_UI_DB_PATH': str(output / 'unused.sqlite3'),
                                'HEALTHVAULT_KNOWLEDGE_MODE': 'lexical'}):
        from healthvault.web.api import create_app
    from healthvault.web.answers import answer_with_model
    app = create_app(controller, model_answerer=lambda q, mode, evidence: answer_with_model(
                        q, mode, evidence, query=calls.callback('generation')),
                     model_question_planner=calls.callback('planning'),
                     model_answer_verifier=calls.callback('verification'), auto_plan_suggestions=planning,
                     answer_repair_enabled=True, skill_evolution_root=output / 'skills')
    client = TestClient(app, raise_server_exceptions=False)
    try:
        registered = client.post('/api/auth/register', json={'username': 'planning-evaluation',
                                                            'password': secrets.token_urlsafe(24)})
        registered.raise_for_status()
        client.headers['X-CSRF-Token'] = registered.json()['csrf_token']
        return client
    except Exception:
        client.close()
        raise


def usage(rows: list[dict]) -> dict:
    known = [row['usage'] for row in rows if isinstance(row.get('usage'), dict)
             and all(type(row['usage'].get(key)) is int and row['usage'][key] >= 0
                     for key in ('prompt_tokens', 'completion_tokens'))]
    return {'api_calls': len(rows), 'unknown_usage_calls': len(rows) - len(known),
            'input_tokens': sum(row['prompt_tokens'] for row in known),
            'output_tokens': sum(row['completion_tokens'] for row in known),
            'latency_ms': round(sum(row.get('latency_ms', 0) for row in rows), 3)}


def pending_rows(questions: list[dict]) -> list[dict]:
    return [{'query_id': question['query_id'], 'question': question['query'], 'arm_id': arm,
             'status': 'not_started', 'application_error': None,
             'quality_review': {'status': 'pending_independent_evidence_review', 'human_confirmation': 'pending',
                                'complete_and_supported': None, 'reference_point_coverage': None,
                                'wrong_refusal': None, 'citation_support': None, 'whole_question_tool_trace': None},
             'model_calls': [], 'usage': usage([])} for question in questions for arm in ARMS]


def collect_pair(plan: dict, questions: list[dict], snapshots: list[dict], calls: dict, output: Path, rows: list[dict]):
    clients, controllers = {}, {}
    try:
        for arm, planning in ARMS.items():
            controllers[arm] = FrozenEvidenceController(output / arm)
            clients[arm] = make_client(output / arm, calls[arm], controllers[arm], planning=planning)
        for index, (question, snapshot) in enumerate(zip(questions, snapshots)):
            for arm in list(ARMS)[::1 if index % 2 == 0 else -1]:
                controller, ledger = controllers[arm], calls[arm]
                controller.snapshot = snapshot
                row = next(row for row in rows if row['query_id'] == question['query_id'] and row['arm_id'] == arm)
                row.update(status='started', raw_evidence=copy.deepcopy(snapshot['sources']))
                result = quality.collect_questions(clients[arm], [question], ledger, output / arm, controller.searches)
                row.update(result['cases'][0])
                row['status'] = ('application_error' if row.get('status') == 'error' or row.get('http_status') != 200
                                 else row['result'].get('status', 'application_error'))
                row['application_error'] = row['status'] == 'application_error'
                row['model_calls'] = copy.deepcopy([call for call in ledger.rows if call['case_id'] == question['query_id']])
                row['usage'] = usage(row['model_calls'])
                if result['status'] == 'stopped_on_error':
                    return 'stopped_on_error'
        return 'completed_with_errors' if any(row['application_error'] for row in rows) else 'completed'
    finally:
        for client in clients.values():
            client.close()


def summarize(plan: dict, rows: list[dict], calls: dict, status: str) -> dict:
    for row in rows:
        row['model_calls'] = copy.deepcopy([call for call in calls[row['arm_id']].rows if call['case_id'] == row['query_id']])
        row['usage'] = usage(row['model_calls'])
        if row['status'] == 'started':
            row['status'] = 'incomplete'
    arms = {}
    for arm in ARMS:
        cases = [row for row in rows if row['arm_id'] == arm]
        uncompleted = sum(row['status'] in {'not_started', 'incomplete'} for row in cases)
        errors = sum(row['application_error'] is True for row in cases)
        arms[arm] = {**usage(calls[arm].rows), 'planned_cases': len(cases), 'uncompleted_cases': uncompleted,
                     'application_error_cases': errors,
                     'application_error_rate': errors / len(cases) if cases and not uncompleted else None,
                     'complete_and_supported_rate': None, 'reference_point_coverage_rate': None,
                     'wrong_refusal_rate': None, 'citation_support_rate': None, 'whole_question_tool_trace_rate': None,
                     'semantic_review_status': 'pending_independent_evidence_review'}
    return {'protocol': PROTOCOL, 'status': status, 'conditions': plan['conditions'], 'cases': rows, 'arms': arms,
            **usage([call for ledger in calls.values() for call in ledger.rows]),
            'quality_status': 'pending_independent_evidence_review', 'human_confirmation': 'pending',
            'untouched_held_out': False, 'scope': plan['scope']}


def run(plan_path: Path, approval: Path, review: Path, output: Path) -> dict:
    if approval is None or review is None:
        raise ValueError('Execution requires new human approval and independent review.')
    plan, config = validate(plan_path, approval, review)
    if output.exists() or not output.resolve().is_relative_to(ROOT):
        raise ValueError('Use a fresh output directory inside the workspace.')
    questions = json.loads(Path(plan['questions']).read_text(encoding='utf-8'))
    snapshots = json.loads(Path(plan['evidence']).read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=False)
    from healthvault.evaluation.deepseek_verifier_run import request_once
    calls = {}
    for arm, planning in ARMS.items():
        (output / arm).mkdir()
        calls[arm] = quality.Calls(plan, output / arm, lambda payload: request_once(config, payload),
            limits={**quality.LIMITS, 'maximum_calls': 120 if planning else 96, 'per_question': 5 if planning else 4},
            phase_caps={**quality.PHASE_CAPS, 'planning': 24 if planning else 0})
    rows, error = pending_rows(questions), None
    try:
        quality.save(ROOT / '.task_outputs/task-109-consumed-plans' / (quality.digest(plan_path) + '.json'),
                     {'plan_sha256': quality.digest(plan_path), 'output': str(output.resolve())})
        status = collect_pair(plan, questions, snapshots, calls, output, rows)
    except (Exception, KeyboardInterrupt) as exc:
        status, error = 'stopped_on_error', type(exc).__name__
    result = summarize(plan, rows, calls, status)
    result.update(plan_sha256=quality.digest(plan_path), model=plan['model'], source_sha256=plan['source_sha256'])
    if error:
        result['error_type'] = error
    quality.save(output / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'validate', 'run'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gold', type=Path, default=TASK100_GOLD)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.output is None:
            parser.error('prepare requires --output')
        result = prepare(args.output, args.gold)
    elif args.mode == 'validate':
        if args.plan is None:
            parser.error('validate requires --plan')
        plan, _ = validate(args.plan, args.approval, args.review)
        result = {'protocol': plan['protocol'], 'status': 'valid', 'api_calls': 0}
    else:
        if any(value is None for value in (args.plan, args.output, args.approval, args.review)):
            parser.error('run requires --plan, --output, --approval and --review')
        result = run(args.plan, args.approval, args.review, args.output)
    print(json.dumps({key: result[key] for key in ('protocol', 'status', 'api_calls', 'limits') if key in result}, ensure_ascii=False))
    return 1 if result.get('status') == 'stopped_on_error' else 0


if __name__ == '__main__':
    raise SystemExit(main())
