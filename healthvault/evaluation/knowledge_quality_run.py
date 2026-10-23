"""Prepare and collect current Web knowledge answers; semantic grades stay separate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'current-knowledge-quality-v1'
PHASE_CAPS = {'planning': 24, 'generation': 48, 'verification': 48}
LIMITS = {'maximum_calls': 120, 'per_question': 5, 'max_output_tokens': 16384,
          'max_input_bytes': 80000, 'timeout_seconds': 120, 'retries': 0}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def public_model(config) -> dict:
    from healthvault.evaluation.deepseek_verifier_run import settings
    return {**settings(config), 'thinking': {'type': 'enabled'}}


def prepare(output: Path, gold_path: Path) -> dict:
    from healthvault.evaluation.current_delivery import implementation_binding
    from healthvault.rag.releases import active_corpus_root, configured_release_root
    from healthvault.rag.retrieval_config import selection
    from healthvault.rag.vector_cache import model_fingerprint
    from healthvault.web.model_runtime import configured_web_model
    corpus = active_corpus_root()
    retrieval = selection(corpus, configured_release_root())
    if retrieval['knowledge_mode'] != 'hybrid' or not retrieval['hybrid_model_path'] or not retrieval['hybrid_index_path']:
        raise ValueError('Configured local hybrid model and index are required.')
    gold = json.loads(gold_path.read_text(encoding='utf-8'))
    if len(gold['queries']) != 24 or len({q['query_id'] for q in gold['queries']}) != 24:
        raise ValueError('Use the unchanged 24-question regression Gold.')
    corpus_path, aliases = corpus / 'chunks/hybrid_v1.json', corpus / 'aliases/aliases.json'
    if digest(corpus_path) != gold['corpus']['artifact_sha256'] or digest(aliases) != gold['corpus']['aliases_sha256']:
        raise ValueError('Gold does not match the current corpus.')
    index, model = Path(retrieval['hybrid_index_path']), Path(retrieval['hybrid_model_path'])
    model_config = public_model(configured_web_model())
    sources = implementation_binding()
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'gold.json', gold)
    save(output / 'sources.json', sources)
    inputs = [output / 'gold.json', output / 'sources.json', corpus_path, aliases]
    inputs += [index / name for name in ('manifest.json', 'index.faiss', 'chunk_ids.json')]
    plan = {'protocol': PROTOCOL, 'model': model_config, 'limits': LIMITS, 'phase_caps': PHASE_CAPS,
            'source_sha256': sources['sha256'], 'bindings': {str(p.resolve()): digest(p) for p in inputs},
            'gold': str((output / 'gold.json').resolve()), 'case_ids': [q['query_id'] for q in gold['queries']],
            'knowledge_mode': 'hybrid', 'corpus_path': str(corpus_path.resolve()),
            'hybrid_index_path': str(index.resolve()), 'hybrid_model_path': str(model.resolve()),
            'embedding_model_sha256': model_fingerprint(model), 'approval_required': True,
            'allow_web_search': False, 'allow_query_rewrite': False, 'api_calls_during_preparation': 0,
            'scope': '24_seen_questions_current_single_arm_web_retrieval_planning_generation_verification',
            'semantic_grader': 'independent_assistant_reading_own_evidence_and_frozen_reference_points',
            'paid_judge_calls': 0, 'human_confirmation': 'pending', 'untouched_held_out': False}
    save(output / 'plan.json', plan)
    return plan


def validate(plan_path: Path, approval: Path, review: Path):
    from healthvault.evaluation.current_delivery import implementation_binding
    from healthvault.rag.vector_cache import model_fingerprint
    from healthvault.web.model_runtime import configured_web_model
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    auth, checked = [json.loads(path.read_text(encoding='utf-8')) for path in (approval, review)]
    sha = digest(plan_path)
    if (auth.get('approved') is not True or auth.get('authorization_source') != 'human_user'
            or auth.get('plan_sha256') != sha or checked.get('decision') != 'PASS' or checked.get('plan_sha256') != sha):
        raise ValueError('New human approval and independent review bound to this exact plan are required.')
    if (plan['protocol'] != PROTOCOL or plan['limits'] != LIMITS or plan['phase_caps'] != PHASE_CAPS
            or plan['source_sha256'] != implementation_binding()['sha256']
            or any(digest(Path(name)) != sha for name, sha in plan['bindings'].items())
            or plan['embedding_model_sha256'] != model_fingerprint(Path(plan['hybrid_model_path']))):
        raise ValueError('Frozen source, input, model or budget changed.')
    gold = json.loads(Path(plan['gold']).read_text(encoding='utf-8'))
    if [q['query_id'] for q in gold['queries']] != plan['case_ids'] or len(plan['case_ids']) != 24:
        raise ValueError('Frozen question coverage changed.')
    config = configured_web_model()
    if public_model(config) != plan['model']:
        raise ValueError('Configured model changed.')
    return plan, config


class Calls:
    def __init__(self, plan: dict, output: Path, request, *, limits: dict | None = None, phase_caps: dict | None = None):
        self.plan, self.output, self.request = plan, output, request
        self.limits, self.phase_caps = dict(limits or LIMITS), dict(phase_caps or PHASE_CAPS)
        self.rows, self.counts = [], Counter()
        self.case_id, self.failed = '', False

    def append(self, row):
        with (self.output / 'calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    def call(self, phase: str, system: str, data: dict, **_kwargs) -> str:
        if (self.failed or phase not in self.phase_caps or self.counts[phase] >= self.phase_caps[phase]
                or len(self.rows) >= self.limits['maximum_calls']
                or sum(row['case_id'] == self.case_id for row in self.rows) >= self.limits['per_question']
                or sum(row['case_id'] == self.case_id and row['phase'] == phase for row in self.rows)
                   >= (1 if phase == 'planning' else 2)):
            raise RuntimeError('Frozen call budget exhausted or a previous call failed.')
        payload = {'model': self.plan['model']['model'], 'reasoning_effort': 'high', 'thinking': {'type': 'enabled'},
                   'stream': False, 'max_tokens': self.limits['max_output_tokens'],
                   'messages': [{'role': 'system', 'content': system},
                                {'role': 'user', 'content': json.dumps(data, ensure_ascii=False)}]}
        if len(json.dumps(payload, ensure_ascii=False).encode()) > self.limits['max_input_bytes']:
            raise ValueError('Model input limit exceeded before transmission.')
        row = {'sequence': len(self.rows) + 1, 'case_id': self.case_id, 'phase': phase, 'status': 'started',
               'request_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), 'usage': None}
        self.rows.append(row)
        self.counts[phase] += 1
        self.append(dict(row))
        started = time.perf_counter()
        try:
            response = self.request(payload)
            row['usage'] = response.get('usage')
            save(self.output / 'responses' / f"{row['sequence']:04d}.json", {'request': payload, 'response': response})
            choice = response['choices'][0]
            text = choice['message']['content']
            row['finish_reason'] = choice.get('finish_reason')
            if choice.get('finish_reason') != 'stop' or choice['message'].get('tool_calls') or not isinstance(text, str):
                row['failure_reason'] = ('output_token_limit' if choice.get('finish_reason') == 'length'
                                         else 'incomplete_or_unexpected_output')
                raise ValueError('Incomplete model output or unexpected tools.')
            try:
                json.loads(text)  # Strict full original JSON; do not trim or extract it.
            except ValueError:
                row['failure_reason'] = 'invalid_original_json'
                raise
            row['status'] = 'completed'
            return text
        except Exception as exc:
            row.update(status='failed', error_type=type(exc).__name__)
            self.failed = True
            raise
        finally:
            row['latency_ms'] = round((time.perf_counter() - started) * 1000, 3)
            self.append(dict(row))

    def callback(self, phase):
        return lambda system, data, **kwargs: self.call(phase, system, data, **kwargs)


def collect_questions(client, questions: list[dict], calls: Calls, output: Path, searches: list) -> dict:
    rows, status = [], 'completed'
    for question in questions:
        calls.case_id = question['query_id']
        offset, started = len(searches), time.perf_counter()
        row = {'query_id': question['query_id'], 'question': question['query']}
        try:
            response = client.post('/api/answers/knowledge', json={'question': question['query'], 'model_consent': True,
                                                                  'allow_web_search': False, 'allow_query_rewrite': False})
            row['http_status'] = response.status_code
            try:
                row['result'] = response.json()
            except ValueError:
                row.update(status='error', error_type='NonJSONResponse', response_text=response.text)
                status = 'stopped_on_error'
        except Exception as exc:
            row.update(status='error', error_type=type(exc).__name__)
            status = 'stopped_on_error'
        row.update(searches=copy.deepcopy(searches[offset:]),
                   latency_ms=round((time.perf_counter() - started) * 1000, 3))
        failed_calls = [call for call in calls.rows if call['case_id'] == calls.case_id and call['status'] == 'failed']
        if failed_calls:
            row['model_failures'] = [{key: call[key] for key in ('sequence', 'phase', 'error_type', 'failure_reason', 'finish_reason')
                                      if key in call} for call in failed_calls]
        rows.append(row)
        save(output / (question['query_id'] + '.json'), row)
        if calls.failed or status == 'stopped_on_error':
            status = 'stopped_on_error'
            break
    known = [r['usage'] for r in calls.rows if isinstance(r.get('usage'), dict)
             and all(type(r['usage'].get(k)) is int and r['usage'][k] >= 0 for k in ('prompt_tokens', 'completion_tokens'))]
    errors = sum(r.get('status') == 'error' or r.get('http_status', 500) != 200 for r in rows)
    if errors and status == 'completed':
        status = 'completed_with_errors'
    return {'protocol': PROTOCOL, 'status': status, 'cases': rows, 'planned_cases': len(questions),
            'api_calls': len(calls.rows), 'unknown_usage_calls': len(calls.rows) - len(known),
            'input_tokens': sum(r['prompt_tokens'] for r in known), 'output_tokens': sum(r['completion_tokens'] for r in known),
            'http_error_cases': errors,
            'quality_status': 'pending_independent_assistant_evidence_review', 'human_confirmation': 'pending'}


def run(plan_path: Path, approval: Path, review: Path, output: Path) -> dict:
    plan, config = validate(plan_path, approval, review)
    if output.exists() or not output.resolve().is_relative_to(ROOT):
        raise ValueError('Use a fresh output directory inside the workspace.')
    output.mkdir(parents=True, exist_ok=False)
    calls = Calls(plan, output, None)
    try:
        save(ROOT / '.task_outputs/task-097-approvals' / (digest(plan_path) + '.json'),
             {'plan_sha256': digest(plan_path), 'output': str(output.resolve())})
        from healthvault.evaluation.deepseek_verifier_run import request_once
        calls.request = lambda payload: request_once(config, payload)
        result = _run_web(plan, calls, output)
    except Exception as exc:
        result = {'protocol': PROTOCOL, 'status': 'stopped_on_error', 'error_type': type(exc).__name__,
                  'api_calls': len(calls.rows), 'calls': calls.rows, 'quality_status': 'not_evaluated'}
    result.update(plan_sha256=digest(plan_path), model=plan['model'], source_sha256=plan['source_sha256'])
    save(output / 'results.json', result)
    return result


def _run_web(plan: dict, calls: Calls, output: Path) -> dict:
    with patch.dict(os.environ, {'HEALTHVAULT_UI_DB_PATH': str(output / 'unused.sqlite3'),
                                'HEALTHVAULT_KNOWLEDGE_MODE': 'hybrid',
                                'HEALTHVAULT_HYBRID_MODEL_PATH': plan['hybrid_model_path'],
                                'HEALTHVAULT_HYBRID_INDEX_PATH': plan['hybrid_index_path']}):
        from fastapi.testclient import TestClient
        from healthvault.web.api import create_app
        from healthvault.ui.controller import HealthVaultUIController
        from healthvault.web.answers import answer_with_model
        controller = HealthVaultUIController(output / 'knowledge.sqlite3', knowledge_release_root=output / 'releases')
        if controller.knowledge_path.resolve() != Path(plan['corpus_path']).resolve() or controller.knowledge_mode != 'hybrid':
            raise ValueError('Actual Web corpus or hybrid runtime differs from the frozen plan.')
        searches = []
        repository = controller.knowledge_repository
        search = repository.search
        def traced_search(query, limit):
            hits = search(query, limit)
            searches.append({'query': query, 'limit': limit, **copy.deepcopy(repository.last_search_metadata or {}),
                             'citation_ids': [hit.document.citation_id for hit in hits]})
            return hits
        repository.search = traced_search
        app = create_app(controller, model_answerer=lambda q, mode, evidence: answer_with_model(q, mode, evidence,
                         query=calls.callback('generation')), model_answer_verifier=calls.callback('verification'),
                         model_question_planner=calls.callback('planning'), skill_evolution_root=output / 'skills')
        client = TestClient(app, raise_server_exceptions=False)
        try:
            registered = client.post('/api/auth/register', json={'username': 'quality-evaluation', 'password': secrets.token_urlsafe(24)})
            registered.raise_for_status()
            client.headers['X-CSRF-Token'] = registered.json()['csrf_token']
            questions = json.loads(Path(plan['gold']).read_text(encoding='utf-8'))['queries']
            # Only question text and IDs reach collection; reference answers never reach a model callback.
            result = collect_questions(client, [{k: q[k] for k in ('query_id', 'query')} for q in questions], calls, output, searches)
            return result
        finally:
            client.close()
            repository.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    result = prepare(args.output, args.gold) if args.mode == 'prepare' else run(args.plan, args.approval, args.review, args.output)
    print(json.dumps({k: result[k] for k in ('protocol', 'api_calls', 'status', 'limits') if k in result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
