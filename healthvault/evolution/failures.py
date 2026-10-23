"""Conservative failure categories; no exception or answer text is persisted."""
from __future__ import annotations

FAILURE_CATEGORIES = frozenset({'none', 'safety', 'external_service', 'tool', 'workflow', 'answer_quality', 'unknown'})
EXTERNAL_SERVICE_CODES = frozenset({
    'APIConnectionError', 'APITimeoutError', 'RateLimitError', 'InternalServerError',
    'ServiceUnavailableError', 'ConnectError', 'ConnectTimeout', 'ReadTimeout',
    'provider_connection_failed', 'provider_timeout', 'provider_rate_limited',
})
QUALITY_CODES = frozenset({'answer_verification_failed', 'invalid_cited_answer', 'invalid_verifier_output',
                           'answer_budget_exceeded', 'answer_incomplete', 'citation_validation_failed'})


def classify_trace_failure(trace: dict) -> str:
    code = str(trace.get('failure_code') or '')
    calls = trace.get('tool_calls') or []
    context = trace.get('signal_context') or {}
    checks = context.get('contract_checks') or {}
    if code.startswith('safety_') or any(str(c.get('failure_code') or '').startswith('safety_') for c in calls):
        return 'safety'
    if code in QUALITY_CODES:
        return 'answer_quality'
    if code.startswith(('workflow_', 'knowledge_', 'tool_evidence_')):
        return 'workflow'
    if any(c.get('result_status') in {'failed', 'rejected', 'error', 'invalid_request'} for c in calls):
        return 'tool'
    if code in EXTERNAL_SERVICE_CODES:
        return 'external_service'
    if code:
        return 'unknown'
    if checks.get('citation') == 'failed' or checks.get('answer') == 'failed':
        return 'answer_quality'
    if checks.get('tool') == 'failed':
        return 'tool'
    if trace.get('result_status') == 'succeeded' and context.get('validator_status') != 'failed' and not code:
        return 'none'
    return 'unknown'


def usage_failure_rank(event: dict) -> int:
    if event.get('hard_failure'):
        return 3
    if event.get('task_status') != 'succeeded' or event.get('validator_status') == 'failed':
        return 1 if event.get('excluded_from_skill_rate') else 2
    return 0
