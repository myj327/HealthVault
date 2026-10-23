"""Separate per-call execution, per-task contracts, quality and paired benefit."""
from __future__ import annotations


def tool_execution_metrics(calls: list[dict]) -> dict:
    completed = sum(call.get('execution_status') == 'ok' for call in calls)
    return {'tool_attempts': len(calls), 'normal_tool_returns': completed,
            'tool_execution_success_rate': completed / len(calls) if calls else None,
            'empty_searches': sum(call.get('name', call.get('tool_name')) == 'search_health_knowledge'
                                  and call.get('result_status') == 'empty' for call in calls),
            'scope': 'per_call_completion; empty retrieval is a normal return, not goal success'}


def repair_metrics(rows: list[dict]) -> dict:
    known = [row for row in rows if type(row.get('passed')) is bool and type(row.get('generation_attempts')) is int]
    repaired = [row for row in known if row['generation_attempts'] > 1]
    first = sum(row['passed'] and row['generation_attempts'] == 1 for row in known)
    rescued = sum(row['passed'] for row in repaired)
    return {'measured_cases': len(known), 'unknown_cases': len(rows) - len(known),
            'first_pass_count': first, 'repair_attempts': len(repaired), 'repair_rescued_count': rescued,
            'first_pass_rate': first / len(known) if known else None,
            'repair_rescue_rate': rescued / len(repaired) if repaired else None,
            'final_pass_rate': sum(row['passed'] for row in known) / len(known) if known else None,
            'scope': 'caller_supplied_goal_judgment; answered status alone is not a passing judgment'}


def paired_skill_metrics(rows: list[dict]) -> dict:
    cases = {}
    for row in rows:
        key, arm = row['case_id'], row['arm_id']
        if arm not in {'baseline', 'candidate'} or arm in cases.setdefault(key, {}):
            raise ValueError('Each case requires one attempt per declared arm.')
        if type(row.get('passed')) is not bool:
            raise ValueError('Unknown goal outcomes cannot be treated as successes.')
        cases[key][arm] = row
    if any(set(arms) != {'baseline', 'candidate'} for arms in cases.values()):
        raise ValueError('Paired comparison is incomplete.')
    improved = [key for key, arms in cases.items() if not arms['baseline']['passed'] and arms['candidate']['passed']]
    regressed = [key for key, arms in cases.items() if arms['baseline']['passed'] and not arms['candidate']['passed']]
    costs = {}
    for field in ('total_tokens', 'latency_ms', 'tool_attempts', 'unnecessary_calls'):
        costs[field + '_delta'] = (sum(arms['candidate'][field] - arms['baseline'][field] for arms in cases.values())
            if all(type(arm.get(field)) in {int, float} for arms in cases.values() for arm in arms.values()) else None)
    return {'paired_cases': len(cases), 'improved_cases': improved, 'regressed_cases': regressed,
            'baseline_passed': sum(arms['baseline']['passed'] for arms in cases.values()),
            'candidate_passed': sum(arms['candidate']['passed'] for arms in cases.values()), **costs,
            'hard_failure_regressions': ([key for key, arms in cases.items()
                                   if not arms['baseline']['hard_failure'] and arms['candidate']['hard_failure']]
                if all(type(arm.get('hard_failure')) is bool for arms in cases.values() for arm in arms.values()) else None),
            'safety_regressions': ([key for key, arms in cases.items()
                                   if not arms['baseline']['safety_failure'] and arms['candidate']['safety_failure']]
                if all(type(arm.get('safety_failure')) is bool for arms in cases.values() for arm in arms.values()) else None),
            'stable_benefit_proven': False}
