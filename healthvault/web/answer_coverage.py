"""Binding checks for model-assessed semantics and question-point coverage."""
from __future__ import annotations
import re

SEMANTIC_DIMENSIONS = ('object', 'relation', 'polarity', 'conditions', 'qualifiers')
# Linguistic anchors, not medical rules or a general entailment classifier.
_QUALIFIERS = (
    ('blood_sample', r'血液|血样|血浆|血清|抽血|\b(?:blood|plasma|serum)\b'),
    ('urine_sample', r'尿液|尿样|\burin(?:e|ary)\b'),
    ('left_side', r'左(?:侧|叶|肺|肾|乳|甲状腺)|\bleft\b'),
    ('right_side', r'右(?:侧|叶|肺|肾|乳|甲状腺)|\bright\b'),
)


def missing_qualifier_anchors(text: str, quotes: str) -> list[str]:
    return [name for name, pattern in _QUALIFIERS
            if re.search(pattern, text, re.I) and not re.search(pattern, quotes, re.I)]


def check_question_coverage(result: dict, checked_claims: list[dict], spans: list[dict], questions: list[str],
                            required_points: list[dict] | None = None) -> dict:
    """Validate the report, not the model's ability to discover every key point."""
    points = result.get('question_coverage')
    by_span = {span['evidence_id']: span for span in spans}
    by_claim = {claim['claim_id']: claim for claim in checked_claims}
    valid = isinstance(points, list) and 1 <= len(points) <= 12
    required = {point['point_id']: point for point in required_points or []}
    if required_points is not None:
        valid = bool(valid and len(points) == len(required) and required)
    checked = []
    for index, point in enumerate(points if isinstance(points, list) else [], 1):
        if not isinstance(point, dict):
            valid = False
            continue
        evidence_ids, claim_ids = point.get('evidence_ids'), point.get('answer_claim_ids')
        good = (set(point) == {'point_id', 'question_quote', 'description', 'evidence_ids', 'answer_claim_ids', 'status'}
                and point.get('point_id') == f'P{index}'
                and isinstance(point.get('description'), str) and bool(point['description'].strip())
                and isinstance(point.get('question_quote'), str) and bool(point['question_quote'].strip())
                and any(point['question_quote'] in question for question in questions)
                and isinstance(evidence_ids, list) and all(isinstance(item, str) and item in by_span for item in evidence_ids)
                and isinstance(claim_ids, list) and all(isinstance(item, str) and item in by_claim for item in claim_ids))
        if good:
            good = len(set(evidence_ids)) == len(evidence_ids) and len(set(claim_ids)) == len(claim_ids)
        failures = [] if good else ['invalid_point_or_evidence_binding']
        if required_points is not None:
            expected = required.get(point.get('point_id'), {})
            matches = bool(expected and all(point.get(key) == expected.get(key) for key in ('question_quote', 'description')))
            if not matches:
                failures.append('fixed_requirement_changed')
            good = bool(good and matches)
        status = point.get('status')
        if good and status == 'covered':
            good = bool(evidence_ids and claim_ids) and all(
                set(by_claim[item]['evidence_ids']) & set(evidence_ids) for item in claim_ids)
        elif good and status == 'omitted':
            good = bool(evidence_ids) and not claim_ids
        elif good and status == 'missing_evidence':
            good = not evidence_ids and not claim_ids
        else:
            good = False
        if not good and not failures:
            failures.append('status_claim_evidence_mismatch')
        checked.append({**point, 'binding_passed': bool(good),
                        'failure_reasons': failures, 'expected_point': required.get(point.get('point_id')),
                        'quotes': [by_span[item]['quote'] for item in evidence_ids] if good else []})
        valid = bool(valid and good)
    complete = bool(valid and checked and all(point['status'] == 'covered' for point in checked))
    valid = bool(valid and result.get('answers_question') is complete)
    return {'completeness_passed': bool(valid and complete), 'coverage_contract_passed': valid,
            'question_points': checked,
            'missing_evidence_points': [point['point_id'] for point in checked if point.get('status') == 'missing_evidence'],
            'omitted_points': [point['point_id'] for point in checked if point.get('status') == 'omitted'],
            'coverage_assessor': 'model', 'requirements_fixed': required_points is not None,
            'human_confirmed': False}
