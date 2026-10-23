"""Keep reviewed factual support and reference-point coverage separate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def answer_quality_metrics(rows: list[dict]) -> dict:
    """Review labels are explicit inputs, never inferred from citation membership."""
    result = {'case_count': len(rows), 'reviewer_kinds': sorted({row.get('reviewer_kind', 'pending') for row in rows})}
    for field, label in (('facts', 'fact_support'), ('reference_points', 'key_point_coverage')):
        items = [item for row in rows for item in row.get(field, [])]
        eligible = [row for row in rows if row['status'] == 'answered'] if field == 'facts' else [row for row in rows if row['answerable']]
        missing_cases = sum(not row.get(field) for row in eligible)
        known = sum(type(item.get('passed')) is bool for item in items)
        passed = sum(item.get('passed') is True for item in items)
        # Unknown labels must not disappear from the denominator or become success.
        result[label] = {'passed': passed, 'total': len(items), 'reviewed': known,
                         'unreviewed_cases': missing_cases,
                         'rate': passed / len(items) if items and known == len(items) and not missing_cases else None}
    answerable = [row for row in rows if row['answerable']]
    unanswerable = [row for row in rows if not row['answerable']]
    result.update(
        answerable_false_refusal_rate=(sum(row['status'] == 'insufficient_evidence' for row in answerable) / len(answerable)
                                      if answerable else None),
        unanswerable_false_answer_rate=(sum(row['status'] == 'answered' for row in unanswerable) / len(unanswerable)
                                       if unanswerable else None),
        errors=sum(row['status'] == 'error' for row in rows),
        human_confirmation_complete=bool(rows) and all(row.get('human_confirmed') is True for row in rows),
    )
    return result


def _review_counts(labels: list[bool | None]) -> dict:
    if any(value is not None and type(value) is not bool for value in labels):
        raise ValueError('Review labels must be true, false or null.')
    passed, failed, pending = labels.count(True), labels.count(False), labels.count(None)
    total = len(labels)
    return {'passed': passed, 'failed': failed, 'pending': pending, 'total': total,
            'rate': passed / total if total and not pending else None,
            'lower_bound': passed / total if total else None,
            'upper_bound': (passed + pending) / total if total else None}


def reviewed_answer_quality(rows: list[dict]) -> dict:
    """Strict whole-question judgments from explicit evidence-review labels."""
    metrics = answer_quality_metrics(rows)
    decisions = []
    for row in rows:
        if type(row['answerable']) is not bool or row['status'] not in {'answered', 'insufficient_evidence', 'error'}:
            raise ValueError('Invalid answerability or original answer status.')
        reasons = []
        if row['status'] == 'error':
            verdict = False
            reasons.append('original_error')
        elif not row['answerable']:
            verdict = row.get('correct_refusal') if row['status'] == 'insufficient_evidence' else False
            reasons.append('correct_refusal' if verdict is True else 'false_answer_or_unreviewed_refusal')
        elif row['status'] != 'answered':
            verdict = False
            reasons.append('false_refusal')
        else:
            labels = []
            for field, reason in (('facts', 'fact_support'), ('reference_points', 'reference_coverage')):
                values = [item.get('passed') for item in row.get(field, [])]
                counts = _review_counts(values)
                labels.extend(values or [None])
                if counts['failed']:
                    reasons.append(reason + '_failed')
                if counts['pending'] or not values:
                    reasons.append(reason + '_pending')
            verdict = False if False in labels else None if None in labels else True
        decisions.append({'query_id': row['query_id'], 'answerable': row['answerable'],
                          'verdict': 'passed' if verdict is True else 'failed' if verdict is False else 'pending',
                          'passed': verdict, 'reasons': reasons})
    metrics['whole_question'] = _review_counts([row['passed'] for row in decisions])
    metrics['answerable_whole_question'] = _review_counts([row['passed'] for row in decisions if row['answerable']])
    metrics['unanswerable_whole_question'] = _review_counts([row['passed'] for row in decisions if not row['answerable']])
    for field, label in (('facts', 'fact_support'), ('reference_points', 'key_point_coverage')):
        counts = _review_counts([item.get('passed') for row in rows for item in row.get(field, [])])
        if metrics[label]['unreviewed_cases']:
            counts.update(rate=None, lower_bound=None, upper_bound=None)
        metrics[label].update(counts)
    metrics['decisions'] = decisions
    return metrics


def load_saved_review(results_path: Path, gold_path: Path, contexts_path: Path, review_path: Path) -> tuple[list[dict], dict]:
    """Bind existing AI labels to unchanged answers, own sources and Gold points."""
    paths = {'results': results_path, 'gold': gold_path, 'contexts': contexts_path, 'review': review_path}
    data = {name: json.loads(path.read_text(encoding='utf-8')) for name, path in paths.items()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    review = data['review']
    for name, key in (('results', 'rag_results_sha256'), ('gold', 'gold_sha256'), ('contexts', 'contexts_sha256')):
        if hashes[name] != review[key]:
            raise ValueError('Saved review input changed: ' + name)

    def indexed(items: list[dict]) -> dict:
        result = {item['query_id']: item for item in items}
        if len(result) != len(items):
            raise ValueError('Duplicate reviewed question.')
        return result

    originals = indexed(data['results']['rows'])
    gold = indexed(data['gold']['queries'])
    contexts = indexed(data['contexts']['queries'])
    reviewed = indexed(review['reviews'])
    if not originals or set(originals) != set(gold) or set(originals) != set(contexts) or set(originals) != set(reviewed):
        raise ValueError('Incomplete saved-answer review coverage.')
    generated, raw_hashes = {}, {}
    ledger_path = results_path.parent / 'calls.jsonl'
    for line in ledger_path.read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event['phase'] != 'generation' or event['status'] == 'started':
            continue
        response_path = results_path.parent / 'responses' / (str(event['sequence']).zfill(4) + '.json')
        response = json.loads(response_path.read_text(encoding='utf-8'))
        content = response['content']
        raw = json.loads(content)  # Full original output, including any trailing text.
        if (event['status'] != 'completed' or response['case_id'] != event['case_id']
                or hashlib.sha256(content.encode()).hexdigest() != event['response_sha256']
                or not isinstance(raw, dict) or set(raw) != {'status', 'claims'} or not isinstance(raw['claims'], list)
                or raw['status'] not in {'answered', 'insufficient_evidence'}
                or (raw['status'] == 'answered') != bool(raw['claims'])):
            raise ValueError('Original generation output is invalid.')
        for claim in raw['claims']:
            if (not isinstance(claim, dict) or set(claim) != {'text', 'citation_id'}
                    or not isinstance(claim['text'], str) or not claim['text'].strip()
                    or not isinstance(claim['citation_id'], str) or not claim['citation_id'].strip()):
                raise ValueError('Original generation claim is invalid.')
        rendered = '\n\n'.join(f"{claim['text'].strip()} [{claim['citation_id']}]" for claim in raw['claims'])
        generated.setdefault(event['case_id'], []).append(rendered)
        raw_hashes[response_path.name] = hashlib.sha256(response_path.read_bytes()).hexdigest()
    rows, case_hashes = [], {}
    for query_id, original in originals.items():
        label, expected, answer = reviewed[query_id], gold[query_id], original['hybrid']
        case_path = results_path.parent / (query_id + '.json')
        if case_path.resolve().parent != results_path.resolve().parent:
            raise ValueError('Invalid saved question ID.')
        case_hash = hashlib.sha256(case_path.read_bytes()).hexdigest()
        if case_hash != label['case_file_sha256'] or json.loads(case_path.read_text(encoding='utf-8')) != original:
            raise ValueError('Original per-question answer changed: ' + query_id)
        case_hashes[query_id] = case_hash
        if (label['arm'] != 'hybrid' or label['answer'] != answer['answer']
                or label['program_status'] != answer['status'] or label['answerable'] != expected['answerable']
                or original['answerable'] != expected['answerable'] or label['question'] != expected['query']
                or answer['question'] != expected['query'] or contexts[query_id]['query'] != expected['query']):
            raise ValueError('Question, answer or eligibility labels drifted: ' + query_id)
        if answer['status'] == 'answered' and answer['answer'] not in generated.get(query_id, []):
            raise ValueError('Saved answer differs from strict raw generation: ' + query_id)
        claims = {claim['claim_id']: claim for claim in answer.get('verification', {}).get('checked_claims', [])}
        facts = label['facts']
        if len({fact['fact_id'] for fact in facts}) != len(facts) or set(claims) != {fact['fact_id'] for fact in facts}:
            raise ValueError('Fact review coverage changed: ' + query_id)
        sources = {source['citation_id']: source for source in answer['sources']}
        for fact in facts:
            raw = claims[fact['fact_id']]
            if any(fact[key] != raw.get(key) for key in ('text', 'context', 'citation_id', 'quotes', 'evidence_ids')):
                raise ValueError('Fact text or citation binding changed: ' + query_id)
            own_source = sources.get(fact['citation_id'], {})
            if not fact['quotes'] or not all(any(quote in own_source.get(field, '') for field in ('title', 'detail'))
                                             for quote in fact['quotes']):
                raise ValueError('Fact quote absent from its own citation: ' + query_id)
        reference = {point['claim_id']: point['text'] for point in expected['reference_claims']}
        points = label['reference_claims']
        if len({point['claim_id'] for point in points}) != len(points) or {p['claim_id']: p['text'] for p in points} != reference:
            raise ValueError('Reference-point review coverage changed: ' + query_id)
        rows.append({'query_id': query_id, 'question': expected['query'], 'answerable': expected['answerable'],
                     'status': answer['status'], 'answer': answer['answer'], 'sources': answer['sources'],
                     'correct_refusal': label['correct_refusal'], 'reviewer_kind': 'assistant', 'human_confirmed': False,
                     'facts': [{**fact, 'passed': fact['supported']} for fact in facts],
                     'reference_points': [{**point, 'passed': point['covered']} for point in points]})
    bindings = {'inputs': {name: {'path': str(paths[name].resolve()), 'sha256': hashes[name]} for name in paths},
                'protocol': 'saved-answer-quality-v1', 'scope': 'TASK083_hybrid_saved_answers_only',
                'scorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'case_files': case_hashes, 'original_implementation_sha256': review['implementation_sha256'],
                'raw_generation_responses': raw_hashes, 'ledger_sha256': hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                'original_review_date': review['review_date'], 'human_confirmation': 'pending',
                'review_kind': 'assistant_labels_reaggregated_with_saved_evidence_checks', 'api_calls': 0}
    return rows, bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'gold', 'contexts', 'review', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    rows, bindings = load_saved_review(args.results, args.gold, args.contexts, args.review)
    metrics = reviewed_answer_quality(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (('bindings.json', bindings), ('reviewed-rows.json', rows), ('summary.json', metrics)):
        with (args.output / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({key: metrics[key] for key in ('case_count', 'whole_question', 'fact_support', 'key_point_coverage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
