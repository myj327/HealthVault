"""Evidence cards from this request's trusted domain tool results only."""
from __future__ import annotations


def evidence_title(item: dict) -> str:
    """Keep the parent document context supplied by this same trusted hit."""
    parent = item.get('source', {}).get('title', '')
    title = item['title']
    return f'{parent} / {title}' if isinstance(parent, str) and parent and parent != title else title


def tool_evidence(name: str, payload: dict) -> list[dict]:
    if payload.get('status') not in {'ok', 'ready', 'partial'}:
        return []
    available = set(payload.get('available_citation_ids', []))
    sources = []

    def record(identifier, date, code, evidence, report_id):
        if isinstance(evidence, dict) and isinstance(evidence.get('quote'), str):
            sources.append({'kind': 'record', 'citation_id': identifier, 'report_id': report_id,
                            'title': f'{date} · {code}', 'detail': evidence['quote'], 'url': '',
                            'evidence_locator': {key: value for key, value in evidence.items() if key != 'quote'}})

    def knowledge(items, kind='knowledge'):
        for item in items:
            if isinstance(item, dict) and all(isinstance(item.get(key), str) for key in ('citation_id', 'title', 'excerpt')):
                sources.append({'kind': kind, 'citation_id': item['citation_id'], 'title': evidence_title(item),
                                'detail': item['excerpt'], 'url': item.get('source', {}).get('url', '')})

    if name in {'search_health_knowledge', 'search_official_health_web'}:
        knowledge(payload.get('results', []), 'official_web' if name == 'search_official_health_web' else 'knowledge')
    elif name == 'query_health_timeline':
        for report in payload.get('timeline', []):
            for item in report.get('indicators', []):
                record(f"REC:{report['report_id']}:{item['code']}", report['report_date'], item['code'],
                       item.get('evidence'), report['report_id'])
    elif name == 'compare_health_indicators':
        for item in payload.get('comparisons', []):
            for endpoint in ('earliest', 'latest'):
                point = item[endpoint]
                record(f"REC:{point['report_id']}:{item['code']}", point['report_date'], item['code'],
                       point.get('evidence'), point['report_id'])
    elif name == 'generate_visit_brief':
        knowledge(payload.get('knowledge_citations', []))
        for item in payload.get('record_citations', []):
            record(item['citation_id'], item['report_date'], item['indicator_code'], item.get('evidence'), item['report_id'])
    return [source for source in sources if source['citation_id'] in available]


def merge_tool_evidence(current: list[dict], found: list[dict]) -> list[dict]:
    by_id = {source['citation_id']: source for source in current}
    for source in found:
        previous = by_id.get(source['citation_id'])
        if previous is not None and any(previous.get(field) != source.get(field) for field in ('title', 'detail')):
            raise ValueError('tool_evidence_citation_conflict')
        by_id[source['citation_id']] = source
    return list(by_id.values())
