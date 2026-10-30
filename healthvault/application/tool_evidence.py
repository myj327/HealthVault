"""Evidence cards from this request's trusted domain tool results only."""
from __future__ import annotations


def evidence_title(item: dict) -> str:
    """Keep the parent document context supplied by this same trusted hit."""
    parent = item.get('source', {}).get('title', '')
    title = item['title']
    return f'{parent} / {title}' if isinstance(parent, str) and parent and parent != title else title


def record_evidence(identifier: str, date: str, code: str, evidence: dict, report_id: str) -> dict:
    """Render trusted record fields together; retain the exact underlying quote."""
    quote = evidence['quote']
    detail = f'报告编号：{report_id}\n报告日期：{date}\n指标：{code}\n原文：{quote}'
    fields = [field.strip() for field in quote.split('|')]
    if len(fields) == 6 and fields[:2] == ['result', code]:
        _, _, value, unit, lower, upper = fields
        suffix = unit if unit == '%' else ' ' + unit
        detail += f'\n原文数值：{value}{suffix}；参考范围 {lower}{suffix}–{upper}{suffix}'
    return {'kind': 'record', 'citation_id': identifier, 'report_id': report_id,
            'title': f'{code} 原始记录', 'detail': detail, 'original_quote': quote, 'url': '',
            'evidence_locator': {key: value for key, value in evidence.items() if key != 'quote'}}


def comparison_evidence(code: str, name: str, comparison: dict, records: dict) -> dict:
    """Carry a tool-computed relationship with both of its original endpoints."""
    identifiers = [comparison[key] for key in ('earliest_record_citation_id', 'latest_record_citation_id')]
    first, last = [records[identifier] for identifier in identifiers]
    unit = comparison['unit']
    suffix = '%' if unit == '%' else ' ' + unit
    change_unit = '个百分点' if unit == '%' else unit
    change = comparison['absolute_change']
    direction = '升高' if change > 0 else '降低' if change < 0 else '不变'
    baseline = comparison['earliest_value']
    relative = f"相对起点变化：{round(change / baseline * 100, 6)}%。" if baseline else '起点为零，不计算相对变化百分比。'
    return {'kind': 'record', 'citation_id': comparison.get('comparison_citation_id', f"REC:derived:{last['report_id']}:{code}"),
            'title': name + ' 跨期数值变化',
            'detail': (f"指标：{name}（{code}）\n"
                       f"起点报告 {first['report_id']}（{first['report_date']}）：{comparison['earliest_value']}{suffix}\n"
                       f"终点报告 {last['report_id']}（{last['report_date']}）：{comparison['latest_value']}{suffix}\n"
                       f"变化方向：{direction}；差值（终点减起点）：{comparison['absolute_change']} {change_unit}；"
                       f"变化幅度：{abs(comparison['absolute_change'])} {change_unit}。{relative}"),
            'url': '', 'derived_from': identifiers,
            'report_links': [{'report_id': record['report_id'], 'title': record['report_date'],
                              'evidence': record['evidence']} for record in (first, last)]}


def brief_comparison_evidence(brief: dict) -> list[dict]:
    records = {item['citation_id']: item for item in brief.get('record_citations', [])}
    return [comparison_evidence(item['code'], item['display_name'], item['comparison'], records)
            for item in brief.get('indicators', []) if item.get('comparison')]


def tool_evidence(name: str, payload: dict) -> list[dict]:
    if payload.get('status') not in {'ok', 'ready', 'partial'}:
        return []
    available = set(payload.get('available_citation_ids', []))
    sources = []

    def record(identifier, date, code, evidence, report_id):
        if isinstance(evidence, dict) and isinstance(evidence.get('quote'), str):
            sources.append(record_evidence(identifier, date, code, evidence, report_id))

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
            derived_id = item.get('comparison_citation_id', f"REC:derived:{item['latest']['report_id']}:{item['code']}")
            if derived_id in available:
                records = {f"REC:{item[end]['report_id']}:{item['code']}": item[end] for end in ('earliest', 'latest')}
                comparison = {**item, 'earliest_value': item['earliest']['value'], 'latest_value': item['latest']['value'],
                              **{f'{end}_record_citation_id': f"REC:{item[end]['report_id']}:{item['code']}"
                                 for end in ('earliest', 'latest')}}
                sources.append(comparison_evidence(item['code'], item['display_name'], comparison, records))
            trend = item.get('trend')
            if trend and trend['citation_id'] in available:
                points = trend['annual_points']
                for point in points:
                    record(f"REC:{point['report_id']}:{item['code']}", point['report_date'], item['code'], point['evidence'], point['report_id'])
                labels = {'increased': '逐年升高', 'decreased': '逐年降低', 'unchanged': '持平',
                          'fluctuating': '有升有降、呈波动', 'insufficient_data': '不足两年，无法判断长期趋势'}
                details = [f"{point['report_date']}：{point['value']} {item['unit']}（报告 {point['report_id']}）" for point in points]
                details += ['每年使用最后一条已确认记录；缺失年份不插值。',
                            '逐年代表值方向：' + labels[trend['direction']]]
                if trend['slope_per_year'] is not None:
                    slope_unit = '个百分点' if item['unit'] == '%' else item['unit']
                    details.append(f"按实际日期线性拟合的年均变化：{trend['slope_per_year']} {slope_unit}/年；仅描述记录，不预测病情。")
                sources.append({'kind': 'record', 'citation_id': trend['citation_id'], 'title': item['display_name'] + ' 年度趋势',
                    'detail': '\n'.join(details), 'url': '', 'derived_from': [f"REC:{point['report_id']}:{item['code']}" for point in points],
                    'report_links': [{'report_id': point['report_id'], 'title': point['report_date'], 'evidence': point['evidence']} for point in points]})
    elif name == 'generate_visit_brief':
        knowledge(payload.get('knowledge_citations', []))
        for item in payload.get('record_citations', []):
            record(item['citation_id'], item['report_date'], item['indicator_code'], item.get('evidence'), item['report_id'])
        sources.extend(brief_comparison_evidence(payload))
    return [source for source in sources if source['citation_id'] in available]


def merge_tool_evidence(current: list[dict], found: list[dict]) -> list[dict]:
    by_id = {source['citation_id']: source for source in current}
    for source in found:
        previous = by_id.get(source['citation_id'])
        if previous is not None and any(previous.get(field) != source.get(field) for field in ('title', 'detail')):
            raise ValueError('tool_evidence_citation_conflict')
        by_id[source['citation_id']] = source
    return list(by_id.values())
