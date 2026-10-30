"""Application completion conditions; independent of Skill learning and prompts."""
from __future__ import annotations

import json
import re
from collections.abc import Collection
from healthvault.adapters.bearcode.final_response_normalizer import FinalResponseNormalizer, parse_final_response

WORKFLOW_PROTOCOL = 'healthvault-workflow-v2'
MAX_KNOWLEDGE_QUERIES = 3
FINAL_OUTPUT_GUIDANCE = (
    '有据事实用自然语言逐项回答，每个事实后写 [实际 citation_id]，例如 [REC:report:indicator]；'
    'ID 必须来自本次工具，保留完整 ID 和方括号，裸 ID 或反引号不算引用，示例 ID 不能直接使用。'
    '工具返回空或资料不足时，终答只输出 {"final_status":"insufficient_evidence","citations":[],"refusal":true}。'
    '越权、未批准写入或其他不支持请求被拒绝时，终答只输出 {"final_status":"rejected","citations":[],"refusal":true}。'
    '这两种终态必须是原始 JSON，不加代码围栏、解释或其他字段；不得编造来源。'
    '可回答的问题仍须先执行必要工具，不能用拒答代替查证。'
)
REQUIRED_TOOLS = {
    'record_timeline': 'query_health_timeline',
    'indicator_comparison': 'compare_health_indicators',
    'visit_preparation': 'generate_visit_brief',
    'knowledge_lookup': 'search_health_knowledge',
    'report_preview': 'preview_health_report',
}


def knowledge_query(query: object) -> str:
    return ' '.join(str(query).casefold().split())


def knowledge_search_issue(query: object, previous_queries: Collection[str], previous_status: str | None) -> str | None:
    expression = knowledge_query(query)
    if expression in previous_queries:
        return 'workflow_search_duplicate_query'
    if len(previous_queries) >= MAX_KNOWLEDGE_QUERIES:
        return 'workflow_search_budget_exceeded'
    if previous_queries and previous_status not in {'empty', 'ok', 'partial'}:
        return 'workflow_search_after_failed_result'
    return None


def workflow_intent(prompt: str) -> str:
    lowered = prompt.casefold()
    routes = (
        (('就诊', '复诊', 'visit', 'appointment', 'brief'), 'visit_preparation'),
        (('对比', '比较', '变化', 'compare', 'comparison', 'change'), 'indicator_comparison'),
        (('时间线', '历史', 'timeline', 'chronological'), 'record_timeline'),
        (('知识', '资料', '指南', '是什么', 'knowledge', 'guideline', 'search'), 'knowledge_lookup'),
        (('预览', 'preview'), 'report_preview'),
    )
    return next((label for words, label in routes if any(word in lowered for word in words)),
                'general_read_only_workflow')


def completion_feedback(text: str, calls: list[dict], intent: str, *, terminal: bool = False) -> str | None:
    required = REQUIRED_TOOLS.get(intent)
    if not terminal and required and not any(call['name'] == required and call['result_status'] in
            {'ok', 'ready', 'empty', 'partial', 'insufficient_data', 'insufficient_evidence'} for call in calls):
        return f'先调用 {required} 并查看结果，再决定回答或证据不足；不要猜测不存在记录。'
    final = parse_final_response(text)
    if text.lstrip().startswith(('```', '{', '[')):
        try:
            final = json.loads(text)
        except ValueError:
            return '原始终答必须是完整合法 JSON，不能追加说明或省略字段。'
    available = {item for call in calls for item in call['available_citation_ids'] if isinstance(item, str)}
    if final is None:
        if text.lstrip().startswith(('{', '[')):
            return '原始终答必须是完整合法 JSON，不能追加说明或省略字段。'
        if not calls:
            return ('尚无本次工具证据；可回答的请求先调用必要工具。若请求越权或不支持，终答只输出 '
                    '{"final_status":"rejected","citations":[],"refusal":true}，不要附加说明。')
        actual = FinalResponseNormalizer().normalize(
            {'final_status': 'ok', 'citations': sorted(available), 'refusal': False}, calls)
        if actual.final_status != 'ok':
            return ('工具结果不能支持肯定答案；终答只输出原始 JSON，不附加说明：'
                    + json.dumps({'final_status': actual.final_status, 'citations': [], 'refusal': True}))
        citations = re.findall(r'\[((?:HVK|HVW|REC):[^\s\[\]]+)\]', text)
        if not citations or not set(citations).issubset(available):
            return ('每个事实必须使用本次工具的实际来源，格式为 [完整 citation_id]；'
                    '裸 ID 或反引号不算引用，不得使用不存在的 ID。')
        return None
    if not isinstance(final, dict):
        return '原始终答必须是 JSON 对象。'
    normalized = FinalResponseNormalizer().normalize(final, calls)
    if (set(final) != {'final_status', 'citations', 'refusal'}
            or not isinstance(final.get('citations'), list)
            or type(final.get('refusal')) is not bool
            or any(not isinstance(item, str) or item not in available for item in final['citations'])
            or final['citations'] != list(normalized.citations)
            or final.get('final_status') != normalized.final_status
            or final['refusal'] is not normalized.refusal
            or final['final_status'] == 'ok' and not final['citations']):
        return '终态、refusal 和 citations 必须与实际工具结果一致；不能靠归一化修正失败。'
    return None
