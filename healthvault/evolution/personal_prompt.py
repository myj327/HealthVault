"""Select the one supported personal workflow delta from the original question."""
from __future__ import annotations

import re


CITATION_GUIDANCE_START = '<!--citation:v1-->'
CITATION_GUIDANCE_END = '<!--/citation-->'

_EXCLUSIONS = (
    ('external_transfer', r'\b(?:upload|transmit|exfiltrat\w*|send|email|post|export)\b.{0,80}'
     r'\b(?:webhook|external|remote|email|server|https?)\b|外传|上传|发送.{0,20}(?:外部|邮箱|网址|服务器)'),
    ('cross_user_request', r'\bcross[- ](?:user|profile|account)\b|'
     r'\b(?:other|another|someone else.s)\b.{0,30}\b(?:user|profile|account|patient|records?)\b|'
     r'跨(?:用户|账户|账号)|他人|别人|其他用户|另一个.{0,8}(?:用户|账户|账号)'),
    ('write_request', r'\b(?:save|persist|persistence|insert|update|delete|overwrite|state-changing)\b|'
     r'\bwrite\b.{0,35}\b(?:record|report|database|profile|account|file)\b|'
     r'写入|保存|删除|确认导入|(?:更新|修改|改写).{0,12}(?:档案|记录|报告|数据库|账户|账号)'),
    ('clinical_request', r'\b(?:diagnose|prescribe|treat|treatment|dosage|medical advice)\b|'
     r'\b(?:what|which|adjust)\b.{0,30}\b(?:medication|medicine|drug|dose)\b|'
     r'\bpredict\b.{0,70}\b(?:disease|develop)\b|'
     r'诊断|治疗|开药|处方|用药|剂量|停药|加药|是否患病'),
)
_SINGLE_SOURCE = (
    r'\b(?:one|single|1)\s+(?:(?:stored|returned|direct|observed|knowledge)\s+){0,3}'
    r'(?:citations?|references?|sources?(?:\s+ids?)?|record\s+(?:ids?|tokens?))\b|'
    r'(?:只|仅|恰好|最多)?(?:引用|保留|使用|给出|提供|选择)?\s*(?:一|1|单一|唯一)'
    r'(?:个|条|处|份|项)?\s*(?:原始|直接|知识|记录|原文)?(?:来源|引用|参考资料|证据)|'
    r'引用.{0,10}(?:一|1)(?:个|条|份)(?:原始|最新|最早)?(?:记录|来源|证据)'
)
_COMPARISON_ENDPOINTS = (
    r'\b(?:both|two|2)\s+(?:(?:comparison|original|stored)\s+)?end[- ]?points?\s+'
    r'(?:citations?|references?|records?|sources?)\b|'
    r'\b(?:cite|citing|retain|include|keep)\b.{0,35}\b(?:both|two)\s+endpoints?\b|'
    r'(?:引用|来源|证据|参考).{0,20}(?:两个?时间点|两个?端点|两端)|'
    r'(?:两个?时间点|两个?端点|两端).{0,20}(?:引用|来源|证据|参考)'
)


def select_personal_citation_guidance(content: str, question: str | None) -> tuple[str, str]:
    """Fail closed for legacy/ambiguous scopes; this does not change tool policy."""
    if not isinstance(question, str) or not question.strip():
        return '', 'unknown_question'
    if '<!--citation' not in content and '<!--/citation' not in content:
        return '', 'scope_marker_missing'
    if (content.count('<!--citation') != 1 or content.count('<!--/citation') != 1
            or content.count(CITATION_GUIDANCE_START) != 1 or content.count(CITATION_GUIDANCE_END) != 1):
        return '', 'scope_marker_invalid'
    start = content.index(CITATION_GUIDANCE_START) + len(CITATION_GUIDANCE_START)
    end = content.index(CITATION_GUIDANCE_END)
    guidance = content[start:end].strip()
    if end < start or not guidance:
        return '', 'scope_marker_invalid'
    question = ' '.join(question.casefold().split())
    for reason, pattern in _EXCLUSIONS:
        if re.search(pattern, question):
            return '', 'excluded_' + reason
    if re.search(_COMPARISON_ENDPOINTS, question):
        return guidance, 'matched_comparison_endpoints'
    if re.search(_SINGLE_SOURCE, question) or (
            re.search(r'\b(?:support|ground|anchor|cite|citing)\b', question)
            and re.search(r'\b(?:one|single|1)\s+(?:stored|returned|observed)\s+observation\b', question)):
        return guidance, 'matched_single_source'
    return '', 'no_citation_selection_requirement'
