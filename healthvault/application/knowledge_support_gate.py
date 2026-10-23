"""Deterministic intent and corpus-support gate for curated health knowledge."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from healthvault.domain.knowledge import KnowledgeHit
from healthvault.repositories.health_knowledge import HealthKnowledgeRepository


@dataclass(frozen=True)
class KnowledgeSupportDecision:
    allowed: bool
    reason_code: str | None = None
    message: str | None = None
    stage: str | None = None


SUPPORT_GATE_RULES: tuple[dict[str, object], ...] = (
    {
        "reason_code": "unsupported_current_guideline",
        "message": "The available sources do not establish a current clinical guideline version.",
        "patterns": (
            r"(?:最新|现行|当前|截至).{0,20}(?:指南|共识|推荐|标准)",
            r"(?:指南|共识|推荐|标准).{0,20}(?:最新|现行|当前|20\d{2})",
            r"\b(?:latest|current|updated)\b.{0,50}\b(?:guideline|recommendation|standard)\b",
        ),
    },
    {
        "reason_code": "unsupported_personal_record",
        "message": "The fixed public knowledge corpus cannot retrieve personal health records.",
        "patterns": (
            r"\b(show|find|retrieve|look up|what (?:is|was))\b.*\bmy\b.*\b(record|history|result|reading|value|level|report|ldl|glucose|a1c|cholesterol)\b",
            r"\bmy\b.*\b(last|previous|latest|yesterday|history|record)\b.*\b(result|reading|value|level|report|ldl|glucose|a1c|cholesterol)\b",
            r"我的.*(?:上次|上个月|去年|历史|记录|报告).*(?:多少|查询|显示|结果|数值|水平)",
        ),
    },
    {
        "reason_code": "unsupported_diagnosis",
        "message": "The knowledge search cannot diagnose a person from symptoms or measurements.",
        "patterns": (
            r"\bdiagnos(?:e|is|ing)\b.*\b(me|my|i have|which disease|which condition)\b",
            r"\b(do i|have i|am i)\b.*\b(disease|condition|diabetes|prediabetes)\b",
            r"(?:根据|结合).*(?:症状|指标|结果|a1c|血糖).*(?:诊断|判断).*(?:病|疾病)",
            r"(?:诊断我|我(?:得了|患有|是不是)).*(?:病|疾病|糖尿病)",
        ),
    },
    {
        "reason_code": "unsupported_treatment",
        "message": "Treatment, medication, and dosage requests are outside this corpus.",
        "patterns": (
            r"\b(treat|treatment|medicine|medication|drug|dose|dosage|prescrib|therapy|cure)\w*\b",
            r"(?:治疗|用药|药物|剂量|处方|吃什么药|怎么治)",
        ),
    },
    {
        "reason_code": "unsupported_future_prediction",
        "message": "The corpus cannot predict an exact future health measurement.",
        "patterns": (
            r"\b(predict|forecast)\w*\b.*\b(exact|exactly|tomorrow|future|next)\b",
            r"\b(exact|exactly|tomorrow|future)\b.*\b(predict|forecast|value|level|reading)\w*\b",
            r"(?:预测|预报|估算).*(?:明天|未来|准确|精确).*(?:数值|水平|多少|结果)?",
            r"(?:明天|未来).*(?:会是多少|准确值|精确值|血糖值|水平)",
        ),
    },
)

_GENERIC_QUERY_TERMS = frozenset(
    {
        "a", "about", "and", "are", "be", "before", "can", "does", "for",
        "from", "health", "how", "is", "it", "laboratory", "lab", "medical",
        "mean", "means", "measure", "measured", "measures", "measurement",
        "my", "of", "person", "range", "reference", "result", "results",
        "should", "test", "tests", "the", "this", "to", "used", "what",
        "when", "which", "why", "with", "your",
        # Question/action words describe the requested operation, not the medical topic.
        "detect", "detected", "detecting", "detects", "explain", "explained",
        "explaining", "explains", "include", "included", "includes", "including",
        "indicate", "indicated", "indicates", "indicating", "interpret",
        "interpretation", "interpreted", "interpreting", "interprets", "know",
        "need", "needed", "needs", "prepare", "prepared", "preparing", "question",
        "questions", "show", "showed", "showing", "shown", "shows", "tell",
        "telling", "tells", "understand", "understanding", "use", "uses", "using",
        # Modality, qualifier, and conversational words are not independent topics.
        "abnormal", "abnormality", "ate", "bad", "blood", "check", "described",
        "diagnose", "diagnosed", "diagnoses", "diagnosing", "diagnosis", "down",
        "draw", "during", "eat", "good", "high", "many", "must", "normal",
        "own", "preparation", "report", "require", "required", "requires", "seeing",
        "special", "write",
    }
)

_GENERIC_CJK_TERMS = (
    "参考范围", "为什么", "意味着", "什么", "如何", "怎么", "是否", "可以",
    "能否", "检查", "化验", "结果", "范围", "解释", "解读", "显示", "检测",
    "发现", "使用", "需要", "应该", "作用", "目的", "帮助", "抽血", "白水",
    "一般", "一天", "任何", "时间", "含义", "建议", "同行", "陪同", "能喝",
    "时候", "期间", "中的", "之前", "之后", "进行", "相关", "哪些", "哪种",
    "自己", "报告", "数值", "正常", "异常", "多少", "以及", "还是", "或者",
    "请问", "告诉", "了解", "知道", "看看", "查看", "准备",
)

_CONCEPT_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "a1c": ("a1c", "hba1c", "glycated hemoglobin", "glycosylated hemoglobin"),
    "appointment": ("appointment", "doctor visit", "medical visit", "clinician visit"),
    "cholesterol": ("cholesterol", "lipid", "lipoprotein"),
    "doctor": ("doctor", "clinician", "provider", "physician"),
    "family": ("family", "friend", "trusted person"),
    "fasting": ("fasting", "fast", "nothing to eat or drink"),
    "glucose": ("glucose", "blood sugar"),
    "laboratory": ("laboratory", "lab"),
}

_CJK_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("糖化血红蛋白", _CONCEPT_EQUIVALENTS["a1c"]),
    ("空腹", _CONCEPT_EQUIVALENTS["fasting"]),
    ("血糖", _CONCEPT_EQUIVALENTS["glucose"]),
    ("胆固醇", _CONCEPT_EQUIVALENTS["cholesterol"]),
    ("血脂", _CONCEPT_EQUIVALENTS["cholesterol"]),
    ("就诊", (*_CONCEPT_EQUIVALENTS["appointment"], "visit preparation")),
    ("医生", _CONCEPT_EQUIVALENTS["doctor"]),
    ("家人", _CONCEPT_EQUIVALENTS["family"]),
)


def _normalized(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def assess_health_knowledge_support(query: str) -> KnowledgeSupportDecision:
    """Classify only the raw query text against frozen deterministic rules."""

    normalized = _normalized(query)
    for rule in SUPPORT_GATE_RULES:
        if any(re.search(pattern, normalized) for pattern in rule["patterns"]):
            return KnowledgeSupportDecision(
                allowed=False,
                reason_code=str(rule["reason_code"]),
                message=str(rule["message"]),
                stage="pre_retrieval",
            )
    return KnowledgeSupportDecision(allowed=True)


def _term_variants(term: str) -> tuple[str, ...]:
    variants = {term}
    if term.endswith("ies") and len(term) > 4:
        variants.add(f"{term[:-3]}y")
    elif term.endswith("s") and len(term) > 3:
        variants.add(term[:-1])
    for equivalents in _CONCEPT_EQUIVALENTS.values():
        if term in equivalents:
            variants.update(equivalents)
    return tuple(sorted(variants))


def _query_core_concepts(query: str) -> tuple[tuple[str, ...], ...]:
    normalized = _normalized(query).replace("_", " ")
    concepts: list[tuple[str, ...]] = []
    for phrase, equivalents in _CJK_CONCEPTS:
        if phrase in normalized:
            concepts.append(equivalents)
            normalized = normalized.replace(phrase, " ")
    for token in re.findall(r"[a-z0-9]+", normalized):
        if len(token) < 3 or token in _GENERIC_QUERY_TERMS:
            continue
        variants = _term_variants(token)
        if variants not in concepts:
            concepts.append(variants)
    cjk_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    for token in cjk_chunks:
        core = token
        for generic in _GENERIC_CJK_TERMS:
            core = core.replace(generic, " ")
        for phrase in core.split():
            if len(phrase) >= 2:
                concepts.append((phrase,))
    return tuple(concepts)


def _evidence_text(hits: tuple[KnowledgeHit, ...]) -> str:
    values: list[str] = []
    for hit in hits:
        document = hit.document
        values.extend(
            (
                document.title,
                document.topic,
                " ".join(document.topic_tags),
                document.summary,
                " ".join(hit.matched_terms),
            )
        )
    return _normalized(" ".join(values).replace("_", " "))


def _variant_in_evidence(variant: str, evidence: str) -> bool:
    if re.fullmatch(r"[a-z0-9 ]+", variant):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])",
                evidence,
            )
        )
    return variant in evidence


def assess_health_knowledge_evidence(
    query: str, hits: tuple[KnowledgeHit, ...]
) -> KnowledgeSupportDecision:
    """Require every query core medical concept to appear in actual Top-K evidence."""

    if not hits:
        return KnowledgeSupportDecision(
            allowed=False,
            reason_code="unsupported_corpus_topic",
            message="The retrieved corpus contains no evidence for the query's core topic.",
            stage="post_retrieval",
        )
    concepts = _query_core_concepts(query)
    if not concepts:
        return KnowledgeSupportDecision(allowed=True)
    evidence = _evidence_text(hits)
    if all(
        any(_variant_in_evidence(variant, evidence) for variant in concept)
        for concept in concepts
    ):
        return KnowledgeSupportDecision(allowed=True)
    return KnowledgeSupportDecision(
        allowed=False,
        reason_code="unsupported_corpus_topic",
        message="The retrieved corpus does not cover the query's core topic.",
        stage="post_retrieval",
    )


def search_with_health_knowledge_gate(
    raw_query: str,
    repository: HealthKnowledgeRepository,
    limit: int,
    *,
    retrieval_query: str | None = None,
) -> tuple[KnowledgeSupportDecision, tuple[KnowledgeHit, ...]]:
    """Apply shared pre-retrieval intent and post-retrieval evidence Gates."""

    decision = assess_health_knowledge_support(raw_query)
    if not decision.allowed:
        return decision, ()
    hits = repository.search(retrieval_query or raw_query, limit)
    evidence_decision = assess_health_knowledge_evidence(raw_query, hits)
    return (evidence_decision, hits) if evidence_decision.allowed else (evidence_decision, ())
