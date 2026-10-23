"""Compose an evidenced visit brief from accepted HealthVault services."""

from __future__ import annotations

from typing import Any

from healthvault.application.compare_health_indicators import (
    compare_health_indicators,
)
from healthvault.application.query_health_timeline import query_health_timeline
from healthvault.domain.comparisons import ComparisonPoint, IndicatorComparison
from healthvault.domain.knowledge import KnowledgeHit
from healthvault.domain.records import ConfirmedIndicator, TimelineReport
from healthvault.domain.visit_briefs import (
    VISIT_BRIEF_NOTICE,
    RecordCitation,
    VisitBriefIndicator,
    VisitBriefIssue,
    VisitBriefQuestion,
    VisitBriefResult,
)
from healthvault.repositories.health_knowledge import HealthKnowledgeRepository
from healthvault.repositories.health_records import HealthRecordRepository


def _invalid(code: str, message: str) -> VisitBriefResult:
    return VisitBriefResult(
        status="invalid_request",
        report_count=0,
        indicator_count=0,
        latest_report=None,
        indicators=(),
        questions=(),
        record_citations=(),
        knowledge_citations=(),
        brief_markdown="",
        issues=(VisitBriefIssue(code, message, "profile_id"),),
    )


def _record_citation(
    report_id: str,
    report_date: str,
    source_id: str,
    source_sha256: str,
    indicator: ConfirmedIndicator,
) -> RecordCitation:
    return RecordCitation(
        citation_id=f"REC:{report_id}:{indicator.code}",
        report_id=report_id,
        report_date=report_date,
        indicator_code=indicator.code,
        source_id=source_id,
        source_sha256=source_sha256,
        evidence_source_id=indicator.evidence_source_id,
        evidence_line_start=indicator.evidence_line_start,
        evidence_line_end=indicator.evidence_line_end,
        evidence_quote=indicator.evidence_quote,
        evidence_locator=indicator.evidence_locator,
    )


def _point_citation(point: ComparisonPoint) -> RecordCitation:
    return _record_citation(
        point.report_id,
        point.report_date,
        point.source_id,
        point.source_sha256,
        point.indicator,
    )


def _comparison_dict(
    comparison: IndicatorComparison,
    earliest_id: str,
    latest_id: str,
) -> dict[str, Any]:
    return {
        "observation_count": comparison.observation_count,
        "earliest_value": comparison.earliest.indicator.value,
        "latest_value": comparison.latest.indicator.value,
        "unit": comparison.unit,
        "absolute_change": comparison.absolute_change,
        "direction": comparison.direction,
        "flag_transition": {
            "from": comparison.earliest.indicator.flag,
            "to": comparison.latest.indicator.flag,
            "changed": (
                comparison.earliest.indicator.flag
                != comparison.latest.indicator.flag
            ),
        },
        "earliest_record_citation_id": earliest_id,
        "latest_record_citation_id": latest_id,
    }


def _citation_refs(citation_ids: tuple[str, ...]) -> str:
    return " ".join(f"[{citation_id}]" for citation_id in citation_ids)


def _render_markdown(
    profile_id: str,
    report_count: int,
    latest_report: dict[str, Any],
    indicators: tuple[VisitBriefIndicator, ...],
    questions: tuple[VisitBriefQuestion, ...],
    record_citations: tuple[RecordCitation, ...],
    knowledge_citations: tuple[KnowledgeHit, ...],
) -> str:
    latest_report_citation_ids = tuple(
        citation.citation_id
        for citation in record_citations
        if citation.report_id == latest_report["report_id"]
    )
    latest_report_refs = _citation_refs(latest_report_citation_ids)
    lines = [
        "# 就诊准备摘要",
        f"> {VISIT_BRIEF_NOTICE}",
        "",
        "## 档案概览",
        f"- profile_id：`{profile_id}`",
        f"- 已确认报告数：{report_count}",
        (
            f"- 最新报告：{latest_report['report_date']} / "
            f"{latest_report['report_id']} / {latest_report['facility']}"
            + (f" {latest_report_refs}" if latest_report_refs else "")
        ),
        "",
        "## 已确认指标",
    ]
    for item in indicators:
        latest = item.latest
        lines.append(
            f"- {item.display_name}：{latest['value']} {latest['unit']} "
            f"（flag: {latest['flag']}）[{latest['record_citation_id']}]"
        )
        if item.comparison is not None:
            comparison = item.comparison
            lines.append(
                "  - 跨期数值变化："
                f"{comparison['earliest_value']} → {comparison['latest_value']} "
                f"{comparison['unit']}；change={comparison['absolute_change']}；"
                f"direction={comparison['direction']} "
                f"[{comparison['earliest_record_citation_id']}] "
                f"[{comparison['latest_record_citation_id']}]"
            )
        if item.knowledge_citation_ids:
            lines.append(
                "  - 一般说明来源："
                f"{_citation_refs(item.knowledge_citation_ids)}"
            )

    lines.extend(["", "## 可向医生询问"])
    for question in questions:
        citation_ids = (
            question.record_citation_ids
            + question.knowledge_citation_ids
        )
        suffix = _citation_refs(citation_ids)
        lines.append(
            f"- {question.text}" + (f" {suffix}" if suffix else "")
        )

    lines.extend(["", "## 记录证据"])
    for citation in record_citations:
        lines.append(
            f"- [{citation.citation_id}] {citation.report_date} / "
            f"{citation.report_id} / {citation.indicator_code}；"
            f"证据：{citation.evidence_quote}"
        )

    lines.extend(["", "## 健康知识来源"])
    for hit in knowledge_citations:
        document = hit.document
        lines.append(
            f"- [{document.citation_id}] {document.title}：{document.summary} "
            f"（body_sha256={document.body_sha256}；"
            f"{document.source.publisher}，{document.source.title}，"
            f"{document.source.url}，retrieved {document.source.retrieved_on}）"
        )
    return "\n".join(lines)


def generate_visit_brief(
    tool_input: Any,
    record_repository: HealthRecordRepository,
    knowledge_repository: HealthKnowledgeRepository,
) -> VisitBriefResult:
    """Compose one deterministic brief without reimplementing source logic."""
    if not isinstance(tool_input, dict):
        return _invalid("invalid_request", "Tool input must be an object.")
    if "profile_id" not in tool_input:
        return _invalid(
            "missing_required_field",
            "Missing required field: profile_id.",
        )
    profile_id = tool_input["profile_id"]
    if not isinstance(profile_id, str) or not profile_id.strip():
        return _invalid(
            "invalid_profile_id",
            "profile_id must be a non-empty string.",
        )

    timeline_result = query_health_timeline(tool_input, record_repository)
    if timeline_result.status == "empty":
        return VisitBriefResult(
            status="empty",
            profile_id=profile_id,
            report_count=0,
            indicator_count=0,
            latest_report=None,
            indicators=(),
            questions=(),
            record_citations=(),
            knowledge_citations=(),
            brief_markdown="",
            issues=(),
        )

    comparison_result = compare_health_indicators(
        tool_input,
        record_repository,
    )
    latest_by_code: dict[str, tuple[TimelineReport, ConfirmedIndicator]] = {}
    for report in timeline_result.timeline:
        for indicator in report.indicators:
            latest_by_code[indicator.code] = (report, indicator)

    knowledge_by_topic: dict[str, KnowledgeHit] = {}
    for code in sorted(latest_by_code):
        _report, indicator = latest_by_code[code]
        hits = knowledge_repository.search(
            f"{code.replace('_', ' ')} {indicator.display_name}",
            5,
        )
        best_topic_hit = next(
            (
                hit
                for hit in hits
                if code == hit.document.topic
                or code in hit.document.topic_tags
            ),
            None,
        )
        if best_topic_hit is not None:
            knowledge_by_topic[code] = best_topic_hit
    visit_hits = knowledge_repository.search(
        "visit preparation doctor visit 就诊准备",
        5,
    )
    visit_hit = next(
        (
            hit
            for hit in visit_hits
            if hit.document.topic == "visit_preparation"
            or "visit_preparation" in hit.document.topic_tags
        ),
        None,
    )
    if visit_hit is not None:
        knowledge_by_topic["visit_preparation"] = visit_hit
    comparison_by_code = {
        comparison.code: comparison
        for comparison in comparison_result.comparisons
    }

    issues = [
        VisitBriefIssue(issue.code, issue.message, issue.indicator_code)
        for issue in comparison_result.issues
    ]
    citations: dict[str, RecordCitation] = {}
    indicators = []
    questions = []
    for code in sorted(latest_by_code):
        report, indicator = latest_by_code[code]
        latest_citation = _record_citation(
            report.report_id,
            report.report_date,
            report.source_id,
            report.source_sha256,
            indicator,
        )
        citations[latest_citation.citation_id] = latest_citation

        comparison = comparison_by_code.get(code)
        comparison_data = None
        question_record_ids = (latest_citation.citation_id,)
        if comparison is not None:
            earliest_citation = _point_citation(comparison.earliest)
            latest_endpoint_citation = _point_citation(comparison.latest)
            citations[earliest_citation.citation_id] = earliest_citation
            citations[latest_endpoint_citation.citation_id] = latest_endpoint_citation
            comparison_data = _comparison_dict(
                comparison,
                earliest_citation.citation_id,
                latest_endpoint_citation.citation_id,
            )
            question_record_ids = tuple(
                dict.fromkeys(
                    (
                        earliest_citation.citation_id,
                        latest_endpoint_citation.citation_id,
                    )
                )
            )

        knowledge_hit = knowledge_by_topic.get(code)
        knowledge_ids = (
            (knowledge_hit.document.citation_id,)
            if knowledge_hit is not None
            else ()
        )
        if knowledge_hit is None:
            issues.append(
                VisitBriefIssue(
                    "knowledge_not_found",
                    f"No knowledge source matched indicator {code}.",
                    code,
                )
            )
        indicators.append(
            VisitBriefIndicator(
                code=code,
                display_name=indicator.display_name,
                latest={
                    "report_id": report.report_id,
                    "report_date": report.report_date,
                    "value": indicator.value,
                    "unit": indicator.unit,
                    "flag": indicator.flag,
                    "record_citation_id": latest_citation.citation_id,
                },
                comparison=comparison_data,
                knowledge_citation_ids=knowledge_ids,
            )
        )
        question_text = (
            f"请医生结合病史和其他检查解释「{indicator.display_name}」的最新结果"
            + ("及跨期数值变化。" if comparison is not None else "。")
        )
        questions.append(
            VisitBriefQuestion(
                question_id=f"ASK:{code}",
                text=question_text,
                indicator_codes=(code,),
                record_citation_ids=question_record_ids,
                knowledge_citation_ids=knowledge_ids,
            )
        )

    visit_hit = knowledge_by_topic.get("visit_preparation")
    visit_knowledge_ids = (
        (visit_hit.document.citation_id,) if visit_hit is not None else ()
    )
    if visit_hit is None:
        issues.append(
            VisitBriefIssue(
                "knowledge_not_found",
                "No visit preparation knowledge source was found.",
                "visit_preparation",
            )
        )
    questions.append(
        VisitBriefQuestion(
            question_id="ASK:visit_preparation",
            text="本次就诊还需要补充哪些药物、过敏、症状或既往资料？",
            indicator_codes=(),
            record_citation_ids=(),
            knowledge_citation_ids=visit_knowledge_ids,
        )
    )

    imaging = []
    for report in timeline_result.timeline:
        for finding in report.imaging_findings:
            identifier = f'REC:{report.report_id}:imaging:{finding.finding_id}'
            citations[identifier] = RecordCitation(identifier,report.report_id,report.report_date,
                'imaging:'+finding.finding_type,report.source_id,report.source_sha256,finding.evidence_source_id,
                finding.evidence_line_start,finding.evidence_line_start,finding.evidence_quote,finding.evidence_locator)
            imaging.append({'citation_id':identifier,'report_date':report.report_date,'body_site':finding.body_site,
                            'laterality':finding.laterality,'location':finding.location,'finding_type':finding.finding_type,
                            'dimensions_mm':list(finding.dimensions_mm),'original_dimensions':finding.original_dimensions,
                            'original_unit':finding.original_unit,'cross_report_entity_matching':'not_established'})
    record_citations = tuple(
        sorted(
            citations.values(),
            key=lambda item: (
                item.report_date,
                item.report_id,
                item.indicator_code,
            ),
        )
    )
    sorted_knowledge_hits = tuple(
        sorted(
            {
                hit.document.citation_id: hit
                for hit in knowledge_by_topic.values()
            }.values(),
            key=lambda hit: hit.document.citation_id,
        )
    )
    issues.sort(key=lambda issue: (issue.indicator_code, issue.code))
    latest_report_model = timeline_result.timeline[-1]
    latest_report = {
        "report_id": latest_report_model.report_id,
        "report_date": latest_report_model.report_date,
        "facility": latest_report_model.facility,
        "source": {
            "source_id": latest_report_model.source_id,
            "sha256": latest_report_model.source_sha256,
        },
    }
    indicators_tuple = tuple(indicators)
    questions_tuple = tuple(questions)
    markdown = _render_markdown(
        profile_id,
        timeline_result.report_count,
        latest_report,
        indicators_tuple,
        questions_tuple,
        record_citations,
        sorted_knowledge_hits,
    )
    if imaging:
        markdown += '\n\n## 已确认影像发现\n' + '\n'.join(
            f"- {item['report_date']} {item['body_site']} {item['location']}：{item['original_dimensions']} {item['original_unit']} [{item['citation_id']}]" for item in imaging)
        markdown += '\n不同报告中的病灶尚未建立同一对象的关联，请由医生对照原件核对。'
    return VisitBriefResult(
        status="partial" if issues else "ok",
        profile_id=profile_id,
        report_count=timeline_result.report_count,
        indicator_count=len(latest_by_code),
        latest_report=latest_report,
        indicators=indicators_tuple,
        questions=questions_tuple,
        record_citations=record_citations,
        knowledge_citations=sorted_knowledge_hits,
        brief_markdown=markdown,
        issues=tuple(issues),
        imaging_findings=tuple(imaging),
    )
