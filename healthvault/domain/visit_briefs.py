"""Domain models for deterministic, evidenced visit preparation briefs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from healthvault.domain.knowledge import KnowledgeHit


VISIT_BRIEF_NOTICE = (
    "Visit preparation aid only; not a diagnosis or treatment recommendation."
)


@dataclass(frozen=True)
class VisitBriefIssue:
    code: str
    message: str
    indicator_code: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RecordCitation:
    citation_id: str
    report_id: str
    report_date: str
    indicator_code: str
    source_id: str
    source_sha256: str
    evidence_source_id: str
    evidence_line_start: int
    evidence_line_end: int
    evidence_quote: str
    evidence_locator: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "report_id": self.report_id,
            "report_date": self.report_date,
            "indicator_code": self.indicator_code,
            "source": {
                "source_id": self.source_id,
                "sha256": self.source_sha256,
            },
            "evidence": {
                "source_id": self.evidence_source_id,
                "line_start": self.evidence_line_start,
                "line_end": self.evidence_line_end,
                "quote": self.evidence_quote,
                **(self.evidence_locator or {}),
            },
        }


@dataclass(frozen=True)
class VisitBriefIndicator:
    code: str
    display_name: str
    latest: dict[str, Any]
    comparison: dict[str, Any] | None
    knowledge_citation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "display_name": self.display_name,
            "latest": self.latest,
            "comparison": self.comparison,
            "knowledge_citation_ids": list(self.knowledge_citation_ids),
        }


@dataclass(frozen=True)
class VisitBriefQuestion:
    question_id: str
    text: str
    indicator_codes: tuple[str, ...]
    record_citation_ids: tuple[str, ...]
    knowledge_citation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "indicator_codes": list(self.indicator_codes),
            "record_citation_ids": list(self.record_citation_ids),
            "knowledge_citation_ids": list(self.knowledge_citation_ids),
        }


@dataclass(frozen=True)
class VisitBriefResult:
    status: str
    report_count: int
    indicator_count: int
    latest_report: dict[str, Any] | None
    indicators: tuple[VisitBriefIndicator, ...]
    questions: tuple[VisitBriefQuestion, ...]
    record_citations: tuple[RecordCitation, ...]
    knowledge_citations: tuple[KnowledgeHit, ...]
    brief_markdown: str
    issues: tuple[VisitBriefIssue, ...]
    profile_id: str | None = None
    imaging_findings: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        available_citation_ids = [
            citation_id
            for citation_id in (
                [item.citation_id for item in self.record_citations]
                + [item.document.citation_id for item in self.knowledge_citations]
            )
            if citation_id.startswith(("REC:", "HVK:"))
        ]
        result = {"status": self.status}
        if self.profile_id is not None:
            result["profile_id"] = self.profile_id
        result.update(
            {
                "report_count": self.report_count,
                "indicator_count": self.indicator_count,
                "latest_report": self.latest_report,
                "indicators": [item.to_dict() for item in self.indicators],
                "questions": [item.to_dict() for item in self.questions],
                "record_citations": [
                    item.to_dict() for item in self.record_citations
                ],
                "knowledge_citations": [
                    item.to_dict() for item in self.knowledge_citations
                ],
                "available_citation_ids": available_citation_ids + [
                    f"REC:derived:{item.latest['report_id']}:{item.code}" for item in self.indicators if item.comparison],
                "recommended_citation_ids": available_citation_ids[:1],
                "brief_markdown": self.brief_markdown,
                "issues": [item.to_dict() for item in self.issues],
                "notice": VISIT_BRIEF_NOTICE,
            }
        )
        if self.imaging_findings:
            result['imaging_findings'] = list(self.imaging_findings)
        return result
