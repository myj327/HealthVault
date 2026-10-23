"""Domain data for explicitly confirmed health records."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ConfirmedIndicator:
    code: str
    display_name: str
    value: float
    unit: str
    reference_lower: float
    reference_upper: float
    flag: str
    evidence_source_id: str
    evidence_line_start: int
    evidence_line_end: int
    evidence_quote: str
    evidence_locator: dict[str, Any] | None = None


@dataclass(frozen=True)
class ConfirmedClinicalItem:
    clinical_item_id: str
    kind: str
    original_text: str
    diagnosis_code: str | None
    normalized_value: str | None
    evidence_source_id: str
    evidence_line_start: int
    evidence_line_end: int
    evidence_span_start: int
    evidence_span_end: int
    evidence_quote: str
    evidence_page: int | None
    evidence_polygon: list[list[float]] | None
    confidence: float | None
    source_type: str
    recorded_by_provider: bool
    confirmation_status: str
    date_status: str | None
    normalized_date: str | None
    date_original: str | None


@dataclass(frozen=True)
class ConfirmedImagingFinding:
    finding_id: str
    finding_type: str
    body_site: str
    laterality: str
    location: str
    dimensions_mm: tuple[float, ...]
    original_dimensions: str
    original_unit: str
    evidence_source_id: str
    evidence_line_start: int
    evidence_quote: str
    evidence_locator: dict[str, Any] | None = None


@dataclass(frozen=True)
class ConfirmedSourceAsset:
    filename: str
    media_type: str
    sha256: str
    content: bytes


@dataclass(frozen=True)
class ConfirmedReportBundle:
    idempotency_key: str
    profile_id: str
    source_id: str
    source_sha256: str
    report_id: str
    report_date: str
    facility: str
    raw_report_text: str
    indicators: tuple[ConfirmedIndicator, ...]
    clinical_items: tuple[ConfirmedClinicalItem, ...] = ()
    imaging_findings: tuple[ConfirmedImagingFinding, ...] = ()
    original_asset: ConfirmedSourceAsset | None = None


@dataclass(frozen=True)
class RepositoryConfirmation:
    status: str
    write_performed: bool
    profile_id: str
    source_id: str
    source_sha256: str
    report_id: str
    report_date: str
    indicator_count: int
    clinical_item_count: int = 0


@dataclass(frozen=True)
class ConfirmationIssue:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ConfirmationResult:
    status: str
    confirmed: bool
    write_performed: bool
    idempotent_replay: bool
    issues: list[dict[str, Any]]
    profile_id: str | None = None
    source_id: str | None = None
    source_sha256: str | None = None
    report_id: str | None = None
    report_date: str | None = None
    indicator_count: int | None = None
    clinical_item_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "confirmed": self.confirmed,
            "write_performed": self.write_performed,
            "idempotent_replay": self.idempotent_replay,
            "issues": self.issues,
        }
        optional_values = {
            "profile_id": self.profile_id,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "report_id": self.report_id,
            "report_date": self.report_date,
            "indicator_count": self.indicator_count,
            "clinical_item_count": self.clinical_item_count,
        }
        result.update(
            {
                key: value
                for key, value in optional_values.items()
                if value is not None
            }
        )
        return result


@dataclass(frozen=True)
class TimelineReport:
    report_id: str
    report_date: str
    facility: str
    source_id: str
    source_sha256: str
    indicators: tuple[ConfirmedIndicator, ...]
    clinical_items: tuple[ConfirmedClinicalItem, ...] = ()
    imaging_findings: tuple[ConfirmedImagingFinding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_date": self.report_date,
            "facility": self.facility,
            "source": {
                "source_id": self.source_id,
                "sha256": self.source_sha256,
            },
            "indicators": [
                {
                    "code": indicator.code,
                    "display_name": indicator.display_name,
                    "value": indicator.value,
                    "unit": indicator.unit,
                    "reference_range": {
                        "lower": indicator.reference_lower,
                        "upper": indicator.reference_upper,
                    },
                    "flag": indicator.flag,
                    "evidence": {
                        "source_id": indicator.evidence_source_id,
                        "line_start": indicator.evidence_line_start,
                        "line_end": indicator.evidence_line_end,
                        "quote": indicator.evidence_quote,
                        **(indicator.evidence_locator or {}),
                    },
                }
                for indicator in self.indicators
            ],
            "clinical_items": [
                {
                    "clinical_item_id": item.clinical_item_id,
                    "kind": item.kind,
                    "original_text": item.original_text,
                    "diagnosis_code": item.diagnosis_code,
                    "normalized_value": item.normalized_value,
                    "evidence": {
                        "source_id": item.evidence_source_id,
                        "line_start": item.evidence_line_start,
                        "line_end": item.evidence_line_end,
                        "span_start": item.evidence_span_start,
                        "span_end": item.evidence_span_end,
                        "quote": item.evidence_quote,
                        "page": item.evidence_page,
                        "polygon": item.evidence_polygon,
                    },
                    "confidence": item.confidence,
                    "source_type": item.source_type,
                    "recorded_by_provider": item.recorded_by_provider,
                    "confirmation_status": item.confirmation_status,
                    "date_status": item.date_status,
                    "normalized_date": item.normalized_date,
                    "date_original": item.date_original,
                }
                for item in self.clinical_items
            ],
            "imaging_findings": [
                {
                    "finding_id": item.finding_id,
                    "finding_type": item.finding_type,
                    "body_site": item.body_site,
                    "laterality": item.laterality,
                    "location": item.location,
                    "dimensions_mm": list(item.dimensions_mm),
                    "original_dimensions": item.original_dimensions,
                    "original_unit": item.original_unit,
                    "evidence": {
                        "source_id": item.evidence_source_id,
                        "line_start": item.evidence_line_start,
                        "quote": item.evidence_quote,
                        **(item.evidence_locator or {}),
                    },
                }
                for item in self.imaging_findings
            ],
            "system_inferences": [],
        }


@dataclass(frozen=True)
class TimelineQueryResult:
    status: str
    report_count: int
    indicator_count: int
    timeline: tuple[TimelineReport, ...]
    issues: list[dict[str, str]]
    profile_id: str | None = None
    clinical_item_count: int = 0
    system_inference_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        available_citation_ids = [
            f"REC:{report.report_id}:{indicator.code}"
            for report in self.timeline
            for indicator in report.indicators
        ]
        result = {
            "status": self.status,
            "report_count": self.report_count,
            "indicator_count": self.indicator_count,
            "clinical_item_count": self.clinical_item_count,
            "system_inference_count": self.system_inference_count,
            "timeline": [report.to_dict() for report in self.timeline],
            "available_citation_ids": available_citation_ids,
            "recommended_citation_ids": available_citation_ids[:1],
            "issues": self.issues,
        }
        if self.profile_id is not None:
            result["profile_id"] = self.profile_id
        return result


def build_idempotency_key(
    profile_id: str,
    source_id: str,
    source_sha256: str,
) -> str:
    """Build the stable business idempotency key specified by TASK-006."""
    raw_key = f"{profile_id}\0{source_id}\0{source_sha256}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
