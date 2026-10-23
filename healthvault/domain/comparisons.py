"""Domain models for deterministic health indicator comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from healthvault.domain.records import ConfirmedIndicator


@dataclass(frozen=True)
class ComparisonPoint:
    report_id: str
    report_date: str
    source_id: str
    source_sha256: str
    indicator: ConfirmedIndicator

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_date": self.report_date,
            "source": {
                "source_id": self.source_id,
                "sha256": self.source_sha256,
            },
            "value": self.indicator.value,
            "reference_range": {
                "lower": self.indicator.reference_lower,
                "upper": self.indicator.reference_upper,
            },
            "flag": self.indicator.flag,
            "evidence": {
                "source_id": self.indicator.evidence_source_id,
                "line_start": self.indicator.evidence_line_start,
                "line_end": self.indicator.evidence_line_end,
                "quote": self.indicator.evidence_quote,
            },
        }


@dataclass(frozen=True)
class IndicatorComparison:
    code: str
    display_name: str
    unit: str
    observation_count: int
    earliest: ComparisonPoint
    latest: ComparisonPoint
    absolute_change: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "display_name": self.display_name,
            "unit": self.unit,
            "observation_count": self.observation_count,
            "earliest": self.earliest.to_dict(),
            "latest": self.latest.to_dict(),
            "absolute_change": self.absolute_change,
            "direction": self.direction,
            "flag_transition": {
                "from": self.earliest.indicator.flag,
                "to": self.latest.indicator.flag,
                "changed": (
                    self.earliest.indicator.flag
                    != self.latest.indicator.flag
                ),
            },
        }


@dataclass(frozen=True)
class ComparisonIssue:
    code: str
    message: str
    indicator_code: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class IndicatorComparisonResult:
    status: str
    report_count: int
    comparisons: tuple[IndicatorComparison, ...]
    issues: tuple[ComparisonIssue, ...]
    profile_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        available_citation_ids = list(
            dict.fromkeys(
                citation_id
                for comparison in self.comparisons
                for citation_id in (
                    f"REC:{comparison.earliest.report_id}:{comparison.code}",
                    f"REC:{comparison.latest.report_id}:{comparison.code}",
                )
            )
        )
        result = {"status": self.status}
        if self.profile_id is not None:
            result["profile_id"] = self.profile_id
        result.update(
            {
                "report_count": self.report_count,
                "comparison_count": len(self.comparisons),
                "comparisons": [
                    comparison.to_dict()
                    for comparison in self.comparisons
                ],
                "available_citation_ids": available_citation_ids,
                "recommended_citation_ids": available_citation_ids[:2],
                "issues": [issue.to_dict() for issue in self.issues],
            }
        )
        return result
