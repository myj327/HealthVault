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
            "reference_range": self.indicator.reference_range_dict(),
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
    selected_years: tuple[int, int] | None = None
    trend: dict[str, Any] | None = None

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
            "relative_change_percent": round(self.absolute_change / self.earliest.indicator.value * 100, 6) if self.earliest.indicator.value else None,
            "comparison_citation_id": (f"REC:derived:{self.earliest.report_id}:{self.latest.report_id}:{self.code}"
                                       if self.selected_years else f"REC:derived:{self.latest.report_id}:{self.code}"),
            "trend": self.trend,
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
    selected_years: tuple[int, int] | None = None

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
        if self.selected_years:
            result['year_selection'] = {'start_year': self.selected_years[0], 'end_year': self.selected_years[1],
                                        'method': 'latest_in_each_year'}
        if self.profile_id is not None:
            result["profile_id"] = self.profile_id
        items = [comparison.to_dict() for comparison in self.comparisons]
        citation_ids = list(available_citation_ids)
        for comparison, item in zip(self.comparisons, items):
            citation_ids.append(item.get('comparison_citation_id', f"REC:derived:{comparison.latest.report_id}:{comparison.code}"))
            trend = item.get('trend')
            if trend:
                citation_ids.append(trend['citation_id'])
                citation_ids.extend(f"REC:{point['report_id']}:{comparison.code}" for point in trend['annual_points'])
        result.update(
            {
                "report_count": self.report_count,
                "comparison_count": len(self.comparisons),
                "comparisons": items,
                "available_citation_ids": list(dict.fromkeys(citation_ids)),
                "recommended_citation_ids": available_citation_ids[:2],
                "issues": [issue.to_dict() for issue in self.issues],
            }
        )
        return result
