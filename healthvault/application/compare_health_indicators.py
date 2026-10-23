"""Compare endpoint observations from confirmed health timelines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from healthvault.domain.comparisons import (
    ComparisonIssue,
    ComparisonPoint,
    IndicatorComparison,
    IndicatorComparisonResult,
)
from healthvault.domain.records import ConfirmedIndicator, TimelineReport
from healthvault.repositories.health_records import HealthRecordRepository


def _invalid(code: str, message: str) -> IndicatorComparisonResult:
    return IndicatorComparisonResult(
        status="invalid_request",
        report_count=0,
        comparisons=(),
        issues=(ComparisonIssue(code, message, "profile_id"),),
    )


def _point(
    report: TimelineReport,
    indicator: ConfirmedIndicator,
) -> ComparisonPoint:
    return ComparisonPoint(
        report_id=report.report_id,
        report_date=report.report_date,
        source_id=report.source_id,
        source_sha256=report.source_sha256,
        indicator=indicator,
    )


def _rounded_change(earliest: float, latest: float) -> float:
    change = round(latest - earliest, 6)
    return 0.0 if change == 0 else change


def compare_health_indicators(
    tool_input: Any,
    repository: HealthRecordRepository,
) -> IndicatorComparisonResult:
    """Compare the earliest and latest values for one exact profile."""
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

    timeline = repository.list_timeline(profile_id)
    if not timeline:
        return IndicatorComparisonResult(
            status="empty",
            profile_id=profile_id,
            report_count=0,
            comparisons=(),
            issues=(),
        )

    observations: dict[str, list[ComparisonPoint]] = defaultdict(list)
    for report in timeline:
        for indicator in report.indicators:
            observations[indicator.code].append(_point(report, indicator))

    comparisons = []
    issues = []
    for indicator_code in sorted(observations):
        points = observations[indicator_code]
        if len(points) < 2:
            issues.append(
                ComparisonIssue(
                    code="insufficient_history",
                    message=(
                        "At least two observations are required for "
                        f"indicator {indicator_code}."
                    ),
                    indicator_code=indicator_code,
                )
            )
            continue

        units = {point.indicator.unit for point in points}
        if len(units) > 1:
            issues.append(
                ComparisonIssue(
                    code="unit_mismatch",
                    message=(
                        "All observations must use the same unit for "
                        f"indicator {indicator_code}."
                    ),
                    indicator_code=indicator_code,
                )
            )
            continue

        earliest = points[0]
        latest = points[-1]
        change = _rounded_change(
            earliest.indicator.value,
            latest.indicator.value,
        )
        if change > 0:
            direction = "increased"
        elif change < 0:
            direction = "decreased"
        else:
            direction = "unchanged"
        comparisons.append(
            IndicatorComparison(
                code=indicator_code,
                display_name=earliest.indicator.display_name,
                unit=earliest.indicator.unit,
                observation_count=len(points),
                earliest=earliest,
                latest=latest,
                absolute_change=change,
                direction=direction,
            )
        )

    if not observations:
        issues.append(
            ComparisonIssue(
                code="no_indicators",
                message="No confirmed health indicators are available.",
                indicator_code="",
            )
        )

    if comparisons and issues:
        status = "partial"
    elif comparisons:
        status = "ok"
    else:
        status = "insufficient_data"
    issues.sort(key=lambda issue: (issue.indicator_code, issue.code))
    return IndicatorComparisonResult(
        status=status,
        profile_id=profile_id,
        report_count=len(timeline),
        comparisons=tuple(comparisons),
        issues=tuple(issues),
    )
