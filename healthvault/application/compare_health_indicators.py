"""Compare endpoint observations from confirmed health timelines."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from hashlib import sha256
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


def _annual_trend(points: list[ComparisonPoint], code: str) -> dict[str, Any]:
    # One actual observation per year, never an invented annual measurement.
    by_year = {int(point.report_date[:4]): point for point in points}
    annual = list(by_year.values())
    values = [point.indicator.value for point in annual]
    changes = [_rounded_change(left, right) for left, right in zip(values, values[1:])]
    direction = ('insufficient_data' if not changes else 'unchanged' if not any(changes)
                 else 'increased' if all(value >= 0 for value in changes)
                 else 'decreased' if all(value <= 0 for value in changes) else 'fluctuating')
    slope = None
    if len(annual) >= 2:
        origin = date.fromisoformat(annual[0].report_date)
        elapsed = [(date.fromisoformat(point.report_date) - origin).days / 365.25 for point in annual]
        mean_x, mean_y = sum(elapsed) / len(elapsed), sum(values) / len(values)
        slope = round(sum((x - mean_x) * (y - mean_y) for x, y in zip(elapsed, values))
                      / sum((x - mean_x) ** 2 for x in elapsed), 6)
    identity = sha256('|'.join(point.report_id for point in annual).encode()).hexdigest()[:16]
    return {'method': 'latest_in_each_year', 'direction': direction, 'slope_per_year': slope,
            'citation_id': f'REC:trend:{identity}:{code}',
            'annual_points': [{'year': year, **point.to_dict()} for year, point in by_year.items()]}


def compare_health_indicators(
    tool_input: Any,
    repository: HealthRecordRepository,
) -> IndicatorComparisonResult:
    """Compare confirmed endpoints or two specified years, with descriptive trends."""
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

    start_year, end_year = tool_input.get('start_year'), tool_input.get('end_year')
    selected_years = None
    if start_year is not None or end_year is not None:
        if (type(start_year) is not int or type(end_year) is not int
                or not 1 <= start_year < end_year <= 9999):
            return _invalid('invalid_year_range', 'Provide both start_year and end_year, with start_year < end_year.')
        selected_years = (start_year, end_year)
    indicator_code = tool_input.get('indicator_code')
    if indicator_code is not None and (not isinstance(indicator_code, str) or not indicator_code.strip()):
        return _invalid('invalid_indicator_code', 'indicator_code must be a non-empty string.')

    timeline = tuple(sorted(repository.list_timeline(profile_id), key=lambda report: (report.report_date, report.report_id)))
    if selected_years:
        timeline = tuple(report for report in timeline if start_year <= int(report.report_date[:4]) <= end_year)
    if not timeline:
        return IndicatorComparisonResult(
            status="empty",
            profile_id=profile_id,
            report_count=0,
            comparisons=(),
            issues=(),
            selected_years=selected_years,
        )

    observations: dict[str, list[ComparisonPoint]] = defaultdict(list)
    for report in timeline:
        for indicator in report.indicators:
            if indicator_code is not None and indicator.code != indicator_code:
                continue
            observations[indicator.code].append(_point(report, indicator))

    comparisons = []
    issues = []
    for indicator_code in sorted(observations):
        points = observations[indicator_code]
        if selected_years:
            first_year = [point for point in points if int(point.report_date[:4]) == start_year]
            last_year = [point for point in points if int(point.report_date[:4]) == end_year]
            if not first_year or not last_year:
                missing = [str(year) for year, found in ((start_year, first_year), (end_year, last_year)) if not found]
                issues.append(ComparisonIssue('missing_year', '缺少 ' + '、'.join(missing) + ' 年的已确认记录；未用其他年份替代。', indicator_code))
                continue
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

        earliest = first_year[-1] if selected_years else points[0]
        latest = last_year[-1] if selected_years else points[-1]
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
                selected_years=selected_years,
                trend=_annual_trend(points, indicator_code),
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
        selected_years=selected_years,
    )
