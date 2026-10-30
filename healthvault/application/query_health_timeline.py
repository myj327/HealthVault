"""Read-only timeline query for confirmed health records."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from healthvault.domain.records import (
    ConfirmationIssue,
    TimelineQueryResult,
)
from healthvault.repositories.health_records import HealthRecordRepository
from healthvault.domain.reports import INDICATOR_ALIASES, INDICATOR_CATALOG


def _invalid(code: str, message: str) -> TimelineQueryResult:
    return TimelineQueryResult(
        status="invalid_request",
        report_count=0,
        indicator_count=0,
        timeline=(),
        issues=[ConfirmationIssue(code, message).to_dict()],
    )


def query_health_timeline(
    tool_input: Any,
    repository: HealthRecordRepository,
) -> TimelineQueryResult:
    """Validate one exact profile identifier and return its stored timeline."""
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

    code = tool_input.get('indicator_code')
    if code is not None:
        if not isinstance(code, str) or not code.strip():
            return _invalid('invalid_indicator_code', 'indicator_code must be a non-empty string.')
        names = {spec.display_name: key for key, spec in INDICATOR_CATALOG.items()}
        code = names.get(code.strip(), INDICATOR_ALIASES.get(code.strip().lower(), code.strip().lower()))
        if code not in INDICATOR_CATALOG:
            return _invalid('invalid_indicator_code', 'Use a supported indicator code or name.')
    timeline = repository.list_timeline(profile_id)
    if code is not None:
        timeline = tuple(replace(report, indicators=tuple(i for i in report.indicators if i.code == code),
                                 clinical_items=(), imaging_findings=())
                         for report in timeline if any(i.code == code for i in report.indicators))
    report_count = len(timeline)
    indicator_count = sum(len(report.indicators) for report in timeline)
    clinical_item_count = sum(len(report.clinical_items) for report in timeline)
    return TimelineQueryResult(
        status="ok" if timeline else "empty",
        profile_id=profile_id,
        indicator_code=code,
        report_count=report_count,
        indicator_count=indicator_count,
        clinical_item_count=clinical_item_count,
        timeline=timeline,
        issues=[],
    )
