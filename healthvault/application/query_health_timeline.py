"""Read-only timeline query for confirmed health records."""

from __future__ import annotations

from typing import Any

from healthvault.domain.records import (
    ConfirmationIssue,
    TimelineQueryResult,
)
from healthvault.repositories.health_records import HealthRecordRepository


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

    timeline = repository.list_timeline(profile_id)
    report_count = len(timeline)
    indicator_count = sum(len(report.indicators) for report in timeline)
    clinical_item_count = sum(len(report.clinical_items) for report in timeline)
    return TimelineQueryResult(
        status="ok" if timeline else "empty",
        profile_id=profile_id,
        report_count=report_count,
        indicator_count=indicator_count,
        clinical_item_count=clinical_item_count,
        timeline=timeline,
        issues=[],
    )
