"""Explicit confirmation gate for persistent health records."""

from __future__ import annotations

import re
from typing import Any

from healthvault.application.report_preview import build_report_preview
from healthvault.domain.records import (
    ConfirmationIssue,
    ConfirmationResult,
    ConfirmedClinicalItem,
    ConfirmedIndicator,
    ConfirmedReportBundle,
    ConfirmedSourceAsset,
    build_idempotency_key,
)
from healthvault.repositories.health_records import HealthRecordRepository


REQUIRED_FIELDS = (
    "profile_id",
    "source_id",
    "report_text",
    "expected_sha256",
    "user_confirmed",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _clinical_core(item: dict[str, Any]) -> tuple[Any, ...]:
    evidence = item.get("evidence") if isinstance(item, dict) else None
    evidence = evidence if isinstance(evidence, dict) else {}
    return (
        item.get("clinical_item_id"),
        item.get("kind"),
        item.get("original_text"),
        item.get("diagnosis_code"),
        item.get("normalized_value"),
        item.get("recorded_by_provider"),
        item.get("date_status"),
        item.get("normalized_date"),
        item.get("date_original"),
        evidence.get("source_id"),
        evidence.get("line_start"),
        evidence.get("line_end"),
        evidence.get("span_start"),
        evidence.get("span_end"),
        evidence.get("quote"),
    )


def _verified_clinical_items(
    server_items: list[dict[str, Any]], submitted: Any
) -> list[dict[str, Any]] | None:
    if submitted is None:
        return server_items
    if not isinstance(submitted, list) or len(submitted) != len(server_items):
        return None
    verified: list[dict[str, Any]] = []
    for server_item, submitted_item in zip(server_items, submitted):
        if not isinstance(submitted_item, dict):
            return None
        if _clinical_core(server_item) != _clinical_core(submitted_item):
            return None
        source_type = submitted_item.get("source_type")
        if source_type not in {"text", "ocr", "ocr_edited"}:
            return None
        item = dict(server_item)
        evidence = dict(server_item["evidence"])
        confidence: float | None = None
        if source_type == "ocr":
            submitted_evidence = submitted_item.get("evidence", {})
            submitted_confidence = submitted_item.get("confidence")
            if (
                not isinstance(submitted_confidence, (int, float))
                or isinstance(submitted_confidence, bool)
                or not 0 <= float(submitted_confidence) <= 1
                or submitted_evidence.get("page") != 1
                or not isinstance(submitted_evidence.get("polygon"), list)
                or not submitted_evidence["polygon"]
            ):
                return None
            confidence = float(submitted_confidence)
            evidence["page"] = 1
            evidence["polygon"] = submitted_evidence["polygon"]
        item["source_type"] = source_type
        item["confidence"] = confidence
        item["evidence"] = evidence
        verified.append(item)
    return verified


def _issue(code: str, message: str) -> dict[str, str]:
    return ConfirmationIssue(code, message).to_dict()


def _rejected(
    status: str,
    code: str,
    message: str,
    *,
    issues: list[dict[str, Any]] | None = None,
) -> ConfirmationResult:
    return ConfirmationResult(
        status=status,
        confirmed=False,
        write_performed=False,
        idempotent_replay=False,
        issues=issues if issues is not None else [_issue(code, message)],
    )


def _validate_input(tool_input: Any) -> ConfirmationResult | None:
    if not isinstance(tool_input, dict):
        return _rejected(
            "invalid_request",
            "invalid_request",
            "Tool input must be an object.",
        )

    missing = [field for field in REQUIRED_FIELDS if field not in tool_input]
    if missing:
        return _rejected(
            "invalid_request",
            "missing_required_field",
            f"Missing required field(s): {', '.join(missing)}.",
        )

    for field in ("profile_id", "source_id", "report_text"):
        value = tool_input[field]
        if not isinstance(value, str) or not value.strip():
            return _rejected(
                "invalid_request",
                f"invalid_{field}",
                f"{field} must be a non-empty string.",
            )

    expected_sha256 = tool_input["expected_sha256"]
    if (
        not isinstance(expected_sha256, str)
        or SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        return _rejected(
            "invalid_request",
            "invalid_expected_sha256",
            "expected_sha256 must be 64 lowercase hexadecimal characters.",
        )

    user_confirmed = tool_input["user_confirmed"]
    if not isinstance(user_confirmed, bool):
        return _rejected(
            "invalid_request",
            "invalid_user_confirmed",
            "user_confirmed must be a JSON boolean.",
        )
    if user_confirmed is not True:
        return _rejected(
            "confirmation_required",
            "confirmation_required",
            "Explicit user confirmation is required before persistence.",
        )
    return None


def confirm_health_records(
    tool_input: Any,
    repository: HealthRecordRepository,
    *,
    preview_clinical_items: list[dict[str, Any]] | None = None,
    original_asset: ConfirmedSourceAsset | None = None,
    evidence_locators: dict[str, dict[str, Any]] | None = None,
) -> ConfirmationResult:
    """Revalidate a report and persist it only after explicit confirmation."""
    input_error = _validate_input(tool_input)
    if input_error is not None:
        return input_error
    assert isinstance(tool_input, dict)

    profile_id = tool_input["profile_id"]
    source_id = tool_input["source_id"]
    report_text = tool_input["report_text"]
    expected_sha256 = tool_input["expected_sha256"]
    preview = build_report_preview(source_id, report_text)
    preview_data = preview.to_dict()
    if preview_data["status"] != "ready" or not preview_data["can_confirm"]:
        return _rejected(
            "invalid_preview",
            "invalid_preview",
            "The report preview is invalid and cannot be confirmed.",
            issues=preview_data["issues"],
        )

    source_sha256 = preview.source.sha256
    if source_sha256 != expected_sha256:
        return _rejected(
            "preview_mismatch",
            "preview_mismatch",
            "The report text does not match the confirmed preview SHA-256.",
        )

    clinical_items = _verified_clinical_items(
        preview_data["clinical_items"], preview_clinical_items
    )
    if clinical_items is None:
        return _rejected(
            "preview_mismatch",
            "preview_mismatch",
            "Clinical items do not match the server-validated report preview.",
        )

    report = preview.report
    assert source_sha256 is not None
    assert report.report_id is not None
    assert report.report_date is not None
    assert report.facility is not None
    bundle = ConfirmedReportBundle(
        idempotency_key=build_idempotency_key(
            profile_id,
            source_id,
            source_sha256,
        ),
        profile_id=profile_id,
        source_id=source_id,
        source_sha256=source_sha256,
        report_id=report.report_id,
        report_date=report.report_date,
        facility=report.facility,
        raw_report_text=report_text,
        original_asset=original_asset,
        indicators=tuple(
            ConfirmedIndicator(
                code=indicator.code,
                display_name=indicator.display_name,
                value=indicator.value,
                unit=indicator.unit,
                reference_lower=indicator.reference_range.lower,
                reference_upper=indicator.reference_range.upper,
                flag=indicator.flag,
                evidence_source_id=indicator.evidence.source_id or source_id,
                evidence_line_start=indicator.evidence.line_start,
                evidence_line_end=indicator.evidence.line_end,
                evidence_quote=indicator.evidence.quote,
                evidence_locator=(evidence_locators or {}).get(indicator.code),
            )
            for indicator in preview.indicators
        ),
        clinical_items=tuple(
            ConfirmedClinicalItem(
                clinical_item_id=item["clinical_item_id"],
                kind=item["kind"],
                original_text=item["original_text"],
                diagnosis_code=item["diagnosis_code"],
                normalized_value=item["normalized_value"],
                evidence_source_id=item["evidence"]["source_id"] or source_id,
                evidence_line_start=item["evidence"]["line_start"],
                evidence_line_end=item["evidence"]["line_end"],
                evidence_span_start=item["evidence"]["span_start"],
                evidence_span_end=item["evidence"]["span_end"],
                evidence_quote=item["evidence"]["quote"],
                evidence_page=item["evidence"].get("page"),
                evidence_polygon=item["evidence"].get("polygon"),
                confidence=item["confidence"],
                source_type=item["source_type"],
                recorded_by_provider=True,
                confirmation_status="confirmed",
                date_status=item["date_status"],
                normalized_date=item["normalized_date"],
                date_original=item["date_original"],
            )
            for item in clinical_items
        ),
    )

    outcome = repository.confirm_report(bundle)
    if outcome.status in {"source_conflict", "report_conflict"}:
        return _rejected(
            outcome.status,
            outcome.status,
            f"The confirmed report conflicts with an existing {outcome.status.split('_')[0]}.",
        )
    if outcome.status not in {"confirmed", "already_confirmed"}:
        raise ValueError(f"Unknown repository confirmation status: {outcome.status}")

    is_replay = outcome.status == "already_confirmed"
    return ConfirmationResult(
        status=outcome.status,
        confirmed=True,
        write_performed=outcome.write_performed,
        idempotent_replay=is_replay,
        issues=[],
        profile_id=outcome.profile_id,
        source_id=outcome.source_id,
        source_sha256=outcome.source_sha256,
        report_id=outcome.report_id,
        report_date=outcome.report_date,
        indicator_count=outcome.indicator_count,
        clinical_item_count=outcome.clinical_item_count,
    )
