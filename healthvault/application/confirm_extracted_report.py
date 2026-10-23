"""Persist a server-validated natural-report preview after explicit review."""

from __future__ import annotations

import hashlib
from typing import Any

from healthvault.domain.records import (
    ConfirmedClinicalItem,
    ConfirmedImagingFinding,
    ConfirmedIndicator,
    ConfirmedReportBundle,
    ConfirmedSourceAsset,
    build_idempotency_key,
)
from healthvault.repositories.health_records import HealthRecordRepository


def confirm_extracted_report(
    profile_id: str,
    source_id: str,
    report_text: str,
    preview: dict[str, Any],
    repository: HealthRecordRepository,
    original_asset: ConfirmedSourceAsset | None = None,
) -> dict[str, Any]:
    """The API must revalidate the server-held preview immediately before calling."""
    source_hash = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
    if not preview.get("can_confirm") or preview.get("source", {}).get("sha256") != source_hash:
        return {"status": "preview_mismatch", "confirmed": False, "write_performed": False}
    metadata = preview["report"]
    bundle = ConfirmedReportBundle(
        idempotency_key=build_idempotency_key(profile_id, source_id, source_hash),
        profile_id=profile_id,
        source_id=source_id,
        source_sha256=source_hash,
        report_id=metadata["report_id"],
        report_date=metadata["report_date"],
        facility=metadata["facility"],
        raw_report_text=report_text,
        original_asset=original_asset,
        indicators=tuple(
            ConfirmedIndicator(
                code=item["code"],
                display_name=item["display_name"],
                value=item["value"],
                unit=item["unit"],
                reference_lower=item["reference_range"]["lower"],
                reference_upper=item["reference_range"]["upper"],
                flag=item["flag"],
                evidence_source_id=source_id,
                evidence_line_start=item["evidence"]["line_start"],
                evidence_line_end=item["evidence"]["line_end"],
                evidence_quote=item["evidence"]["quote"],
                evidence_locator={key: value for key, value in item["evidence"].items() if key in {"span_start", "span_end", "regions", "fields"}},
            )
            for item in preview["indicators"]
        ),
        clinical_items=tuple(
            ConfirmedClinicalItem(
                clinical_item_id=item["clinical_item_id"], kind=item["kind"],
                original_text=item["original_text"], diagnosis_code=item.get("diagnosis_code"),
                normalized_value=item.get("normalized_value"),
                evidence_source_id=source_id, evidence_line_start=item["evidence"]["line_start"],
                evidence_line_end=item["evidence"]["line_end"],
                evidence_span_start=item["evidence"]["span_start"], evidence_span_end=item["evidence"]["span_end"],
                evidence_quote=item["evidence"]["quote"], evidence_page=item["evidence"].get("page"),
                evidence_polygon=item["evidence"].get("polygon"), confidence=item.get("confidence"),
                source_type=item["source_type"], recorded_by_provider=True, confirmation_status="confirmed",
                date_status=item.get("date_status"), normalized_date=item.get("normalized_date"),
                date_original=item.get("date_original"),
            ) for item in preview.get("clinical_items", [])
        ),
        imaging_findings=tuple(
            ConfirmedImagingFinding(
                finding_id=item["finding_id"],
                finding_type=item["finding_type"],
                body_site=item["body_site"],
                laterality=item["laterality"],
                location=item["location"],
                dimensions_mm=tuple(item["dimensions_mm"]),
                original_dimensions=item["original_dimensions"],
                original_unit=item["original_unit"],
                evidence_source_id=source_id,
                evidence_line_start=item["evidence"]["line_start"],
                evidence_quote=item["evidence"]["quote"],
                evidence_locator={key: value for key, value in item["evidence"].items() if key in {"span_start", "span_end", "regions", "fields"}},
            )
            for item in preview["imaging_findings"]
        ),
    )
    outcome = repository.confirm_report(bundle)
    if outcome.status in {"source_conflict", "report_conflict"}:
        return {"status": outcome.status, "confirmed": False, "write_performed": False}
    return {
        "status": outcome.status,
        "confirmed": True,
        "write_performed": outcome.write_performed,
        "idempotent_replay": outcome.status == "already_confirmed",
        "profile_id": profile_id,
        "source_id": source_id,
        "report_id": outcome.report_id,
        "report_date": outcome.report_date,
        "indicator_count": outcome.indicator_count,
        "imaging_finding_count": len(bundle.imaging_findings),
        "clinical_item_count": len(bundle.clinical_items),
        "issues": [],
    }
