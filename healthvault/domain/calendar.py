"""Domain values for the application-local HealthVault calendar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CalendarEvent:
    event_id: str
    profile_id: str
    source: str
    event_type: str
    title: str
    scheduled_date: str
    status: str
    remind_before_days: int
    source_report_id: str | None
    source_clinical_item_id: str | None
    source_report_date: str | None
    source_original_text: str | None
    source_evidence: dict[str, Any] | None
    source_normalized_date: str | None
    created_at: str
    updated_at: str
    reminder_acknowledged_at: str | None

    def to_dict(self, *, reminder_status: str) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "profile_id": self.profile_id,
            "source": self.source,
            "event_type": self.event_type,
            "title": self.title,
            "scheduled_date": self.scheduled_date,
            "status": self.status,
            "remind_before_days": self.remind_before_days,
            "reminder_status": reminder_status,
            "source_report_id": self.source_report_id,
            "source_clinical_item_id": self.source_clinical_item_id,
            "source_report_date": self.source_report_date,
            "source_original_text": self.source_original_text,
            "source_evidence": self.source_evidence,
            "source_normalized_date": self.source_normalized_date,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reminder_acknowledged_at": self.reminder_acknowledged_at,
        }
