"""Application service for deterministic in-app calendar state transitions."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from healthvault.domain.calendar import CalendarEvent
from healthvault.infrastructure.sqlite_health_calendar import (
    SQLiteHealthCalendarRepository,
)


def _valid_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _date_value(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _valid_reminder(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 365


def _reminder_status(event: CalendarEvent, as_of: date) -> str:
    if event.status != "scheduled":
        return "inactive"
    if event.reminder_acknowledged_at is not None:
        return "acknowledged"
    scheduled = date.fromisoformat(event.scheduled_date)
    due_date = scheduled - timedelta(days=event.remind_before_days)
    if as_of < due_date:
        return "not_due"
    if as_of > scheduled:
        return "overdue"
    return "due"


class HealthCalendarService:
    def __init__(self, repository: SQLiteHealthCalendarRepository) -> None:
        self.repository = repository

    def _event_result(
        self, status: str, event: CalendarEvent | None, as_of: date | None = None
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"status": status}
        if event is not None:
            reference_date = as_of or date.fromisoformat(event.scheduled_date)
            result["event"] = event.to_dict(
                reminder_status=_reminder_status(event, reference_date)
            )
        return result

    def sync(self, profile_id: str) -> dict[str, Any]:
        if not _valid_text(profile_id):
            return {"status": "invalid_request"}
        return {"status": "ok", **self.repository.sync_record_events(profile_id)}

    def view(self, profile_id: str, as_of: str) -> dict[str, Any]:
        as_of_date = _date_value(as_of)
        if not _valid_text(profile_id) or as_of_date is None:
            return {
                "status": "invalid_request",
                "event_count": 0,
                "events": [],
                "pending_confirmation": [],
                "upcoming_due": [],
                "ended": [],
                "needs_date_confirmation_count": 0,
            }
        events = [
            event.to_dict(reminder_status=_reminder_status(event, as_of_date))
            for event in self.repository.list_events(profile_id)
        ]
        return {
            "status": "ok",
            "profile_id": profile_id,
            "as_of": as_of,
            "event_count": len(events),
            "events": events,
            "pending_confirmation": [
                event for event in events if event["status"] == "pending_confirmation"
            ],
            "upcoming_due": [
                event for event in events if event["status"] == "scheduled"
            ],
            "ended": [
                event for event in events if event["status"] in {"completed", "cancelled"}
            ],
            "unresolved_plans": self.repository.unresolved_plans(profile_id),
            "needs_date_confirmation_count": self.repository.count_needs_date_confirmation(
                profile_id
            ),
        }

    def schedule_plan(self, profile_id: str, clinical_item_id: str, scheduled_date: str, remind_before_days: int = 1) -> dict[str, Any]:
        if not _valid_text(profile_id) or not _valid_text(clinical_item_id) or _date_value(scheduled_date) is None or not _valid_reminder(remind_before_days):
            return {"status": "invalid_request"}
        event = self.repository.prepare_plan_event(profile_id, clinical_item_id, scheduled_date, remind_before_days)
        if event is None:
            return {"status": "not_found"}
        return self.activate(profile_id, event.event_id, scheduled_date=scheduled_date, remind_before_days=remind_before_days)

    def create_user_event(
        self,
        profile_id: str,
        idempotency_key: str,
        title: str,
        scheduled_date: str,
        remind_before_days: int,
    ) -> dict[str, Any]:
        if (
            not _valid_text(profile_id)
            or not _valid_text(idempotency_key)
            or not _valid_text(title)
            or _date_value(scheduled_date) is None
            or not _valid_reminder(remind_before_days)
        ):
            return {"status": "invalid_request"}
        status, event = self.repository.create_user_event(
            profile_id,
            idempotency_key,
            title.strip(),
            scheduled_date,
            remind_before_days,
        )
        return self._event_result(status, event)

    def activate(
        self,
        profile_id: str,
        event_id: str,
        *,
        scheduled_date: str | None = None,
        title: str | None = None,
        remind_before_days: int | None = None,
    ) -> dict[str, Any]:
        event = self.repository.get_event(profile_id, event_id)
        if event is None:
            return {"status": "not_found"}
        if event.source != "record_extracted":
            return {"status": "invalid_transition"}
        resolved_title = title.strip() if _valid_text(title) else event.title
        resolved_date = scheduled_date or event.scheduled_date
        resolved_reminder = (
            remind_before_days
            if remind_before_days is not None
            else event.remind_before_days
        )
        if _date_value(resolved_date) is None or not _valid_reminder(resolved_reminder):
            return {"status": "invalid_request"}
        if event.status == "scheduled":
            unchanged = (
                event.title == resolved_title
                and event.scheduled_date == resolved_date
                and event.remind_before_days == resolved_reminder
            )
            return self._event_result(
                "already_activated" if unchanged else "invalid_transition", event
            )
        if event.status != "pending_confirmation":
            return {"status": "invalid_transition"}
        updated = self.repository.update_event(
            profile_id,
            event_id,
            title=resolved_title,
            scheduled_date=resolved_date,
            remind_before_days=resolved_reminder,
            activate=True,
        )
        return self._event_result("activated" if updated else "invalid_transition", updated)

    def update(
        self,
        profile_id: str,
        event_id: str,
        *,
        scheduled_date: str | None = None,
        title: str | None = None,
        remind_before_days: int | None = None,
    ) -> dict[str, Any]:
        event = self.repository.get_event(profile_id, event_id)
        if event is None:
            return {"status": "not_found"}
        if event.status != "scheduled":
            return {"status": "invalid_transition"}
        resolved_title = title.strip() if _valid_text(title) else event.title
        resolved_date = scheduled_date or event.scheduled_date
        resolved_reminder = (
            remind_before_days
            if remind_before_days is not None
            else event.remind_before_days
        )
        if _date_value(resolved_date) is None or not _valid_reminder(resolved_reminder):
            return {"status": "invalid_request"}
        updated = self.repository.update_event(
            profile_id,
            event_id,
            title=resolved_title,
            scheduled_date=resolved_date,
            remind_before_days=resolved_reminder,
        )
        return self._event_result("updated" if updated else "invalid_transition", updated)

    def _transition(
        self, profile_id: str, event_id: str, target: str
    ) -> dict[str, Any]:
        event = self.repository.get_event(profile_id, event_id)
        if event is None:
            return {"status": "not_found"}
        if event.status == target:
            return self._event_result(f"already_{target}", event)
        allowed = ("scheduled",) if target == "completed" else (
            "scheduled",
            "pending_confirmation",
        )
        if event.status not in allowed:
            return {"status": "invalid_transition"}
        updated = self.repository.set_status(profile_id, event_id, target, allowed)
        return self._event_result(target if updated else "invalid_transition", updated)

    def complete(self, profile_id: str, event_id: str) -> dict[str, Any]:
        return self._transition(profile_id, event_id, "completed")

    def cancel(self, profile_id: str, event_id: str) -> dict[str, Any]:
        return self._transition(profile_id, event_id, "cancelled")

    def acknowledge(
        self, profile_id: str, event_id: str, as_of: str
    ) -> dict[str, Any]:
        as_of_date = _date_value(as_of)
        event = self.repository.get_event(profile_id, event_id)
        if event is None:
            return {"status": "not_found"}
        if as_of_date is None:
            return {"status": "invalid_request"}
        if event.reminder_acknowledged_at is not None:
            return self._event_result("already_acknowledged", event, as_of_date)
        if _reminder_status(event, as_of_date) not in {"due", "overdue"}:
            return self._event_result("not_due", event, as_of_date)
        updated = self.repository.acknowledge_reminder(profile_id, event_id)
        return self._event_result("acknowledged", updated, as_of_date)
