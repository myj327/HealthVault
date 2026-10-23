"""Repository port and stable persistence failures for health calendar events."""

from __future__ import annotations

from typing import Protocol

from healthvault.domain.calendar import CalendarEvent


class HealthCalendarPersistenceError(RuntimeError):
    """Raised when a calendar write cannot be committed."""


class HealthCalendarQueryError(RuntimeError):
    """Raised when calendar state cannot be queried."""


class HealthCalendarRepository(Protocol):
    def sync_record_events(self, profile_id: str) -> dict: ...

    def list_events(self, profile_id: str) -> tuple[CalendarEvent, ...]: ...

    def get_event(self, profile_id: str, event_id: str) -> CalendarEvent | None: ...
