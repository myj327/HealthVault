"""SQLite persistence for record-extracted and user-created calendar events."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from healthvault.domain.calendar import CalendarEvent
from healthvault.repositories.health_calendar import (
    HealthCalendarPersistenceError,
    HealthCalendarQueryError,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar_events (
    event_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    title TEXT NOT NULL,
    scheduled_date TEXT NOT NULL,
    status TEXT NOT NULL,
    remind_before_days INTEGER NOT NULL,
    source_report_id TEXT,
    source_clinical_item_id TEXT,
    source_report_date TEXT,
    source_original_text TEXT,
    source_evidence_json TEXT,
    source_normalized_date TEXT,
    user_idempotency_key TEXT,
    idempotency_payload_sha256 TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reminder_acknowledged_at TEXT,
    UNIQUE(profile_id, source_clinical_item_id),
    UNIQUE(profile_id, user_idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_calendar_profile_date
ON calendar_events(profile_id, scheduled_date, event_id);
"""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _event_id(profile_id: str, source: str, stable_key: str) -> str:
    return hashlib.sha256(
        f"{profile_id}\0{source}\0{stable_key}".encode("utf-8")
    ).hexdigest()


def _payload_hash(title: str, scheduled_date: str, remind_before_days: int) -> str:
    return hashlib.sha256(
        f"{title}\0{scheduled_date}\0{remind_before_days}".encode("utf-8")
    ).hexdigest()


class SQLiteHealthCalendarRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        connection = self._connect()
        try:
            connection.executescript(SCHEMA)
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _to_event(row: sqlite3.Row) -> CalendarEvent:
        evidence = (
            json.loads(row["source_evidence_json"])
            if row["source_evidence_json"] is not None
            else None
        )
        return CalendarEvent(
            event_id=row["event_id"],
            profile_id=row["profile_id"],
            source=row["source"],
            event_type=row["event_type"],
            title=row["title"],
            scheduled_date=row["scheduled_date"],
            status=row["status"],
            remind_before_days=row["remind_before_days"],
            source_report_id=row["source_report_id"],
            source_clinical_item_id=row["source_clinical_item_id"],
            source_report_date=row["source_report_date"],
            source_original_text=row["source_original_text"],
            source_evidence=evidence,
            source_normalized_date=row["source_normalized_date"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            reminder_acknowledged_at=row["reminder_acknowledged_at"],
        )

    def sync_record_events(self, profile_id: str) -> dict[str, int]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT
                    reports.report_id,
                    reports.report_date,
                    items.*
                FROM clinical_items AS items
                JOIN confirmed_reports AS reports
                    ON reports.id = items.confirmed_report_id
                WHERE reports.profile_id = ?
                  AND items.confirmation_status = 'confirmed'
                  AND items.kind IN ('visit_type', 'follow_up_plan')
                ORDER BY reports.report_date, items.evidence_line_start, items.id
                """,
                (profile_id,),
            ).fetchall()
            created_count = 0
            existing_count = 0
            needs_date_confirmation_count = 0
            now = _utc_timestamp()
            for row in rows:
                if row["kind"] == "visit_type":
                    event_type = "historical_visit"
                    title = f"病历记载就诊：{row['original_text']}"
                    scheduled_date = row["report_date"]
                    status = "completed"
                    remind_before_days = 0
                    source_normalized_date = None
                else:
                    normalized_date = row["normalized_date"]
                    if row["date_status"] != "normalized" or not normalized_date:
                        needs_date_confirmation_count += 1
                        continue
                    try:
                        normalized_value = date.fromisoformat(normalized_date)
                        source_date = date.fromisoformat(row["report_date"])
                    except ValueError:
                        needs_date_confirmation_count += 1
                        continue
                    if normalized_value <= source_date:
                        needs_date_confirmation_count += 1
                        continue
                    event_type = "follow_up"
                    title = f"病历复诊计划：{row['original_text']}"
                    scheduled_date = normalized_date
                    status = "pending_confirmation"
                    remind_before_days = 1
                    source_normalized_date = normalized_date

                evidence = {
                    "source_id": row["evidence_source_id"],
                    "line_start": row["evidence_line_start"],
                    "line_end": row["evidence_line_end"],
                    "span_start": row["evidence_span_start"],
                    "span_end": row["evidence_span_end"],
                    "quote": row["evidence_quote"],
                    "page": row["evidence_page"],
                    "polygon": json.loads(row["evidence_polygon_json"])
                    if row["evidence_polygon_json"] is not None
                    else None,
                }
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO calendar_events (
                        event_id, profile_id, source, event_type, title,
                        scheduled_date, status, remind_before_days,
                        source_report_id, source_clinical_item_id,
                        source_report_date, source_original_text,
                        source_evidence_json, source_normalized_date,
                        created_at, updated_at
                    ) VALUES (?, ?, 'record_extracted', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _event_id(profile_id, "record_extracted", row["clinical_item_id"]),
                        profile_id,
                        event_type,
                        title,
                        scheduled_date,
                        status,
                        remind_before_days,
                        row["report_id"],
                        row["clinical_item_id"],
                        row["report_date"],
                        row["original_text"],
                        json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                        source_normalized_date,
                        now,
                        now,
                    ),
                )
                if cursor.rowcount == 1:
                    created_count += 1
                else:
                    existing_count += 1
            event_count = connection.execute(
                "SELECT COUNT(*) FROM calendar_events WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()[0]
            connection.commit()
            return {
                "created_count": created_count,
                "existing_count": existing_count,
                "needs_date_confirmation_count": needs_date_confirmation_count,
                "event_count": event_count,
            }
        except sqlite3.Error as exc:
            connection.rollback()
            raise HealthCalendarPersistenceError(
                "Failed to synchronize confirmed clinical items into the calendar."
            ) from exc
        finally:
            connection.close()

    def count_needs_date_confirmation(self, profile_id: str) -> int:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT items.date_status, items.normalized_date, reports.report_date
                FROM clinical_items AS items
                JOIN confirmed_reports AS reports
                    ON reports.id = items.confirmed_report_id
                WHERE reports.profile_id = ?
                  AND items.confirmation_status = 'confirmed'
                  AND items.kind = 'follow_up_plan'
                  AND NOT EXISTS (SELECT 1 FROM calendar_events AS events
                      WHERE events.profile_id = reports.profile_id
                        AND events.source_clinical_item_id = items.clinical_item_id)
                """,
                (profile_id,),
            ).fetchall()
            needs_confirmation = 0
            for row in rows:
                if row["date_status"] != "normalized" or not row["normalized_date"]:
                    needs_confirmation += 1
                    continue
                try:
                    normalized = date.fromisoformat(row["normalized_date"])
                    source_date = date.fromisoformat(row["report_date"])
                except ValueError:
                    needs_confirmation += 1
                    continue
                if normalized <= source_date:
                    needs_confirmation += 1
            return needs_confirmation
        except sqlite3.Error as exc:
            raise HealthCalendarQueryError(
                "Failed to count follow-up plans needing date confirmation."
            ) from exc
        finally:
            connection.close()

    def unresolved_plans(self, profile_id: str) -> list[dict]:
        connection = self._connect()
        try:
            rows = connection.execute("""
                SELECT items.*, reports.report_id, reports.report_date
                FROM clinical_items AS items JOIN confirmed_reports AS reports
                    ON reports.id = items.confirmed_report_id
                WHERE reports.profile_id = ? AND items.confirmation_status = 'confirmed'
                    AND items.kind = 'follow_up_plan'
                    AND NOT EXISTS (SELECT 1 FROM calendar_events AS events
                        WHERE events.profile_id = reports.profile_id
                          AND events.source_clinical_item_id = items.clinical_item_id)
                ORDER BY reports.report_date, items.id
            """, (profile_id,)).fetchall()
            return [{"clinical_item_id": row["clinical_item_id"], "original_text": row["original_text"],
                     "report_id": row["report_id"], "report_date": row["report_date"],
                     "date_status": row["date_status"], "evidence_quote": row["evidence_quote"],
                     "evidence_line_start": row["evidence_line_start"]} for row in rows]
        finally:
            connection.close()

    def prepare_plan_event(self, profile_id: str, clinical_item_id: str, scheduled_date: str, remind_before_days: int) -> CalendarEvent | None:
        """Use an explicitly chosen date without rewriting the original report fact."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""
                SELECT items.*, reports.report_id, reports.report_date
                FROM clinical_items AS items JOIN confirmed_reports AS reports
                    ON reports.id = items.confirmed_report_id
                WHERE reports.profile_id = ? AND items.clinical_item_id = ?
                    AND items.kind = 'follow_up_plan' AND items.confirmation_status = 'confirmed'
            """, (profile_id, clinical_item_id)).fetchone()
            if row is None:
                connection.rollback()
                return None
            evidence = {"source_id": row["evidence_source_id"], "line_start": row["evidence_line_start"],
                        "line_end": row["evidence_line_end"], "span_start": row["evidence_span_start"],
                        "span_end": row["evidence_span_end"], "quote": row["evidence_quote"], "page": row["evidence_page"],
                        "polygon": json.loads(row["evidence_polygon_json"]) if row["evidence_polygon_json"] else None}
            event_id = _event_id(profile_id, "record_extracted", clinical_item_id)
            now = _utc_timestamp()
            connection.execute("""
                INSERT OR IGNORE INTO calendar_events (
                    event_id, profile_id, source, event_type, title, scheduled_date, status, remind_before_days,
                    source_report_id, source_clinical_item_id, source_report_date, source_original_text,
                    source_evidence_json, source_normalized_date, created_at, updated_at
                ) VALUES (?, ?, 'record_extracted', 'follow_up', ?, ?, 'pending_confirmation', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (event_id, profile_id, f"病历复诊计划：{row['original_text']}", scheduled_date, remind_before_days,
                   row["report_id"], clinical_item_id, row["report_date"], row["original_text"],
                   json.dumps(evidence, ensure_ascii=False, sort_keys=True), row["normalized_date"], now, now))
            event = connection.execute("SELECT * FROM calendar_events WHERE profile_id = ? AND event_id = ?", (profile_id, event_id)).fetchone()
            connection.commit()
            return self._to_event(event)
        except sqlite3.Error as exc:
            connection.rollback()
            raise HealthCalendarPersistenceError("Failed to prepare a confirmed follow-up plan.") from exc
        finally:
            connection.close()

    def list_events(self, profile_id: str) -> tuple[CalendarEvent, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM calendar_events
                WHERE profile_id = ?
                ORDER BY scheduled_date, event_id
                """,
                (profile_id,),
            ).fetchall()
            return tuple(self._to_event(row) for row in rows)
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise HealthCalendarQueryError("Failed to query calendar events.") from exc
        finally:
            connection.close()

    def get_event(self, profile_id: str, event_id: str) -> CalendarEvent | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM calendar_events WHERE profile_id = ? AND event_id = ?",
                (profile_id, event_id),
            ).fetchone()
            return self._to_event(row) if row is not None else None
        finally:
            connection.close()

    def create_user_event(
        self,
        profile_id: str,
        idempotency_key: str,
        title: str,
        scheduled_date: str,
        remind_before_days: int,
    ) -> tuple[str, CalendarEvent]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM calendar_events
                WHERE profile_id = ? AND user_idempotency_key = ?
                """,
                (profile_id, idempotency_key),
            ).fetchone()
            payload_hash = _payload_hash(title, scheduled_date, remind_before_days)
            if existing is not None:
                status = (
                    "already_exists"
                    if existing["idempotency_payload_sha256"] == payload_hash
                    else "idempotency_conflict"
                )
                connection.commit()
                return status, self._to_event(existing)
            now = _utc_timestamp()
            event_id = _event_id(profile_id, "user_created", idempotency_key)
            connection.execute(
                """
                INSERT INTO calendar_events (
                    event_id, profile_id, source, event_type, title,
                    scheduled_date, status, remind_before_days,
                    user_idempotency_key, idempotency_payload_sha256,
                    created_at, updated_at
                ) VALUES (?, ?, 'user_created', 'user_event', ?, ?, 'scheduled', ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    profile_id,
                    title,
                    scheduled_date,
                    remind_before_days,
                    idempotency_key,
                    payload_hash,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM calendar_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            connection.commit()
            return "created", self._to_event(row)
        except sqlite3.Error as exc:
            connection.rollback()
            raise HealthCalendarPersistenceError("Failed to create calendar event.") from exc
        finally:
            connection.close()

    def update_event(
        self,
        profile_id: str,
        event_id: str,
        *,
        title: str,
        scheduled_date: str,
        remind_before_days: int,
        activate: bool = False,
    ) -> CalendarEvent | None:
        connection = self._connect()
        try:
            allowed_status = "pending_confirmation" if activate else "scheduled"
            cursor = connection.execute(
                """
                UPDATE calendar_events
                SET title = ?, scheduled_date = ?, remind_before_days = ?,
                    reminder_acknowledged_at = CASE
                        WHEN scheduled_date != ? OR remind_before_days != ? THEN NULL
                        ELSE reminder_acknowledged_at END,
                    status = ?, updated_at = ?
                WHERE profile_id = ? AND event_id = ? AND status = ?
                """,
                (
                    title,
                    scheduled_date,
                    remind_before_days,
                    scheduled_date,
                    remind_before_days,
                    "scheduled",
                    _utc_timestamp(),
                    profile_id,
                    event_id,
                    allowed_status,
                ),
            )
            connection.commit()
            return self.get_event(profile_id, event_id) if cursor.rowcount == 1 else None
        except sqlite3.Error as exc:
            connection.rollback()
            raise HealthCalendarPersistenceError("Failed to update calendar event.") from exc
        finally:
            connection.close()

    def set_status(
        self, profile_id: str, event_id: str, status: str, allowed_statuses: tuple[str, ...]
    ) -> CalendarEvent | None:
        placeholders = ",".join("?" for _ in allowed_statuses)
        connection = self._connect()
        try:
            cursor = connection.execute(
                f"""
                UPDATE calendar_events
                SET status = ?, updated_at = ?
                WHERE profile_id = ? AND event_id = ? AND status IN ({placeholders})
                """,
                (status, _utc_timestamp(), profile_id, event_id, *allowed_statuses),
            )
            connection.commit()
            return self.get_event(profile_id, event_id) if cursor.rowcount == 1 else None
        except sqlite3.Error as exc:
            connection.rollback()
            raise HealthCalendarPersistenceError("Failed to change calendar status.") from exc
        finally:
            connection.close()

    def acknowledge_reminder(
        self, profile_id: str, event_id: str
    ) -> CalendarEvent | None:
        connection = self._connect()
        try:
            now = _utc_timestamp()
            cursor = connection.execute(
                """
                UPDATE calendar_events
                SET reminder_acknowledged_at = ?, updated_at = ?
                WHERE profile_id = ? AND event_id = ?
                  AND status = 'scheduled' AND reminder_acknowledged_at IS NULL
                """,
                (now, now, profile_id, event_id),
            )
            connection.commit()
            return self.get_event(profile_id, event_id) if cursor.rowcount == 1 else None
        except sqlite3.Error as exc:
            connection.rollback()
            raise HealthCalendarPersistenceError("Failed to acknowledge reminder.") from exc
        finally:
            connection.close()
