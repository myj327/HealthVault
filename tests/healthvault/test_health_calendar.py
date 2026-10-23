import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.ui.controller import HealthVaultUIController


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "health_reports"
CLINICAL_FIXTURE = FIXTURE_ROOT / "synthetic_clinical_zh.txt"
APP_PATH = Path(__file__).parents[2] / "healthvault" / "ui" / "app.py"


def clinical_text() -> str:
    return CLINICAL_FIXTURE.read_text(encoding="utf-8")


def confirm_text(
    controller: HealthVaultUIController,
    profile_id: str,
    source_id: str,
    text: str,
) -> dict:
    preview = controller.preview_report(source_id, text)
    return controller.confirm_preview(
        profile_id=profile_id,
        preview_state=preview,
        current_source_id=source_id,
        current_report_text=text,
    )


def event_by_type(view: dict, event_type: str) -> dict:
    return next(event for event in view["events"] if event["event_type"] == event_type)


class HealthCalendarTests(unittest.TestCase):
    def test_sync_uses_only_confirmed_visit_and_normalized_follow_up(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "calendar.sqlite3")
            profile = "calendar-sync-profile"
            self.assertEqual(
                "confirmed",
                confirm_text(controller, profile, "clinical-normalized", clinical_text())[
                    "status"
                ],
            )

            first = controller.sync_calendar(profile)
            repeated = controller.sync_calendar(profile)
            view = controller.calendar(profile, "2026-06-20")
            self.assertEqual((2, 0), (first["created_count"], first["existing_count"]))
            self.assertEqual((0, 2), (repeated["created_count"], repeated["existing_count"]))
            self.assertEqual(2, view["event_count"])
            self.assertEqual(
                {"historical_visit", "follow_up"},
                {event["event_type"] for event in view["events"]},
            )
            self.assertEqual("completed", event_by_type(view, "historical_visit")["status"])
            self.assertEqual(
                "pending_confirmation", event_by_type(view, "follow_up")["status"]
            )

            fuzzy = (
                clinical_text()
                .replace("HV-SYN-CLINICAL-ZH-001", "HV-SYN-CLINICAL-FUZZY-001")
                .replace("2026年7月2日复查", "近期复查")
            )
            confirm_text(controller, profile, "clinical-fuzzy", fuzzy)
            fuzzy_sync = controller.sync_calendar(profile)
            self.assertEqual(1, fuzzy_sync["needs_date_confirmation_count"])
            self.assertEqual(3, controller.calendar(profile, "2026-06-20")["event_count"])

            invalid_profile = "calendar-invalid-date-profile"
            confirm_text(controller, invalid_profile, "clinical-invalid", clinical_text())
            connection = sqlite3.connect(Path(directory) / "calendar.sqlite3")
            try:
                connection.execute(
                    """
                    UPDATE clinical_items SET normalized_date = 'not-a-date'
                    WHERE kind = 'follow_up_plan' AND confirmed_report_id IN (
                        SELECT id FROM confirmed_reports WHERE profile_id = ?
                    )
                    """,
                    (invalid_profile,),
                )
                connection.commit()
            finally:
                connection.close()
            invalid_sync = controller.sync_calendar(invalid_profile)
            self.assertEqual(1, invalid_sync["needs_date_confirmation_count"])
            self.assertEqual(
                1,
                controller.calendar(invalid_profile, "2026-06-20")[
                    "needs_date_confirmation_count"
                ],
            )
            self.assertEqual(1, invalid_sync["event_count"])

    def test_record_activation_and_update_preserve_source_and_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "calendar.sqlite3")
            profile = "calendar-record-profile"
            confirm_text(controller, profile, "clinical-source", clinical_text())
            controller.sync_calendar(profile)
            pending = event_by_type(controller.calendar(profile, "2026-06-20"), "follow_up")
            immutable = {
                key: pending[key]
                for key in (
                    "source",
                    "source_report_id",
                    "source_clinical_item_id",
                    "source_report_date",
                    "source_original_text",
                    "source_evidence",
                    "source_normalized_date",
                )
            }

            activated = controller.activate_calendar_event(
                profile,
                pending["event_id"],
                scheduled_date="2026-07-03",
                title="人工核对后的复诊",
                remind_before_days=3,
            )
            activated_again = controller.activate_calendar_event(
                profile,
                pending["event_id"],
                scheduled_date="2026-07-03",
                title="人工核对后的复诊",
                remind_before_days=3,
            )
            updated = controller.update_calendar_event(
                profile,
                pending["event_id"],
                scheduled_date="2026-07-04",
                title="复诊安排",
                remind_before_days=2,
            )
            denied = controller.update_calendar_event(
                "another-profile",
                pending["event_id"],
                title="should not change",
            )

            self.assertEqual("activated", activated["status"])
            self.assertEqual("already_activated", activated_again["status"])
            self.assertEqual("updated", updated["status"])
            self.assertEqual("record_extracted", updated["event"]["source"])
            self.assertEqual(immutable, {key: updated["event"][key] for key in immutable})
            self.assertEqual("not_found", denied["status"])

    def test_user_events_are_idempotent_mutable_completed_or_cancelled(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "calendar.sqlite3")
            profile = "calendar-user-profile"
            first = controller.create_calendar_event(
                profile, "user-key-1", "体检预约", "2026-08-10", 2
            )
            replay = controller.create_calendar_event(
                profile, "user-key-1", "体检预约", "2026-08-10", 2
            )
            conflict = controller.create_calendar_event(
                profile, "user-key-1", "伪造覆盖", "2026-08-11", 1
            )
            changed = controller.update_calendar_event(
                profile,
                first["event"]["event_id"],
                title="年度体检",
                scheduled_date="2026-08-12",
                remind_before_days=4,
            )
            completed = controller.complete_calendar_event(
                profile, first["event"]["event_id"]
            )
            completed_again = controller.complete_calendar_event(
                profile, first["event"]["event_id"]
            )

            second = controller.create_calendar_event(
                profile, "user-key-2", "自建复查", "2026-09-01", 1
            )
            cancelled = controller.cancel_calendar_event(
                profile, second["event"]["event_id"]
            )
            cancelled_again = controller.cancel_calendar_event(
                profile, second["event"]["event_id"]
            )

            self.assertEqual("created", first["status"])
            self.assertEqual("already_exists", replay["status"])
            self.assertEqual("idempotency_conflict", conflict["status"])
            self.assertEqual(("updated", "年度体检"), (changed["status"], changed["event"]["title"]))
            self.assertEqual("completed", completed["event"]["status"])
            self.assertEqual("already_completed", completed_again["status"])
            self.assertEqual("cancelled", cancelled["event"]["status"])
            self.assertEqual("already_cancelled", cancelled_again["status"])
            self.assertEqual(2, controller.calendar(profile, "2026-08-01")["event_count"])

    def test_reminder_boundaries_acknowledgement_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "calendar.sqlite3"
            controller = HealthVaultUIController(database)
            created = controller.create_calendar_event(
                "reminder-profile", "reminder-key", "复诊", "2026-07-10", 2
            )
            event_id = created["event"]["event_id"]

            expected = {
                "2026-07-07": "not_due",
                "2026-07-08": "due",
                "2026-07-10": "due",
                "2026-07-11": "overdue",
            }
            for as_of, reminder_status in expected.items():
                event = controller.calendar("reminder-profile", as_of)["events"][0]
                self.assertEqual(reminder_status, event["reminder_status"])

            acknowledged = controller.acknowledge_calendar_reminder(
                "reminder-profile", event_id, "2026-07-08"
            )
            replay = controller.acknowledge_calendar_reminder(
                "reminder-profile", event_id, "2026-07-08"
            )
            restarted = HealthVaultUIController(database)
            persisted = restarted.calendar("reminder-profile", "2026-07-11")["events"][0]
            self.assertEqual("acknowledged", acknowledged["event"]["reminder_status"])
            self.assertEqual("already_acknowledged", replay["status"])
            self.assertEqual("acknowledged", persisted["reminder_status"])

    def test_existing_database_migrates_without_changing_record_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "old-records.sqlite3"
            record_repository = SQLiteHealthRecordRepository(database)
            text = clinical_text()
            confirmed = confirm_health_records(
                {
                    "profile_id": "migration-profile",
                    "source_id": "migration-source",
                    "report_text": text,
                    "expected_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "user_confirmed": True,
                },
                record_repository,
            ).to_dict()
            connection = sqlite3.connect(database)
            try:
                before = connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name = 'calendar_events'"
                ).fetchone()[0]
            finally:
                connection.close()

            controller = HealthVaultUIController(database)
            timeline = controller.timeline("migration-profile")
            comparison = controller.comparisons("migration-profile")
            synced = controller.sync_calendar("migration-profile")
            self.assertEqual("confirmed", confirmed["status"])
            self.assertEqual(0, before)
            self.assertEqual((1, 4), (timeline["report_count"], timeline["clinical_item_count"]))
            self.assertEqual("insufficient_data", comparison["status"])
            self.assertEqual(2, synced["event_count"])

    def test_streamlit_adds_health_calendar_tab_and_sections(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "calendar.sqlite3"
            controller = HealthVaultUIController(database)
            confirm_text(controller, "demo-profile", "clinical-ui", clinical_text())
            controller.sync_calendar("demo-profile")
            with patch.dict(
                "os.environ", {"HEALTHVAULT_UI_DB_PATH": str(database)}
            ):
                app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

        self.assertEqual([], list(app.exception))
        self.assertEqual(
            ["导入与确认", "健康时间线", "知识检索", "就诊摘要", "健康日历"],
            [tab.label for tab in app.tabs],
        )
        markdown = [element.value for element in app.markdown]
        for label in ("待确认事件", "即将到期", "历史与已结束", "待修正计划"):
            self.assertTrue(any(label in value for value in markdown))


if __name__ == "__main__":
    unittest.main()
