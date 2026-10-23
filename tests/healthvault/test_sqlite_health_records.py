import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.application.report_preview import build_report_preview
from healthvault.domain.records import build_idempotency_key
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.repositories.health_records import HealthRecordPersistenceError


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "health_reports"


def load_fixture(name="synthetic_checkup_2024.txt"):
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def confirmation_input(
    *,
    profile_id="synthetic-profile-001",
    source_id="synthetic-2024",
    report_text=None,
    expected_sha256=None,
    user_confirmed=True,
    **extra,
):
    text = report_text if report_text is not None else load_fixture()
    resolved_sha256 = expected_sha256
    if resolved_sha256 is None:
        resolved_sha256 = (
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            if isinstance(text, str)
            else "0" * 64
        )
    values = {
        "profile_id": profile_id,
        "source_id": source_id,
        "report_text": text,
        "expected_sha256": resolved_sha256,
        "user_confirmed": user_confirmed,
    }
    values.update(extra)
    return values


class SQLiteHealthRecordRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "healthvault-test.sqlite3"
        self.repository = SQLiteHealthRecordRepository(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def fetch_all(self, sql, parameters=()):
        connection = sqlite3.connect(self.database_path)
        try:
            connection.row_factory = sqlite3.Row
            return connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

    def counts(self):
        reports = self.fetch_all("SELECT COUNT(*) AS count FROM confirmed_reports")
        records = self.fetch_all("SELECT COUNT(*) AS count FROM health_records")
        return reports[0]["count"], records[0]["count"]

    def test_rejected_requests_leave_both_tables_empty(self):
        old_text = load_fixture()
        changed_text = old_text.replace("5.4", "5.6", 1)
        missing_confirmation = confirmation_input()
        missing_confirmation.pop("user_confirmed")
        cases = (
            (None, "invalid_request"),
            (missing_confirmation, "invalid_request"),
            (
                confirmation_input(user_confirmed=False),
                "confirmation_required",
            ),
            (confirmation_input(user_confirmed=1), "invalid_request"),
            (confirmation_input(user_confirmed="true"), "invalid_request"),
            (confirmation_input(profile_id=""), "invalid_request"),
            (confirmation_input(source_id=None), "invalid_request"),
            (confirmation_input(report_text=[]), "invalid_request"),
            (
                confirmation_input(expected_sha256="not-a-sha"),
                "invalid_request",
            ),
            (
                confirmation_input(
                    report_text=load_fixture("synthetic_invalid_unit.txt")
                ),
                "invalid_preview",
            ),
            (
                confirmation_input(
                    report_text=load_fixture("synthetic_malformed.txt")
                ),
                "invalid_preview",
            ),
            (
                confirmation_input(
                    report_text=changed_text,
                    expected_sha256=hashlib.sha256(old_text.encode("utf-8")).hexdigest(),
                ),
                "preview_mismatch",
            ),
        )

        for tool_input, expected_status in cases:
            with self.subTest(expected_status=expected_status):
                result = confirm_health_records(tool_input, self.repository).to_dict()
                self.assertEqual(expected_status, result["status"])
                self.assertFalse(result["confirmed"])
                self.assertFalse(result["write_performed"])
                self.assertEqual((0, 0), self.counts())

    def test_first_confirmation_atomically_persists_report_records_and_evidence(self):
        tool_input = confirmation_input(
            indicators=[{"code": "forged", "value": 999}],
        )
        preview = build_report_preview(
            tool_input["source_id"],
            tool_input["report_text"],
        ).to_dict()

        result = confirm_health_records(tool_input, self.repository).to_dict()

        self.assertEqual("confirmed", result["status"])
        self.assertTrue(result["confirmed"])
        self.assertTrue(result["write_performed"])
        self.assertFalse(result["idempotent_replay"])
        self.assertEqual(3, result["indicator_count"])
        self.assertEqual((1, 3), self.counts())

        reports = self.fetch_all("SELECT * FROM confirmed_reports")
        self.assertEqual(1, len(reports))
        report = reports[0]
        self.assertEqual(tool_input["profile_id"], report["profile_id"])
        self.assertEqual(tool_input["source_id"], report["source_id"])
        self.assertEqual(tool_input["expected_sha256"], report["source_sha256"])
        self.assertEqual(preview["report"]["report_id"], report["report_id"])
        self.assertEqual(preview["report"]["report_date"], report["report_date"])
        self.assertEqual(preview["report"]["facility"], report["facility"])
        self.assertEqual(tool_input["report_text"], report["raw_report_text"])
        self.assertTrue(report["confirmed_at"].endswith("Z"))
        self.assertEqual(
            build_idempotency_key(
                tool_input["profile_id"],
                tool_input["source_id"],
                tool_input["expected_sha256"],
            ),
            report["idempotency_key"],
        )

        records = self.fetch_all(
            "SELECT * FROM health_records ORDER BY id"
        )
        expected_by_code = {
            indicator["code"]: indicator
            for indicator in preview["indicators"]
        }
        self.assertEqual(set(expected_by_code), {row["indicator_code"] for row in records})
        for row in records:
            expected = expected_by_code[row["indicator_code"]]
            self.assertEqual(expected["display_name"], row["display_name"])
            self.assertEqual(expected["value"], row["value"])
            self.assertEqual(expected["unit"], row["unit"])
            self.assertEqual(
                expected["reference_range"]["lower"],
                row["reference_lower"],
            )
            self.assertEqual(
                expected["reference_range"]["upper"],
                row["reference_upper"],
            )
            self.assertEqual(expected["flag"], row["flag"])
            evidence = expected["evidence"]
            self.assertEqual(evidence["source_id"], row["evidence_source_id"])
            self.assertEqual(evidence["line_start"], row["evidence_line_start"])
            self.assertEqual(evidence["line_end"], row["evidence_line_end"])
            self.assertEqual(evidence["quote"], row["evidence_quote"])

        foreign_keys = self.fetch_all("PRAGMA foreign_key_list(health_records)")
        self.assertEqual("confirmed_reports", foreign_keys[0]["table"])

    def test_cross_instance_replay_uses_database_state_without_changing_rows(self):
        tool_input = confirmation_input()
        first = confirm_health_records(tool_input, self.repository).to_dict()
        original_report = dict(self.fetch_all("SELECT * FROM confirmed_reports")[0])
        original_records = [
            dict(row)
            for row in self.fetch_all("SELECT * FROM health_records ORDER BY id")
        ]

        second_repository = SQLiteHealthRecordRepository(self.database_path)
        second = confirm_health_records(tool_input, second_repository).to_dict()

        self.assertEqual("confirmed", first["status"])
        self.assertEqual("already_confirmed", second["status"])
        self.assertTrue(second["confirmed"])
        self.assertFalse(second["write_performed"])
        self.assertTrue(second["idempotent_replay"])
        self.assertEqual(3, second["indicator_count"])
        self.assertEqual((1, 3), self.counts())
        self.assertEqual(
            original_report,
            dict(self.fetch_all("SELECT * FROM confirmed_reports")[0]),
        )
        self.assertEqual(
            original_records,
            [dict(row) for row in self.fetch_all("SELECT * FROM health_records ORDER BY id")],
        )

    def test_source_report_conflicts_and_profile_partition_do_not_overwrite(self):
        original_input = confirmation_input()
        confirm_health_records(original_input, self.repository)
        original_report = dict(self.fetch_all("SELECT * FROM confirmed_reports")[0])
        original_records = [
            dict(row)
            for row in self.fetch_all("SELECT * FROM health_records ORDER BY id")
        ]

        source_conflict_input = confirmation_input(
            source_id=original_input["source_id"],
            report_text=load_fixture("synthetic_checkup_2025.txt"),
        )
        source_conflict = confirm_health_records(
            source_conflict_input,
            self.repository,
        ).to_dict()
        self.assertEqual("source_conflict", source_conflict["status"])
        self.assertEqual((1, 3), self.counts())

        alternate_text = original_input["report_text"].replace("5.4", "5.6", 1)
        report_conflict_input = confirmation_input(
            source_id="synthetic-alternate-source",
            report_text=alternate_text,
        )
        report_conflict = confirm_health_records(
            report_conflict_input,
            self.repository,
        ).to_dict()
        self.assertEqual("report_conflict", report_conflict["status"])
        self.assertEqual((1, 3), self.counts())
        self.assertEqual(
            original_report,
            dict(self.fetch_all("SELECT * FROM confirmed_reports")[0]),
        )
        self.assertEqual(
            original_records,
            [dict(row) for row in self.fetch_all("SELECT * FROM health_records ORDER BY id")],
        )

        other_profile = confirm_health_records(
            confirmation_input(profile_id="synthetic-profile-002"),
            self.repository,
        ).to_dict()
        self.assertEqual("confirmed", other_profile["status"])
        self.assertEqual((2, 6), self.counts())

    def test_indicator_failure_rolls_back_report_and_all_records(self):
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                """
                CREATE TRIGGER fail_synthetic_indicator
                BEFORE INSERT ON health_records
                WHEN NEW.indicator_code = 'hba1c'
                BEGIN
                    SELECT RAISE(ABORT, 'synthetic indicator failure');
                END
                """
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaises(HealthRecordPersistenceError):
            confirm_health_records(confirmation_input(), self.repository)

        self.assertEqual((0, 0), self.counts())


if __name__ == "__main__":
    unittest.main()
