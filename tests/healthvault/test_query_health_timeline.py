import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.application.query_health_timeline import query_health_timeline
from healthvault.application.report_preview import build_report_preview
from healthvault.domain.records import TimelineReport
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.repositories.health_records import HealthRecordQueryError


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "health_reports"


def load_fixture(name):
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def confirm_fixture(repository, profile_id, source_id, filename):
    report_text = load_fixture(filename)
    result = confirm_health_records(
        {
            "profile_id": profile_id,
            "source_id": source_id,
            "report_text": report_text,
            "expected_sha256": hashlib.sha256(
                report_text.encode("utf-8")
            ).hexdigest(),
            "user_confirmed": True,
        },
        repository,
    ).to_dict()
    if result["status"] != "confirmed":
        raise AssertionError(f"Synthetic fixture confirmation failed: {result}")
    return report_text


class RecordingRepository:
    def __init__(self):
        self.queries = []

    def list_timeline(self, profile_id):
        self.queries.append(profile_id)
        return ()


class QueryHealthTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "timeline-test.sqlite3"
        self.repository = SQLiteHealthRecordRepository(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def fetch_all(self, sql, parameters=()):
        connection = sqlite3.connect(self.database_path)
        try:
            connection.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in connection.execute(sql, parameters).fetchall()
            ]
        finally:
            connection.close()

    def database_snapshot(self):
        return {
            "reports": self.fetch_all(
                "SELECT * FROM confirmed_reports ORDER BY id"
            ),
            "records": self.fetch_all(
                "SELECT * FROM health_records ORDER BY id"
            ),
        }

    def test_invalid_input_never_calls_repository(self):
        cases = (
            None,
            "not-an-object",
            {},
            {"profile_id": ""},
            {"profile_id": "   "},
            {"profile_id": 7},
            {"profile_id": []},
        )

        for tool_input in cases:
            with self.subTest(tool_input=tool_input):
                repository = RecordingRepository()
                result = query_health_timeline(tool_input, repository).to_dict()
                self.assertEqual("invalid_request", result["status"])
                self.assertEqual(0, result["report_count"])
                self.assertEqual(0, result["indicator_count"])
                self.assertEqual([], result["timeline"])
                self.assertTrue(result["issues"])
                self.assertEqual([], repository.queries)

    def test_empty_unconfirmed_and_injection_like_profiles_are_empty(self):
        report_text = load_fixture("synthetic_checkup_2024.txt")
        rejected = confirm_health_records(
            {
                "profile_id": "synthetic-unconfirmed",
                "source_id": "synthetic-unconfirmed-source",
                "report_text": report_text,
                "expected_sha256": hashlib.sha256(
                    report_text.encode("utf-8")
                ).hexdigest(),
                "user_confirmed": False,
            },
            self.repository,
        ).to_dict()
        self.assertEqual("confirmation_required", rejected["status"])

        for profile_id in (
            "synthetic-profile-missing",
            "synthetic-unconfirmed",
            "' OR 1=1 --",
        ):
            with self.subTest(profile_id=profile_id):
                result = query_health_timeline(
                    {"profile_id": profile_id},
                    self.repository,
                ).to_dict()
                self.assertEqual("empty", result["status"])
                self.assertEqual(profile_id, result["profile_id"])
                self.assertEqual(0, result["report_count"])
                self.assertEqual(0, result["indicator_count"])
                self.assertEqual([], result["timeline"])

    def test_two_year_timeline_is_sorted_and_matches_stored_evidence(self):
        profile_id = "synthetic-profile-timeline"
        text_2025 = confirm_fixture(
            self.repository,
            profile_id,
            "synthetic-2025",
            "synthetic_checkup_2025.txt",
        )
        text_2024 = confirm_fixture(
            self.repository,
            profile_id,
            "synthetic-2024",
            "synthetic_checkup_2024.txt",
        )
        before = self.database_snapshot()

        first = query_health_timeline(
            {"profile_id": profile_id},
            self.repository,
        ).to_dict()
        second = query_health_timeline(
            {"profile_id": profile_id},
            self.repository,
        ).to_dict()

        self.assertEqual(first, second)
        self.assertEqual(before, self.database_snapshot())
        self.assertEqual("ok", first["status"])
        self.assertEqual(2, first["report_count"])
        self.assertEqual(6, first["indicator_count"])
        self.assertEqual(
            ["2024-05-20", "2025-06-18"],
            [report["report_date"] for report in first["timeline"]],
        )

        expected = (
            (
                "synthetic-2024",
                text_2024,
                [5.4, 5.5, 3.1],
            ),
            (
                "synthetic-2025",
                text_2025,
                [6.5, 5.9, 2.8],
            ),
        )
        for actual_report, (source_id, report_text, values) in zip(
            first["timeline"],
            expected,
        ):
            preview = build_report_preview(source_id, report_text).to_dict()
            self.assertEqual(preview["report"]["report_id"], actual_report["report_id"])
            self.assertEqual(preview["report"]["report_date"], actual_report["report_date"])
            self.assertEqual(preview["report"]["facility"], actual_report["facility"])
            self.assertEqual(source_id, actual_report["source"]["source_id"])
            self.assertEqual(
                preview["source"]["sha256"],
                actual_report["source"]["sha256"],
            )
            self.assertEqual(
                ["fasting_glucose", "hba1c", "ldl_cholesterol"],
                [indicator["code"] for indicator in actual_report["indicators"]],
            )
            self.assertEqual(
                values,
                [indicator["value"] for indicator in actual_report["indicators"]],
            )
            expected_by_code = {
                indicator["code"]: indicator
                for indicator in preview["indicators"]
            }
            for indicator in actual_report["indicators"]:
                expected_indicator = expected_by_code[indicator["code"]]
                self.assertEqual(expected_indicator, indicator)

        flags_2025 = {
            indicator["code"]: indicator["flag"]
            for indicator in first["timeline"][1]["indicators"]
        }
        self.assertEqual("high", flags_2025["fasting_glucose"])
        self.assertEqual("normal", flags_2025["hba1c"])
        self.assertEqual("normal", flags_2025["ldl_cholesterol"])

        serialized = str(first)
        for forbidden in (
            "raw_report_text",
            "idempotency_key",
            "confirmed_at",
            "confirmed_report_id",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_profile_isolation_and_same_day_sorting_are_stable(self):
        base_text = load_fixture("synthetic_checkup_2024.txt")
        text_b = base_text.replace("HV-SYN-2024-001", "HV-SYN-SAME-B")
        text_a = base_text.replace("HV-SYN-2024-001", "HV-SYN-SAME-A")
        second_profile = "synthetic-profile-isolated"
        confirm_fixture(
            self.repository,
            second_profile,
            "synthetic-isolated-source",
            "synthetic_checkup_2024.txt",
        )

        # A separate profile can safely store both same-day variants.
        variant_profile = "synthetic-same-day-variants"
        for source_id, report_text in (
            ("synthetic-source-z", text_b),
            ("synthetic-source-a", text_a),
        ):
            result = confirm_health_records(
                {
                    "profile_id": variant_profile,
                    "source_id": source_id,
                    "report_text": report_text,
                    "expected_sha256": hashlib.sha256(
                        report_text.encode("utf-8")
                    ).hexdigest(),
                    "user_confirmed": True,
                },
                self.repository,
            ).to_dict()
            self.assertEqual("confirmed", result["status"])

        variants = query_health_timeline(
            {"profile_id": variant_profile},
            self.repository,
        ).to_dict()
        self.assertEqual(
            ["HV-SYN-SAME-A", "HV-SYN-SAME-B"],
            [report["report_id"] for report in variants["timeline"]],
        )
        isolated = query_health_timeline(
            {"profile_id": second_profile},
            self.repository,
        ).to_dict()
        self.assertEqual(1, isolated["report_count"])
        self.assertEqual(
            "synthetic-isolated-source",
            isolated["timeline"][0]["source"]["source_id"],
        )

    def test_profile_identifier_is_queried_without_normalization(self):
        exact_profile_id = " synthetic-profile-exact "
        confirm_fixture(
            self.repository,
            exact_profile_id,
            "synthetic-exact-source",
            "synthetic_checkup_2024.txt",
        )

        exact_result = query_health_timeline(
            {"profile_id": exact_profile_id},
            self.repository,
        ).to_dict()
        normalized_result = query_health_timeline(
            {"profile_id": exact_profile_id.strip()},
            self.repository,
        ).to_dict()

        self.assertEqual("ok", exact_result["status"])
        self.assertEqual(exact_profile_id, exact_result["profile_id"])
        self.assertEqual("empty", normalized_result["status"])

    def test_repository_returns_domain_models_and_wraps_sqlite_failure(self):
        confirm_fixture(
            self.repository,
            "synthetic-domain-model",
            "synthetic-domain-source",
            "synthetic_checkup_2024.txt",
        )
        timeline = self.repository.list_timeline("synthetic-domain-model")
        self.assertIsInstance(timeline, tuple)
        self.assertIsInstance(timeline[0], TimelineReport)
        self.assertNotIsInstance(timeline[0], sqlite3.Row)

        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("DROP TABLE health_records")
            connection.commit()
        finally:
            connection.close()

        with self.assertRaises(HealthRecordQueryError):
            self.repository.list_timeline("synthetic-domain-model")


if __name__ == "__main__":
    unittest.main()
