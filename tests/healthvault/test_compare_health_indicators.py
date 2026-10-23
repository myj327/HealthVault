import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from healthvault.application.compare_health_indicators import (
    compare_health_indicators,
)
from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.domain.records import ConfirmedIndicator, TimelineReport
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


def make_indicator(
    code,
    value,
    unit="mmol/L",
    flag="normal",
    source_id="synthetic-source",
):
    return ConfirmedIndicator(
        code=code,
        display_name=f"Synthetic {code}",
        value=value,
        unit=unit,
        reference_lower=0.0,
        reference_upper=10.0,
        flag=flag,
        evidence_source_id=source_id,
        evidence_line_start=4,
        evidence_line_end=4,
        evidence_quote=f"result|{code}|{value}|{unit}|0.0|10.0",
    )


def make_report(report_id, report_date, indicators, source_id=None):
    source_id = source_id or f"source-{report_id}"
    return TimelineReport(
        report_id=report_id,
        report_date=report_date,
        facility="Synthetic Clinic",
        source_id=source_id,
        source_sha256=f"sha256-{report_id}",
        indicators=tuple(indicators),
    )


class RecordingRepository:
    def __init__(self, timeline=(), error=None):
        self.timeline = timeline
        self.error = error
        self.queries = []

    def list_timeline(self, profile_id):
        self.queries.append(profile_id)
        if self.error is not None:
            raise self.error
        return self.timeline


class CompareHealthIndicatorsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "compare-test.sqlite3"
        self.repository = SQLiteHealthRecordRepository(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def fetch_all(self, sql):
        connection = sqlite3.connect(self.database_path)
        try:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(sql).fetchall()]
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
                result = compare_health_indicators(
                    tool_input,
                    repository,
                ).to_dict()
                self.assertEqual("invalid_request", result["status"])
                self.assertEqual(0, result["report_count"])
                self.assertEqual(0, result["comparison_count"])
                self.assertEqual([], result["comparisons"])
                self.assertTrue(result["issues"])
                self.assertEqual([], repository.queries)

    def test_empty_history_and_query_error_remain_distinct(self):
        empty_repository = RecordingRepository()
        empty = compare_health_indicators(
            {"profile_id": "synthetic-empty"},
            empty_repository,
        ).to_dict()

        self.assertEqual("empty", empty["status"])
        self.assertEqual("synthetic-empty", empty["profile_id"])
        self.assertEqual(0, empty["report_count"])
        self.assertEqual(0, empty["comparison_count"])
        self.assertEqual([], empty["comparisons"])
        self.assertEqual([], empty["issues"])
        self.assertEqual(["synthetic-empty"], empty_repository.queries)

        failing_repository = RecordingRepository(
            error=HealthRecordQueryError("synthetic query failure")
        )
        with self.assertRaisesRegex(
            HealthRecordQueryError,
            "synthetic query failure",
        ):
            compare_health_indicators(
                {"profile_id": "synthetic-error"},
                failing_repository,
            )
        self.assertEqual(["synthetic-error"], failing_repository.queries)

    def test_one_report_returns_one_issue_for_each_indicator(self):
        profile_id = "synthetic-one-year"
        confirm_fixture(
            self.repository,
            profile_id,
            "synthetic-one-year-source",
            "synthetic_checkup_2024.txt",
        )

        result = compare_health_indicators(
            {"profile_id": profile_id},
            self.repository,
        ).to_dict()

        self.assertEqual("insufficient_data", result["status"])
        self.assertEqual(1, result["report_count"])
        self.assertEqual(0, result["comparison_count"])
        self.assertEqual([], result["comparisons"])
        self.assertEqual(
            [
                ("fasting_glucose", "insufficient_history"),
                ("hba1c", "insufficient_history"),
                ("ldl_cholesterol", "insufficient_history"),
            ],
            [
                (issue["indicator_code"], issue["code"])
                for issue in result["issues"]
            ],
        )

    def test_two_year_comparison_is_exact_isolated_read_only_and_stable(self):
        profile_id = "synthetic-two-year"
        confirm_fixture(
            self.repository,
            profile_id,
            "synthetic-2025",
            "synthetic_checkup_2025.txt",
        )
        confirm_fixture(
            self.repository,
            profile_id,
            "synthetic-2024",
            "synthetic_checkup_2024.txt",
        )
        confirm_fixture(
            self.repository,
            "synthetic-other-profile",
            "synthetic-other-source",
            "synthetic_checkup_2024.txt",
        )
        before = self.database_snapshot()

        first = compare_health_indicators(
            {"profile_id": profile_id},
            self.repository,
        ).to_dict()
        second = compare_health_indicators(
            {"profile_id": profile_id},
            self.repository,
        ).to_dict()

        self.assertEqual(first, second)
        self.assertEqual(before, self.database_snapshot())
        self.assertEqual("ok", first["status"])
        self.assertEqual(profile_id, first["profile_id"])
        self.assertEqual(2, first["report_count"])
        self.assertEqual(3, first["comparison_count"])
        self.assertEqual([], first["issues"])
        self.assertEqual(
            ["fasting_glucose", "hba1c", "ldl_cholesterol"],
            [comparison["code"] for comparison in first["comparisons"]],
        )

        expected = {
            "fasting_glucose": (5.4, 6.5, 1.1, "increased", "normal", "high"),
            "hba1c": (5.5, 5.9, 0.4, "increased", "normal", "normal"),
            "ldl_cholesterol": (3.1, 2.8, -0.3, "decreased", "normal", "normal"),
        }
        for comparison in first["comparisons"]:
            earliest, latest, change, direction, old_flag, new_flag = expected[
                comparison["code"]
            ]
            self.assertEqual(2, comparison["observation_count"])
            self.assertEqual(earliest, comparison["earliest"]["value"])
            self.assertEqual(latest, comparison["latest"]["value"])
            self.assertEqual(change, comparison["absolute_change"])
            self.assertEqual(direction, comparison["direction"])
            self.assertEqual(
                {
                    "from": old_flag,
                    "to": new_flag,
                    "changed": old_flag != new_flag,
                },
                comparison["flag_transition"],
            )
            self.assertEqual(
                "synthetic-2024",
                comparison["earliest"]["source"]["source_id"],
            )
            self.assertEqual(
                "synthetic-2025",
                comparison["latest"]["source"]["source_id"],
            )
            for point in (comparison["earliest"], comparison["latest"]):
                self.assertEqual(point["source"]["source_id"], point["evidence"]["source_id"])
                self.assertIn("sha256", point["source"])
                self.assertEqual({"lower", "upper"}, set(point["reference_range"]))

        serialized = json.dumps(first, ensure_ascii=False)
        for forbidden in (
            "raw_report_text",
            "idempotency_key",
            "confirmed_at",
            "confirmed_report_id",
            "synthetic-other-source",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_partial_unit_mismatch_unchanged_and_middle_observation(self):
        timeline = (
            make_report(
                "report-1",
                "2023-01-01",
                (
                    make_indicator("stable", -0.0),
                    make_indicator("three_points", 1.1234561),
                    make_indicator("unit_conflict", 2.0, unit="mmol/L"),
                    make_indicator("single", 3.0),
                ),
            ),
            make_report(
                "report-2",
                "2024-01-01",
                (make_indicator("three_points", 2.0),),
            ),
            make_report(
                "report-3",
                "2025-01-01",
                (
                    make_indicator("stable", 0.0),
                    make_indicator("three_points", 2.1234568),
                    make_indicator("unit_conflict", 2.0, unit="mg/dL"),
                ),
            ),
        )
        repository = RecordingRepository(timeline=timeline)

        result = compare_health_indicators(
            {"profile_id": " synthetic-exact-profile "},
            repository,
        ).to_dict()

        self.assertEqual([" synthetic-exact-profile "], repository.queries)
        self.assertEqual("partial", result["status"])
        self.assertEqual(3, result["report_count"])
        self.assertEqual(2, result["comparison_count"])
        self.assertEqual(
            ["stable", "three_points"],
            [comparison["code"] for comparison in result["comparisons"]],
        )
        stable = result["comparisons"][0]
        self.assertEqual(0.0, stable["absolute_change"])
        self.assertEqual("unchanged", stable["direction"])
        self.assertEqual("0.0", str(stable["absolute_change"]))
        three_points = result["comparisons"][1]
        self.assertEqual(3, three_points["observation_count"])
        self.assertEqual("report-1", three_points["earliest"]["report_id"])
        self.assertEqual("report-3", three_points["latest"]["report_id"])
        self.assertEqual(1.000001, three_points["absolute_change"])
        self.assertEqual(
            [
                ("single", "insufficient_history"),
                ("unit_conflict", "unit_mismatch"),
            ],
            [
                (issue["indicator_code"], issue["code"])
                for issue in result["issues"]
            ],
        )


if __name__ == "__main__":
    unittest.main()
