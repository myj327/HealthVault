import hashlib
import unittest
from pathlib import Path

from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.application.report_preview import build_report_preview
from healthvault.domain.records import RepositoryConfirmation


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "health_reports"


def load_fixture(name="synthetic_checkup_2024.txt"):
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def valid_input(**overrides):
    report_text = load_fixture()
    values = {
        "profile_id": "synthetic-profile-001",
        "source_id": "synthetic-2024",
        "report_text": report_text,
        "expected_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        "user_confirmed": True,
    }
    values.update(overrides)
    return values


class RecordingRepository:
    def __init__(self, status="confirmed"):
        self.status = status
        self.bundles = []

    def confirm_report(self, bundle):
        self.bundles.append(bundle)
        return RepositoryConfirmation(
            status=self.status,
            write_performed=self.status == "confirmed",
            profile_id=bundle.profile_id,
            source_id=bundle.source_id,
            source_sha256=bundle.source_sha256,
            report_id=bundle.report_id,
            report_date=bundle.report_date,
            indicator_count=len(bundle.indicators),
        )


class ConfirmHealthRecordsApplicationTests(unittest.TestCase):
    def assert_rejected(self, result, status):
        encoded = result.to_dict()
        self.assertEqual(status, encoded["status"])
        self.assertFalse(encoded["confirmed"])
        self.assertFalse(encoded["write_performed"])
        self.assertFalse(encoded["idempotent_replay"])
        self.assertTrue(encoded["issues"])

    def test_confirmation_gate_is_strict_and_prevents_repository_calls(self):
        cases = (
            (valid_input(user_confirmed=False), "confirmation_required"),
            (valid_input(user_confirmed=1), "invalid_request"),
            (valid_input(user_confirmed="true"), "invalid_request"),
            ({key: value for key, value in valid_input().items() if key != "user_confirmed"}, "invalid_request"),
            (None, "invalid_request"),
            ("not-a-dict", "invalid_request"),
        )

        for tool_input, expected_status in cases:
            with self.subTest(tool_input=tool_input):
                repository = RecordingRepository()
                result = confirm_health_records(tool_input, repository)
                self.assert_rejected(result, expected_status)
                self.assertEqual([], repository.bundles)

    def test_required_string_and_sha_validation_prevents_repository_calls(self):
        cases = (
            {"profile_id": ""},
            {"profile_id": 7},
            {"source_id": "  "},
            {"source_id": None},
            {"report_text": ""},
            {"report_text": []},
            {"expected_sha256": "abc"},
            {"expected_sha256": "A" * 64},
            {"expected_sha256": None},
        )

        for override in cases:
            with self.subTest(override=override):
                repository = RecordingRepository()
                result = confirm_health_records(valid_input(**override), repository)
                self.assert_rejected(result, "invalid_request")
                self.assertEqual([], repository.bundles)

    def test_invalid_preview_and_sha_mismatch_never_reach_repository(self):
        invalid_cases = (
            "synthetic_invalid_unit.txt",
            "synthetic_malformed.txt",
        )
        for filename in invalid_cases:
            with self.subTest(filename=filename):
                repository = RecordingRepository()
                report_text = load_fixture(filename)
                result = confirm_health_records(
                    valid_input(
                        report_text=report_text,
                        expected_sha256=hashlib.sha256(
                            report_text.encode("utf-8")
                        ).hexdigest(),
                    ),
                    repository,
                )
                self.assert_rejected(result, "invalid_preview")
                self.assertEqual([], repository.bundles)
                self.assertTrue(result.to_dict()["issues"])

        repository = RecordingRepository()
        original = load_fixture()
        changed = original.replace("5.4", "5.6", 1)
        result = confirm_health_records(
            valid_input(report_text=changed),
            repository,
        )
        self.assert_rejected(result, "preview_mismatch")
        self.assertEqual([], repository.bundles)

    def test_success_uses_only_server_preview_and_ignores_forged_fields(self):
        repository = RecordingRepository()
        tool_input = valid_input(
            indicators=[{"code": "forged", "value": 999}],
            report_id="FORGED-REPORT",
            evidence={"quote": "forged"},
        )

        result = confirm_health_records(tool_input, repository).to_dict()

        self.assertEqual("confirmed", result["status"])
        self.assertTrue(result["confirmed"])
        self.assertTrue(result["write_performed"])
        self.assertFalse(result["idempotent_replay"])
        self.assertEqual(3, result["indicator_count"])
        self.assertEqual(1, len(repository.bundles))
        bundle = repository.bundles[0]
        preview = build_report_preview(
            tool_input["source_id"],
            tool_input["report_text"],
        ).to_dict()
        self.assertEqual(preview["report"]["report_id"], bundle.report_id)
        self.assertEqual(
            [item["code"] for item in preview["indicators"]],
            [item.code for item in bundle.indicators],
        )
        self.assertNotIn("forged", {item.code for item in bundle.indicators})

    def test_unknown_indicator_stays_in_raw_report_but_not_structured_records(self):
        report_text = load_fixture() + "result|unmapped_marker|9|unknown-unit|0|10\n"
        repository = RecordingRepository()
        preview = build_report_preview("synthetic-2024", report_text).to_dict()

        self.assertEqual("ready", preview["status"])
        self.assertEqual("pending_mapping", preview["warnings"][0]["status"])
        result = confirm_health_records(
            valid_input(
                report_text=report_text,
                expected_sha256=hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
            ),
            repository,
        ).to_dict()

        self.assertEqual("confirmed", result["status"])
        bundle = repository.bundles[0]
        self.assertEqual(3, len(bundle.indicators))
        self.assertIn("unmapped_marker", bundle.raw_report_text)
        self.assertNotIn("unmapped_marker", {item.code for item in bundle.indicators})

    def test_repository_conflicts_are_returned_as_stable_rejections(self):
        for status in ("source_conflict", "report_conflict"):
            with self.subTest(status=status):
                result = confirm_health_records(
                    valid_input(),
                    RecordingRepository(status=status),
                )
                self.assert_rejected(result, status)
                self.assertEqual(status, result.to_dict()["issues"][0]["code"])


if __name__ == "__main__":
    unittest.main()
