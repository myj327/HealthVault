import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.application.report_preview import build_report_preview
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.ocr.pipeline import OCRPipeline
from healthvault.repositories.health_records import HealthRecordPersistenceError
from healthvault.ui.controller import HealthVaultUIController


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "health_reports"
TXT_FIXTURE = FIXTURE_ROOT / "synthetic_clinical_zh.txt"
IMAGE_FIXTURE = FIXTURE_ROOT / "images" / "synthetic_clinical_en.png"
APP_PATH = Path(__file__).parents[2] / "healthvault" / "ui" / "app.py"


def fixture_text() -> str:
    return TXT_FIXTURE.read_text(encoding="utf-8")


def report_with(*extra_lines: str, report_date: str = "2025-01-31") -> str:
    return "\n".join(
        (
            "# Synthetic test data only; no real person",
            "report_id: HV-SYN-CLINICAL-DATE-001",
            f"report_date: {report_date}",
            "facility: Synthetic Date Clinic",
            *extra_lines,
        )
    )


def table_count(database: Path, table: str) -> int:
    connection = sqlite3.connect(database)
    try:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        connection.close()


class ClinicalItemTests(unittest.TestCase):
    def test_only_explicit_chinese_and_english_labels_create_candidates(self):
        chinese = build_report_preview("clinical-zh", fixture_text()).to_dict()
        english = build_report_preview(
            "clinical-en",
            report_with(
                "Diagnosis: Hypertension [I10]",
                "Doctor Advice：Reduce sodium intake.",
                "Visit Type: Follow-up",
                "Follow-up Plan: 2025-03-01 review",
            ),
        ).to_dict()
        abnormal_without_labels = build_report_preview(
            "no-diagnosis",
            report_with("result|fasting_glucose|9.9|mmol/L|3.9|6.1"),
        ).to_dict()

        for preview in (chinese, english, abnormal_without_labels):
            self.assertEqual("ready", preview["status"])
        self.assertEqual(
            {"diagnosis", "provider_advice", "visit_type", "follow_up_plan"},
            {item["kind"] for item in chinese["clinical_items"]},
        )
        self.assertEqual("E11.9", chinese["clinical_items"][0]["diagnosis_code"])
        self.assertEqual("I10", english["clinical_items"][0]["diagnosis_code"])
        self.assertEqual([], abnormal_without_labels["clinical_items"])
        self.assertTrue(all(item["recorded_by_provider"] for item in chinese["clinical_items"]))
        self.assertTrue(all(len(item["clinical_item_id"]) == 64 for item in chinese["clinical_items"]))
        for item in chinese["clinical_items"]:
            evidence = item["evidence"]
            self.assertEqual(
                item["original_text"],
                fixture_text()[evidence["span_start"] : evidence["span_end"]],
            )

    def test_dates_are_normalized_only_when_deterministic(self):
        explicit = build_report_preview(
            "explicit-date",
            report_with("复诊计划：2025年3月2日复查"),
        ).to_dict()["clinical_items"][0]
        relative = build_report_preview(
            "relative-date",
            report_with("复诊计划：1个月后复查"),
        ).to_dict()["clinical_items"][0]
        fuzzy = build_report_preview(
            "fuzzy-date",
            report_with("复诊计划：近期复查"),
        ).to_dict()["clinical_items"][0]
        missing_base = build_report_preview(
            "missing-base",
            report_with("Follow-up Plan: In 2 weeks", report_date="not-a-date"),
        ).to_dict()["clinical_items"][0]
        conflicting = build_report_preview(
            "conflicting-date",
            report_with("复诊计划：2025-03-02 或 2025-03-09 复查"),
        ).to_dict()["clinical_items"][0]

        self.assertEqual(("normalized", "2025-03-02"), (explicit["date_status"], explicit["normalized_date"]))
        self.assertEqual(("normalized", "2025-02-28"), (relative["date_status"], relative["normalized_date"]))
        self.assertEqual(("needs_date_confirmation", None), (fuzzy["date_status"], fuzzy["normalized_date"]))
        self.assertEqual(("needs_date_confirmation", None), (missing_base["date_status"], missing_base["normalized_date"]))
        self.assertEqual(("needs_date_confirmation", None), (conflicting["date_status"], conflicting["normalized_date"]))

    def test_real_ocr_keeps_block_evidence_but_edited_text_drops_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = OCRPipeline(Path(directory) / "ocr")
            result = pipeline.process(IMAGE_FIXTURE.read_bytes(), IMAGE_FIXTURE.name)
            controller = HealthVaultUIController(Path(directory) / "records.sqlite3")
            preview = controller.preview_ocr_report(
                source_id=f"image:{result['image_sha256']}",
                report_text=result["text"],
                image_sha256=result["image_sha256"],
                ocr_text_sha256=result["text_sha256"],
                ocr_blocks=result["blocks"],
                reviewed=True,
            )
            items = preview["preview"]["clinical_items"]
            self.assertEqual(4, len(items))
            self.assertTrue(all(item["source_type"] == "ocr" for item in items))
            self.assertTrue(all(item["confidence"] is not None for item in items))
            self.assertTrue(all(item["evidence"]["page"] == 1 for item in items))
            self.assertTrue(all(item["evidence"]["polygon"] for item in items))

            edited = controller.preview_ocr_report(
                source_id=f"image:{result['image_sha256']}",
                report_text=result["text"] + "\n# manually reviewed edit",
                image_sha256=result["image_sha256"],
                ocr_text_sha256=result["text_sha256"],
                ocr_blocks=result["blocks"],
                reviewed=True,
            )["preview"]["clinical_items"]
            self.assertTrue(all(item["source_type"] == "ocr_edited" for item in edited))
            self.assertTrue(all(item["confidence"] is None for item in edited))

            changed = controller.confirm_preview(
                profile_id="ocr-clinical-profile",
                preview_state=preview,
                current_source_id=f"image:{result['image_sha256']}",
                current_report_text=result["text"] + "\n# changed after preview",
                current_image_sha256=result["image_sha256"],
                ocr_reviewed=True,
            )
            self.assertEqual("preview_mismatch", changed["status"])
            self.assertEqual(0, table_count(Path(directory) / "records.sqlite3", "clinical_items"))

            confirmed = controller.confirm_preview(
                profile_id="ocr-clinical-profile",
                preview_state=preview,
                current_source_id=f"image:{result['image_sha256']}",
                current_report_text=result["text"],
                current_image_sha256=result["image_sha256"],
                ocr_reviewed=True,
            )
            self.assertEqual("confirmed", confirmed["status"])
            self.assertEqual(4, confirmed["clinical_item_count"])
            self.assertEqual(
                4, controller.timeline("ocr-clinical-profile")["clinical_item_count"]
            )

    def test_confirmation_is_atomic_idempotent_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "records.sqlite3"
            controller = HealthVaultUIController(database)
            preview = controller.preview_report("clinical-zh", fixture_text())
            request = {
                "profile_id": "clinical-profile",
                "preview_state": preview,
                "current_source_id": "clinical-zh",
                "current_report_text": fixture_text(),
            }
            first = controller.confirm_preview(**request)
            second = controller.confirm_preview(**request)
            restarted = HealthVaultUIController(database)
            timeline = restarted.timeline("clinical-profile")

            self.assertEqual("confirmed", first["status"])
            self.assertEqual("already_confirmed", second["status"])
            self.assertEqual(4, first["clinical_item_count"])
            self.assertEqual(4, second["clinical_item_count"])
            self.assertEqual(4, table_count(database, "clinical_items"))
            self.assertEqual(4, timeline["clinical_item_count"])
            self.assertEqual(4, len(timeline["timeline"][0]["clinical_items"]))
            self.assertEqual([], timeline["timeline"][0]["system_inferences"])

            connection = sqlite3.connect(database)
            try:
                connection.execute("DROP TABLE clinical_items")
                connection.commit()
            finally:
                connection.close()
            migrated = HealthVaultUIController(database).timeline("clinical-profile")
            self.assertEqual(1, migrated["report_count"])
            self.assertEqual(0, table_count(database, "clinical_items"))

    def test_clinical_insert_failure_and_stale_preview_write_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "records.sqlite3"
            controller = HealthVaultUIController(database)
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    """
                    CREATE TRIGGER fail_clinical_insert
                    BEFORE INSERT ON clinical_items
                    BEGIN
                        SELECT RAISE(ABORT, 'synthetic clinical failure');
                    END
                    """
                )
                connection.commit()
            finally:
                connection.close()

            preview = controller.preview_report("clinical-zh", fixture_text())
            with self.assertRaises(HealthRecordPersistenceError):
                controller.confirm_preview(
                    profile_id="atomic-profile",
                    preview_state=preview,
                    current_source_id="clinical-zh",
                    current_report_text=fixture_text(),
                )
            self.assertEqual(0, table_count(database, "confirmed_reports"))
            self.assertEqual(0, table_count(database, "health_records"))
            self.assertEqual(0, table_count(database, "clinical_items"))

            stale = controller.confirm_preview(
                profile_id="atomic-profile",
                preview_state=preview,
                current_source_id="clinical-zh",
                current_report_text=fixture_text() + "\n# changed",
            )
            self.assertEqual("preview_mismatch", stale["status"])
            self.assertEqual(0, table_count(database, "clinical_items"))

    def test_streamlit_timeline_labels_records_and_zero_system_inference(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "records.sqlite3"
            controller = HealthVaultUIController(database)
            preview = controller.preview_report("clinical-zh", fixture_text())
            controller.confirm_preview(
                profile_id="demo-profile",
                preview_state=preview,
                current_source_id="clinical-zh",
                current_report_text=fixture_text(),
            )
            with patch.dict(
                "os.environ", {"HEALTHVAULT_UI_DB_PATH": str(database)}
            ):
                app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

        self.assertEqual([], list(app.exception))
        visible_markdown = [element.value for element in app.markdown]
        self.assertTrue(any("病历明确记载" in value for value in visible_markdown))
        self.assertTrue(any("系统推断：0" in value for value in visible_markdown))


if __name__ == "__main__":
    unittest.main()
