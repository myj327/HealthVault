import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.ui.controller import HealthVaultUIController
from healthvault.ui.demo_data import seed_demo_profile


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "health_reports"
APP_PATH = Path(__file__).parents[2] / "healthvault" / "ui" / "app.py"


def fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def preview_and_confirm(
    controller: HealthVaultUIController,
    profile_id: str,
    source_id: str,
    report_text: str,
) -> dict:
    preview = controller.preview_report(source_id, report_text)
    return controller.confirm_preview(
        profile_id=profile_id,
        preview_state=preview,
        current_source_id=source_id,
        current_report_text=report_text,
    )


class HealthVaultUIControllerTests(unittest.TestCase):
    def test_demo_profile_seed_is_idempotent_and_populates_primary_views(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "healthvault.sqlite3")

            first = seed_demo_profile(controller, "demo-profile")
            second = seed_demo_profile(controller, "demo-profile")
            timeline = controller.timeline("demo-profile")
            comparisons = controller.comparisons("demo-profile")
            calendar = controller.calendar("demo-profile", "2026-09-14")

        self.assertEqual(("ready", 3, 0), (
            first["status"],
            first["inserted_reports"],
            first["existing_reports"],
        ))
        self.assertEqual(("ready", 0, 3), (
            second["status"],
            second["inserted_reports"],
            second["existing_reports"],
        ))
        self.assertEqual(3, timeline["report_count"])
        self.assertGreaterEqual(len(comparisons["comparisons"]), 3)
        self.assertGreaterEqual(
            len(calendar["pending_confirmation"]) + len(calendar["upcoming_due"]),
            1,
        )

    def test_no_confirmation_invalid_preview_and_changed_text_never_write(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "healthvault.sqlite3")
            profile_id = "ui-synthetic-profile"
            valid_text = fixture("synthetic_checkup_2024.txt")

            controller.preview_report("source-2024", valid_text)
            self.assertEqual(0, controller.timeline(profile_id)["report_count"])

            invalid = controller.preview_report("source-invalid", "not a report")
            invalid_result = controller.confirm_preview(
                profile_id=profile_id,
                preview_state=invalid,
                current_source_id="source-invalid",
                current_report_text="not a report",
            )
            self.assertEqual("invalid_preview", invalid_result["status"])

            valid = controller.preview_report("source-2024", valid_text)
            changed_result = controller.confirm_preview(
                profile_id=profile_id,
                preview_state=valid,
                current_source_id="source-2024",
                current_report_text=valid_text + "\n# changed after preview",
            )
            self.assertEqual("preview_mismatch", changed_result["status"])
            self.assertEqual(0, controller.timeline(profile_id)["report_count"])

    def test_two_year_confirmation_is_idempotent_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "healthvault.sqlite3"
            first = HealthVaultUIController(database)
            profile_id = "ui-synthetic-profile"
            text_2024 = fixture("synthetic_checkup_2024.txt")
            text_2025 = fixture("synthetic_checkup_2025.txt")

            self.assertEqual(
                "confirmed",
                preview_and_confirm(first, profile_id, "source-2024", text_2024)["status"],
            )
            self.assertEqual(
                "already_confirmed",
                preview_and_confirm(first, profile_id, "source-2024", text_2024)["status"],
            )
            self.assertEqual(
                "confirmed",
                preview_and_confirm(first, profile_id, "source-2025", text_2025)["status"],
            )

            restarted = HealthVaultUIController(database)
            timeline = restarted.timeline(profile_id)
            comparison = restarted.comparisons(profile_id)
            self.assertEqual(2, timeline["report_count"])
            self.assertEqual(
                ["2024-05-20", "2025-06-18"],
                [report["report_date"] for report in timeline["timeline"]],
            )
            self.assertGreaterEqual(len(comparison["comparisons"]), 1)

    def test_offline_knowledge_search_keeps_clickable_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "healthvault.sqlite3")
            search = controller.search_knowledge("HbA1c", limit=3)
            self.assertEqual("ok", search["status"])
            hit = search["results"][0]
            self.assertTrue(hit["citation_id"].startswith("HVK:"))
            self.assertTrue(hit["source"]["url"].startswith("https://"))
            self.assertTrue(hit["source"]["publisher"])
            self.assertTrue(hit["excerpt"])

    def test_offline_visit_brief_keeps_record_and_knowledge_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "healthvault.sqlite3")
            profile_id = "ui-synthetic-profile"
            for year in (2024, 2025):
                preview_and_confirm(
                    controller,
                    profile_id,
                    f"source-{year}",
                    fixture(f"synthetic_checkup_{year}.txt"),
                )

            brief = controller.visit_brief(profile_id)
            self.assertIn(brief["status"], {"ok", "partial"})
            self.assertIn("[REC:", brief["brief_markdown"])
            self.assertIn("[HVK:", brief["brief_markdown"])
            self.assertIn("https://", brief["brief_markdown"])

    def test_streamlit_homepage_has_five_tabs_and_no_exception(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"HEALTHVAULT_UI_DB_PATH": str(Path(directory) / "healthvault.sqlite3")},
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

        self.assertEqual([], list(app.exception))
        self.assertEqual(
            ["导入与确认", "健康时间线", "知识检索", "就诊摘要", "健康日历"],
            [tab.label for tab in app.tabs],
        )
        self.assertTrue(
            any("仅使用合成演示数据" in warning.value for warning in app.warning)
        )


if __name__ == "__main__":
    unittest.main()
