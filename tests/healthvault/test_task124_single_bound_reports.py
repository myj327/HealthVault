"""Synthetic regressions; no patient images or report text are stored here."""
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from contextlib import closing
from unittest.mock import patch

from fastapi.testclient import TestClient

from healthvault.application.confirm_extracted_report import confirm_extracted_report
from healthvault.domain.comparisons import ComparisonPoint
from healthvault.infrastructure.sqlite_health_records import SCHEMA, SQLiteHealthRecordRepository
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.extraction import build_natural_preview, _normalize_indicator


REPORT = "\n".join((
    "合成检验报告，仅供测试",
    "出生日期：2002-03-04 标本：血清 检测日期：2053-07-08",
    "甘油三酯 1.30 适宜 <1.80mmol/L 说明：低密度脂蛋白胆固醇目标需结合分层",
    "增高1.80-2.50",
    "很高>2.50",
    "总胆固醇 4.20 适宜 <=5.40mmol/L 心血管危险分层",
    "低密度脂蛋白胆固醇 2.10 适宜 ≤3.50mmol/L",
    "高密度脂蛋白胆固醇 1.30 >1.10mmol/L 风险说明",
    "低危 <3.5 mmol/L 健康人群",
    "中高危 <2.7 mmol/L",
    "LDL-C≥4.8 mmol/L",
    "极高危 <1.9 mmol/L",
))


def blocks_for(text):
    offset = 0
    blocks = []
    for index, line in enumerate(text.splitlines()):
        blocks.append(dict(text=line, span_start=offset, span_end=offset+len(line), page=1,
                           polygon=[[0, index*20], [600, index*20], [600, index*20+15], [0, index*20+15]],
                           page_width=800, page_height=400))
        offset += len(line)+1
    return blocks


class SyntheticOCRController(HealthVaultUIController):
    def process_image(self, image_bytes, filename):
        return {"text": REPORT, "text_sha256": hashlib.sha256(REPORT.encode("utf-8")).hexdigest(), "blocks": blocks_for(REPORT),
                "confidence": {"average": 0.99, "minimum": 0.99, "low_confidence_blocks": []}}


class SingleBoundReportTests(unittest.TestCase):
    def test_four_results_exclude_side_panel_and_bind_original_fields(self):
        preview = build_natural_preview("synthetic", REPORT, {}, blocks=blocks_for(REPORT))
        self.assertTrue(preview["can_confirm"], preview["issues"])
        self.assertEqual("2053-07-08", preview["report"]["report_date"])
        expected = [("triglycerides", 1.3, None, 1.8, "<1.80"),
                    ("total_cholesterol", 4.2, None, 5.4, "<=5.40"),
                    ("ldl_cholesterol", 2.1, None, 3.5, "≤3.50"),
                    ("hdl_cholesterol", 1.3, 1.1, None, ">1.10")]
        self.assertEqual(expected, [(i["code"], i["value"], i["reference_range"]["lower"],
                                    i["reference_range"]["upper"], i["reference_range"]["text"])
                                   for i in preview["indicators"]])
        self.assertEqual([], preview["clinical_items"])
        for item in preview["indicators"]:
            evidence = item["evidence"]
            self.assertTrue(evidence["quote"].endswith("mmol/L"))
            self.assertEqual(evidence["quote"], REPORT[evidence["span_start"]:evidence["span_end"]])
            for field in evidence["fields"].values():
                self.assertEqual(field["text"], REPORT[field["span_start"]:field["span_end"]])
                self.assertTrue(field["regions"])
            self.assertEqual(item["reference_range"]["text"], evidence["fields"]["reference_range"]["text"])
            self.assertEqual("2053-07-08", evidence["fields"]["report_date"]["text"])

    def test_strict_inclusive_comparators_and_unit_alias(self):
        for operator, flag in (("<", "high"), (">", "low"), ("<=", "normal"),
                               (">=", "normal"), ("≤", "normal"), ("≥", "normal")):
            with self.subTest(operator=operator):
                preview = build_natural_preview("boundary", f"检测日期：2053-07-08\n甘油三酯 1.8 {operator}1.80mmol/L", {})
                self.assertTrue(preview["can_confirm"])
                self.assertEqual(flag, preview["indicators"][0]["flag"])
                self.assertEqual(operator, preview["indicators"][0]["reference_range"]["operator"])
        alias = build_natural_preview("alias", "检测日期：2053-07-08\n肌酐 80 <100µmol/L", {})
        self.assertTrue(alias["can_confirm"])
        self.assertEqual("umol/L", alias["indicators"][0]["unit"])
        self.assertEqual("<100", alias["indicators"][0]["reference_range"]["text"])

    def test_targets_and_invented_values_or_bounds_are_not_measurements(self):
        for row in ("低密度脂蛋白胆固醇 <1.9mmol/L", "LDL-C≥4.8 mmol/L", "甘油三酯 1.3 mg/L <1.8"):
            with self.subTest(row=row):
                self.assertFalse(build_natural_preview("target", "检测日期：2053-07-08\n"+row, {})["can_confirm"])
        quote = "甘油三酯 1.30 适宜 <1.80mmol/L"
        good = dict(code="triglycerides", value=1.3, unit="mmol/L", reference_lower=None, reference_upper=1.8, quote=quote)
        for change in ({"value": 1.8}, {"reference_lower": 0}, {"reference_upper": 1.3},
                       {"unit": "mg/L"}, {"code": "ldl_cholesterol"}):
            with self.subTest(change=change):
                self.assertIsNone(_normalize_indicator({**good, **change}, quote, "synthetic")[0])
        # A model's larger quote must not reattach the neighbouring advice panel.
        extended = quote + " 说明：低密度脂蛋白胆固醇目标需结合分层"
        item, warning = _normalize_indicator({**good, "quote": extended}, extended, "synthetic")
        self.assertIsNone(warning)
        self.assertEqual(quote, item["evidence"]["quote"])

    def test_date_conflicts_and_existing_closed_ranges(self):
        row = "空腹血糖 5.4 mmol/L 参考 3.9-6.1"
        valid = build_natural_preview("closed", "出生日期：2002-03-04 检测日期：2053-07-08\n"+row, {})
        self.assertTrue(valid["can_confirm"])
        self.assertEqual({"lower": 3.9, "upper": 6.1}, valid["indicators"][0]["reference_range"])
        conflict = build_natural_preview("conflict", "检测日期：2053-07-08\n报告日期：2053-07-09\n"+row, {})
        self.assertFalse(conflict["can_confirm"])
        self.assertIsNone(conflict["report"]["report_date"])

    def test_image_review_confirm_and_reopen_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "HEALTHVAULT_UI_DB_PATH": str(Path(directory)/"bootstrap.sqlite3"),
            "HEALTHVAULT_KNOWLEDGE_MODE": "lexical",
            "HEALTHVAULT_KNOWLEDGE_RELEASE_ROOT": str(Path(directory)/"knowledge"),
        }), patch("healthvault.web.api.extract_with_model") as model:
            from healthvault.web.api import create_app
            database = Path(directory)/"records.sqlite3"
            with TestClient(create_app(SyntheticOCRController(database))) as client:
                user = client.app.state.auth_store.register("synthetic-lipids", "synthetic-password-123", user_id="synthetic-lipids")
                token, csrf = client.app.state.auth_store.create_session(user)
                client.cookies.set("hv_session", token)
                client.headers.update({"X-CSRF-Token": csrf})
                upload = client.post("/api/drafts/image", data={"profile_id": "synthetic-lipids"},
                                     files={"file": ("synthetic.png", b"\x89PNG\r\n\x1a\nsynthetic-image", "image/png")})
                self.assertEqual(200, upload.status_code, upload.text)
                draft = upload.json()
                self.assertIsNone(draft["preview"])
                reviewed = client.put(f"/api/drafts/{draft['token']}", json={
                    "profile_id": "synthetic-lipids", "report_text": REPORT, "reviewed": True, "model_consent": False})
                self.assertEqual(200, reviewed.status_code, reviewed.text)
                draft = reviewed.json()
                body = {"profile_id": "synthetic-lipids", "report_text": REPORT,
                        "preview_sha256": draft["preview_sha256"], "acknowledged": True}
                confirm = client.post(f"/api/drafts/{draft['token']}/confirm", json=body)
                self.assertEqual(200, confirm.status_code, confirm.text)
                self.assertEqual("confirmed", confirm.json()["status"])
                replay = client.post(f"/api/drafts/{draft['token']}/confirm", json=body)
                self.assertEqual("already_confirmed", replay.json()["status"])
                timeline = client.get("/api/profiles/synthetic-lipids/timeline").json()
                self.assertEqual(1, timeline["report_count"])
                self.assertEqual(4, len(timeline["timeline"][0]["indicators"]))
            model.assert_not_called()
            stored = SQLiteHealthRecordRepository(database).list_timeline("synthetic-lipids")
            self.assertEqual(1, len(stored))
            expected = {i["code"]: i["reference_range"] for i in draft["preview"]["indicators"]}
            self.assertEqual(expected, {i["code"]: i["reference_range"] for i in stored[0].to_dict()["indicators"]})
            for indicator in stored[0].indicators:
                point = ComparisonPoint(stored[0].report_id, stored[0].report_date, stored[0].source_id, stored[0].source_sha256, indicator)
                self.assertEqual(expected[indicator.code], point.to_dict()["reference_range"])

    @staticmethod
    def seed_legacy(connection):
        schema = SCHEMA.replace("reference_lower REAL,", "reference_lower REAL NOT NULL,").replace("reference_upper REAL,", "reference_upper REAL NOT NULL,")
        connection.executescript(schema)
        connection.execute("ALTER TABLE health_records ADD COLUMN evidence_locator_json TEXT")
        connection.execute("CREATE INDEX legacy_indicator_index ON health_records(indicator_code)")
        connection.execute("INSERT INTO confirmed_reports VALUES (1, 'legacy-key', 'synthetic-lipids', 'legacy', 'hash', 'legacy-report', '2052-07-08', 'synthetic', 'synthetic original', '2052-07-08')")
        connection.execute("INSERT INTO health_records VALUES (7, 1, 'fasting_glucose', '空腹血糖', 5.4, 'mmol/L', 3.9, 6.1, 'normal', 'legacy', 1, 1, '空腹血糖 5.4 mmol/L 3.9-6.1', ?)", (json.dumps({"span_start": 0, "span_end": 26}),))
        connection.commit()

    def test_legacy_migration_preserves_rows_constraints_index_and_locators(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory)/"legacy.sqlite3"
            with closing(sqlite3.connect(database)) as connection:
                self.seed_legacy(connection)
                original = connection.execute("SELECT * FROM health_records").fetchall()
            repo = SQLiteHealthRecordRepository(database)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(original, connection.execute("SELECT * FROM health_records").fetchall())
                self.assertEqual([], connection.execute("PRAGMA foreign_key_check").fetchall())
                self.assertIsNotNone(connection.execute("SELECT name FROM sqlite_master WHERE name = 'legacy_indicator_index'").fetchone())
                self.assertTrue(all(not row[3] for row in connection.execute("PRAGMA table_info(health_records)") if row[1].startswith("reference_")))
            preview = build_natural_preview("new", REPORT, {})
            result = confirm_extracted_report("synthetic-lipids", "new", REPORT, preview, repo)
            self.assertEqual("confirmed", result["status"])
            reopened = SQLiteHealthRecordRepository(database)
            self.assertEqual(2, len(reopened.list_timeline("synthetic-lipids")))
            self.assertEqual("already_confirmed", confirm_extracted_report("synthetic-lipids", "new", REPORT, preview, reopened)["status"])

    def test_legacy_migration_rolls_back_if_rename_fails(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            self.seed_legacy(connection)
            connection.row_factory = sqlite3.Row
            original = [tuple(row) for row in connection.execute("SELECT * FROM health_records")]
            connection.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_ALTER_TABLE else sqlite3.SQLITE_OK)
            with self.assertRaises(sqlite3.DatabaseError):
                SQLiteHealthRecordRepository._allow_single_bound_ranges(connection)
            connection.set_authorizer(None)
            self.assertEqual(original, [tuple(row) for row in connection.execute("SELECT * FROM health_records")])
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name = 'health_records_nullable'").fetchone())


if __name__ == "__main__":
    unittest.main()
