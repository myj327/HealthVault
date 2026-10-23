import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from healthvault.ocr.pipeline import OCRProcessingError
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app
from healthvault.web.extraction import build_natural_preview
from healthvault.web.model_runtime import ModelServiceError


TEXT = "报告日期: 2026-10-01\n空腹血糖\n5.6 mmol/L 参考 3.9-6.1\n心脏超声：需结合原报告核对。"


class DurableReportIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "records.sqlite3"
        self.controller = HealthVaultUIController(self.database, knowledge_release_root=self.root / "releases")
        self.app = create_app(self.controller, model_extractor=Mock(side_effect=ModelServiceError(502, "synthetic failure")))
        self.client = TestClient(self.app)
        user = self.client.post("/api/auth/register", json={"username": "intake-user", "password": "synthetic-password-123"}).json()
        self.profile = user["profile_id"]
        self.client.headers.update({"X-CSRF-Token": user["csrf_token"]})

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_rule_preview_and_unknown_text_survive_restart_and_old_token_is_invalid(self):
        response = self.client.post("/api/drafts/text", json={"profile_id": self.profile, "source_id": "durable-source", "report_text": TEXT})
        self.assertEqual(200, response.status_code, response.text)
        draft = response.json()
        self.assertEqual(1, len(draft["preview"]["indicators"]))
        self.assertEqual(2, draft["preview"]["indicators"][0]["evidence"]["line_start"])
        self.assertEqual(3, draft["preview"]["indicators"][0]["evidence"]["line_end"])
        self.assertIn("心脏超声", str(draft["preview"]["other_findings"]))
        self.app.state.model_extractor.assert_not_called()
        reopened = TestClient(create_app(HealthVaultUIController(self.database, knowledge_release_root=self.root / "releases")))
        reopened.cookies.update(self.client.cookies)
        reopened.headers.update(self.client.headers)
        try:
            self.assertEqual(TEXT, reopened.get(f"/api/drafts/{draft['token']}?profile_id={self.profile}").json()["report_text"])
            self.assertEqual(1, len(reopened.get(f"/api/profiles/{self.profile}/uploads").json()["uploads"]))
            updated = reopened.put(f"/api/drafts/{draft['token']}", json={"profile_id": self.profile, "report_text": TEXT + "\n补充原文", "reviewed": False}).json()
            self.assertEqual(TEXT, updated["original_text"])
            self.assertEqual(404, reopened.get(f"/api/drafts/{draft['token']}?profile_id={self.profile}").status_code)
            self.assertEqual(0, reopened.get(f"/api/profiles/{self.profile}/timeline").json()["report_count"])
            confirmed = reopened.post(f"/api/drafts/{updated['token']}/confirm", json={
                "profile_id": self.profile, "acknowledged": True,
                "preview_sha256": updated["preview_sha256"], "report_text": updated["report_text"],
            })
            self.assertEqual(200, confirmed.status_code, confirmed.text)
            report_id = updated["preview"]["report"]["report_id"]
            detail = reopened.get(f"/api/profiles/{self.profile}/reports/{report_id}").json()
            self.assertEqual(TEXT, detail["original_text"])
            self.assertEqual(updated["report_text"], detail["report_text"])
            self.assertEqual(0, len(reopened.get(f"/api/profiles/{self.profile}/uploads").json()["uploads"]))
            self.assertEqual(409, reopened.put(f"/api/drafts/{updated['token']}", json={
                "profile_id": self.profile, "report_text": TEXT, "reviewed": False,
            }).status_code)
        finally:
            reopened.close()

    def test_model_and_ocr_failure_retain_owned_originals(self):
        draft = self.client.post("/api/drafts/text", json={"profile_id": self.profile, "source_id": "model-failure", "report_text": TEXT, "auto_extract": True, "model_consent": True}).json()
        self.assertEqual("extraction_failed", draft["processing_status"])
        self.assertEqual(TEXT, draft["original_text"])
        self.controller.process_image = Mock(side_effect=OCRProcessingError("synthetic OCR failure"))
        raw = b"\x89PNG\r\n\x1a\nsynthetic-original"
        image = self.client.post("/api/drafts/image", data={"profile_id": self.profile}, files={"file": ("source.png", raw, "image/png")}).json()
        self.assertEqual("extraction_failed", image["processing_status"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), image["image_sha256"])
        self.assertEqual(raw, self.client.get(image["image_url"]).content)
        other = TestClient(self.app)
        try:
            other.post("/api/auth/register", json={"username": "other-intake", "password": "synthetic-password-123"})
            self.assertEqual(404, other.get(image["image_url"]).status_code)
        finally:
            other.close()

    def test_wrong_indicator_and_swapped_reference_value_are_not_verified(self):
        source = "报告日期: 2026-10-01\n空腹血糖 5.6 mmol/L 参考 3.9-6.1"
        for code, value in (("total_cholesterol", 5.6), ("fasting_glucose", 3.9)):
            with self.subTest(code=code, value=value):
                preview = build_natural_preview("columns", source, {"indicators": [{"code": code, "value": value, "unit": "mmol/L", "reference_lower": 3.9, "reference_upper": 6.1, "quote": source.splitlines()[1]}]})
                self.assertEqual([("fasting_glucose", 5.6)], [(item["code"], item["value"]) for item in preview["indicators"]])
                self.assertIn("ambiguous_indicator_columns", [item["code"] for item in preview["warnings"]])


if __name__ == "__main__":
    unittest.main()
