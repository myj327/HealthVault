import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app
from healthvault.web.evidence import layout_blocks
from healthvault.web.extraction import build_natural_preview


class WebCalendarTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "records.sqlite3"
        self.controller = HealthVaultUIController(self.database, knowledge_release_root=self.root / "releases")
        self.extractor = Mock(side_effect=AssertionError("No paid model is needed"))
        self.app = create_app(self.controller, model_extractor=self.extractor)
        self.client = TestClient(self.app)
        account = self.client.post("/api/auth/register", json={"username": "calendar-owner", "password": "synthetic-password-123"}).json()
        self.profile = account["profile_id"]
        self.client.headers.update({"X-CSRF-Token": account["csrf_token"]})
        self.base = f"/api/profiles/{self.profile}/calendar"

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def import_report(self, text):
        response = self.client.post("/api/drafts/text", json={"profile_id": self.profile, "source_id": "clinical-source", "report_text": text})
        self.assertEqual(200, response.status_code, response.text)
        draft = response.json()
        self.assertTrue(draft["preview"]["can_confirm"], draft)
        response = self.client.post(f"/api/drafts/{draft['token']}/confirm", json={"profile_id": self.profile, "acknowledged": True, "report_text": text, "preview_sha256": draft["preview_sha256"]})
        self.assertEqual(200, response.status_code, response.text)
        return draft, response.json()

    def test_natural_clinical_only_report_is_saved_and_month_end_is_clamped(self):
        text = "报告日期: 2026-01-31\n临床诊断：甲状腺结节\n医生建议：遵医嘱复查\n复查计划：1个月后复查超声"
        draft, result = self.import_report(text)
        self.assertEqual(4, result["clinical_item_count"])
        self.extractor.assert_not_called()
        self.assertEqual([], draft["preview"]["indicators"])
        with closing(sqlite3.connect(self.database)) as connection:
            rows = connection.execute("SELECT kind, original_text, normalized_date, confirmation_status FROM clinical_items").fetchall()
        self.assertIn(("diagnosis", "甲状腺结节", None, "confirmed"), rows)
        self.assertIn(("follow_up_plan", "1个月后复查超声", "2026-02-28", "confirmed"), rows)
        view = self.client.get(self.base).json()
        self.assertEqual(1, len(view["pending_confirmation"]))
        self.assertEqual(1, len(view["unresolved_plans"]))
        self.assertEqual("inactive", view["pending_confirmation"][0]["reminder_status"])

    def test_report_to_activated_completed_calendar_survives_restart_and_sync_is_idempotent(self):
        self.import_report("报告日期: 2026-10-01\n复诊计划：2周后复诊")
        event = self.client.get(self.base).json()["pending_confirmation"][0]
        url = f"{self.base}/events/{event['event_id']}"
        self.assertEqual(409, self.client.post(url + "/activate", json={}).status_code)
        response = self.client.post(url + "/activate", json={"acknowledged": True})
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("scheduled", response.json()["event"]["status"])
        self.assertEqual("2周后复诊", response.json()["event"]["source_original_text"])
        self.assertEqual(1, self.client.get(self.base).json()["event_count"])
        self.assertEqual("completed", self.client.post(url + "/complete").json()["event"]["status"])
        reopened = TestClient(create_app(HealthVaultUIController(self.database, knowledge_release_root=self.root / "releases")))
        reopened.cookies.update(self.client.cookies)
        try:
            view = reopened.get(self.base).json()
            self.assertEqual(1, view["event_count"])
            self.assertEqual("completed", view["ended"][0]["status"])
        finally:
            reopened.close()

    def test_ambiguous_plan_requires_chosen_date_without_rewriting_source(self):
        self.import_report("报告日期: 2026-10-01\n复诊计划：近期复诊")
        view = self.client.get(self.base).json()
        self.assertEqual(0, view["event_count"])
        self.assertEqual(1, view["needs_date_confirmation_count"])
        item = view["unresolved_plans"][0]
        url = f"{self.base}/plans/{item['clinical_item_id']}/schedule"
        body = {"scheduled_date": "2026-10-20", "acknowledged": True}
        self.assertEqual(409, self.client.post(url, json={**body, "acknowledged": False}).status_code)
        self.assertEqual(409, self.client.post(url, json={**body, "scheduled_date": "invalid"}).status_code)
        for _ in range(2):
            response = self.client.post(url, json=body)
            self.assertEqual(200, response.status_code, response.text)
            self.assertIsNone(response.json()["event"]["source_normalized_date"])
        view = self.client.get(self.base).json()
        self.assertEqual(1, view["event_count"])
        self.assertEqual([], view["unresolved_plans"])
        self.assertEqual(0, view["needs_date_confirmation_count"])
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(("近期复诊", None, "needs_date_confirmation"), connection.execute("SELECT original_text, normalized_date, date_status FROM clinical_items").fetchone())

    def test_calendar_mutations_and_evidence_are_profile_scoped(self):
        self.import_report("报告日期: 2026-10-01\n复诊计划：近期复诊")
        item = self.client.get(self.base).json()["unresolved_plans"][0]
        other = TestClient(self.app)
        try:
            account = other.post("/api/auth/register", json={"username": "calendar-other", "password": "synthetic-password-123"}).json()
            other.headers.update({"X-CSRF-Token": account["csrf_token"]})
            self.assertEqual(404, other.get(self.base).status_code)
            otherbase = f"/api/profiles/{account['profile_id']}/calendar"
            response = other.post(f"{otherbase}/plans/{item['clinical_item_id']}/schedule", json={"scheduled_date": "2026-10-20", "acknowledged": True})
            self.assertEqual(404, response.status_code)
            self.assertEqual(0, other.get(otherbase).json()["event_count"])
        finally:
            other.close()

    def test_manual_events_reject_conflicting_retries_and_support_edit_cancel(self):
        body = {"idempotency_key": "user-plan-1", "title": "用户安排复诊", "scheduled_date": "2026-10-10"}
        first = self.client.post(self.base + "/events", json=body)
        self.assertEqual(200, first.status_code, first.text)
        event_id = first.json()["event"]["event_id"]
        self.assertEqual("already_exists", self.client.post(self.base + "/events", json=body).json()["status"])
        self.assertEqual(409, self.client.post(self.base + "/events", json={**body, "scheduled_date": "2026-10-11"}).status_code)
        url = f"{self.base}/events/{event_id}"
        self.assertEqual("2026-10-12", self.client.patch(url, json={"scheduled_date": "2026-10-12"}).json()["event"]["scheduled_date"])
        self.assertEqual("cancelled", self.client.post(url + "/cancel").json()["event"]["status"])

    def test_clinical_evidence_is_bound_to_physical_field_and_no_diagnosis_is_inferred(self):
        blocks = [{"text": line, "page": 1, "polygon": [[0, y], [220, y], [220, y+15], [0, y+15]], "confidence": 0.98} for y, line in ((0, "报告日期: 2026-10-01"), (30, "诊断：甲状腺结节"), (60, "复诊计划：1个月后复查"))]
        text, blocks = layout_blocks(blocks)
        preview = build_natural_preview("region-source", text, {}, blocks=blocks, original_text=text)
        for item in preview["clinical_items"]:
            evidence = item["evidence"]
            self.assertEqual(item["original_text"], text[evidence["span_start"]:evidence["span_end"]])
            self.assertEqual(1, evidence["page"])
            self.assertTrue(evidence["polygon"])
            self.assertTrue(evidence["fields"]["original_text"]["regions"])
        unsupported = build_natural_preview("no-inference", "报告日期: 2026-10-01\n血糖偏高，请结合检查判断", {"clinical_items": [{"kind": "diagnosis", "original_text": "糖尿病"}]})
        self.assertEqual([], unsupported["clinical_items"])


if __name__ == "__main__":
    unittest.main()
