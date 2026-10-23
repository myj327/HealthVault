import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf as fitz
from PIL import Image
from fastapi.testclient import TestClient

from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.agent_tasks import AgentTaskConflict, WebAgentTaskStore
from healthvault.web.api import create_app
from healthvault.web.extraction import build_natural_preview, extract_with_model


REPORT = "\n".join((
    "检查报告编号: IMG-2025",
    "报告日期: 2025-06-01",
    "机构: 测试医院",
    "空腹血糖 5.6 mmol/L 参考 3.9-6.1",
    "甲状腺右叶结节 0.8×0.6 cm",
    "子宫肌瘤 2.3×1.9 cm",
))


def model_payload():
    return {
        "report_id": "IMG-2025",
        "report_date": "2025-06-01",
        "date_quote": "报告日期: 2025-06-01",
        "facility": "测试医院",
        "indicators": [{
            "code": "fasting_glucose", "value": 5.6, "unit": "mmol/L",
            "reference_lower": 3.9, "reference_upper": 6.1,
            "quote": "空腹血糖 5.6 mmol/L 参考 3.9-6.1",
        }],
        "imaging_findings": [
            {
                "finding_type": "nodule", "body_site": "甲状腺", "laterality": "right",
                "location": "右叶", "dimensions": [0.8, 0.6], "unit": "cm",
                "quote": "甲状腺右叶结节 0.8×0.6 cm",
            },
            {
                "finding_type": "fibroid", "body_site": "子宫",
                "laterality": "unspecified", "location": "",
                "dimensions": [2.3, 1.9], "unit": "cm",
                "quote": "子宫肌瘤 2.3×1.9 cm",
            },
        ],
    }


class NaturalOCRController(HealthVaultUIController):
    def process_image(self, image_bytes: bytes, filename: str) -> dict:
        return {
            "text": REPORT,
            "text_sha256": hashlib.sha256(REPORT.encode("utf-8")).hexdigest(),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "blocks": [],
            "confidence": {"average": 0.99, "minimum": 0.99, "low_confidence_blocks": []},
        }


class WebExtractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        controller = NaturalOCRController(Path(self.directory.name) / "records.sqlite3")
        self.client = TestClient(create_app(controller, model_extractor=lambda _: model_payload()))
        user = self.client.app.state.auth_store.register(
            "synthetic-demo-a", "synthetic-password-123", user_id="demo-a"
        )
        token, csrf = self.client.app.state.auth_store.create_session(user)
        self.client.cookies.set("hv_session", token)
        self.client.headers.update({"X-CSRF-Token": csrf})

    def tearDown(self):
        self.client.close()
        self.directory.cleanup()

    def confirm(self, draft):
        return self.client.post(f"/api/drafts/{draft['token']}/confirm", json={
            "profile_id": "demo-a", "report_text": draft["report_text"],
            "preview_sha256": draft["preview_sha256"], "acknowledged": True,
        })

    def test_natural_text_extracts_and_persists_labs_and_imaging(self):
        response = self.client.post("/api/drafts/text", json={
            "profile_id": "demo-a", "source_id": "natural-2025",
            "report_text": REPORT, "auto_extract": True, "model_consent": True,
        })
        self.assertEqual(200, response.status_code, response.text)
        draft = response.json()
        self.assertTrue(draft["preview"]["can_confirm"])
        self.assertEqual(1, len(draft["preview"]["indicators"]))
        self.assertEqual(
            [[8.0, 6.0], [23.0, 19.0]],
            [finding["dimensions_mm"] for finding in draft["preview"]["imaging_findings"]],
        )
        self.assertEqual("confirmed", self.confirm(draft).json()["status"])
        timeline = self.client.get("/api/profiles/demo-a/timeline").json()
        self.assertEqual(1, timeline["report_count"])
        self.assertEqual(2, len(timeline["timeline"][0]["imaging_findings"]))
        self.assertEqual("0.8×0.6", timeline["timeline"][0]["imaging_findings"][0]["original_dimensions"])

    def test_default_extraction_uses_harness_model_with_profile_context(self):
        with patch("healthvault.web.extraction.run_model_task", return_value=json.dumps(model_payload(), ensure_ascii=False)) as run:
            payload = extract_with_model(REPORT, profile_id="demo-a")
        self.assertEqual("IMG-2025", payload["report_id"])
        self.assertEqual("demo-a", run.call_args.kwargs["profile_id"])

    def test_unmatched_model_numbers_or_site_are_rejected(self):
        bad = model_payload()
        bad["imaging_findings"][0]["dimensions"] = [0.9, 0.6]
        bad["imaging_findings"][1]["body_site"] = "甲状腺"
        preview = build_natural_preview("bad", REPORT, bad)
        self.assertTrue(preview["can_confirm"])
        self.assertEqual(1, len(preview["indicators"]))
        self.assertEqual(2, len(preview["imaging_findings"]))
        self.assertTrue(any(item["code"] == "unverified_finding" for item in preview["warnings"]))
        self.assertEqual([8.0, 6.0], preview["imaging_findings"][0]["dimensions_mm"])

    def test_imaging_keeps_original_precision_and_chinese_unit(self):
        report = REPORT.replace("0.8×0.6 cm", "0.80×0.60 厘米")
        payload = model_payload()
        payload["imaging_findings"][0]["dimensions"] = [0.8, 0.6]
        payload["imaging_findings"][0]["unit"] = "厘米"
        payload["imaging_findings"][0]["quote"] = "甲状腺右叶结节 0.80×0.60 厘米"
        preview = build_natural_preview("precision", report, payload)
        finding = preview["imaging_findings"][0]
        self.assertEqual("0.80×0.60", finding["original_dimensions"])
        self.assertEqual("厘米", finding["original_unit"])
        self.assertEqual([8.0, 6.0], finding["dimensions_mm"])

    def test_uploaded_image_requires_review_before_model_extraction(self):
        image = b"\x89PNG\r\n\x1a\nsynthetic-image"
        upload = self.client.post("/api/drafts/image", data={"profile_id": "demo-a"}, files={
            "file": ("test.png", image, "image/png"),
        })
        self.assertEqual(200, upload.status_code, upload.text)
        draft = upload.json()
        self.assertIsNone(draft["preview"])
        reviewed = self.client.put(f"/api/drafts/{draft['token']}", json={
            "profile_id": "demo-a", "report_text": draft["report_text"], "reviewed": True,
            "model_consent": True,
        })
        self.assertEqual(200, reviewed.status_code, reviewed.text)
        self.assertEqual(2, len(reviewed.json()["preview"]["imaging_findings"]))
        confirmed = self.confirm(reviewed.json())
        self.assertEqual(200, confirmed.status_code, confirmed.text)
        self.assertEqual("confirmed", confirmed.json()["status"])
        detail = self.client.get("/api/profiles/demo-a/reports/IMG-2025").json()
        self.assertEqual(REPORT, detail["report_text"])
        self.assertEqual("test.png", detail["original_asset"]["filename"])
        self.assertEqual(image, self.client.get("/api/profiles/demo-a/reports/IMG-2025/original").content)

    def test_pdf_upload_extracts_selectable_text_for_review(self):
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Report date: 2025-06-01\nThyroid right nodule 8 x 6 mm\nReport ID: PDF-1")
        raw = pdf.tobytes()
        pdf.close()
        response = self.client.post("/api/drafts/pdf", data={"profile_id": "demo-a"}, files={
            "file": ("report.pdf", raw, "application/pdf"),
        })
        self.assertEqual(200, response.status_code, response.text)
        draft = response.json()
        self.assertEqual("pdf", draft["input_kind"])
        self.assertIn("Thyroid right nodule", draft["report_text"])
        self.assertIsNone(draft["preview"])
        self.assertEqual(raw, self.client.get(draft["image_url"]).content)

    def test_confirmed_lab_pdf_original_is_retrievable(self):
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72),
                         "report_id: LAB-PDF-2025\nreport_date: 2025-06-01\n"
                         "facility: Synthetic Lab\nresult|fasting_glucose|5.6|mmol/L|3.9|6.1")
        raw = pdf.tobytes()
        pdf.close()
        upload = self.client.post("/api/drafts/pdf", data={"profile_id": "demo-a"}, files={
            "file": ("blood.pdf", raw, "application/pdf"),
        })
        self.assertEqual(200, upload.status_code, upload.text)
        draft = upload.json()
        reviewed = self.client.put(f"/api/drafts/{draft['token']}", json={
            "profile_id": "demo-a", "report_text": draft["report_text"], "reviewed": True,
        })
        self.assertEqual(200, reviewed.status_code, reviewed.text)
        confirmed = self.confirm(reviewed.json())
        self.assertEqual(200, confirmed.status_code, confirmed.text)
        self.assertEqual("confirmed", confirmed.json()["status"])
        detail = self.client.get("/api/profiles/demo-a/reports/LAB-PDF-2025").json()
        self.assertEqual("blood.pdf", detail["original_asset"]["filename"])
        original = self.client.get("/api/profiles/demo-a/reports/LAB-PDF-2025/original")
        self.assertEqual("application/pdf", original.headers["content-type"])
        self.assertEqual(raw, original.content)

    def test_scanned_pdf_uses_page_ocr_before_review(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (100, 100), "white").save(image_bytes, format="PNG")
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_image(page.rect, stream=image_bytes.getvalue())
        raw = pdf.tobytes()
        pdf.close()
        response = self.client.post("/api/drafts/pdf", data={"profile_id": "demo-a"}, files={
            "file": ("scan.pdf", raw, "application/pdf"),
        })
        self.assertEqual(200, response.status_code, response.text)
        draft = response.json()
        self.assertIn("甲状腺右叶结节", draft["report_text"])
        self.assertIsNone(draft["preview"])

    def test_task_store_restores_only_own_profile_and_checks_version(self):
        path = Path(self.directory.name) / "tasks.sqlite3"
        first = WebAgentTaskStore(path).save(
            "demo-a", "task-2025", expected_version=0,
            messages=[{"role": "system", "content": "read only"}],
            folded=[], dialogue=[{"role": "user", "text": "compare reports"}],
        )
        self.assertEqual(1, first["version"])
        self.assertEqual(first, WebAgentTaskStore(path).get("demo-a", "task-2025"))
        self.assertIsNone(WebAgentTaskStore(path).get("demo-b", "task-2025"))
        with self.assertRaises(AgentTaskConflict):
            WebAgentTaskStore(path).save(
                "demo-a", "task-2025", expected_version=0,
                messages=[], folded=[], dialogue=[],
            )

    def test_agent_task_routes_expose_profile_scoped_saved_state(self):
        store = self.client.app.state.agent_tasks
        store.save(
            "demo-a", "task-2025", expected_version=0,
            messages=[{"role": "system", "content": "read only"}],
            folded=[], dialogue=[{"role": "user", "text": "check reports"}],
        )
        own = self.client.get("/api/profiles/demo-a/agent/tasks/task-2025")
        other = self.client.get("/api/profiles/demo-b/agent/tasks/task-2025")
        self.assertEqual(200, own.status_code)
        self.assertEqual(1, own.json()["version"])
        self.assertEqual(404, other.status_code)
        self.assertEqual(1, len(self.client.get("/api/profiles/demo-a/agent/tasks").json()["tasks"]))
        self.assertEqual("check reports", self.client.get("/api/profiles/demo-a/agent/tasks").json()["tasks"][0]["title"])
        self.assertEqual(422, self.client.get("/api/profiles/demo-a/agent/tasks/invalid!").status_code)


if __name__ == "__main__":
    unittest.main()
