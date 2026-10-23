import hashlib
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from healthvault.domain.reports import INDICATOR_CATALOG
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app


FIXTURES = Path(__file__).parents[1] / "fixtures" / "health_reports"
REPORT = (FIXTURES / "synthetic_checkup_2024.txt").read_text(encoding="utf-8")


class FakeOCRController(HealthVaultUIController):
    def process_image(self, image_bytes: bytes, filename: str) -> dict:
        text = REPORT
        return {
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "blocks": [
                {
                    "index": index,
                    "page": 1,
                    "text": line,
                    "polygon": [[0, index * 12], [100, index * 12],
                                [100, index * 12 + 10], [0, index * 12 + 10]],
                    "confidence": 0.96,
                }
                for index, line in enumerate(text.splitlines(), 1)
            ],
            "confidence": {
                "average": 0.96,
                "minimum": 0.96,
                "low_confidence_blocks": [],
            },
        }


class WebAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        controller = FakeOCRController(Path(self.directory.name) / "records.sqlite3")
        self.client = TestClient(create_app(controller))
        user = self.client.app.state.auth_store.register(
            "synthetic-demo-a", "synthetic-password-123", user_id="demo-a"
        )
        token, csrf = self.client.app.state.auth_store.create_session(user)
        self.client.cookies.set("hv_session", token)
        self.client.headers.update({"X-CSRF-Token": csrf})

    def tearDown(self):
        self.client.close()
        self.directory.cleanup()

    def text_draft(self, source_id="synthetic-2024"):
        response = self.client.post(
            "/api/drafts/text",
            json={
                "profile_id": "demo-a",
                "source_id": source_id,
                "report_text": REPORT,
            },
        )
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def confirm(self, draft, **changes):
        body = {
            "profile_id": "demo-a",
            "report_text": draft["report_text"],
            "preview_sha256": draft["preview_sha256"],
            "acknowledged": True,
        }
        body.update(changes)
        return self.client.post(f"/api/drafts/{draft['token']}/confirm", json=body)

    def test_indicator_catalog_exposes_the_parser_catalog(self):
        response = self.client.get("/api/indicators")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(len(INDICATOR_CATALOG), payload["count"])
        self.assertGreaterEqual(payload["count"], 60)
        self.assertEqual(set(INDICATOR_CATALOG), {item["code"] for item in payload["indicators"]})
        self.assertEqual("mmHg", next(item["unit"] for item in payload["indicators"] if item["code"] == "systolic_bp"))

    def test_new_checkup_indicators_confirm_and_enter_comparisons(self):
        for year, systolic, uric_acid in ((2025, 122, 290), (2026, 128, 310)):
            report_text = "\n".join((
                f"report_id: HV-SYN-CHECKUP-{year}",
                f"report_date: {year}-06-01",
                "facility: Synthetic Lab",
                f"result|SBP|{systolic}|mmHg|0|300",
                f"result|uric_acid|{uric_acid}|umol/L|0|500",
            ))
            draft_response = self.client.post("/api/drafts/text", json={
                "profile_id": "demo-a", "source_id": f"synthetic-checkup-{year}", "report_text": report_text,
            })
            self.assertEqual(200, draft_response.status_code, draft_response.text)
            draft = draft_response.json()
            self.assertTrue(draft["preview"]["can_confirm"])
            self.assertEqual("confirmed", self.confirm(draft).json()["status"])

        timeline = self.client.get("/api/profiles/demo-a/timeline").json()
        comparisons = self.client.get("/api/profiles/demo-a/comparisons").json()
        self.assertEqual(2, timeline["report_count"])
        self.assertEqual(4, timeline["indicator_count"])
        self.assertEqual(
            {"systolic_bp", "uric_acid"},
            {item["code"] for item in comparisons["comparisons"]},
        )
        self.assertEqual("mmHg", next(item["unit"] for item in comparisons["comparisons"] if item["code"] == "systolic_bp"))

    def test_text_preview_requires_explicit_confirmation_and_persists_timeline(self):
        draft = self.text_draft()
        self.assertTrue(draft["preview"]["can_confirm"])
        before = self.client.get("/api/profiles/demo-a/timeline").json()
        self.assertEqual(0, before["report_count"])

        first = self.confirm(draft)
        second = self.confirm(draft)
        self.assertEqual("confirmed", first.json()["status"])
        self.assertEqual("already_confirmed", second.json()["status"])
        self.assertEqual(1, self.client.get("/api/profiles/demo-a/timeline").json()["report_count"])
        self.assertEqual(404, self.client.get("/api/profiles/demo-b/timeline").status_code)
        self.assertEqual(200, self.client.get("/api/profiles/demo-a/comparisons").status_code)
        detail = self.client.get("/api/profiles/demo-a/reports/HV-SYN-2024-001").json()
        self.assertEqual(REPORT, detail["report_text"])
        self.assertIsNone(detail["original_asset"])
        self.assertEqual(404, self.client.get("/api/profiles/demo-a/reports/HV-SYN-2024-001/original").status_code)

    def test_forgery_and_stale_preview_do_not_write(self):
        draft = self.text_draft()
        self.assertEqual(404, self.confirm(draft, profile_id="demo-b").status_code)
        self.assertEqual(409, self.confirm(draft, acknowledged=False).status_code)
        self.assertEqual(409, self.confirm(draft, preview_sha256="0" * 64).status_code)
        self.assertEqual(409, self.confirm(draft, report_text=REPORT + "\n# changed").status_code)
        self.assertEqual(
            422,
            self.confirm(draft, user_confirmed=True).status_code,
        )

        updated = self.client.put(
            f"/api/drafts/{draft['token']}",
            json={
                "profile_id": "demo-a",
                "report_text": REPORT + "\n# reviewed again",
                "reviewed": False,
            },
        ).json()
        self.assertNotEqual(draft["token"], updated["token"])
        self.assertNotEqual(draft["preview_sha256"], updated["preview_sha256"])
        self.assertEqual(404, self.confirm(draft).status_code)
        self.assertEqual(0, self.client.get("/api/profiles/demo-a/timeline").json()["report_count"])

    def test_image_requires_ocr_review_and_keeps_original_evidence(self):
        image = b"\x89PNG\r\n\x1a\nsynthetic-image"
        uploaded = self.client.post(
            "/api/drafts/image",
            data={"profile_id": "demo-a"},
            files={"file": ("synthetic.png", image, "image/png")},
        )
        self.assertEqual(200, uploaded.status_code, uploaded.text)
        draft = uploaded.json()
        self.assertIsNone(draft["preview"])
        self.assertFalse(draft["ocr_reviewed"])
        self.assertEqual(image, self.client.get(draft["image_url"]).content)
        self.assertEqual(404, self.client.get(
            draft["image_url"].replace("demo-a", "demo-b")
        ).status_code)
        self.assertEqual(422, self.confirm(draft).status_code)

        reviewed = self.client.put(
            f"/api/drafts/{draft['token']}",
            json={"profile_id": "demo-a", "report_text": draft["report_text"], "reviewed": True},
        ).json()
        self.assertTrue(reviewed["ocr_reviewed"])
        self.assertTrue(reviewed["preview"]["can_confirm"])
        self.assertEqual(404, self.confirm(draft, preview_sha256=reviewed["preview_sha256"]).status_code)
        self.assertEqual("confirmed", self.confirm(reviewed).json()["status"])
        self.assertEqual(1, self.client.get("/api/profiles/demo-a/timeline").json()["report_count"])
        report_id = self.client.get("/api/profiles/demo-a/timeline").json()["timeline"][0]["report_id"]
        url = f"/api/profiles/demo-a/reports/{report_id}"
        detail = self.client.get(url).json()
        self.assertEqual(REPORT, detail["report_text"])
        self.assertEqual("synthetic.png", detail["original_asset"]["filename"])
        self.assertEqual(hashlib.sha256(image).hexdigest(), detail["original_asset"]["sha256"])
        original = self.client.get(url + "/original")
        self.assertEqual(image, original.content)
        self.assertEqual("image/png", original.headers["content-type"])
        self.assertEqual("no-store", original.headers["cache-control"])
        self.assertEqual(404, self.client.get(url.replace("demo-a", "demo-b")).status_code)
        self.assertEqual(404, self.client.get(url.replace("demo-a", "demo-b") + "/original").status_code)
        with TestClient(create_app(FakeOCRController(Path(self.directory.name) / "records.sqlite3"))) as reopened:
            reopened.cookies.update(self.client.cookies)
            self.assertEqual(image, reopened.get(url + "/original").content)

    def test_invalid_preview_and_cross_profile_update_are_blocked(self):
        draft = self.client.post(
            "/api/drafts/text",
            json={"profile_id": "demo-a", "source_id": "bad", "report_text": "not a report"},
        ).json()
        self.assertFalse(draft["preview"]["can_confirm"])
        self.assertEqual(409, self.confirm(draft).status_code)
        self.assertEqual(404, self.client.put(
            f"/api/drafts/{draft['token']}",
            json={"profile_id": "demo-b", "report_text": REPORT, "reviewed": True},
        ).status_code)
        self.assertEqual(0, self.client.get("/api/profiles/demo-a/timeline").json()["report_count"])


    def test_failed_repreview_keeps_old_draft(self):
        draft = self.text_draft()
        controller = self.client.app.state.controller
        original = controller.preview_report

        def fail_preview(*_args):
            raise RuntimeError("synthetic preview failure")

        controller.preview_report = fail_preview
        try:
            with self.assertRaises(RuntimeError):
                self.client.put(
                    f"/api/drafts/{draft['token']}",
                    json={
                        "profile_id": "demo-a",
                        "report_text": REPORT + "\n# edit",
                        "reviewed": False,
                    },
                )
        finally:
            controller.preview_report = original
        self.assertEqual("confirmed", self.confirm(draft).json()["status"])

if __name__ == "__main__":
    unittest.main()


