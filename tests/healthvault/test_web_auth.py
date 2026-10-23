import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app


REPORT = "\n".join((
    "report_id: AUTH-SYN-2025",
    "report_date: 2025-06-01",
    "facility: Synthetic Lab",
    "result|fasting_glucose|5.6|mmol/L|3.9|6.1",
))


class WebAuthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        controller = HealthVaultUIController(Path(self.directory.name) / "records.sqlite3")
        self.app = create_app(controller)
        self.alice = TestClient(self.app)
        self.bob = TestClient(self.app)

    def tearDown(self):
        self.alice.close()
        self.bob.close()
        self.directory.cleanup()

    def register(self, client, username):
        response = client.post("/api/auth/register", json={
            "username": username, "password": "synthetic-password-123",
        })
        self.assertEqual(200, response.status_code, response.text)
        client.headers.update({"X-CSRF-Token": response.json()["csrf_token"]})
        return response

    def test_login_cookie_logout_and_unauthenticated_gate(self):
        self.assertEqual(401, self.alice.get("/api/profiles/demo-a/timeline").status_code)
        registered = self.register(self.alice, "alice-test")
        self.assertIn("httponly", registered.headers["set-cookie"].lower())
        self.assertIn("samesite=strict", registered.headers["set-cookie"].lower())
        self.assertEqual("alice-test", self.alice.get("/api/auth/me").json()["username"])
        self.assertEqual(200, self.alice.post("/api/auth/logout").status_code)
        self.assertEqual(401, self.alice.get("/api/auth/me").status_code)
        self.assertEqual(401, self.alice.post("/api/auth/login", json={
            "username": "alice-test", "password": "wrong-password",
        }).status_code)
        self.assertEqual(200, self.alice.post("/api/auth/login", json={
            "username": "alice-test", "password": "synthetic-password-123",
        }).status_code)

    def test_accounts_cannot_read_each_others_reports_drafts_or_agent_tasks(self):
        a = self.register(self.alice, "alice-test").json()["profile_id"]
        b = self.register(self.bob, "bob-test").json()["profile_id"]
        self.assertNotEqual(a, b)
        draft = self.alice.post("/api/drafts/text", json={
            "profile_id": a, "source_id": "auth-source", "report_text": REPORT,
        }).json()
        confirmed = self.alice.post(f"/api/drafts/{draft['token']}/confirm", json={
            "profile_id": a, "report_text": REPORT,
            "preview_sha256": draft["preview_sha256"], "acknowledged": True,
        })
        self.assertEqual("confirmed", confirmed.json()["status"])
        self.assertEqual(1, self.alice.get(f"/api/profiles/{a}/timeline").json()["report_count"])
        self.assertEqual(0, self.bob.get(f"/api/profiles/{b}/timeline").json()["report_count"])
        self.assertEqual(404, self.bob.get(f"/api/profiles/{a}/timeline").status_code)
        self.assertEqual(404, self.bob.get(f"/api/profiles/{a}/reports/AUTH-SYN-2025").status_code)
        self.assertEqual(404, self.bob.get(f"/api/profiles/{a}/agent/tasks").status_code)
        self.assertEqual(404, self.bob.post("/api/drafts/text", json={
            "profile_id": a, "source_id": "forged", "report_text": REPORT,
        }).status_code)
        self.assertEqual(404, self.bob.post(f"/api/drafts/{draft['token']}/confirm", json={
            "profile_id": b, "report_text": REPORT,
            "preview_sha256": draft["preview_sha256"], "acknowledged": True,
        }).status_code)

    def test_csrf_and_cross_site_write_are_rejected(self):
        a = self.register(self.alice, "alice-test").json()["profile_id"]
        body = {"profile_id": a, "source_id": "auth-source", "report_text": REPORT}
        csrf = self.alice.headers.pop("X-CSRF-Token")
        self.assertEqual(403, self.alice.post("/api/drafts/text", json=body).status_code)
        self.alice.headers.update({"X-CSRF-Token": csrf})
        self.assertEqual(403, self.alice.post("/api/drafts/text", json=body,
                                              headers={"Origin": "https://evil.example"}).status_code)
        self.assertEqual(200, self.alice.post("/api/drafts/text", json=body).status_code)

    def test_model_use_requires_explicit_consent(self):
        a = self.register(self.alice, "alice-test").json()["profile_id"]
        natural = self.alice.post("/api/drafts/text", json={
            "profile_id": a, "source_id": "natural", "report_text": "synthetic text",
            "auto_extract": True,
        })
        self.assertEqual(409, natural.status_code)
        self.assertEqual(409, self.alice.post("/api/answers/knowledge", json={
            "question": "What does this mean?",
        }).status_code)
        self.assertEqual(409, self.alice.post(f"/api/profiles/{a}/answers/visit", json={
            "question": "What should I ask?",
        }).status_code)
        self.assertEqual(409, self.alice.post(f"/api/profiles/{a}/agent/tasks/task-consent/messages", json={
            "prompt": "Please explain my report", "expected_version": 0,
        }).status_code)

    def test_public_admin_name_has_no_role_and_local_grant_revoke_is_effective(self):
        registered = self.register(self.alice, "admin").json()
        self.assertFalse(registered["is_admin"])
        self.assertEqual(403, self.alice.get("/api/admin/skills/candidates").status_code)
        self.app.state.auth_store.set_admin("admin", enabled=True)
        self.assertTrue(self.alice.get("/api/auth/me").json()["is_admin"])
        self.assertEqual(200, self.alice.get("/api/admin/skills/candidates").status_code)
        self.app.state.auth_store.set_admin("admin", enabled=False)
        self.assertEqual(403, self.alice.post("/api/admin/skills/rollback").status_code)


if __name__ == "__main__":
    unittest.main()
