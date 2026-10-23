import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient as FastAPITestClient

from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.answers import answer_with_model
from healthvault.web.api import create_app
from healthvault.web.model_runtime import configured_web_model


FIXTURES = Path(__file__).parents[1] / "fixtures" / "health_reports"
REPORT = (FIXTURES / "synthetic_checkup_2024.txt").read_text(encoding="utf-8")


class StubController:
    def __init__(self, hits=None, records=None):
        self.hits = hits or []
        self.records = records or []
        self.directory = tempfile.TemporaryDirectory()
        self.record_repository = SQLiteHealthRecordRepository(
            Path(self.directory.name) / "records.sqlite3"
        )

    def search_knowledge(self, _query, limit=4):
        return {"status": "ok", "results": self.hits[:limit]}

    def visit_brief(self, profile_id):
        return {"profile_id": profile_id, "record_citations": self.records, "knowledge_citations": [], "indicators": []}


class TestClient(FastAPITestClient):
    def __init__(self, app):
        super().__init__(app)
        user = app.state.auth_store.register(
            "synthetic-demo-a", "synthetic-password-123", user_id="demo-a"
        )
        token, csrf = app.state.auth_store.create_session(user)
        self.cookies.set("hv_session", token)
        self.headers.update({"X-CSRF-Token": csrf})

    def post(self, url, **kwargs):
        if url == "/api/answers/knowledge" or url.endswith("/answers/visit"):
            kwargs["json"] = {**kwargs["json"], "model_consent": True}
        return super().post(url, **kwargs)


class WebAnswerTests(unittest.TestCase):
    def test_knowledge_question_returns_answer_and_retrieved_source(self):
        calls = []
        hit = {"citation_id": "HVK:test-1", "title": "合成知识", "excerpt": "空腹血糖用于描述采血时的血糖水平。", "source": {"url": "https://example.org/source"}}
        def fake_model(question, mode, evidence):
            calls.append((question, mode, evidence))
            return "空腹血糖是一次检测结果。[HVK:test-1]"
        with TestClient(create_app(StubController(hits=[hit]), fake_model)) as client:
            response = client.post("/api/answers/knowledge", json={"question": "空腹血糖是什么？"})
            history = client.get("/api/profiles/demo-a/answers/history")
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("answered", response.json()["status"])
        self.assertEqual("HVK:test-1", response.json()["sources"][0]["citation_id"])
        self.assertEqual("HVK:test-1", history.json()["answers"][0]["sources"][0]["citation_id"])
        self.assertEqual("no-store", response.headers["cache-control"])
        self.assertEqual("knowledge", calls[0][1])
        self.assertEqual("HVK:test-1", calls[0][2]["sources"][0]["citation_id"])

    def test_missing_knowledge_evidence_does_not_call_model(self):
        def forbidden(*_args):
            raise AssertionError("model must not run")
        with TestClient(create_app(StubController(), forbidden)) as client:
            response = client.post("/api/answers/knowledge", json={"question": "没有资料的问题是什么？"})
        self.assertEqual(200, response.status_code)
        self.assertEqual("insufficient_evidence", response.json()["status"])
        self.assertEqual([], response.json()["sources"])

    def test_compound_indicator_question_retrieves_both_topics(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "records.sqlite3")
            captured = []
            def fake_model(_question, _mode, evidence):
                captured.append(evidence)
                citations = "".join(f"[{source['citation_id']}]" for source in evidence["sources"])
                return f"这两项检查反映不同时间范围的血糖信息。{citations}"
            with TestClient(create_app(controller, fake_model)) as client:
                response = client.post("/api/answers/knowledge", json={"question": "糖化血红蛋白反映什么？它和空腹血糖有什么区别？"})
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("answered", response.json()["status"])
        self.assertEqual(1, len(captured))
        self.assertGreaterEqual(len(response.json()["sources"]), 2)

    def test_unsupported_treatment_question_never_uses_topic_fallback(self):
        hit = {"citation_id": "HVK:test-1", "title": "合成知识", "excerpt": "合成内容", "source": {"url": ""}}
        def forbidden(*_args):
            raise AssertionError("model must not run")
        with TestClient(create_app(StubController(hits=[hit]), forbidden)) as client:
            response = client.post("/api/answers/knowledge", json={"question": "糖化血红蛋白高了该吃什么药？"})
        self.assertEqual("insufficient_evidence", response.json()["status"])

    def test_empty_visit_and_invalid_question_do_not_call_model(self):
        def forbidden(*_args):
            raise AssertionError("model must not run")
        with TestClient(create_app(StubController(), forbidden)) as client:
            empty = client.post("/api/profiles/demo-a/answers/visit", json={"question": "我该问医生什么？"})
            invalid = client.post("/api/answers/knowledge", json={"question": "   "})
        self.assertEqual("insufficient_evidence", empty.json()["status"])
        self.assertEqual(422, invalid.status_code)

    def test_visit_treatment_request_is_blocked_before_model(self):
        record = {
            "citation_id": "REC:test-1", "report_date": "2025-06-01",
            "indicator_code": "fasting_glucose", "evidence": {"quote": "synthetic result"},
        }
        def forbidden(*_args):
            raise AssertionError("unsafe intent must not call model")
        with TestClient(create_app(StubController(records=[record]), forbidden)) as client:
            response = client.post("/api/profiles/demo-a/answers/visit", json={"question": "我该吃什么药？"})
        self.assertEqual(200, response.status_code)
        self.assertEqual("insufficient_evidence", response.json()["status"])

    def test_visit_uses_only_confirmed_records_for_current_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "records.sqlite3")
            calls = []
            def fake_model(_question, mode, evidence):
                calls.append((mode, evidence))
                return f"可以与医生讨论这项结果。[{evidence['sources'][0]['citation_id']}]"
            with TestClient(create_app(controller, fake_model)) as client:
                before = client.post("/api/profiles/demo-a/answers/visit", json={"question": "最近的结果如何？"})
                draft = client.post("/api/drafts/text", json={"profile_id": "demo-a", "source_id": "synthetic-2024", "report_text": REPORT}).json()
                still_draft = client.post("/api/profiles/demo-a/answers/visit", json={"question": "最近的结果如何？"})
                confirmed = client.post(f"/api/drafts/{draft['token']}/confirm", json={"profile_id": "demo-a", "report_text": REPORT, "preview_sha256": draft["preview_sha256"], "acknowledged": True})
                other = client.post("/api/profiles/demo-b/answers/visit", json={"question": "最近的结果如何？"})
                own = client.post("/api/profiles/demo-a/answers/visit", json={"question": "最近的结果如何？"})
            self.assertEqual("insufficient_evidence", before.json()["status"])
            self.assertEqual("insufficient_evidence", still_draft.json()["status"])
            self.assertEqual("confirmed", confirmed.json()["status"])
            self.assertEqual(404, other.status_code)
            self.assertEqual("answered", own.json()["status"])
            self.assertEqual(1, len(calls))
            self.assertEqual("visit", calls[0][0])
            self.assertTrue(all(source["kind"] in {"record", "knowledge"} for source in calls[0][1]["sources"]))
            self.assertNotIn(REPORT, str(calls[0][1]))

    def test_unrecognized_model_citation_is_rejected(self):
        hit = {"citation_id": "HVK:test-1", "title": "空腹血糖", "excerpt": "空腹血糖是一次检测。", "source": {"url": ""}}
        with TestClient(create_app(StubController(hits=[hit]), lambda *_: "结论。[HVK:invented]")) as client:
            response = client.post("/api/answers/knowledge", json={"question": "空腹血糖是什么？"})
        self.assertEqual(502, response.status_code)

    def test_model_transport_reuses_configured_key_without_network(self):
        with patch.dict("os.environ", {
            "HEALTHVAULT_OPENAI_API_KEY": "synthetic-secret",
            "HEALTHVAULT_OPENAI_BASE_URL": "https://example.org/v1",
            "HEALTHVAULT_OPENAI_MODEL": "synthetic-model",
        }, clear=True), patch("healthvault.web.answers.run_model_task", return_value=json.dumps({"status": "answered", "claims": [{"text": "回答。", "citation_id": "HVK:test-1"}]})) as run:
            config = configured_web_model()
            answer = answer_with_model("问题？", "knowledge", {"sources": [{"citation_id": "HVK:test-1"}]}, profile_id="demo-a")
        self.assertEqual("回答。 [HVK:test-1]", answer)
        self.assertEqual("synthetic-secret", config.api_key)
        self.assertEqual("https://example.org/v1", config.api_base)
        self.assertEqual("synthetic-model", config.model)
        self.assertEqual("demo-a", run.call_args.kwargs["profile_id"])


if __name__ == "__main__":
    unittest.main()
