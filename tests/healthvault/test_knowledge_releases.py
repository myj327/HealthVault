import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT
from healthvault.rag.releases import KnowledgeReleaseStore
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app


def update_payload(version="test-2026-10-01-v1"):
    source = json.loads((DEFAULT_CORPUS_ROOT / "sources" / "niddk-a1c-test.json").read_text(encoding="utf-8"))
    source["sections"][0]["blocks"][0]["text"] += " A synthetic release marker amberquartz identifies this offline fixture."
    source["source_sha256"] = hashlib.sha256(json.dumps(source["sections"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    source["retrieved_on"] = "2026-10-01"
    return {"corpus_version": version, "source_updates": [source], "source_metadata": {source["source_id"]: {"document_kind": "general_health", "updated_on": "2026-09-30", "published_on": None}}, "probes": [{"query": "amberquartz", "expected_source_id": source["source_id"]}]}


class KnowledgeReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = KnowledgeReleaseStore(self.root / "releases")

    def tearDown(self):
        self.temp.cleanup()

    def test_one_source_stage_publish_restart_and_rollback(self):
        payload = update_payload()
        record = self.store.stage(payload, "owner")
        self.assertEqual("pending_review", record["status"])
        self.assertEqual(["niddk-a1c-test"], record["changed_source_ids"])
        self.assertEqual(DEFAULT_CORPUS_ROOT, self.store.active_root)
        candidate = self.root / "releases" / "candidates" / record["candidate_id"] / "corpus"
        before = json.loads((DEFAULT_CORPUS_ROOT / "sources" / "medlineplus-hba1c-test.json").read_text(encoding="utf-8"))
        after = json.loads((candidate / "sources" / "medlineplus-hba1c-test.json").read_text(encoding="utf-8"))
        self.assertEqual(before["sections"], after["sections"])
        self.assertEqual(before["source_sha256"], after["source_sha256"])
        self.assertEqual("published", self.store.decide(record["candidate_id"], "approve", "owner")["status"])
        controller = HealthVaultUIController(self.root / "records.sqlite3", knowledge_release_root=self.store.root)
        self.assertEqual(payload["corpus_version"], controller.knowledge_repository.index_metadata["corpus_version"])
        hits = controller.knowledge_repository.search("amberquartz", 5)
        self.assertTrue(hits)
        self.assertTrue(all(hit.document.citation_id.startswith("HVK:" + payload["corpus_version"] + ":") for hit in hits))
        self.assertEqual("rolled_back", self.store.rollback("owner")["status"])
        controller.reload_knowledge()
        self.assertEqual(DEFAULT_CORPUS_ROOT.name, controller.knowledge_repository.index_metadata["corpus_version"])

    def test_tampered_and_stale_candidates_cannot_publish(self):
        first = self.store.stage(update_payload(), "owner")
        second = self.store.stage(update_payload("test-2026-10-01-v2"), "owner")
        self.store.decide(first["candidate_id"], "approve", "owner")
        self.assertEqual("stale", self.store.decide(second["candidate_id"], "approve", "owner")["status"])
        third = self.store.stage(update_payload("test-2026-10-01-v3"), "owner")
        path = self.store.root / "candidates" / third["candidate_id"] / "corpus" / "chunks" / "hybrid_v1.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.store.decide(third["candidate_id"], "approve", "owner")

    def test_unapproved_source_and_undated_guideline_are_rejected(self):
        payload = update_payload()
        payload["source_updates"][0]["url"] = "https://unapproved.invalid/knowledge"
        with self.assertRaises(ValueError):
            self.store.stage(payload, "owner")
        payload = update_payload()
        payload["source_metadata"]["niddk-a1c-test"]["document_kind"] = "formal_guideline"
        with self.assertRaises(ValueError):
            self.store.stage(payload, "owner")
        payload["source_metadata"]["niddk-a1c-test"]["published_on"] = "2026-09-01"
        with self.assertRaisesRegex(ValueError, "source version"):
            self.store.stage(payload, "owner")

    def test_admin_api_activation_rebinds_web_repository_and_normal_users_cannot_manage(self):
        controller = HealthVaultUIController(self.root / "records.sqlite3", knowledge_release_root=self.store.root)
        client = TestClient(create_app(controller))
        try:
            account = client.post("/api/auth/register", json={"username": "release-owner", "password": "synthetic-password-123"}).json()
            client.headers.update({"X-CSRF-Token": account["csrf_token"]})
            self.assertEqual(403, client.post("/api/admin/knowledge/candidates", json=update_payload()).status_code)
            client.app.state.auth_store.set_admin("release-owner", enabled=True)
            staged = client.post("/api/admin/knowledge/candidates", json=update_payload())
            self.assertEqual(200, staged.status_code, staged.text)
            candidate_id = staged.json()["candidate_id"]
            approved = client.post(f"/api/admin/knowledge/candidates/{candidate_id}/decision", json={"decision": "approve"})
            self.assertEqual(200, approved.status_code, approved.text)
            result = client.get("/api/knowledge", params={"query": "amberquartz"})
            self.assertIn("test-2026-10-01-v1", result.text)
            self.assertEqual("lexical", controller.knowledge_mode)
            self.assertEqual(200, client.post("/api/admin/knowledge/rollback").status_code)
            self.assertEqual(DEFAULT_CORPUS_ROOT.name, controller.knowledge_repository.index_metadata["corpus_version"])
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
