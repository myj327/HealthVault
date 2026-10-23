import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from healthvault.infrastructure.local_health_knowledge import (
    DEFAULT_HEALTH_KNOWLEDGE_PATH,
    LocalHealthKnowledgeRepository,
)
from healthvault.repositories.health_knowledge import HealthKnowledgeQueryError
from healthvault.rag.chunking import estimate_tokens


class LocalHealthKnowledgeRepositoryTests(unittest.TestCase):
    def test_default_wrapper_exposes_versioned_traceable_fts_hits(self):
        repository = LocalHealthKnowledgeRepository()

        self.assertEqual("fts5", repository.index_metadata["engine"])
        self.assertEqual("bm25", repository.index_metadata["ranking"])
        self.assertEqual("trigram", repository.index_metadata["tokenizer"])
        self.assertEqual("hybrid_v1", repository.index_metadata["strategy"])
        self.assertGreaterEqual(repository.index_metadata["chunk_count"], 80)

        hit = repository.search("HbA1c", 1)[0]
        self.assertTrue(hit.document.citation_id.startswith("HVK:r1-2026-08-24-v1:"))
        self.assertEqual(hit.document.citation_id, hit.document.chunk_id)
        self.assertEqual("r1-2026-08-24-v1", hit.document.corpus_version)
        self.assertIn("hba1c", hit.document.topic_tags)
        self.assertEqual(64, len(hit.document.body_sha256))
        self.assertEqual(64, len(hit.document.source.sha256))
        self.assertTrue(hit.document.source.url.startswith("https://"))
        self.assertEqual(1, hit.rank)

    def test_missing_broken_and_custom_artifacts_have_explicit_behavior(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.json"
            with self.assertRaises(HealthKnowledgeQueryError):
                LocalHealthKnowledgeRepository(missing_path)

            broken_path = Path(temp_dir) / "broken.json"
            broken_path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(HealthKnowledgeQueryError):
                LocalHealthKnowledgeRepository(broken_path)

            corpus_root = Path(temp_dir) / "custom-corpus"
            shutil.copytree(DEFAULT_HEALTH_KNOWLEDGE_PATH.parents[1], corpus_root)
            custom_path = corpus_root / "chunks" / "hybrid_v1.json"
            custom = LocalHealthKnowledgeRepository(custom_path)
            self.assertTrue(custom.search("空腹血糖", 1))

            artifact = json.loads(custom_path.read_text(encoding="utf-8"))
            forged_body = " ".join(["FORGED EVIDENCE."] * 30)
            artifact["chunks"][0]["body"] = forged_body
            artifact["chunks"][0]["estimated_tokens"] = estimate_tokens(forged_body)
            artifact_bytes = (
                json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            custom_path.write_bytes(artifact_bytes)
            manifest_path = corpus_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["strategies"]["hybrid_v1"]["artifact_sha256"] = (
                hashlib.sha256(artifact_bytes).hexdigest()
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(HealthKnowledgeQueryError):
                LocalHealthKnowledgeRepository(custom_path)

    def test_alias_normalization_ranking_limit_empty_and_injection_are_stable(self):
        repository = LocalHealthKnowledgeRepository()
        cases = (
            ("HbA1c", "hba1c"),
            ("  ＨｂＡ１ｃ\t", "hba1c"),
            ("空腹血糖", "fasting_glucose"),
            ("LDL cholesterol", "ldl_cholesterol"),
            ("就诊准备", "visit_preparation"),
        )
        for query, topic in cases:
            with self.subTest(query=query):
                hits = repository.search(query, 3)
                self.assertTrue(hits)
                self.assertIn(topic, hits[0].document.topic_tags)
                self.assertEqual(list(range(1, len(hits) + 1)), [hit.rank for hit in hits])
                self.assertTrue(all(hit.matched_terms for hit in hits))

        query = "LDL cholesterol and fasting glucose and A1C"
        first = repository.search(query, 2)
        self.assertEqual(first, repository.search(query, 2))
        self.assertEqual(2, len(first))
        self.assertEqual((), repository.search("unrelated synthetic query", 3))
        self.assertEqual((), repository.search('\" OR 1=1 --', 3))


if __name__ == "__main__":
    unittest.main()
