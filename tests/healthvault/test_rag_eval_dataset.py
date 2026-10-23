import json
import tempfile
import unittest
from pathlib import Path

from healthvault.evaluation.rag.dataset import (
    DEFAULT_DATASET_PATH,
    LEGACY_DATASET_PATH,
    TASK013_QUERY_REGISTRY_PATH,
    load_query_relevance_dataset,
)
from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    SQLiteFTSHealthKnowledgeRepository,
)


class RagEvalDatasetTests(unittest.TestCase):
    def test_fixed_dataset_schema_distribution_hashes_and_evidence_are_valid(self):
        dataset = load_query_relevance_dataset(DEFAULT_DATASET_PATH)

        self.assertEqual("rag-eval-r1-v2", dataset.eval_version)
        self.assertEqual("active", dataset.validity_status)
        self.assertEqual(60, len(dataset.queries))
        self.assertEqual(20, sum(query.split == "dev" for query in dataset.queries))
        self.assertEqual(40, sum(query.split == "test" for query in dataset.queries))
        self.assertEqual(50, sum(query.answerable for query in dataset.queries))
        self.assertEqual(
            40,
            sum(
                query.answerable and query.rewrite_category == "direct"
                for query in dataset.queries
            ),
        )
        self.assertEqual(
            10,
            sum(
                query.answerable and query.rewrite_category != "direct"
                for query in dataset.queries
            ),
        )
        self.assertEqual(10, sum(not query.answerable for query in dataset.queries))
        self.assertEqual(64, len(dataset.dataset_sha256))
        self.assertEqual(64, len(dataset.frozen_test_sha256))

    def test_v1_is_retained_but_invalidated_and_v2_test_has_no_task013_query_leakage(self):
        legacy = load_query_relevance_dataset(LEGACY_DATASET_PATH)
        current = load_query_relevance_dataset(DEFAULT_DATASET_PATH)
        registry = json.loads(
            Path(TASK013_QUERY_REGISTRY_PATH).read_text(encoding="utf-8")
        )
        excluded = {" ".join(query.split()).casefold() for query in registry["queries"]}

        self.assertEqual("invalidated", legacy.validity_status)
        self.assertIn("held-out leakage", legacy.invalidated_reason)
        self.assertFalse(
            excluded.intersection(
                " ".join(query.query.split()).casefold()
                for query in current.queries
                if query.split == "test"
            )
        )
        self.assertEqual(17, sum(q.split == "dev" and q.answerable for q in current.queries))
        self.assertEqual(3, sum(q.split == "dev" and not q.answerable for q in current.queries))
        self.assertEqual(33, sum(q.split == "test" and q.answerable for q in current.queries))
        self.assertEqual(7, sum(q.split == "test" and not q.answerable for q in current.queries))
        self.assertEqual(2, sum(q.split == "dev" and q.language == "mixed" for q in current.queries))
        self.assertEqual(5, sum(q.split == "test" and q.language == "mixed" for q in current.queries))

    def test_v2_pooled_qrels_cover_ranked_candidates_and_known_reviewer_corrections(self):
        dataset = load_query_relevance_dataset(DEFAULT_DATASET_PATH)
        by_id = {query.query_id: query for query in dataset.queries}

        self.assertGreater(
            sum(len(query.relevance["natural_section"].chunk_ids) for query in dataset.queries),
            50,
        )
        self.assertIn(
            "HVK:r1-2026-08-24-v1:niddk-diabetes-tests:003:9a52fd667ccc",
            by_id["RAG-002"].relevance["natural_section"].chunk_ids,
        )
        self.assertIn(
            "HVK:r1-2026-08-24-v1:odphp-get-screened:000:e8e5eeb18025",
            by_id["RAG-017"].relevance["hybrid_v1"].chunk_ids,
        )
        self.assertTrue(by_id["RAG-060"].answerable)
        self.assertEqual("hba1c", by_id["RAG-060"].category)
        self.assertIn(
            "HVK:r1-2026-08-24-v1:niddk-a1c-test:013:d94bf3c0e467",
            by_id["RAG-060"].relevance["hybrid_v1"].chunk_ids,
        )
        for query in dataset.queries:
            for strategy in ("natural_section", "hybrid_v1"):
                self.assertTrue(
                    set(query.relevance[strategy].chunk_ids).issubset(
                        query.judgment_pool[strategy]
                    )
                )

    def test_eval_data_is_outside_corpus_and_never_enters_the_fts_index(self):
        dataset = load_query_relevance_dataset(DEFAULT_DATASET_PATH)
        raw = json.loads(Path(DEFAULT_DATASET_PATH).read_text(encoding="utf-8"))
        repository = SQLiteFTSHealthKnowledgeRepository()
        try:
            self.assertNotIn("knowledge", Path(DEFAULT_DATASET_PATH).parts)
            self.assertEqual(
                repository.index_metadata["chunk_count"], len(repository._chunks)
            )
            for query in dataset.queries:
                self.assertNotIn(query.query_id, repository._chunks)
            serialized_index = "\n".join(repository._normalized_index_text.values())
            self.assertNotIn(raw["eval_version"], serialized_index)
            self.assertNotIn("reference_claims", serialized_index)
        finally:
            repository.close()

    def test_extra_dataset_fields_and_tampered_frozen_test_are_rejected(self):
        raw = json.loads(Path(DEFAULT_DATASET_PATH).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            with self.subTest("extra field"):
                extra = dict(raw)
                extra["unexpected"] = True
                path.write_text(json.dumps(extra), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exact schema"):
                    load_query_relevance_dataset(path)
            with self.subTest("frozen test mutation"):
                mutated = json.loads(json.dumps(raw))
                mutated["queries"][20]["query"] += " changed"
                path.write_text(json.dumps(mutated), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Frozen test hash mismatch"):
                    load_query_relevance_dataset(path)


if __name__ == "__main__":
    unittest.main()
