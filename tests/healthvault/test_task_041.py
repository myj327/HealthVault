import json
import tempfile
import unittest
from pathlib import Path

from healthvault.evaluation.rag import task_041
from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT, load_chunk_artifact


class Task041Tests(unittest.TestCase):
    def test_r2_artifacts_validate_without_changing_the_r1_default(self):
        r1 = load_chunk_artifact(DEFAULT_CORPUS_ROOT / "chunks" / "hybrid_v1.json")
        r2 = load_chunk_artifact(
            task_041.R2_ROOT / "chunks" / "hybrid_v1.json",
            expected_corpus_version=task_041.CORPUS_VERSION,
        )
        self.assertEqual("r1-2026-08-24-v1", r1.corpus_version)
        self.assertEqual(task_041.CORPUS_VERSION, r2.corpus_version)
        self.assertEqual(88, len(r2.chunks))

    def test_unsigned_draft_has_twenty_cases_but_no_human_gold(self):
        draft = task_041.draft_queries()
        self.assertEqual(20, draft["query_count"])
        self.assertEqual("not_signed", draft["gold_status"])
        self.assertTrue(all(item["human_annotation"]["answerable"] is None for item in draft["queries"]))
        self.assertTrue(all(not item["human_annotation"]["qrels"] for item in draft["queries"]))

    def test_draft_has_no_collision_with_prior_heldout_tracks(self):
        self.assertEqual([], task_041.collision_report(task_041.draft_queries()))

    def test_pool_is_union_of_bm25_and_hybrid_without_relevance_labels(self):
        pool = task_041.build_candidate_pool(task_041.draft_queries(), corpus_root=task_041.R2_ROOT)
        self.assertEqual(["bm25", "hybrid"], pool["arms"])
        self.assertEqual(20, len(pool["cases"]))
        self.assertTrue(all(case["human_relevance"] == "pending" for case in pool["cases"]))
        self.assertEqual("lexical", pool["arm_metadata"]["hybrid"]["retrieval_mode"])
        self.assertTrue(all(case["candidates"] for case in pool["cases"][:13]))
        self.assertTrue(all(set(item["arm_ranks"]).issubset({"bm25", "hybrid"}) for case in pool["cases"] for item in case["candidates"]))

    def test_judge_schema_and_pending_preflight_are_strict(self):
        self.assertEqual({"passed": True, "reason": "supported"}, task_041.validate_judge_result('{"passed": true, "reason": "supported"}'))
        with self.assertRaises(ValueError):
            task_041.validate_judge_result('{"passed": true, "reason": "x", "score": 1}')
        with tempfile.TemporaryDirectory() as directory:
            task_041.write_unsigned_annotation_package(directory, corpus_root=task_041.R2_ROOT)
            preflight = task_041.build_answer_judge_preflight(corpus_root=task_041.R2_ROOT, annotation_directory=directory)
        self.assertEqual("LOCAL_HYBRID_INDEX_GATE_PENDING", preflight["status"])
        self.assertEqual(0, preflight["api_calls_made"])
        self.assertEqual(64, len(preflight["preflight_sha256"]))


if __name__ == "__main__":
    unittest.main()
