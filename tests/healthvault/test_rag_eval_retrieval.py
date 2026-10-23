import unittest

from healthvault.evaluation.rag.dataset import (
    DEFAULT_DATASET_PATH,
    load_query_relevance_dataset,
)
from healthvault.evaluation.rag.retrieval import (
    build_retrieval_run_manifest,
    canonical_json_bytes,
    compare_retrieval_strategies,
)
from healthvault.evaluation.rag.generation import prepare_test_generation_inputs


class RagEvalRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_query_relevance_dataset(DEFAULT_DATASET_PATH)
        cls.result = compare_retrieval_strategies(cls.dataset)

    def test_strategies_use_identical_retrieval_configuration_and_query_order(self):
        natural = self.result["natural_section"]
        hybrid = self.result["hybrid_v1"]

        self.assertEqual(natural["repository"], hybrid["repository"])
        self.assertEqual(5, natural["k"])
        self.assertEqual(
            [query["query_id"] for query in natural["queries"]],
            [query["query_id"] for query in hybrid["queries"]],
        )
        self.assertEqual(60, len(natural["queries"]))

    def test_all_required_groups_metrics_and_absolute_deltas_are_reported(self):
        required_groups = {
            "overall", "split/dev", "split/test", "answerable/true",
            "answerable/false", "language/en", "language/zh", "language/mixed",
            "category/fasting_glucose", "category/hba1c",
            "category/ldl_cholesterol", "category/lab_preparation",
            "category/result_communication", "category/visit_preparation",
            "rewrite_category/direct", "rewrite_category/synonym",
            "rewrite_category/mixed_language", "rewrite_category/unanswerable",
        }
        for strategy in ("natural_section", "hybrid_v1"):
            groups = self.result[strategy]["groups"]
            self.assertTrue(required_groups.issubset(groups))
            for group in groups.values():
                for metric in (
                    "recall_at_1", "recall_at_3", "recall_at_5",
                    "precision_at_3", "mrr_at_5", "ndcg_at_3",
                    "no_answer_accuracy", "unanswerable_false_positive_rate",
                ):
                    self.assertIn(metric, group)
        self.assertEqual(
            set(self.result["natural_section"]["groups"]),
            set(self.result["hybrid_minus_natural"]),
        )

    def test_result_is_byte_stable(self):
        rebuilt = compare_retrieval_strategies(self.dataset)
        self.assertEqual(canonical_json_bytes(self.result), canonical_json_bytes(rebuilt))

    def test_run_manifest_separates_timestamp_and_binds_code_config_and_result(self):
        manifest = build_retrieval_run_manifest(
            self.result,
            executed_at="2026-08-25T20:30:00+08:00",
            code_head="320254658709b05e285d7d547e52881cccb0c679",
        )

        self.assertEqual("offline_retrieval", manifest["run_type"])
        self.assertEqual(120, manifest["query_executions"]["successful"])
        self.assertEqual({"successful": 0, "failed": 0}, manifest["api_attempts"])
        self.assertEqual(64, len(manifest["configuration_sha256"]))
        self.assertEqual(64, len(manifest["result_sha256"]))

    def test_generation_inputs_use_only_frozen_test_rankings(self):
        for strategy in ("natural_section", "hybrid_v1"):
            cases, contexts = prepare_test_generation_inputs(
                self.dataset, self.result, strategy
            )
            self.assertEqual(40, len(cases))
            self.assertTrue(all(case["split"] == "test" for case in cases))
            self.assertEqual(
                {case["query_id"] for case in cases}, set(contexts)
            )
            self.assertTrue(
                all(
                    set(item) == {"citation_id", "body"}
                    for context in contexts.values()
                    for item in context
                )
            )

    def test_every_top_five_candidate_was_in_the_manual_judgment_pool(self):
        queries = {query.query_id: query for query in self.dataset.queries}
        for strategy in ("natural_section", "hybrid_v1"):
            for result in self.result[strategy]["queries"]:
                self.assertTrue(
                    set(result["ranked_chunk_ids"]).issubset(
                        queries[result["query_id"]].judgment_pool[strategy]
                    )
                )


if __name__ == "__main__":
    unittest.main()
