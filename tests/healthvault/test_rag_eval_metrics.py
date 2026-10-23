import unittest

from healthvault.evaluation.rag.metrics import (
    aggregate_retrieval_metrics,
    rank_scored_documents,
)


class RagEvalMetricsTests(unittest.TestCase):
    def test_standard_metrics_handle_multiple_relevant_documents_and_empty_results(self):
        evaluations = (
            ({"a", "b"}, ("b", "a", "x")),
            ({"c"}, ()),
        )

        metrics = aggregate_retrieval_metrics(evaluations)

        self.assertEqual(0.25, metrics["recall_at_1"])
        self.assertEqual(0.5, metrics["recall_at_3"])
        self.assertEqual(0.5, metrics["recall_at_5"])
        self.assertEqual(0.333333, metrics["precision_at_3"])
        self.assertEqual(0.5, metrics["mrr_at_5"])
        self.assertEqual(0.5, metrics["ndcg_at_3"])

    def test_tied_scores_are_stable_and_duplicate_documents_are_removed(self):
        ranked = rank_scored_documents(
            (("z", 1.0), ("b", 2.0), ("a", 1.0), ("b", 0.5))
        )

        self.assertEqual(("b", "a", "z"), ranked)

    def test_unanswerable_metrics_count_empty_and_false_positive_results(self):
        metrics = aggregate_retrieval_metrics(
            ((set(), ("unexpected",)), (set(), ()))
        )

        self.assertEqual(0.5, metrics["no_answer_accuracy"])
        self.assertEqual(0.5, metrics["unanswerable_false_positive_rate"])
        self.assertIsNone(metrics["recall_at_1"])


if __name__ == "__main__":
    unittest.main()
