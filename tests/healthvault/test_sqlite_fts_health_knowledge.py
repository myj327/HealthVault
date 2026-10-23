import threading
import unittest

from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    SQLiteFTSHealthKnowledgeRepository,
)


class SQLiteFTSHealthKnowledgeRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.repository = SQLiteFTSHealthKnowledgeRepository()

    def tearDown(self):
        self.repository.close()

    def test_fts5_trigram_aliases_and_mixed_queries_are_reproducible(self):
        cases = (
            ("HbA1c", "hba1c"),
            ("糖化血红蛋白", "hba1c"),
            ("ＦＰＧ", "fasting_glucose"),
            ("空腹血糖", "fasting_glucose"),
            ("LDL", "ldl_cholesterol"),
            ("低密度脂蛋白", "ldl_cholesterol"),
            ("血糖 A1C结果沟通", "hba1c"),
            ("就诊准备", "visit_preparation"),
        )
        self.assertEqual("fts5", self.repository.index_metadata["engine"])
        self.assertEqual("trigram", self.repository.index_metadata["tokenizer"])
        self.assertEqual(":memory:", self.repository.index_metadata["storage"])
        for query, expected_topic in cases:
            with self.subTest(query=query):
                first = self.repository.search(query, 3)
                second = self.repository.search(query, 3)
                self.assertEqual(first, second)
                self.assertTrue(first)
                self.assertEqual(expected_topic, first[0].document.topic)
                self.assertEqual(1, first[0].rank)
                self.assertIsInstance(first[0].score, float)

    def test_query_syntax_injection_limit_ties_and_unknown_are_safe(self):
        for query in (
            '" OR *',
            "a1c' UNION SELECT * FROM knowledge_chunks --",
            "() NEAR / { } : ^",
            "unrelated synthetic astronomy query",
            "治療方案",
        ):
            with self.subTest(query=query):
                self.assertEqual((), self.repository.search(query, 5))

        hits = self.repository.search("blood glucose A1C fasting", 2)
        self.assertEqual(2, len(hits))
        self.assertEqual(
            sorted(hits, key=lambda hit: (-hit.score, hit.document.citation_id)),
            list(hits),
        )
        self.assertEqual(hits, self.repository.search("blood glucose A1C fasting", 2))

    def test_out_of_scope_corpus_queries_return_no_hits(self):
        for query in (
            "blood clotting tests",
            "complete blood count",
            "autoantibody test",
            "Book traversal links Previous Next",
        ):
            with self.subTest(query=query):
                self.assertEqual((), self.repository.search(query, 5))

    def test_read_only_index_can_be_searched_from_another_thread(self):
        result: dict[str, object] = {}

        def search_from_streamlit_script_thread() -> None:
            try:
                result["hits"] = self.repository.search("HbA1c", 3)
            except Exception as exc:  # Captured to assert the regression boundary.
                result["exception"] = exc

        thread = threading.Thread(target=search_from_streamlit_script_thread)
        thread.start()
        thread.join()

        self.assertNotIn("exception", result)
        self.assertTrue(result["hits"])


if __name__ == "__main__":
    unittest.main()
