import unittest

from healthvault.application.search_health_knowledge import (
    search_health_knowledge,
)
from healthvault.domain.knowledge import (
    KnowledgeDocument,
    KnowledgeHit,
    KnowledgeSource,
)
from healthvault.infrastructure.local_health_knowledge import (
    LocalHealthKnowledgeRepository,
)
from healthvault.repositories.health_knowledge import HealthKnowledgeQueryError


NOTICE = "Educational source retrieval only; not medical advice."


def make_hit():
    return KnowledgeHit(
        document=KnowledgeDocument(
            citation_id="HVK-SYNTHETIC-001",
            topic="synthetic_topic",
            title="Synthetic knowledge card",
            summary="A short educational excerpt explaining the synthetic term.",
            search_terms=("synthetic term",),
            source=KnowledgeSource(
                publisher="Synthetic Publisher",
                title="Synthetic Official Source",
                url="https://example.test/official",
                retrieved_on="2026-08-24",
            ),
        ),
        matched_terms=("synthetic term",),
    )


class RecordingKnowledgeRepository:
    def __init__(self, hits=(), error=None):
        self.hits = hits
        self.error = error
        self.queries = []

    def search(self, query, limit):
        self.queries.append((query, limit))
        if self.error is not None:
            raise self.error
        return self.hits[:limit]


class SearchHealthKnowledgeTests(unittest.TestCase):
    def test_invalid_input_never_calls_repository_and_always_has_notice(self):
        cases = (
            None,
            "not-an-object",
            {},
            {"query": ""},
            {"query": "   "},
            {"query": 7},
            {"query": []},
            {"query": "a1c", "limit": True},
            {"query": "a1c", "limit": "2"},
            {"query": "a1c", "limit": 0},
            {"query": "a1c", "limit": -1},
            {"query": "a1c", "limit": 6},
        )

        for tool_input in cases:
            with self.subTest(tool_input=tool_input):
                repository = RecordingKnowledgeRepository()
                result = search_health_knowledge(
                    tool_input,
                    repository,
                ).to_dict()
                self.assertEqual("invalid_request", result["status"])
                self.assertEqual(0, result["result_count"])
                self.assertEqual([], result["results"])
                self.assertTrue(result["issues"])
                self.assertEqual(NOTICE, result["notice"])
                self.assertEqual([], repository.queries)

    def test_default_limit_exact_query_empty_and_ok_results(self):
        empty_repository = RecordingKnowledgeRepository()
        empty = search_health_knowledge(
            {"query": "  exact Query  "},
            empty_repository,
        ).to_dict()

        self.assertEqual("empty", empty["status"])
        self.assertEqual("  exact Query  ", empty["query"])
        self.assertEqual(3, empty["limit"])
        self.assertEqual(0, empty["result_count"])
        self.assertEqual([], empty["results"])
        self.assertEqual("unsupported_corpus_topic", empty["issues"][0]["code"])
        self.assertEqual(NOTICE, empty["notice"])
        self.assertEqual([("exact Query", 3)], empty_repository.queries)

        hit = make_hit()
        repository = RecordingKnowledgeRepository(hits=(hit,))
        ok = search_health_knowledge(
            {"query": "synthetic term", "limit": 1},
            repository,
        ).to_dict()
        self.assertEqual("ok", ok["status"])
        self.assertEqual(1, ok["limit"])
        self.assertEqual(1, ok["result_count"])
        result = ok["results"][0]
        self.assertEqual("HVK-SYNTHETIC-001", result["citation_id"])
        self.assertEqual("HVK-SYNTHETIC-001", result["chunk_id"])
        self.assertEqual("synthetic_topic", result["topic"])
        self.assertEqual("Synthetic knowledge card", result["title"])
        self.assertEqual(
            "A short educational excerpt explaining the synthetic term.", result["excerpt"]
        )
        self.assertEqual(["synthetic term"], result["matched_terms"])
        self.assertEqual("Synthetic Publisher", result["source"]["publisher"])
        self.assertEqual("https://example.test/official", result["source"]["url"])
        self.assertNotIn("profile_id", str(ok))

    def test_repository_error_is_not_converted_to_empty(self):
        repository = RecordingKnowledgeRepository(
            error=HealthKnowledgeQueryError("synthetic corpus failure")
        )

        with self.assertRaisesRegex(
            HealthKnowledgeQueryError,
            "synthetic corpus failure",
        ):
            search_health_knowledge({"query": "a1c"}, repository)
        self.assertEqual([("a1c", 3)], repository.queries)

    def test_default_corpus_result_is_traceable_safe_and_deterministic(self):
        repository = LocalHealthKnowledgeRepository()
        tool_input = {"query": "  HbA1c  ", "limit": 1}

        first = search_health_knowledge(tool_input, repository).to_dict()
        second = search_health_knowledge(tool_input, repository).to_dict()

        self.assertEqual(first, second)
        self.assertEqual(
            {
                "status",
                "query",
                "limit",
                "result_count",
                "results",
                "available_citation_ids",
                "recommended_citation_ids",
                "issues",
                "notice",
            },
            set(first),
        )
        self.assertEqual("  HbA1c  ", first["query"])
        self.assertEqual(NOTICE, first["notice"])
        self.assertTrue(
            first["results"][0]["citation_id"].startswith(
                "HVK:r1-2026-08-24-v1:"
            )
        )
        self.assertEqual(
            first["results"][0]["citation_id"],
            first["results"][0]["chunk_id"],
        )
        self.assertEqual("r1-2026-08-24-v1", first["results"][0]["corpus_version"])
        self.assertIn("hba1c", first["results"][0]["topic_tags"])
        self.assertEqual(64, len(first["results"][0]["body_sha256"]))
        self.assertEqual(1, first["results"][0]["rank"])
        self.assertEqual(["a1c", "hba1c"], first["results"][0]["matched_terms"])
        source = first["results"][0]["source"]
        self.assertIn(source["publisher"], {"CDC", "MedlinePlus", "NIDDK"})
        self.assertTrue(source["url"].startswith("https://"))
        self.assertTrue(source["source_id"])
        self.assertEqual(64, len(source["sha256"]))
        serialized = str(first)
        for forbidden in (
            "profile_id",
            "database_path",
            "raw_report_text",
            "treatment plan",
            "治疗方案",
        ):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
