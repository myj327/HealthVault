import json
import tempfile
import unittest
from pathlib import Path

from healthvault.application.knowledge_support_gate import (
    SUPPORT_GATE_RULES,
    assess_health_knowledge_support,
    search_with_health_knowledge_gate,
)
from healthvault.application.search_health_knowledge import search_health_knowledge
from healthvault.domain.knowledge import KnowledgeDocument, KnowledgeHit, KnowledgeSource
from healthvault.evaluation.rag.task_027 import run_task_027
from healthvault.infrastructure.local_health_knowledge import (
    LocalHealthKnowledgeRepository,
)


ROOT = Path(__file__).parents[2]
DEV_CASES = ROOT / "healthvault/evaluation/rag/task-027/gate-dev-cases.json"


def _hit(
    *,
    title="Synthetic A1C evidence",
    topic="hba1c",
    topic_tags=("hba1c",),
    summary="A1C is a laboratory blood test.",
    matched_terms=("a1c",),
) -> KnowledgeHit:
    return KnowledgeHit(
        document=KnowledgeDocument(
            citation_id="HVK:synthetic:evidence",
            topic=topic,
            title=title,
            summary=summary,
            search_terms=matched_terms,
            source=KnowledgeSource(
                publisher="Synthetic",
                title="Synthetic source",
                url="https://example.test/evidence",
                retrieved_on="2026-09-01",
            ),
            topic_tags=topic_tags,
        ),
        matched_terms=matched_terms,
        rank=1,
        score=1.0,
    )


class RecordingRepository:
    def __init__(self, hits=()):
        self.hits = tuple(hits)
        self.calls = []

    def search(self, query, limit):
        self.calls.append((query, limit))
        return self.hits[:limit]


class Task027KnowledgeSupportGateTests(unittest.TestCase):
    def test_v2_dev_cases_are_independent_and_pre_gate_has_only_four_intents(self):
        payload = json.loads(DEV_CASES.read_text(encoding="utf-8"))
        self.assertEqual([], payload["gold_case_ids_used"])
        self.assertEqual(2, payload["schema_version"])
        self.assertNotIn(
            "unsupported_corpus_topic",
            {rule["reason_code"] for rule in SUPPORT_GATE_RULES},
        )
        for case in payload["cases"]:
            with self.subTest(case_id=case["case_id"]):
                decision = assess_health_knowledge_support(case["query"])
                actual = "allow" if decision.allowed else decision.reason_code
                self.assertEqual(case["expected_pre_gate"], actual)

    def test_four_pre_gate_rejections_return_empty_and_never_call_repository(self):
        rejected = (
            ("Retrieve the values in my private record from last year.", "unsupported_personal_record"),
            ("Use these symptoms to diagnose me.", "unsupported_diagnosis"),
            ("What prescription dose should I take?", "unsupported_treatment"),
            ("Forecast my exact glucose value tomorrow.", "unsupported_future_prediction"),
        )
        for query, reason in rejected:
            with self.subTest(reason=reason):
                repository = RecordingRepository((_hit(),))
                result = search_health_knowledge(
                    {"query": query, "limit": 5}, repository
                ).to_dict()
                self.assertEqual("empty", result["status"])
                self.assertEqual([], result["results"])
                self.assertEqual(reason, result["issues"][0]["code"])
                self.assertEqual([], repository.calls)

    def test_real_repository_rejects_uncovered_thyroid_queries_and_action_words(self):
        repository = LocalHealthKnowledgeRepository()
        self.addCleanup(repository.close)
        queries = (
            "What reference range should be used for a thyroid ultrasound?",
            "Can an A1C test detect or show a thyroid ultrasound abnormality?",
        )

        for query in queries:
            with self.subTest(query=query):
                result = search_health_knowledge(
                    {"query": query, "limit": 5}, repository
                ).to_dict()
                self.assertEqual("empty", result["status"])
                self.assertEqual([], result["results"])
                self.assertEqual(
                    "unsupported_corpus_topic", result["issues"][0]["code"]
                )

    def test_action_words_alone_never_cover_a_missing_medical_topic(self):
        evidence = _hit(
            title="How to interpret laboratory results",
            topic="result_communication",
            topic_tags=("result_communication",),
            summary="Results can detect and show changes over time.",
            matched_terms=("interpret", "detect", "show"),
        )
        repository = RecordingRepository((evidence,))

        result = search_health_knowledge(
            {"query": "How do I interpret, detect, or show a thyroid ultrasound?"},
            repository,
        ).to_dict()

        self.assertEqual("empty", result["status"])
        self.assertEqual("unsupported_corpus_topic", result["issues"][0]["code"])

    def test_real_repository_keeps_three_supported_health_topics(self):
        repository = LocalHealthKnowledgeRepository()
        self.addCleanup(repository.close)
        queries = (
            "What does an A1C test measure?",
            "How should I prepare for a fasting blood test?",
            "How can I prepare questions for a doctor appointment?",
        )

        for query in queries:
            with self.subTest(query=query):
                result = search_health_knowledge(
                    {"query": query, "limit": 5}, repository
                ).to_dict()
                self.assertEqual("ok", result["status"])
                self.assertGreater(result["result_count"], 0)

    def test_core_topic_coverage_uses_top_five_text_metadata_and_matched_terms(self):
        query = "What does an A1C laboratory test measure?"
        evidence = _hit(
            title="Laboratory overview",
            topic="blood_test",
            topic_tags=("blood_glucose",),
            summary="A laboratory blood test measures average glucose over time.",
            matched_terms=("a1c",),
        )
        repository = RecordingRepository((evidence,))
        decision, hits = search_with_health_knowledge_gate(query, repository, 5)

        self.assertTrue(decision.allowed)
        self.assertEqual((evidence,), hits)
        self.assertEqual([(query, 5)], repository.calls)

    def test_supported_rewritten_query_preserves_repository_query_hits_and_schema(self):
        evidence = _hit()
        repository = RecordingRepository((evidence,))
        raw_query = "What does an A1C laboratory test measure?"
        decision, hits = search_with_health_knowledge_gate(
            raw_query,
            repository,
            5,
            retrieval_query="unchanged frozen rewrite",
        )

        self.assertTrue(decision.allowed)
        self.assertEqual((evidence,), hits)
        self.assertEqual([("unchanged frozen rewrite", 5)], repository.calls)

    def test_v3_zero_api_recompute_preserves_answerable_top_five(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run = run_task_027(output)
            manifest = json.loads(
                (output / "task-027-v3-run-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(3, manifest["schema_version"])
            self.assertTrue((output / "task-027-v3-retrieval-results.json").is_file())

        self.assertEqual("task-027-gate-v3", run["task_version"])
        self.assertTrue(run["answerable_top5_unchanged"])
        self.assertEqual(16, run["answerable_top5_checked"])
        self.assertEqual([], run["answerable_top5_mismatches"])
        metrics = run["metrics"]
        self.assertEqual(0.688542, metrics["recall_at_5"])
        self.assertEqual(0.516667, metrics["mrr_at_5"])
        self.assertLessEqual(metrics["unanswerable_false_positive_rate"], 0.25)
        self.assertEqual(0, run["api_calls"])
        self.assertEqual(0, run["input_tokens"])
        self.assertEqual(0, run["output_tokens"])
        self.assertEqual(0.0, run["cost_usd"])


if __name__ == "__main__":
    unittest.main()
