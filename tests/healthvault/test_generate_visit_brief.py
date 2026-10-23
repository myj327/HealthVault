import hashlib
import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path

from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.application.generate_visit_brief import generate_visit_brief
from healthvault.domain.records import ConfirmedIndicator, TimelineReport
from healthvault.infrastructure.local_health_knowledge import (
    LocalHealthKnowledgeRepository,
)
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.repositories.health_knowledge import HealthKnowledgeQueryError
from healthvault.repositories.health_records import HealthRecordQueryError


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "health_reports"
NOTICE = "Visit preparation aid only; not a diagnosis or treatment recommendation."


def load_fixture(name):
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def confirm_fixture(repository, profile_id, source_id, filename):
    report_text = load_fixture(filename)
    result = confirm_health_records(
        {
            "profile_id": profile_id,
            "source_id": source_id,
            "report_text": report_text,
            "expected_sha256": hashlib.sha256(
                report_text.encode("utf-8")
            ).hexdigest(),
            "user_confirmed": True,
        },
        repository,
    ).to_dict()
    if result["status"] != "confirmed":
        raise AssertionError(f"Synthetic fixture confirmation failed: {result}")


def make_indicator(code, value, unit="synthetic-unit", flag="normal"):
    return ConfirmedIndicator(
        code=code,
        display_name=f"Synthetic {code}",
        value=value,
        unit=unit,
        reference_lower=0.0,
        reference_upper=10.0,
        flag=flag,
        evidence_source_id=f"source-{code}",
        evidence_line_start=4,
        evidence_line_end=4,
        evidence_quote=f"result|{code}|{value}|{unit}|0.0|10.0",
    )


def make_report(report_id, report_date, indicators):
    return TimelineReport(
        report_id=report_id,
        report_date=report_date,
        facility="Synthetic Clinic",
        source_id=f"source-{report_id}",
        source_sha256=f"sha256-{report_id}",
        indicators=tuple(indicators),
    )


class CountingRecordRepository:
    def __init__(self, delegate=None, responses=()):
        self.delegate = delegate
        self.responses = tuple(responses)
        self.queries = []

    def list_timeline(self, profile_id):
        self.queries.append(profile_id)
        if self.delegate is not None:
            return self.delegate.list_timeline(profile_id)
        response = self.responses[len(self.queries) - 1]
        if isinstance(response, Exception):
            raise response
        return response


class CountingKnowledgeRepository:
    def __init__(self, delegate=None, error=None):
        self.delegate = delegate
        self.error = error
        self.queries = []

    def search(self, query, limit):
        self.queries.append((query, limit))
        if self.error is not None:
            raise self.error
        return self.delegate.search(query, limit) if self.delegate else ()


class GenerateVisitBriefTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "visit-brief.sqlite3"
        self.sqlite_repository = SQLiteHealthRecordRepository(self.database_path)
        self.knowledge_repository = LocalHealthKnowledgeRepository()

    def tearDown(self):
        self.temp_dir.cleanup()

    def fetch_all(self, sql):
        connection = sqlite3.connect(self.database_path)
        try:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(sql).fetchall()]
        finally:
            connection.close()

    def database_snapshot(self):
        return {
            "reports": self.fetch_all(
                "SELECT * FROM confirmed_reports ORDER BY id"
            ),
            "records": self.fetch_all(
                "SELECT * FROM health_records ORDER BY id"
            ),
        }

    def test_invalid_input_calls_neither_repository(self):
        cases = (
            None,
            "not-an-object",
            {},
            {"profile_id": ""},
            {"profile_id": "   "},
            {"profile_id": 7},
            {"profile_id": []},
        )
        for tool_input in cases:
            with self.subTest(tool_input=tool_input):
                records = CountingRecordRepository()
                knowledge = CountingKnowledgeRepository()
                result = generate_visit_brief(
                    tool_input,
                    records,
                    knowledge,
                ).to_dict()
                self.assertEqual("invalid_request", result["status"])
                self.assertEqual(0, result["report_count"])
                self.assertEqual(0, result["indicator_count"])
                self.assertIsNone(result["latest_report"])
                self.assertEqual([], result["indicators"])
                self.assertEqual([], result["questions"])
                self.assertEqual([], result["record_citations"])
                self.assertEqual([], result["knowledge_citations"])
                self.assertEqual("", result["brief_markdown"])
                self.assertTrue(result["issues"])
                self.assertEqual(NOTICE, result["notice"])
                self.assertEqual([], records.queries)
                self.assertEqual([], knowledge.queries)

    def test_empty_calls_records_once_and_skips_comparison_and_knowledge(self):
        records = CountingRecordRepository(responses=((),))
        knowledge = CountingKnowledgeRepository()

        result = generate_visit_brief(
            {"profile_id": " synthetic-empty "},
            records,
            knowledge,
        ).to_dict()

        self.assertEqual("empty", result["status"])
        self.assertEqual(" synthetic-empty ", result["profile_id"])
        self.assertEqual(0, result["report_count"])
        self.assertEqual(0, result["indicator_count"])
        self.assertIsNone(result["latest_report"])
        self.assertEqual([], result["indicators"])
        self.assertEqual([], result["questions"])
        self.assertEqual([], result["record_citations"])
        self.assertEqual([], result["knowledge_citations"])
        self.assertEqual("", result["brief_markdown"])
        self.assertEqual([], result["issues"])
        self.assertEqual([" synthetic-empty "], records.queries)
        self.assertEqual([], knowledge.queries)

    def test_record_and_knowledge_errors_propagate_at_each_call_boundary(self):
        first_error = CountingRecordRepository(
            responses=(HealthRecordQueryError("first record failure"),)
        )
        with self.assertRaisesRegex(HealthRecordQueryError, "first record failure"):
            generate_visit_brief(
                {"profile_id": "synthetic-first-error"},
                first_error,
                CountingKnowledgeRepository(),
            )

        timeline = (make_report("report-1", "2025-01-01", (make_indicator("x", 1.0),)),)
        second_error = CountingRecordRepository(
            responses=(timeline, HealthRecordQueryError("second record failure"))
        )
        second_knowledge = CountingKnowledgeRepository()
        with self.assertRaisesRegex(HealthRecordQueryError, "second record failure"):
            generate_visit_brief(
                {"profile_id": "synthetic-second-error"},
                second_error,
                second_knowledge,
            )
        self.assertEqual(2, len(second_error.queries))
        self.assertEqual([], second_knowledge.queries)

        records = CountingRecordRepository(responses=(timeline, timeline))
        knowledge = CountingKnowledgeRepository(
            error=HealthKnowledgeQueryError("knowledge failure")
        )
        with self.assertRaisesRegex(HealthKnowledgeQueryError, "knowledge failure"):
            generate_visit_brief(
                {"profile_id": "synthetic-knowledge-error"},
                records,
                knowledge,
            )
        self.assertEqual(2, len(records.queries))
        self.assertEqual(1, len(knowledge.queries))

    def test_one_year_is_partial_with_latest_questions_and_knowledge(self):
        profile_id = "synthetic-one-year-brief"
        confirm_fixture(
            self.sqlite_repository,
            profile_id,
            "synthetic-one-year-source",
            "synthetic_checkup_2024.txt",
        )
        records = CountingRecordRepository(delegate=self.sqlite_repository)
        knowledge = CountingKnowledgeRepository(delegate=self.knowledge_repository)

        result = generate_visit_brief(
            {"profile_id": profile_id},
            records,
            knowledge,
        ).to_dict()

        self.assertEqual("partial", result["status"])
        self.assertEqual(1, result["report_count"])
        self.assertEqual(3, result["indicator_count"])
        self.assertEqual(3, len(result["indicators"]))
        self.assertTrue(all(item["comparison"] is None for item in result["indicators"]))
        self.assertEqual(4, len(result["questions"]))
        self.assertEqual(3, len(result["record_citations"]))
        self.assertEqual(4, len(result["knowledge_citations"]))
        self.assertEqual(
            [
                ("fasting_glucose", "insufficient_history"),
                ("hba1c", "insufficient_history"),
                ("ldl_cholesterol", "insufficient_history"),
            ],
            [
                (issue["indicator_code"], issue["code"])
                for issue in result["issues"]
            ],
        )
        self.assertEqual([profile_id, profile_id], records.queries)
        self.assertEqual(4, len(knowledge.queries))
        self.assertTrue(all(limit == 5 for _query, limit in knowledge.queries))
        self.assertTrue(
            all(
                query.casefold().startswith(prefix)
                for (query, _limit), prefix in zip(
                    knowledge.queries[:3],
                    ("fasting glucose", "hba1c", "ldl cholesterol"),
                )
            )
        )
        self.assertEqual(
            "visit preparation doctor visit 就诊准备",
            knowledge.queries[-1][0],
        )

    def test_two_year_brief_is_exact_evidenced_isolated_read_only_and_stable(self):
        profile_id = "synthetic-two-year-brief"
        confirm_fixture(
            self.sqlite_repository,
            profile_id,
            "synthetic-2025",
            "synthetic_checkup_2025.txt",
        )
        confirm_fixture(
            self.sqlite_repository,
            profile_id,
            "synthetic-2024",
            "synthetic_checkup_2024.txt",
        )
        confirm_fixture(
            self.sqlite_repository,
            "synthetic-other-profile",
            "synthetic-other-source",
            "synthetic_checkup_2024.txt",
        )
        before = self.database_snapshot()

        first_records = CountingRecordRepository(delegate=self.sqlite_repository)
        first_knowledge = CountingKnowledgeRepository(
            delegate=self.knowledge_repository
        )
        first = generate_visit_brief(
            {"profile_id": profile_id},
            first_records,
            first_knowledge,
        ).to_dict()
        second = generate_visit_brief(
            {"profile_id": profile_id},
            CountingRecordRepository(delegate=self.sqlite_repository),
            CountingKnowledgeRepository(delegate=self.knowledge_repository),
        ).to_dict()

        self.assertEqual(first, second)
        self.assertEqual(before, self.database_snapshot())
        self.assertEqual([profile_id, profile_id], first_records.queries)
        self.assertEqual(4, len(first_knowledge.queries))
        self.assertEqual("ok", first["status"])
        self.assertEqual(2, first["report_count"])
        self.assertEqual(3, first["indicator_count"])
        self.assertEqual(
            {
                "report_id": "HV-SYN-2025-001",
                "report_date": "2025-06-18",
                "facility": "Synthetic Community Clinic",
                "source": {
                    "source_id": "synthetic-2025",
                    "sha256": hashlib.sha256(
                        load_fixture("synthetic_checkup_2025.txt").encode("utf-8")
                    ).hexdigest(),
                },
            },
            first["latest_report"],
        )
        self.assertEqual(
            ["fasting_glucose", "hba1c", "ldl_cholesterol"],
            [item["code"] for item in first["indicators"]],
        )
        expected = {
            "fasting_glucose": (6.5, "high", 1.1, "increased"),
            "hba1c": (5.9, "normal", 0.4, "increased"),
            "ldl_cholesterol": (2.8, "normal", -0.3, "decreased"),
        }
        for item in first["indicators"]:
            value, flag, change, direction = expected[item["code"]]
            self.assertEqual(value, item["latest"]["value"])
            self.assertEqual(flag, item["latest"]["flag"])
            self.assertEqual(change, item["comparison"]["absolute_change"])
            self.assertEqual(direction, item["comparison"]["direction"])
            self.assertEqual(1, len(item["knowledge_citation_ids"]))
            self.assertIn(
                item["latest"]["record_citation_id"],
                item["comparison"].values(),
            )

        self.assertEqual(4, len(first["questions"]))
        self.assertEqual("ASK:visit_preparation", first["questions"][-1]["question_id"])
        self.assertEqual(1, len(first["questions"][-1]["knowledge_citation_ids"]))
        record_ids = {item["citation_id"] for item in first["record_citations"]}
        knowledge_ids = {item["citation_id"] for item in first["knowledge_citations"]}
        self.assertEqual(6, len(record_ids))
        self.assertEqual(4, len(knowledge_ids))
        self.assertTrue(
            all(identifier.startswith("HVK:r1-2026-08-24-v1:") for identifier in knowledge_ids)
        )
        knowledge_by_id = {
            item["citation_id"]: item for item in first["knowledge_citations"]
        }
        for item in first["indicators"]:
            knowledge_id = item["knowledge_citation_ids"][0]
            self.assertIn(item["code"], knowledge_by_id[knowledge_id]["topic_tags"])
        visit_id = first["questions"][-1]["knowledge_citation_ids"][0]
        self.assertIn("visit_preparation", knowledge_by_id[visit_id]["topic_tags"])
        for item in first["indicators"]:
            self.assertIn(item["latest"]["record_citation_id"], record_ids)
            comparison = item["comparison"]
            self.assertIn(comparison["earliest_record_citation_id"], record_ids)
            self.assertIn(comparison["latest_record_citation_id"], record_ids)
        for citation in first["record_citations"]:
            self.assertEqual(
                citation["source"]["source_id"],
                citation["evidence"]["source_id"],
            )
            self.assertTrue(citation["evidence"]["quote"].startswith("result|"))

        markdown = first["brief_markdown"]
        expected_headers = (
            "# 就诊准备摘要",
            "## 档案概览",
            "## 已确认指标",
            "## 可向医生询问",
            "## 记录证据",
            "## 健康知识来源",
        )
        positions = [markdown.index(header) for header in expected_headers]
        self.assertEqual(sorted(positions), positions)
        markdown_ids = set(
            re.findall(r"\[((?:REC:[^\]]+)|(?:HVK:[^\]]+))\]", markdown)
        )
        self.assertEqual(record_ids | knowledge_ids, markdown_ids)
        self.assertIn("数值变化", markdown)
        serialized = json.dumps(first, ensure_ascii=False)
        for forbidden in (
            "raw_report_text",
            "idempotency_key",
            "confirmed_at",
            "confirmed_report_id",
            "synthetic-other-source",
            "改善",
            "恶化",
            "确诊",
            "风险分级",
            "治疗建议",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_partial_unknown_and_unit_conflict_keep_latest_from_earlier_report(self):
        timeline = (
            make_report(
                "report-2024",
                "2024-01-01",
                (
                    make_indicator("unit_conflict", 1.0, unit="unit-a"),
                    make_indicator("unknown_indicator", 2.0),
                ),
            ),
            make_report(
                "report-2025",
                "2025-01-01",
                (make_indicator("unit_conflict", 3.0, unit="unit-b"),),
            ),
        )
        records = CountingRecordRepository(responses=(timeline, timeline))
        knowledge = CountingKnowledgeRepository(delegate=self.knowledge_repository)

        result = generate_visit_brief(
            {"profile_id": "synthetic-partial"},
            records,
            knowledge,
        ).to_dict()

        self.assertEqual("partial", result["status"])
        by_code = {item["code"]: item for item in result["indicators"]}
        self.assertIsNone(by_code["unit_conflict"]["comparison"])
        self.assertIsNone(by_code["unknown_indicator"]["comparison"])
        self.assertEqual(
            "report-2024",
            by_code["unknown_indicator"]["latest"]["report_id"],
        )
        self.assertEqual(
            {
                ("unit_conflict", "unit_mismatch"),
                ("unknown_indicator", "insufficient_history"),
                ("unit_conflict", "knowledge_not_found"),
                ("unknown_indicator", "knowledge_not_found"),
            },
            {
                (issue["indicator_code"], issue["code"])
                for issue in result["issues"]
            },
        )
        self.assertEqual(2, len(records.queries))
        self.assertEqual(3, len(knowledge.queries))


if __name__ == "__main__":
    unittest.main()
