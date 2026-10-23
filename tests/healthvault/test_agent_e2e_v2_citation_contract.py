"""TASK-031 focused checks for canonical Agent-facing citation IDs."""

from __future__ import annotations

import json
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.domain.comparisons import (
    ComparisonIssue,
    ComparisonPoint,
    IndicatorComparison,
    IndicatorComparisonResult,
)
from healthvault.domain.knowledge import KnowledgeDocument, KnowledgeHit, KnowledgeSearchResult, KnowledgeSource
from healthvault.domain.records import ConfirmedIndicator, TimelineQueryResult, TimelineReport
from healthvault.domain.visit_briefs import RecordCitation, VisitBriefResult
from healthvault.evaluation.e2e import task_031


def _indicator(code: str = "hba1c") -> ConfirmedIndicator:
    return ConfirmedIndicator(
        code=code, display_name=code, value=5.2, unit="%", reference_lower=4.0,
        reference_upper=6.0, flag="normal", evidence_source_id="synthetic-source",
        evidence_line_start=1, evidence_line_end=1, evidence_quote="synthetic",
    )


def _report(report_id: str) -> TimelineReport:
    return TimelineReport(
        report_id=report_id, report_date="2033-01-01", facility="Synthetic Clinic",
        source_id="synthetic-source", source_sha256="synthetic-sha", indicators=(_indicator(),),
    )


def _knowledge_hit(citation_id: str) -> KnowledgeHit:
    return KnowledgeHit(
        document=KnowledgeDocument(
            citation_id=citation_id, topic="synthetic", title="Synthetic", summary="Synthetic",
            search_terms=("synthetic",), source=KnowledgeSource("Synthetic", "Synthetic", "https://example.invalid", "2026-01-01"),
        ),
        matched_terms=("synthetic",), rank=1, score=1.0,
    )


class Task031CitationContractTests(unittest.TestCase):
    def test_timeline_exposes_only_real_record_ids_and_empty_stays_empty(self) -> None:
        result = TimelineQueryResult("ok", 1, 1, (_report("R-1"),), []).to_dict()
        self.assertEqual(["REC:R-1:hba1c"], result["available_citation_ids"])
        self.assertEqual("R-1", result["timeline"][0]["report_id"])
        empty = TimelineQueryResult("empty", 0, 0, (), []).to_dict()
        self.assertEqual([], empty["available_citation_ids"])

    def test_comparison_exposes_real_endpoint_ids_without_changing_comparison(self) -> None:
        indicator = _indicator()
        comparison = IndicatorComparison(
            code="hba1c", display_name="hba1c", unit="%", observation_count=2,
            earliest=ComparisonPoint("R-1", "2033-01-01", "s", "sha", indicator),
            latest=ComparisonPoint("R-2", "2033-06-01", "s", "sha", indicator),
            absolute_change=0.0, direction="unchanged",
        )
        result = IndicatorComparisonResult("partial", 2, (comparison,), (ComparisonIssue("x", "x", "other"),)).to_dict()
        self.assertEqual(["REC:R-1:hba1c", "REC:R-2:hba1c"], result["available_citation_ids"])
        self.assertEqual("unchanged", result["comparisons"][0]["direction"])

    def test_knowledge_filters_noncanonical_ids_and_empty_stays_empty(self) -> None:
        canonical = "HVK:r1:source:001:abc"
        result = KnowledgeSearchResult("ok", 2, (_knowledge_hit(canonical), _knowledge_hit("HVK-SYNTHETIC")), ()).to_dict()
        self.assertEqual([canonical], result["available_citation_ids"])
        self.assertEqual(2, result["result_count"])
        empty = KnowledgeSearchResult("empty", 3, (), ()).to_dict()
        self.assertEqual([], empty["available_citation_ids"])

    def test_visit_brief_exposes_only_embedded_record_and_knowledge_ids(self) -> None:
        record = RecordCitation("REC:R-1:hba1c", "R-1", "2033-01-01", "hba1c", "s", "sha", "s", 1, 1, "q")
        knowledge = _knowledge_hit("HVK:r1:source:001:abc")
        result = VisitBriefResult("partial", 1, 1, None, (), (), (record,), (knowledge,), "", ()).to_dict()
        self.assertEqual(["REC:R-1:hba1c", "HVK:r1:source:001:abc"], result["available_citation_ids"])
        empty = VisitBriefResult("empty", 0, 0, None, (), (), (), (), "", ()).to_dict()
        self.assertEqual([], empty["available_citation_ids"])

    def test_v2_preserves_v1_bytes_and_changes_only_e2e_016_outcome(self) -> None:
        before = task_031.V1_DATASET_PATH.read_bytes()
        dataset = task_031.load_dataset()
        after = task_031.V1_DATASET_PATH.read_bytes()
        self.assertEqual(before, after)
        self.assertFalse(dataset["held_out"])
        case = next(item for item in dataset["cases"] if item["case_id"] == "E2E-016")
        self.assertEqual(["partial"], case["expected"]["required_tool_result_statuses"])

    def test_real_agent_loop_fake_transport_receives_v2_prompt_and_canonical_ids(self) -> None:
        from healthvault.evaluation.skill.benchmark.task_022 import CallLedger, ScriptedFakeTransport

        case = next(item for item in task_031.load_dataset()["cases"] if item["case_id"] == "E2E-001")
        citation = case["expected"]["allowed_citation_ids"][0]
        transport = ScriptedFakeTransport([
            {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "task031-call", "type": "function",
                "function": {"name": "query_health_timeline", "arguments": "{}"},
            }]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            {"choices": [{"message": {"role": "assistant", "content": json.dumps({
                "final_status": "ok", "citations": [citation], "refusal": False,
            }), "tool_calls": None}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}},
        ])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            ledger = CallLedger(root / "ledger.jsonl", run_config_sha256="f" * 64, max_transport_attempts=8)
            with patch("agents.agent.print_tool_call"), patch("agents.agent.print_tool_result"):
                result = asyncio.run(task_031.run_e2e_case(
                    case, model_config=task_031.v1.resolve_model_config({}), transport=transport,
                    ledger=ledger, runtime_root=root, api_key="fake",
                ))
        self.assertTrue(result["passed"], result)
        prompt = next(message["content"] for message in reversed(transport.requests[0]["messages"]) if message.get("role") == "user")
        self.assertIn("cite between 1 and 2", prompt)
        self.assertIn("Status mapping", prompt)
        self.assertNotIn(citation, prompt)
        tool_messages = [message["content"] for message in transport.requests[1]["messages"] if message.get("role") == "tool"]
        self.assertTrue(any(citation in content and "available_citation_ids" in content for content in tool_messages))
        self.assertEqual(0, result["attempt"]["api_calls"])

    def test_v2_preflight_is_zero_api_and_hash_bound(self) -> None:
        dataset = task_031.load_dataset()
        preflight = task_031.build_preflight(dataset, task_031.v1.resolve_model_config({}))
        self.assertEqual(0, preflight["api_calls_made"])
        self.assertEqual(37, preflight["estimated_model_calls"])
        with self.assertRaises(task_031.ApiGateError):
            task_031.assert_api_gate(preflight=preflight, supplied_hash="wrong", approved=True, api_key="fake")

    def test_v1_to_v2_delta_uses_the_frozen_v1_summary(self) -> None:
        baseline = task_031._v1_summary()
        self.assertEqual(
            {key: 0.0 for key in task_031.DELTA_METRICS},
            task_031.v1_to_v2_delta(baseline),
        )


if __name__ == "__main__":
    unittest.main()
