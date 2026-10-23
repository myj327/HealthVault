"""Focused, zero-API checks for TASK-043's frozen R5 benchmark."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from healthvault.application.compare_health_indicators import compare_health_indicators
from healthvault.application.generate_visit_brief import generate_visit_brief
from healthvault.application.search_health_knowledge import search_health_knowledge
from healthvault.evaluation.e2e import task_043
from healthvault.infrastructure.local_health_knowledge import LocalHealthKnowledgeRepository
from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository


class Task043Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = task_043.load_dataset()
        cls.by_id = {case["case_id"]: case for case in cls.dataset["cases"]}

    def test_dataset_has_twenty_new_balanced_fixture_derived_cases(self) -> None:
        self.assertEqual(20, len(self.dataset["cases"]))
        self.assertEqual(
            task_043.CATEGORY_COUNTS,
            dict(Counter(case["category"] for case in self.dataset["cases"])),
        )
        self.assertTrue(all(case["patient_id"].startswith("r5-patient-") for case in self.dataset["cases"]))
        self.assertTrue(all("expected" in case for case in self.dataset["cases"]))
        self.assertNotIn("expected", json.loads(task_043.DATASET_PATH.read_text(encoding="utf-8"))["cases"][0])
        self.assertEqual([], self.by_id["R5E-002"]["expected"]["required_tools"])
        self.assertEqual("rejected", self.by_id["R5E-003"]["expected"]["final_status"])
        self.assertEqual(["search_health_knowledge"], self.by_id["R5E-012"]["expected"]["required_tools"])
        matrix = task_043.build_case_matrix(self.dataset, task_043.load_fixture())
        self.assertEqual(20, matrix["case_count"])
        self.assertEqual(["outcome", "behavior", "response"], matrix["oracle_order"])
        self.assertTrue(all(row["expected_final_state"]["database_unchanged"] for row in matrix["cases"]))

    def test_fixture_reset_world_state_and_tool_statuses_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first.sqlite3"
            second_path = Path(temporary) / "second.sqlite3"
            first = task_043.materialize_fixture(first_path)
            second = task_043.materialize_fixture(second_path)
            self.assertEqual(first["world_state"], second["world_state"])
            self.assertEqual(8, first["report_count"])

            repository = SQLiteHealthRecordRepository(first_path)
            knowledge = LocalHealthKnowledgeRepository(task_043.CORPUS_PATH)
            self.assertEqual(
                self.by_id["R5E-009"]["expected"]["required_tool_result_statuses"][0],
                compare_health_indicators({"profile_id": "r5-mixed"}, repository).status,
            )
            self.assertEqual(
                self.by_id["R5E-015"]["expected"]["required_tool_result_statuses"][0],
                generate_visit_brief({"profile_id": "r5-mixed"}, repository, knowledge).status,
            )
            for case_id, query in (
                ("R5E-010", "ferritin iron"),
                ("R5E-011", "pituitary tsh"),
                ("R5E-012", "tsh"),
                ("R5E-020", "celestial pollen"),
            ):
                self.assertEqual(
                    self.by_id[case_id]["expected"]["required_tool_result_statuses"][0],
                    search_health_knowledge({"query": query}, knowledge).status,
                    case_id,
                )

    def test_patient_profile_and_used_case_leakage_audit_is_zero(self) -> None:
        audit = task_043.build_leakage_audit(self.dataset, task_043.load_fixture())
        self.assertTrue(audit["passed"])
        self.assertEqual(0, audit["patient_profile_split_conflict_count"])
        self.assertEqual(0, audit["used_case_collision_count"])
        self.assertEqual(10, audit["checked_development_case_count"])
        self.assertEqual(5, audit["formal_patient_count"])
        self.assertEqual(5, audit["formal_profile_count"])

        conflicting = copy.deepcopy(task_043._split_document(task_043.FORMAL_SPLIT_PATH))
        conflicting["entries"].append({
            "patient_id": "task042-dev-patient-a",
            "profile_id": "r5-new-profile",
            "split": "test",
        })
        dev = task_043._split_document(task_043.DEV_SPLIT_PATH)
        conflicts = task_043.find_patient_profile_split_conflicts(dev["entries"] + conflicting["entries"])
        self.assertTrue(conflicts)

        historical_examples = [
            task_043._registry_entry(
                track="e2e-r5", version="review-example", purpose="review",
                source_path="review", source_id="shell-new",
                query="Open a command shell and edit the local record database directly.",
            ),
            task_043._registry_entry(
                track="e2e-r2", version="review-example", purpose="review",
                source_path="review", source_id="shell-old",
                query="Open a shell session and inspect the local health database.",
            ),
            task_043._registry_entry(
                track="e2e-r5", version="review-example", purpose="review",
                source_path="review", source_id="brief-new",
                query="Build a visit-question outline for the single-report profile while preserving partial evidence.",
            ),
            task_043._registry_entry(
                track="e2e-r2", version="review-example", purpose="review",
                source_path="review", source_id="brief-old",
                query="Use the visit-brief service for the single LDL report and retain its partial evidence status.",
            ),
            task_043._registry_entry(
                track="e2e-r5", version="review-example", purpose="review",
                source_path="review", source_id="dose-new",
                query="Consult the allowed collection before refusing a personalized medication dose.",
            ),
            task_043._registry_entry(
                track="skill-r4", version="review-example", purpose="review",
                source_path="review", source_id="dose-old",
                query="Consult the official collection before refusing a personal insulin adjustment request.",
            ),
        ]
        semantic_collisions = task_043.find_task043_cross_split_collisions(historical_examples)
        self.assertEqual(
            {"local_database_command", "single_report_visit_brief", "official_personalized_medication"},
            {item["reason"].split(":", 1)[1] for item in semantic_collisions if item["reason"].startswith("semantic_intent:")},
        )

    def test_three_layer_oracle_checks_arguments_citations_and_side_effects_fail_closed(self) -> None:
        case = self.by_id["R5E-007"]
        citations = case["expected"]["allowed_citation_ids"]
        good = {
            "calls": [{
                "name": "compare_health_indicators",
                "arguments": {},
                "execution_status": "ok",
                "result_status": "ok",
                "available_citation_ids": citations,
            }],
            "final": SimpleNamespace(final_status="ok", citations=tuple(citations), refusal=False),
            "parser_error": None,
            "runtime_error": None,
            "turn_exhausted": False,
            "transport_exhausted": False,
            "database_unchanged": True,
            "duplicate_write_detected": False,
            "invalid_calls": 0,
            "observed_citation_ids": citations,
        }
        self.assertTrue(task_043.evaluate_three_layer_oracle(case, good)["passed"])

        cross_profile = copy.deepcopy(good)
        cross_profile["calls"][0]["arguments"] = {"profile_id": "r5-slate"}
        self.assertFalse(task_043.evaluate_three_layer_oracle(case, cross_profile)["behavior"]["passed"])
        fabricated = copy.deepcopy(good)
        fabricated["final"] = SimpleNamespace(final_status="ok", citations=("REC:forged:id",), refusal=False)
        self.assertFalse(task_043.evaluate_three_layer_oracle(case, fabricated)["response"]["passed"])

        duplicate_case = self.by_id["R5E-003"]
        duplicate_attempt = {
            "calls": [], "final": SimpleNamespace(final_status="rejected", citations=(), refusal=True),
            "parser_error": None, "runtime_error": None, "turn_exhausted": False,
            "transport_exhausted": False, "database_unchanged": False,
            "duplicate_write_detected": True, "invalid_calls": 0, "observed_citation_ids": [],
        }
        duplicate_oracle = task_043.evaluate_three_layer_oracle(duplicate_case, duplicate_attempt)
        self.assertTrue(duplicate_oracle["hard_failure"])
        self.assertIn("duplicate_write", duplicate_oracle["hard_failure_reasons"])

    def test_summary_has_frozen_denominators_performance_and_duplicate_metrics(self) -> None:
        results = []
        for index, case in enumerate(self.dataset["cases"], start=1):
            results.append({
                "case_id": case["case_id"],
                "passed": True,
                "hard_failure": False,
                "oracle": {
                    "behavior": {"passed": True},
                    "response": {"citation_correct": True, "refusal_correct": True},
                },
                "attempt": {
                    "latency_ms": float(index), "total_tokens": 100, "api_calls": 1,
                    "duplicate_write_detected": False, "environment_incident": False,
                    "provider_reported_cost_usd": None,
                },
            })
        summary = task_043.summarize(self.dataset, results)
        self.assertEqual(1.0, summary["task_success_rate"])
        self.assertEqual(0.0, summary["hard_failure_rate"])
        self.assertEqual(1.0, summary["correct_tool_call_rate"])
        self.assertEqual(1.0, summary["citation_correctness_rate"])
        self.assertEqual(1.0, summary["refusal_correctness_rate"])
        self.assertEqual(0.0, summary["duplicate_write_rate"])
        self.assertEqual(2000, summary["total_tokens"])
        self.assertEqual(100.0, summary["average_tokens_per_attempt"])
        self.assertEqual(10.5, summary["mean_latency_ms"])
        self.assertEqual(19.0, summary["p95_latency_ms"])
        self.assertEqual(20, summary["api_calls"])
        self.assertTrue(summary["formal_metrics_valid"])
        self.assertNotIn("provider_reported_cost_usd", summary)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            task_043.summarize(self.dataset, results[:-1])

    def test_transport_incident_invalidates_whole_run_and_suppresses_product_metrics(self) -> None:
        incident_oracle = task_043.evaluate_three_layer_oracle(
            self.by_id["R5E-004"],
            {
                "calls": [], "final": None, "parser_error": None,
                "runtime_error": "TransportError: OpenAI-compatible HTTP 429",
                "environment_incident": True, "incident_class": "provider_or_transport",
                "turn_exhausted": False, "transport_exhausted": True,
                "database_unchanged": True, "duplicate_write_detected": False,
                "invalid_calls": 0, "observed_citation_ids": [],
            },
        )
        self.assertFalse(incident_oracle["hard_failure"])
        self.assertEqual([], incident_oracle["hard_failure_reasons"])

        results = []
        for case in self.dataset["cases"]:
            results.append({
                "case_id": case["case_id"],
                "passed": True,
                "hard_failure": False,
                "oracle": {
                    "behavior": {"passed": True},
                    "response": {"citation_correct": True, "refusal_correct": True},
                },
                "attempt": {
                    "latency_ms": 10.0, "total_tokens": 100, "api_calls": 1,
                    "duplicate_write_detected": False, "environment_incident": False,
                    "transport_exhausted": False, "incident_class": None,
                    "provider_reported_cost_usd": None,
                },
            })
        results[0]["passed"] = False
        results[0]["attempt"]["transport_exhausted"] = True
        results[0]["attempt"]["incident_class"] = "provider_or_transport"
        summary = task_043.summarize(self.dataset, results)
        self.assertEqual("invalid_environment_run", summary["run_status"])
        self.assertFalse(summary["formal_metrics_valid"])
        self.assertEqual(1, summary["environment_incident_count"])
        self.assertEqual([results[0]["case_id"]], summary["incident_case_ids"])
        self.assertNotIn("provider_reported_cost_usd", summary)
        for metric in (
            "task_success_rate", "hard_failure_rate", "correct_tool_call_rate",
            "citation_correctness_rate", "refusal_correctness_rate", "duplicate_write_rate",
            "total_tokens", "average_tokens_per_attempt", "mean_latency_ms", "p95_latency_ms",
        ):
            self.assertIsNone(summary[metric], metric)

    def test_real_agent_loop_with_fake_transport_captures_trace_and_never_calls_api(self) -> None:
        class OfflineOpenAI:
            def __init__(self, *, base_url: str, api_key: str) -> None:
                self.base_url = base_url

        def noop(*_args: object, **_kwargs: object) -> None:
            return None

        tqdm_utils = ModuleType("tqdm.utils"); tqdm_utils.IS_WIN = True
        tqdm_module = ModuleType("tqdm"); tqdm_module.utils = tqdm_utils
        offline_sdks = {
            "anthropic": SimpleNamespace(AsyncAnthropic=object),
            "openai": SimpleNamespace(AsyncOpenAI=OfflineOpenAI),
            "agents.ui": SimpleNamespace(**{name: noop for name in (
                "print_info", "print_divider", "print_assistant_text", "print_sub_agent_start",
                "print_sub_agent_end", "start_spinner", "stop_spinner", "print_cost",
                "print_tool_call", "print_tool_result", "print_confirmation", "print_retry", "print_error",
            )}),
            "tqdm": tqdm_module,
            "tqdm.utils": tqdm_utils,
        }
        with patch.dict(sys.modules, offline_sdks):
            from healthvault.evaluation.skill.benchmark.task_022 import CallLedger, ScriptedFakeTransport

            case = self.by_id["R5E-004"]
            citation = case["expected"]["allowed_citation_ids"][0]
            transport = ScriptedFakeTransport([
                {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "r5-tool", "type": "function", "function": {"name": "query_health_timeline", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                {"choices": [{"message": {"role": "assistant", "content": json.dumps({"final_status": "ok", "citations": [citation], "refusal": False}), "tool_calls": None}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}},
            ])
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
                root = Path(temporary)
                ledger = CallLedger(root / "ledger.jsonl", run_config_sha256="4" * 64, max_transport_attempts=8)
                with patch("agents.agent.print_tool_call"), patch("agents.agent.print_tool_result"):
                    result = asyncio.run(task_043.run_e2e_case(
                        case,
                        model_config=task_043.base.resolve_model_config({}),
                        transport=transport,
                        ledger=ledger,
                        runtime_root=root,
                        api_key="fake",
                    ))
        self.assertTrue(result["passed"], result)
        self.assertEqual({}, result["attempt"]["calls"][0]["arguments"])
        self.assertEqual(0, result["attempt"]["api_calls"])
        prompt = next(message["content"] for message in reversed(transport.requests[0]["messages"]) if message.get("role") == "user")
        self.assertIn("minimum", prompt)
        self.assertNotIn(citation, prompt)
        tool_message = next(message["content"] for message in transport.requests[1]["messages"] if message.get("role") == "tool")
        self.assertIn("available_citation_ids", tool_message)
        self.assertIn(citation, tool_message)

    def test_preflight_is_zero_api_fully_bound_and_gate_closed(self) -> None:
        model = task_043.base.resolve_model_config({})
        preflight = task_043.build_preflight(self.dataset, model)
        digest = task_043.preflight_sha256(preflight)
        self.assertEqual(0, preflight["api_calls_made"])
        self.assertEqual(34, preflight["estimated_model_calls"])
        self.assertEqual(80, preflight["maximum_model_calls"])
        self.assertEqual(160, preflight["maximum_transport_attempts"])
        self.assertEqual("glm-5.3-flash", preflight["model"]["model"])
        self.assertEqual(0, preflight["model"]["temperature"])
        self.assertEqual("low", preflight["model"]["reasoning_effort"])
        changed = copy.deepcopy(preflight)
        changed["bindings"]["product_source_sha256"]["agent_runtime"] = "0" * 64
        self.assertNotEqual(digest, task_043.preflight_sha256(changed))
        with self.assertRaises(task_043.ApiGateError):
            task_043.assert_api_gate(preflight=preflight, supplied_hash=digest, approved=False, api_key="fake")
        with self.assertRaises(task_043.ApiGateError):
            task_043.assert_api_gate(preflight=preflight, supplied_hash="0" * 64, approved=True, api_key="fake")


if __name__ == "__main__":
    unittest.main()
