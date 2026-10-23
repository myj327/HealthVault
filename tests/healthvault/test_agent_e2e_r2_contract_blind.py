"""Focused offline checks for TASK-032's untouched R2 contract blind test."""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.evaluation.e2e import task_032


class Task032ContractBlindTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = task_032.load_dataset()

    def test_dataset_is_balanced_and_separate_from_all_frozen_tracks(self) -> None:
        self.assertEqual(20, len(self.dataset["cases"]))
        self.assertEqual(task_032.CATEGORY_COUNTS, dict(__import__("collections").Counter(
            case["category"] for case in self.dataset["cases"]
        )))
        self.assertEqual([], task_032.find_held_out_collisions(self.dataset["used_case_registry"]["entries"]))
        self.assertEqual(4, len(self.dataset["leakage_sources"]))

    def test_independent_fixture_materializes_new_reports_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "r2.sqlite3"
            result = task_032.materialize_fixture(database)
            self.assertTrue(database.is_file())
        self.assertEqual("agent-e2e-r2-fixture-v1", result["snapshot_id"])
        self.assertEqual(7, result["report_count"])
        fixture_text = task_032.FIXTURE_PATH.read_text(encoding="utf-8")
        self.assertIn("R32-NOV-01", fixture_text)
        self.assertNotIn("R24-ATL-01", fixture_text)

    def test_oracles_are_derived_from_fixture_tool_state_and_policy(self) -> None:
        by_id = {case["case_id"]: case for case in self.dataset["cases"]}
        self.assertEqual(["ok"], by_id["R2E-005"]["expected"]["required_tool_result_statuses"])
        self.assertEqual(["insufficient_data"], by_id["R2E-007"]["expected"]["required_tool_result_statuses"])
        self.assertEqual(["partial"], by_id["R2E-016"]["expected"]["required_tool_result_statuses"])
        self.assertEqual([], by_id["R2E-003"]["expected"]["allowed_citation_ids"])
        self.assertEqual("rejected", by_id["R2E-017"]["expected"]["final_status"])
        self.assertIn("REC:R32-NOV-01:fasting_glucose", by_id["R2E-001"]["expected"]["allowed_citation_ids"])

    def test_manifest_binds_same_runtime_prompt_model_and_only_contract_view_varies(self) -> None:
        manifest = task_032.build_run_manifest(self.dataset, task_032.v1.resolve_model_config({}))
        arms = manifest["arms"]
        self.assertTrue(arms["fixed_alternating_by_case"])
        self.assertFalse(arms["legacy_contract"]["top_level_available_citation_ids"])
        self.assertTrue(arms["canonical_contract"]["top_level_available_citation_ids"])
        self.assertEqual("glm-5.3-flash", manifest["model"]["model"])
        self.assertEqual(0, manifest["model"]["temperature"])
        self.assertEqual("low", manifest["model"]["reasoning_effort"])

    def test_legacy_adapter_removes_only_the_top_level_contract_field(self) -> None:
        payload = {"status": "ok", "available_citation_ids": ["REC:R32-NOV-01:fasting_glucose"], "timeline": [{"citation_id": "REC:R32-NOV-01:fasting_glucose"}]}
        legacy = task_032._adapt_payload(payload, "legacy_contract")
        canonical = task_032._adapt_payload(payload, "canonical_contract")
        self.assertNotIn("available_citation_ids", legacy)
        self.assertEqual(payload["timeline"], legacy["timeline"])
        self.assertEqual(payload, canonical)

    def test_real_agent_loop_fake_transport_sees_the_two_contract_views(self) -> None:
        from healthvault.evaluation.skill.benchmark.task_022 import CallLedger, ScriptedFakeTransport

        case = next(item for item in self.dataset["cases"] if item["case_id"] == "R2E-001")
        citation = case["expected"]["allowed_citation_ids"][0]
        tool_messages: dict[str, str] = {}
        prompts: list[str] = []
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            ledger = CallLedger(root / "ledger.jsonl", run_config_sha256="2" * 64, max_transport_attempts=8)
            for arm_id in task_032.ARM_IDS:
                transport = ScriptedFakeTransport([
                    {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": arm_id, "type": "function", "function": {"name": "query_health_timeline", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                    {"choices": [{"message": {"role": "assistant", "content": json.dumps({"final_status": "ok", "citations": [citation], "refusal": False}), "tool_calls": None}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}},
                ])
                with patch("agents.agent.print_tool_call"), patch("agents.agent.print_tool_result"):
                    result = asyncio.run(task_032.run_e2e_case(case, arm_id=arm_id, model_config=task_032.v1.resolve_model_config({}), transport=transport, ledger=ledger, runtime_root=root, api_key="fake"))
                self.assertTrue(result["passed"], result)
                prompts.append(next(message["content"] for message in reversed(transport.requests[0]["messages"]) if message.get("role") == "user"))
                tool_messages[arm_id] = next(message["content"] for message in transport.requests[1]["messages"] if message.get("role") == "tool")
        self.assertEqual(prompts[0], prompts[1])
        self.assertNotIn("available_citation_ids", tool_messages["legacy_contract"])
        self.assertIn("available_citation_ids", tool_messages["canonical_contract"])
        self.assertIn(citation, tool_messages["canonical_contract"])

    def test_summary_recomputes_per_arm_metrics_and_delta(self) -> None:
        results = []
        for arm_id in task_032.ARM_IDS:
            for case in self.dataset["cases"]:
                results.append({"arm_id": arm_id, "case_id": case["case_id"], "passed": True, "hard_failure": False, "oracle": {"behavior": {"passed": True}, "response": {"citation_correct": True}}, "attempt": {"api_calls": 0}})
        summary = task_032.summarize(self.dataset, results)
        self.assertEqual(20, summary["arms"]["legacy_contract"]["attempts"])
        self.assertEqual(0.0, summary["canonical_minus_legacy"]["task_success_rate"])

    def test_preflight_is_zero_api_and_hash_binds_arm_contract(self) -> None:
        model = task_032.v1.resolve_model_config({})
        preflight = task_032.build_preflight(self.dataset, model)
        digest = task_032.preflight_sha256(preflight)
        self.assertEqual(0, preflight["api_calls_made"])
        self.assertEqual(74, preflight["estimated_model_calls"])
        self.assertEqual(160, preflight["maximum_model_calls"])
        changed = copy.deepcopy(preflight)
        changed["bindings"]["arms_sha256"] = "0" * 64
        self.assertNotEqual(digest, task_032.preflight_sha256(changed))
        with self.assertRaises(task_032.ApiGateError):
            task_032.assert_api_gate(preflight=preflight, supplied_hash=digest, approved=False, api_key="fake")


if __name__ == "__main__":
    unittest.main()
