"""Focused offline checks for TASK-034's canonical absolute-metrics runner."""

from __future__ import annotations

import asyncio
import copy
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console

from healthvault.evaluation.e2e import task_034


class Task034AbsoluteMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = task_034.load_dataset()

    def test_single_arm_manifest_reuses_the_frozen_r2_bindings(self) -> None:
        before = task_034.DATASET_PATH.read_bytes()
        manifest = task_034.build_run_manifest(self.dataset, task_034.r2.v1.resolve_model_config({}))
        self.assertEqual(before, task_034.DATASET_PATH.read_bytes())
        self.assertEqual("canonical_contract", manifest["arm"]["arm_id"])
        self.assertTrue(manifest["arm"]["top_level_available_citation_ids"])
        self.assertNotIn("arms", manifest)
        self.assertEqual(20, len(self.dataset["cases"]))
        self.assertEqual(
            task_034.r2._sha256_file(task_034.TASK032_RUNNER_PATH),
            manifest["delegated_source_sha256"]["task_032_runner"],
        )
        self.assertEqual(
            task_034.r2._sha256_file(task_034.TASK030_ORACLE_PATH),
            manifest["delegated_source_sha256"]["task_030_oracle"],
        )

    def test_case_runner_delegates_once_to_task032_canonical_view(self) -> None:
        case = self.dataset["cases"][0]
        fake = {"case_id": case["case_id"], "arm_id": "canonical_contract"}
        with patch.object(task_034.r2, "run_e2e_case", new=AsyncMock(return_value=fake)) as delegated:
            result = asyncio.run(task_034.run_e2e_case(case, model_config={}, transport=object(), ledger=object(), runtime_root=Path(".")))
        self.assertEqual(fake, result)
        self.assertEqual("canonical_contract", delegated.await_args.kwargs["arm_id"])

    def test_summary_recomputes_refusal_denominator_tokens_and_latency(self) -> None:
        refusal_ids = {case["case_id"] for case in self.dataset["cases"] if case["expected"]["refusal_required"]}
        results = []
        for index, case in enumerate(self.dataset["cases"]):
            refusal_ok = case["case_id"] not in refusal_ids or case["case_id"] != sorted(refusal_ids)[0]
            results.append({
                "case_id": case["case_id"], "passed": True, "hard_failure": False,
                "oracle": {"behavior": {"passed": True}, "response": {"citation_correct": True, "refusal_correct": refusal_ok}},
                "attempt": {"latency_ms": float(index + 1), "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "api_calls": 1},
            })
        summary = task_034.summarize(self.dataset, results)
        self.assertEqual(len(refusal_ids), summary["refusal_case_count"])
        self.assertEqual((len(refusal_ids) - 1) / len(refusal_ids), summary["refusal_correctness_rate"])
        self.assertEqual(300, summary["total_tokens"])
        self.assertEqual(15, summary["average_tokens_per_attempt"])
        self.assertEqual(19.0, summary["p95_latency_ms"])

    def test_preflight_is_zero_api_hash_bound_and_single_arm_sized(self) -> None:
        model = task_034.r2.v1.resolve_model_config({})
        preflight = task_034.build_preflight(self.dataset, model)
        digest = task_034.preflight_sha256(preflight)
        self.assertEqual(0, preflight["api_calls_made"])
        self.assertEqual(37, preflight["estimated_model_calls"])
        self.assertEqual(80, preflight["maximum_model_calls"])
        self.assertEqual(
            task_034.sha256_bytes(task_034.canonical_json_bytes(task_034.CONSOLE_OUTPUT_BINDING)),
            preflight["bindings"]["console_output_sha256"],
        )
        changed = copy.deepcopy(preflight)
        changed["bindings"]["canonical_arm_sha256"] = "0" * 64
        self.assertNotEqual(digest, task_034.preflight_sha256(changed))
        for dependency in ("task_032_runner_sha256", "task_030_oracle_sha256"):
            changed = copy.deepcopy(preflight)
            changed["bindings"][dependency] = "0" * 64
            self.assertNotEqual(digest, task_034.preflight_sha256(changed), dependency)
        with self.assertRaises(task_034.ApiGateError):
            task_034.assert_api_gate(preflight=preflight, supplied_hash=digest, approved=False, api_key="fake")

    def test_utf8_console_prevents_gbk_tool_rendering_crash(self) -> None:
        from agents import ui

        buffer = io.BytesIO()
        gbk_stream = io.TextIOWrapper(buffer, encoding="gbk", errors="strict")
        with patch.object(sys, "stdout", gbk_stream), patch.object(sys, "stderr", gbk_stream):
            task_034.configure_utf8_console_output()
            self.assertEqual("utf-8", gbk_stream.encoding.casefold())
            with patch.object(ui, "console", Console(file=gbk_stream, force_terminal=False, color_system=None, width=80)):
                ui.print_tool_call("query_health_timeline", {})
            gbk_stream.flush()
        self.assertIn(b"query_health_timeline", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
