"""Focused offline checks for TASK-036's generic citation policy and R3 runner."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch

from healthvault.application.citation_policy import actual_available_citation_ids
from healthvault.evaluation.e2e import task_036
from healthvault.prompt import build_healthvault_system_prompt


class Task036Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = task_036.load_dataset()

    def test_product_policy_accepts_only_top_level_canonical_ids_and_deduplicates(self) -> None:
        payload = {
            "available_citation_ids": ["REC:R36-AMB-01:fasting_glucose", "REC:R36-AMB-01:fasting_glucose", "HVK:r1:source:001:abc", "bad-id"],
            "nested": {"citation_id": "REC:forged:nested"},
        }
        self.assertEqual(["REC:R36-AMB-01:fasting_glucose", "HVK:r1:source:001:abc"], actual_available_citation_ids(payload))
        self.assertEqual([], actual_available_citation_ids({"nested": payload}))
        prompt = build_healthvault_system_prompt()
        self.assertIn("top-level available_citation_ids", prompt)
        self.assertIn("REC:* only", prompt)
        self.assertIn("official evidence before a safety refusal", prompt)

    def test_r3_is_balanced_resettable_and_collision_free_without_changing_r2(self) -> None:
        r2_before = task_036.r2.DATASET_PATH.read_bytes()
        self.assertEqual(20, len(self.dataset["cases"]))
        self.assertEqual(task_036.CATEGORY_COUNTS, dict(Counter(case["category"] for case in self.dataset["cases"])))
        self.assertEqual([], task_036.find_held_out_collisions(self.dataset["used_case_registry"]["entries"]))
        with tempfile.TemporaryDirectory() as temporary:
            outcome = task_036.materialize_fixture(Path(temporary) / "r3.sqlite3")
        self.assertEqual({"snapshot_id": "agent-e2e-r3-fixture-v1", "report_count": 7}, outcome)
        self.assertTrue(all("R36-" not in item for item in r2_before.decode("utf-8").split()))
        self.assertEqual(r2_before, task_036.r2.DATASET_PATH.read_bytes())

    def test_oracle_is_fixture_and_status_derived_for_official_unsafe_read_then_refuse(self) -> None:
        by_id = {case["case_id"]: case for case in self.dataset["cases"]}
        unsafe = by_id["R3E-011"]["expected"]
        self.assertEqual(["search_health_knowledge"], unsafe["required_tools"])
        self.assertEqual(["empty"], unsafe["required_tool_result_statuses"])
        self.assertEqual("rejected", unsafe["final_status"])
        self.assertEqual([], unsafe["allowed_citation_ids"])
        self.assertEqual(["partial"], by_id["R3E-016"]["expected"]["required_tool_result_statuses"])

    def test_real_agent_loop_fake_transport_receives_prompt_and_top_level_ids(self) -> None:
        class OfflineOpenAI:
            def __init__(self, *, base_url: str, api_key: str) -> None:
                self.base_url = base_url

        def noop(*_args: object, **_kwargs: object) -> None:
            return None

        tqdm_utils = ModuleType("tqdm.utils")
        tqdm_utils.IS_WIN = True
        tqdm_module = ModuleType("tqdm")
        tqdm_module.utils = tqdm_utils
        offline_sdks = {
            "anthropic": SimpleNamespace(AsyncAnthropic=object),
            "openai": SimpleNamespace(AsyncOpenAI=OfflineOpenAI),
            "agents.ui": SimpleNamespace(**{
                name: noop for name in (
                    "print_info", "print_divider", "print_assistant_text", "print_sub_agent_start",
                    "print_sub_agent_end", "start_spinner", "stop_spinner", "print_cost",
                    "print_tool_call", "print_tool_result", "print_confirmation", "print_retry",
                    "print_error",
                )
            }),
            "tqdm": tqdm_module,
            "tqdm.utils": tqdm_utils,
        }
        with patch.dict(sys.modules, offline_sdks):
            from healthvault.evaluation.skill.benchmark.task_022 import CallLedger, ScriptedFakeTransport
            case = next(item for item in self.dataset["cases"] if item["case_id"] == "R3E-001")
            citation = case["expected"]["allowed_citation_ids"][0]
            transport = ScriptedFakeTransport([
                {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "r3-tool", "type": "function", "function": {"name": "query_health_timeline", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                {"choices": [{"message": {"role": "assistant", "content": json.dumps({"final_status": "ok", "citations": [citation], "refusal": False}), "tool_calls": None}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}},
            ])
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
                root = Path(temporary); ledger = CallLedger(root / "ledger.jsonl", run_config_sha256="3" * 64, max_transport_attempts=8)
                with patch("agents.agent.print_tool_call"), patch("agents.agent.print_tool_result"):
                    result = asyncio.run(task_036.run_e2e_case(case, model_config=task_036.v1.resolve_model_config({}), transport=transport, ledger=ledger, runtime_root=root, api_key="fake"))
        self.assertTrue(result["passed"], result)
        prompt = next(message["content"] for message in reversed(transport.requests[0]["messages"]) if message.get("role") == "user")
        self.assertIn("minimum sufficient evidence", prompt)
        self.assertNotIn(citation, prompt)
        tool_message = next(message["content"] for message in transport.requests[1]["messages"] if message.get("role") == "tool")
        self.assertIn("available_citation_ids", tool_message)
        self.assertIn(citation, tool_message)
        self.assertEqual(0, result["attempt"]["api_calls"])

    def test_preflight_is_zero_api_and_binds_generic_policy_dependencies(self) -> None:
        model = task_036.v1.resolve_model_config({}); preflight = task_036.build_preflight(self.dataset, model); digest = task_036.preflight_sha256(preflight)
        self.assertEqual(0, preflight["api_calls_made"])
        self.assertEqual(38, preflight["estimated_model_calls"])
        self.assertEqual(80, preflight["maximum_model_calls"])
        self.assertEqual("glm-5.3-flash", preflight["model"]["model"])
        self.assertEqual(0, preflight["model"]["temperature"])
        self.assertEqual("low", preflight["model"]["reasoning_effort"])
        for binding in ("citation_policy_sha256", "product_prompt_sha256", "oracle_sha256"):
            changed = copy.deepcopy(preflight); changed["bindings"][binding] = "0" * 64
            self.assertNotEqual(digest, task_036.preflight_sha256(changed), binding)
        with patch.object(task_036.v1, "_git_provenance", side_effect=[
            {"commit": "fixed", "dirty": True, "dirty_paths_count": 25},
            {"commit": "fixed", "dirty": True, "dirty_paths_count": 999},
        ]):
            first, second = task_036.build_preflight(self.dataset, model), task_036.build_preflight(self.dataset, model)
        self.assertEqual(task_036.preflight_sha256(first), task_036.preflight_sha256(second))
        self.assertEqual("unbound; task-source hashes are authoritative", first["git"]["working_tree_state"])
        task_036.assert_api_gate(preflight=preflight, supplied_hash=digest, approved=True, api_key="fake")
        with self.assertRaises(task_036.ApiGateError):
            task_036.assert_api_gate(preflight=preflight, supplied_hash=digest, approved=False, api_key="fake")


if __name__ == "__main__":
    unittest.main()
