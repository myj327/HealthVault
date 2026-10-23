"""Focused offline checks for TASK-030's independent e2e benchmark."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from healthvault.evaluation.e2e import task_030


def _final(status: str, citations: list[str], refusal: bool) -> SimpleNamespace:
    return SimpleNamespace(final_status=status, citations=tuple(citations), refusal=refusal)


def _valid_attempt(case: dict[str, object]) -> dict[str, object]:
    expected = case["expected"]  # type: ignore[index]
    required = expected["required_tools"]  # type: ignore[index]
    execution = expected["required_tool_statuses"]  # type: ignore[index]
    result = expected["required_tool_result_statuses"]  # type: ignore[index]
    citations = list(expected["allowed_citation_ids"][: expected["min_citations"]])  # type: ignore[index]
    return {
        "calls": [
            {
                "name": name,
                "execution_status": status,
                "result_status": result_status,
                "available_citation_ids": citations,
            }
            for name, status, result_status in zip(required, execution, result)
        ],
        "final": _final(expected["final_status"], citations, expected["refusal_required"]),  # type: ignore[index]
        "parser_error": None,
        "runtime_error": None,
        "turn_exhausted": False,
        "transport_exhausted": False,
        "database_unchanged": True,
        "invalid_calls": 0,
        "observed_citation_ids": citations,
    }


class Task030OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = task_030.load_dataset()

    def test_dataset_is_balanced_and_cross_track_clean(self) -> None:
        self.assertEqual(len(self.dataset["cases"]), 20)
        self.assertEqual(
            dict(__import__("collections").Counter(case["category"] for case in self.dataset["cases"])),
            task_030.EXPECTED_CATEGORY_COUNTS,
        )
        self.assertEqual(
            task_030.find_held_out_collisions(self.dataset["used_case_registry"]["entries"]), []
        )

    def test_three_layer_oracle_is_independently_sensitive(self) -> None:
        case = next(case for case in self.dataset["cases"] if case["case_id"] == "E2E-001")
        good = _valid_attempt(case)
        self.assertTrue(task_030.evaluate_three_layer_oracle(case, good)["passed"])

        outcome_bad = copy.deepcopy(good)
        outcome_bad["calls"][0]["result_status"] = "empty"
        verdict = task_030.evaluate_three_layer_oracle(case, outcome_bad)
        self.assertFalse(verdict["outcome"]["passed"])
        self.assertTrue(verdict["behavior"]["passed"])

        behavior_bad = copy.deepcopy(good)
        behavior_bad["calls"][0]["name"] = "search_health_knowledge"
        verdict = task_030.evaluate_three_layer_oracle(case, behavior_bad)
        self.assertFalse(verdict["behavior"]["passed"])
        self.assertIn("unauthorized_tool", verdict["hard_failure_reasons"])

        response_bad = copy.deepcopy(good)
        response_bad["final"] = _final("ok", ["REC:not-observed"], False)
        verdict = task_030.evaluate_three_layer_oracle(case, response_bad)
        self.assertFalse(verdict["response"]["passed"])
        self.assertTrue(verdict["outcome"]["passed"])

        data_bad = copy.deepcopy(good)
        data_bad["database_unchanged"] = False
        verdict = task_030.evaluate_three_layer_oracle(case, data_bad)
        self.assertFalse(verdict["outcome"]["passed"])
        self.assertIn("destructive_data_change", verdict["hard_failure_reasons"])

    def test_permission_refusal_and_unauthorized_tool_are_distinct(self) -> None:
        case = next(case for case in self.dataset["cases"] if case["case_id"] == "E2E-017")
        good = _valid_attempt(case)
        verdict = task_030.evaluate_three_layer_oracle(case, good)
        self.assertTrue(verdict["passed"])
        self.assertFalse(verdict["hard_failure"])

        bad = copy.deepcopy(good)
        bad["calls"] = [{"name": "query_health_timeline", "execution_status": "ok", "result_status": "ok"}]
        bad["invalid_calls"] = 1
        verdict = task_030.evaluate_three_layer_oracle(case, bad)
        self.assertTrue(verdict["hard_failure"])
        self.assertIn("unauthorized_tool", verdict["hard_failure_reasons"])

    def test_preflight_is_zero_api_and_gate_is_hash_bound(self) -> None:
        model = task_030.resolve_model_config({})
        preflight = task_030.build_preflight(self.dataset, model)
        digest = task_030.preflight_sha256(preflight)
        self.assertEqual(preflight["api_calls_made"], 0)
        self.assertEqual(preflight["estimated_model_calls"], 37)
        self.assertEqual(preflight["maximum_model_calls"], 80)
        with self.assertRaises(task_030.ApiGateError):
            task_030.assert_api_gate(preflight=preflight, supplied_hash=digest, approved=False, api_key="key")
        with self.assertRaises(task_030.ApiGateError):
            task_030.assert_api_gate(preflight=preflight, supplied_hash="wrong", approved=True, api_key="key")

    def test_preflight_hash_binds_all_non_secret_request_configuration(self) -> None:
        model = task_030.resolve_model_config({})
        baseline = task_030.preflight_sha256(task_030.build_preflight(self.dataset, model))
        mutations = {
            "protocol": "alternative-compatible-protocol",
            "api_base": "https://example.invalid/v1",
            "thinking": {"type": "disabled", "clear_thinking": False},
            "reasoning_effort": "medium",
        }
        for field, replacement in mutations.items():
            changed = copy.deepcopy(model)
            changed[field] = replacement
            changed_hash = task_030.preflight_sha256(task_030.build_preflight(self.dataset, changed))
            self.assertNotEqual(baseline, changed_hash, field)

    def test_frozen_fixture_materializes_to_the_requested_database_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "fixture.sqlite3"
            result = task_030.materialize_e2e_fixture(database_path)

            self.assertTrue(database_path.is_file())
            self.assertEqual(result["snapshot_id"], "task-024-synthetic-db-r2-v1")


if __name__ == "__main__":
    unittest.main()
