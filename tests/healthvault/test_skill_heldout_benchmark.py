import copy
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from healthvault.evaluation.skill.benchmark import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MANIFEST_PATH,
    FakeArmRunner,
    build_manifest,
    load_dataset,
    load_database_fixture,
    load_manifest,
    materialize_database_fixture,
    run_benchmark,
    summarize_attempts,
    validate_attempt,
    validate_attempts,
)
from healthvault.evaluation.used_cases import validate_used_case_registry


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_SKILL = ROOT / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"
LIFECYCLE_FIXTURE = ROOT / "healthvault" / "evaluation" / "skill" / "task-020-synthetic-fixture.json"


class SkillHeldOutBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DEFAULT_DATASET_PATH)
        cls.manifest = load_manifest(DEFAULT_MANIFEST_PATH, cls.dataset, workspace_root=ROOT)

    def test_dataset_freezes_twenty_unique_balanced_synthetic_cases(self):
        cases = self.dataset["cases"]
        self.assertEqual("skill-heldout-r1-v1", self.dataset["dataset_version"])
        self.assertEqual(20, len(cases))
        self.assertEqual(20, len({case["case_id"] for case in cases}))
        self.assertEqual(20, len({case["case_family_id"] for case in cases}))
        self.assertEqual(
            {"timeline": 5, "indicator_comparison": 5, "knowledge_retrieval": 5, "visit_brief": 5},
            {task_type: sum(case["task_type"] == task_type for case in cases)
             for task_type in {case["task_type"] for case in cases}},
        )
        self.assertTrue(all(case["fixture"]["data_classification"] == "synthetic" for case in cases))
        self.assertTrue(all(case["expected"]["prohibited_writes"] for case in cases))
        self.assertTrue(all(case["validator"]["kind"] == "deterministic_exact_v1" for case in cases))

    def test_used_case_registry_is_valid_and_cross_track_collision_is_rejected(self):
        registry = validate_used_case_registry(self.dataset["used_case_registry"])
        benchmark_entries = [entry for entry in registry["entries"] if entry["track"] == "skill_benchmark"]
        self.assertEqual(20, len(benchmark_entries))
        inventory_entries = [entry for entry in registry["entries"] if entry["track"] in {"agent_e2e", "long_fault"}]
        self.assertGreaterEqual(len(inventory_entries), 7)
        self.assertTrue(all("HealthVaultAgentRuntimeTests.test_" in entry["upstream_id"] for entry in inventory_entries))
        mutated = copy.deepcopy(self.dataset)
        benchmark = next(entry for entry in mutated["used_case_registry"]["entries"] if entry["track"] == "skill_benchmark")
        prior = next(entry for entry in mutated["used_case_registry"]["entries"] if entry["track"] != "skill_benchmark")
        benchmark["case_family_id"] = prior["case_family_id"]
        with self.assertRaisesRegex(ValueError, "held-out collision"):
            load_dataset(mutated)
        stale = copy.deepcopy(self.dataset)
        rag_query = next(entry["query"] for entry in stale["used_case_registry"]["entries"] if entry["track"] == "rag")
        stale["cases"][0]["prompt"] = rag_query
        with self.assertRaisesRegex(ValueError, "held-out collision"):
            load_dataset(stale)

    def test_manifest_binds_three_isolated_arms_and_all_shared_inputs(self):
        self.assertEqual(["no_skill", "active_v1", "candidate_v2"], [arm["arm_id"] for arm in self.manifest["arms"]])
        self.assertEqual(3, len({arm["skill_sha256"] for arm in self.manifest["arms"]}))
        self.assertEqual([case["case_id"] for case in self.dataset["cases"]], self.manifest["shared_config"]["case_order"])
        validator_source = inspect.getsource(validate_attempt).encode("utf-8")
        self.assertEqual(hashlib.sha256(validator_source).hexdigest(), self.manifest["shared_config"]["validator_sha256"])
        broken = copy.deepcopy(self.manifest)
        broken["arms"].pop()
        with self.assertRaisesRegex(ValueError, "three benchmark arms"):
            load_manifest(broken, self.dataset, workspace_root=ROOT)
        mutated_dataset = copy.deepcopy(self.dataset)
        mutated_dataset["cases"][0]["prompt"] += " changed"
        with self.assertRaisesRegex(ValueError, "dataset hash"):
            load_manifest(self.manifest, mutated_dataset, workspace_root=ROOT)

    def test_fake_smoke_runs_exactly_sixty_bound_attempts(self):
        attempts = run_benchmark(self.dataset, self.manifest, FakeArmRunner())
        self.assertEqual(60, len(attempts))
        validate_attempts(self.dataset, self.manifest, attempts)
        with self.assertRaisesRegex(ValueError, "duplicate attempt"):
            validate_attempts(self.dataset, self.manifest, attempts + [attempts[0]])
        with self.assertRaisesRegex(ValueError, "missing attempt"):
            validate_attempts(self.dataset, self.manifest, attempts[:-1])
        broken = copy.deepcopy(attempts)
        broken[0]["run_config_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "run config"):
            validate_attempts(self.dataset, self.manifest, broken)

    def test_deterministic_validator_blocks_each_required_failure_boundary(self):
        case = self.dataset["cases"][0]
        valid = FakeArmRunner()(self.manifest["arms"][0], case, self.manifest)
        self.assertTrue(validate_attempt(case, valid)["task_success"])
        mutations = []
        wrong_tool = copy.deepcopy(valid); wrong_tool["tool_calls"][0]["name"] = "write_file"; mutations.append(wrong_tool)
        invalid_citation = copy.deepcopy(valid); invalid_citation["citations"] = ["forged-citation"]; mutations.append(invalid_citation)
        wrong_refusal = copy.deepcopy(valid); wrong_refusal["refusal"] = not case["expected"]["refusal_required"]; mutations.append(wrong_refusal)
        hard_failure = copy.deepcopy(valid); hard_failure["hard_failure"] = True; mutations.append(hard_failure)
        write_attempt = copy.deepcopy(valid); write_attempt["write_attempts"] = ["calendar_events"]; mutations.append(write_attempt)
        for attempt in mutations:
            self.assertFalse(validate_attempt(case, attempt)["task_success"])
        unresolved_retry = copy.deepcopy(valid)
        unresolved_retry["tool_calls"][-1]["status"] = "retryable_failure"
        self.assertFalse(validate_attempt(case, unresolved_retry)["task_success"])
        order_case = copy.deepcopy(case)
        order_case["expected"]["tool_set"] = ["query_health_timeline", "compare_health_indicators"]
        order_case["expected"]["tool_order"] = ["query_health_timeline", "compare_health_indicators"]
        wrong_order = FakeArmRunner()(self.manifest["arms"][0], order_case, self.manifest)
        wrong_order["tool_calls"].reverse()
        self.assertFalse(validate_attempt(order_case, wrong_order)["task_success"])

    def test_scripted_metrics_and_regression_are_exactly_recomputable(self):
        attempts = run_benchmark(self.dataset, self.manifest, FakeArmRunner())
        attempts[0]["invalid_calls"] = 1
        attempts[0]["steps"] = 9
        attempts[21]["hard_failure"] = True
        summary = summarize_attempts(self.dataset, self.manifest, attempts)
        self.assertEqual(19 / 20, summary["arms"]["no_skill"]["task_success_rate"])
        self.assertEqual(1, summary["arms"]["no_skill"]["invalid_calls"])
        self.assertEqual(19 / 20, summary["arms"]["active_v1"]["task_success_rate"])
        self.assertEqual(1 / 20, summary["arms"]["active_v1"]["hard_failure_rate"])
        self.assertEqual(1, summary["regression_vs_baseline"]["active_v1"]["regressed_cases"])
        self.assertEqual(1 / 20, summary["regression_vs_baseline"]["active_v1"]["regression_rate"])
        self.assertEqual(0, summary["regression_vs_baseline"]["candidate_v2"]["regressed_cases"])
        self.assertEqual(0.0, summary["regression_vs_baseline"]["candidate_v2"]["regression_rate"])

    def test_database_fixture_materializes_and_all_gold_citations_resolve(self):
        fixture = load_database_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "task-021.sqlite3"
            result = materialize_database_fixture(database_path)
            self.assertEqual(fixture["report_count"], result["report_count"])
            self.assertTrue(database_path.is_file())
        bad = copy.deepcopy(self.dataset)
        knowledge = next(case for case in bad["cases"] if case["task_type"] == "knowledge_retrieval" and case["expected"]["citations"])
        knowledge["expected"]["citations"] = ["HVK:r1-2026-08-24-v1:forged:000:000000000000"]
        knowledge["fixture"]["evidence"] = list(knowledge["expected"]["citations"])
        with self.assertRaisesRegex(ValueError, "citation does not resolve"):
            load_dataset(bad)

    def test_smoke_is_zero_api_not_claimable_and_does_not_mutate_skill_state(self):
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (ACTIVE_SKILL, LIFECYCLE_FIXTURE)}
        summary = summarize_attempts(
            self.dataset,
            self.manifest,
            run_benchmark(self.dataset, self.manifest, FakeArmRunner()),
        )
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (ACTIVE_SKILL, LIFECYCLE_FIXTURE)}
        self.assertEqual(before, after)
        self.assertEqual("mock_smoke", summary["run_kind"])
        self.assertEqual("not_claimable", summary["claim_status"])
        self.assertEqual(0, summary["api_calls"])
        self.assertEqual(0, summary["tokens"])
        self.assertEqual(0.0, summary["cost_usd"])
        self.assertEqual(0, summary["candidate_generation_calls"])
        self.assertEqual(0, summary["judge_calls"])


if __name__ == "__main__":
    unittest.main()
