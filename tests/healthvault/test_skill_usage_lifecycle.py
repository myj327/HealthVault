import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agents.skills import load_project_skill_resources, project_skill_manifest
from healthvault.evolution.synthetic_demo import SKILL_V1, SKILL_V2
from healthvault.evolution.usage_demo import (
    activate_synthetic_v2,
    prepare_synthetic_v1,
    run_usage_demo,
)
from healthvault.evolution.usage_lifecycle import (
    SkillLifecycleManager,
    SkillUsageEvaluator,
    SkillUsageEvent,
)


SKILL_NAME = "HealthVault-Task-Execution"
V1_SHA = hashlib.sha256(SKILL_V1.encode("utf-8")).hexdigest()
V2_SHA = hashlib.sha256(SKILL_V2.encode("utf-8")).hexdigest()


def event(
    run_id: str,
    skill_sha256: str = V1_SHA,
    *,
    task_type: str = "timeline",
    injected: bool = True,
    task_status: str = "succeeded",
    validator_status: str = "passed",
    hard_failure: bool = False,
    failure_code: str | None = None,
) -> SkillUsageEvent:
    return SkillUsageEvent(
        run_id=run_id,
        task_id=f"task-{run_id}",
        task_type=task_type,
        skill_name=SKILL_NAME,
        skill_sha256=skill_sha256,
        retrieved=True,
        injected=injected,
        used=injected,
        task_status=task_status,
        validator_status=validator_status,
        hard_failure=hard_failure,
        failure_code=failure_code,
        tool_count=2,
        step_count=3,
        retry_count=0,
        input_tokens=0,
        output_tokens=0,
        latency_ms=10.0,
        cost_usd=0.0,
    )


class SkillUsageLifecycleTests(unittest.TestCase):
    def test_usage_event_schema_rejects_invalid_hash_counts_and_cost(self):
        with self.assertRaises(ValueError):
            replace(event("bad-hash"), skill_sha256="ABC")
        with self.assertRaises(ValueError):
            replace(event("bad-count"), retry_count=-1)
        with self.assertRaises(ValueError):
            replace(event("bad-cost"), cost_usd=-0.01)

    def test_keep_uses_only_injected_target_events_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = SkillUsageEvaluator(Path(temp_dir))
            events = [event(f"keep-{index}") for index in range(5)]
            events.extend([
                event("retrieved-only", injected=False),
                replace(event("other-skill"), skill_name="Other-Skill"),
                event("other-hash", "f" * 64),
                event("keep-0", task_status="failed", validator_status="failed"),
            ])
            first = evaluator.evaluate(events, SKILL_NAME, V1_SHA, "timeline")
            second = evaluator.evaluate(events, SKILL_NAME, V1_SHA, "timeline")

        self.assertEqual("keep", first.decision)
        self.assertEqual(5, first.effective_samples)
        self.assertEqual(1.0, first.success_rate)
        self.assertEqual(1, first.retrieved_not_injected)
        self.assertEqual(2, first.excluded_other_version)
        self.assertEqual(1, first.duplicate_runs)
        self.assertEqual(10, first.total_tool_count)
        self.assertEqual(15, first.total_step_count)
        self.assertEqual(50.0, first.total_latency_ms)
        self.assertEqual(0, first.total_tokens)
        self.assertEqual(0.0, first.total_cost_usd)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_keep_allows_four_of_five_success_without_hard_or_validator_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = SkillUsageEvaluator(Path(temp_dir))
            events = [event(f"threshold-{index}") for index in range(4)]
            events.append(event(
                "threshold-nonsuccess",
                task_status="failed",
                validator_status="passed",
            ))
            result = evaluator.evaluate(events, SKILL_NAME, V1_SHA, "timeline")

        self.assertEqual(5, result.effective_samples)
        self.assertEqual(4, result.success_count)
        self.assertEqual(0.8, result.success_rate)
        self.assertEqual(0, result.hard_failures)
        self.assertEqual(0, result.validator_failures)
        self.assertEqual("keep", result.decision)

    def test_non_target_same_run_cannot_preempt_target_and_event_ids_include_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evaluator = SkillUsageEvaluator(root)
            events = [
                replace(event("shared-run"), skill_name="Other-Skill"),
                event("shared-run"),
                event("same-run-other-hash", "f" * 64),
                event("same-run-other-hash"),
                event("same-run-other-task", task_type="knowledge"),
                event("same-run-other-task"),
                event("target-3"),
                event("target-4"),
            ]
            result = evaluator.evaluate(events, SKILL_NAME, V1_SHA, "timeline")
            persisted = list((root / "usage-eval" / "events").glob("*.json"))

        self.assertEqual(5, result.effective_samples)
        self.assertEqual(5, result.success_count)
        self.assertEqual("keep", result.decision)
        self.assertEqual(3, result.excluded_other_version)
        self.assertEqual(0, result.duplicate_runs)
        self.assertEqual(8, len(persisted))

    def test_insufficient_and_mixed_windows_watch_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepare_synthetic_v1(root)
            active = root / ".bear" / "skills" / SKILL_NAME / "SKILL.md"
            before = active.read_bytes()
            evaluator = SkillUsageEvaluator(root / ".bear" / "skill-evolution" / "healthvault")
            lifecycle = SkillLifecycleManager(
                root / ".bear" / "skills",
                root / ".bear" / "skill-evolution" / "healthvault",
            )
            insufficient = evaluator.evaluate(
                [event(f"watch-{index}", task_type="knowledge") for index in range(3)],
                SKILL_NAME,
                V1_SHA,
                "knowledge",
            )
            mixed_events = [event(f"mixed-{index}", task_type="brief") for index in range(4)]
            mixed_events.append(event(
                "mixed-failure",
                task_type="brief",
                task_status="failed",
                validator_status="failed",
            ))
            mixed = evaluator.evaluate(mixed_events, SKILL_NAME, V1_SHA, "brief")
            insufficient_result = lifecycle.apply(insufficient)
            mixed_result = lifecycle.apply(mixed)
            after = active.read_bytes()

        self.assertEqual("watch", insufficient.decision)
        self.assertEqual("watch", mixed.decision)
        self.assertEqual(0.8, mixed.success_rate)
        self.assertEqual(("watch",), insufficient_result.actions)
        self.assertEqual(("watch",), mixed_result.actions)
        self.assertEqual(before, after)

    def test_safety_hard_failure_disables_immediately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = SkillUsageEvaluator(Path(temp_dir))
            result = evaluator.evaluate([
                event(
                    "safety-v2",
                    V2_SHA,
                    task_type="visit_brief",
                    task_status="failed",
                    validator_status="failed",
                    hard_failure=True,
                    failure_code="safety_forbidden_tool",
                )
            ], SKILL_NAME, V2_SHA, "visit_brief")

        self.assertEqual("disable", result.decision)
        self.assertEqual(1, result.effective_samples)
        self.assertEqual(1, result.hard_failures)
        self.assertEqual(1, result.validator_failures)

    def test_disable_rolls_back_archives_and_blocks_v2_injection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepare_synthetic_v1(root)
            activate_synthetic_v2(root)
            evolution_root = root / ".bear" / "skill-evolution" / "healthvault"
            evaluator = SkillUsageEvaluator(evolution_root)
            lifecycle = SkillLifecycleManager(root / ".bear" / "skills", evolution_root)
            evaluation = evaluator.evaluate([
                event(
                    "safety-v2",
                    V2_SHA,
                    task_type="visit_brief",
                    task_status="failed",
                    validator_status="failed",
                    hard_failure=True,
                    failure_code="safety_forbidden_tool",
                )
            ], SKILL_NAME, V2_SHA, "visit_brief")
            result = lifecycle.apply(evaluation)
            result_again = lifecycle.apply(evaluation)
            manifest = project_skill_manifest(load_project_skill_resources(
                root / ".bear" / "skills", (SKILL_NAME,)
            ))
            archived = list((evolution_root / "archive").glob("*.json"))
            v2_evidence = evolution_root / "versions" / V2_SHA / "SKILL.md"
            v2_injectable = lifecycle.is_injectable(V2_SHA)
            v1_injectable = lifecycle.is_injectable(V1_SHA)
            v2_evidence_exists = v2_evidence.is_file()

        self.assertEqual(("disable", "rollback", "archive"), result.actions)
        self.assertEqual("already_applied", result_again.status)
        self.assertEqual(V1_SHA, result.final_sha256)
        self.assertEqual(V1_SHA, manifest[0]["sha256"])
        self.assertFalse(v2_injectable)
        self.assertTrue(v1_injectable)
        self.assertEqual(1, len(archived))
        self.assertTrue(v2_evidence_exists)

    def test_fixed_demo_has_three_decisions_zero_api_and_no_candidate_generation(self):
        fixture_path = (
            Path(__file__).parents[2]
            / "healthvault"
            / "evaluation"
            / "skill"
            / "task-020-synthetic-fixture.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "healthvault.evolution.candidates.CandidateGenerator.generate",
            side_effect=AssertionError("usage evaluation must not generate candidates"),
        ) as generate_candidate:
            summary_path = Path(temp_dir) / "summary.json"
            summary = run_usage_demo(
                Path(temp_dir) / "workspace",
                fixture_path=fixture_path,
                summary_path=summary_path,
            )
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual("synthetic", summary["data_classification"])
        self.assertEqual("keep", summary["v1_keep"]["decision"])
        self.assertEqual("watch", summary["v1_watch"]["decision"])
        self.assertEqual("disable", summary["v2_safety"]["decision"])
        self.assertEqual(["disable", "rollback", "archive"], summary["lifecycle"]["actions"])
        self.assertEqual("archived", summary["archive_evidence"]["status"])
        self.assertFalse(summary["archive_evidence"]["physical_delete"])
        self.assertEqual(V1_SHA, summary["final_whitelist_manifest"]["sha256"])
        self.assertEqual(0, summary["api_calls"])
        self.assertEqual(0, summary["tokens"])
        self.assertEqual(0.0, summary["cost_usd"])
        self.assertEqual(0, summary["candidate_generation_calls"])
        self.assertEqual(summary, written)
        generate_candidate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
