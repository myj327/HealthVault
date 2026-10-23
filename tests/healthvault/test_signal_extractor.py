import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from healthvault.evolution.candidates import CandidateGenerator
from healthvault.evolution.publisher import SkillPublisher
from healthvault.evolution.signal_extractor import (
    SEMANTIC_EXTRACTION_CONFIG,
    ExperienceCard,
    SignalExtractor,
    TaskObservation,
    eligible_cards,
    generate_candidate_once,
    process_persisted_trace_batch,
    process_task_end_batch,
    run_fixed_demo,
)
from healthvault.evolution.synthetic_demo import SKILL_V1, SKILL_V2
from healthvault.evolution.trace import TaskTraceRecorder
from healthvault.evolution.verifier import CandidateVerifier


SKILL_SHA = hashlib.sha256(SKILL_V1.encode("utf-8")).hexdigest()


def observation(
    index: int,
    *,
    task_id: str | None = None,
    feedback: str | None = None,
    user_confirmed: bool = False,
    tool_calls: list[dict[str, str]] | None = None,
    failure_code: str | None = None,
    result_status: str = "succeeded",
    validator_status: str = "passed",
    intent: str = "retrieve a chronological record timeline",
    preconditions: tuple[str, ...] = ("read-only record access",),
) -> TaskObservation:
    return TaskObservation(
        trace={
            "trace_id": f"TRACE-{index}",
            "task_id": task_id or f"TASK-{index}",
            "skill": {"sha256": SKILL_SHA},
            "result_status": result_status,
            "failure_code": failure_code,
            "tool_calls": tool_calls or [
                {"tool_name": "query_health_timeline", "result_status": "ok"},
            ],
        },
        intent=intent,
        preconditions=preconditions,
        expected_outcome="validated timeline result",
        validator_status=validator_status,
        feedback=feedback,
        user_confirmed=user_confirmed,
    )


class SignalExtractorTests(unittest.TestCase):
    def test_persisted_runtime_traces_are_processed_by_manual_batch_once(self):
        async def fake_tool(_name, _input, _state):
            return json.dumps({"status": "ok", "validator_code": "citation_valid"})

        async def fake_candidate(prompt, _schema):
            trace_ids = [
                trace_id for evidence in json.loads(prompt)["evidence"]
                for trace_id in evidence["trace_ids"]
            ]
            return {
                "action": "evolve", "skill_name": "HealthVault-Task-Execution",
                "content": SKILL_V2, "rationale": "persisted trace workflow",
                "trace_ids": trace_ids,
            }

        context = {
            "intent": "retrieve a chronological record timeline",
            "preconditions": ["read-only record access"],
            "expected_outcome": "validated timeline result",
            "validator_status": "passed",
            "contract_checks": {"tool": "passed", "citation": "passed"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for index in range(1, 4):
                recorder = TaskTraceRecorder(
                    root / "evolution", request_id=f"request-{index}", task_id=f"TASK-{index}",
                    skill={"name": "HealthVault-Task-Execution", "sha256": SKILL_SHA},
                    executor=fake_tool, signal_context=context,
                )
                asyncio.run(recorder.execute("query_health_timeline", {}))
                paths.append(recorder.finish("succeeded"))
            generator = CandidateGenerator(root / "evolution", fake_candidate)
            report, result = asyncio.run(process_persisted_trace_batch(
                paths, feedback_by_trace_id={}, extractor=SignalExtractor(), generator=generator,
                action="evolve", verifier=CandidateVerifier("HealthVault-Task-Execution"),
                active_content=SKILL_V1,
            ))

        self.assertEqual(1, len(report.cards))
        self.assertEqual(0, report.semantic_extractor_calls)
        self.assertEqual("pending_manual_review", result.status)
        self.assertEqual(1, result.generator_calls)
        self.assertEqual({"TASK-1", "TASK-2", "TASK-3"}, set(report.cards[0].task_ids))

    def test_deterministic_retry_recovery_makes_a_strong_card(self):
        recovered = observation(1, tool_calls=[
            {"tool_name": "query_health_timeline", "result_status": "retryable_failure"},
            {"tool_name": "query_health_timeline", "result_status": "ok"},
        ])
        report = asyncio.run(SignalExtractor().extract([recovered]))

        self.assertEqual(1, len(report.cards))
        card = report.cards[0]
        self.assertEqual("deterministic", card.verification_source)
        self.assertEqual("strong", card.verification_strength)
        self.assertEqual(("TRACE-1",), card.trace_ids)
        self.assertIn("retryable", card.lesson)
        self.assertEqual((card,), eligible_cards(report.cards))

    def test_three_distinct_successes_form_one_verified_pattern(self):
        report = asyncio.run(SignalExtractor().extract([
            observation(1), observation(2), observation(3),
        ]))

        self.assertEqual(1, len(report.cards))
        card = report.cards[0]
        self.assertEqual("strong", card.verification_strength)
        self.assertEqual(3, card.evidence_count)
        self.assertEqual(("TASK-1", "TASK-2", "TASK-3"), card.task_ids)
        self.assertEqual(("TRACE-1", "TRACE-2", "TRACE-3"), card.trace_ids)

    def test_medium_strength_needs_three_distinct_tasks_and_weak_never_promotes(self):
        def card(index: int, strength: str) -> ExperienceCard:
            return ExperienceCard(
                card_id=f"CARD-{index}-{strength}", intent="route a record timeline",
                preconditions=("read-only",), lesson="Use the relevant read-only timeline tool.",
                tool_path=("query_health_timeline:ok",),
                expected_outcome="validated timeline result", when_not_to_use=("different intent",),
                trace_ids=(f"TRACE-{index}",), task_ids=(f"TASK-{index}",),
                skill_sha256=SKILL_SHA, evidence_count=1,
                verification_source="semantic_judge" if strength == "medium" else "deterministic",
                verification_strength=strength, reusable=True,
            )

        self.assertEqual((), eligible_cards([card(1, "medium"), card(2, "medium")]))
        promoted = eligible_cards([card(1, "medium"), card(2, "medium"), card(3, "medium")])
        self.assertEqual(1, len(promoted))
        self.assertEqual(3, promoted[0].evidence_count)
        self.assertEqual((), eligible_cards([card(1, "weak"), card(2, "weak"), card(3, "weak")]))

    def test_patient_facts_and_engineering_defects_are_not_skill_evidence(self):
        patient = observation(1, intent="explain patient glucose 9.1")
        defect = observation(2, failure_code="environment_unicodeencode_gbk")
        report = asyncio.run(SignalExtractor().extract([patient, defect]))

        self.assertEqual((), report.cards)
        self.assertEqual(("TRACE-1",), report.ignored_trace_ids)
        self.assertEqual(("TRACE-2:engineering_defect",), report.engineering_issues)

    def test_failed_corrections_use_one_strict_semantic_batch_and_do_not_publish(self):
        calls: list[tuple[dict, dict, dict]] = []

        async def fake_semantic(payload, schema, config):
            calls.append((payload, schema, config))
            return {"corrections": [{
                "trace_id": "TRACE-1", "intent": "route a record timeline",
                "preconditions": ["read-only record access"],
                "lesson": "Use the timeline tool only for timeline requests.",
                "when_not_to_use": ["comparison requests"], "reusable": True,
            }]}

        async def should_not_generate(_prompt, _schema):
            raise AssertionError("a single medium correction must not generate a candidate")

        failed = observation(
            1, feedback="This is wrong; use the timeline tool instead.",
            result_status="rejected", validator_status="failed",
            tool_calls=[{"tool_name": "query_health_timeline", "result_status": "rejected"}],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_path = Path(temp_dir) / "SKILL.md"
            skill_path.write_bytes(SKILL_V1.encode("utf-8"))
            before = skill_path.read_bytes()
            generator = CandidateGenerator(Path(temp_dir) / "evolution", should_not_generate)
            report, result = asyncio.run(process_task_end_batch(
                [failed], extractor=SignalExtractor(fake_semantic), generator=generator,
                action="evolve", verifier=CandidateVerifier("HealthVault-Task-Execution"),
                active_content=SKILL_V1,
            ))
            after = skill_path.read_bytes()

        self.assertEqual(1, report.semantic_extractor_calls)
        self.assertEqual(1, len(calls))
        self.assertEqual(1, len(calls[0][0]["observations"]))
        self.assertEqual(SEMANTIC_EXTRACTION_CONFIG, calls[0][2])
        self.assertEqual("semantic_judge", report.cards[0].verification_source)
        self.assertEqual("medium", report.cards[0].verification_strength)
        self.assertEqual("no_eligible_cards", result.status)
        self.assertEqual(0, result.generator_calls)
        self.assertEqual(before, after)

    def test_one_eligible_batch_generates_one_candidate_and_never_publishes(self):
        report = asyncio.run(SignalExtractor().extract([
            observation(1), observation(2), observation(3),
        ]))
        captured: list[dict] = []

        async def fake_candidate(prompt, _schema):
            payload = json.loads(prompt)
            captured.append(payload)
            trace_ids = [item for evidence in payload["evidence"] for item in evidence["trace_ids"]]
            return {
                "action": "evolve", "skill_name": "HealthVault-Task-Execution",
                "content": SKILL_V2, "rationale": "verified workflow",
                "trace_ids": trace_ids,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_path = root / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(SKILL_V1, encoding="utf-8")
            before = skill_path.read_bytes()
            generator = CandidateGenerator(root / "evolution", fake_candidate)
            result = asyncio.run(generate_candidate_once(
                report, generator=generator, action="evolve",
                verifier=CandidateVerifier("HealthVault-Task-Execution"), active_content=SKILL_V1,
            ))
            publisher = SkillPublisher(root / ".bear" / "skills", root / "evolution")
            unapproved = publisher.publish_candidate(
                result.candidate, verification=result.verification, approved=False,
            )
            after = skill_path.read_bytes()

        self.assertEqual(1, result.generator_calls)
        self.assertEqual(1, len(captured))
        self.assertEqual("pending_manual_review", result.status)
        self.assertEqual("verified", result.verification.status)
        self.assertEqual({"TRACE-1", "TRACE-2", "TRACE-3"}, set(result.candidate.trace_ids))
        self.assertEqual("approval_required", unapproved.status)
        self.assertEqual(before, after)

    def test_fixed_demo_is_zero_api_and_stops_at_manual_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = asyncio.run(run_fixed_demo(temp_dir))

        self.assertEqual("pending_manual_review", summary["status"])
        self.assertEqual("verified", summary["verification"])
        self.assertEqual(1, summary["generator_calls"])
        self.assertEqual(1, summary["semantic_extractor_calls"])
        self.assertEqual(2, len(summary["experience_cards"]))
        self.assertTrue(summary["active_skill_unchanged"])
        self.assertFalse(summary["published"])
        self.assertEqual(0, summary["api_calls"])


if __name__ == "__main__":
    unittest.main()
