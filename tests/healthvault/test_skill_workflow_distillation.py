import asyncio
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from healthvault.evaluation.skill.benchmark.task_022 import (
    KEY_ENV,
    ScriptedFakeTransport,
    TransportError,
)
from healthvault.evolution.candidates import CandidateGenerator, TaskEndProcessor
from healthvault.evolution.usage_lifecycle import SkillUsageEvaluator, SkillUsageEvent
from healthvault.evolution.verifier import CandidateVerifier
from healthvault.evolution.workflow_distillation import (
    ACTIVE_SKILL_PATH,
    CANDIDATE_MODEL_ENV,
    CandidateModelSideQuery,
    DEFAULT_TRACE_PATH,
    FAKE_CANDIDATE_CONTENT,
    ApiGateError,
    WorkflowTraceError,
    build_candidate_artifact,
    build_preflight,
    distill_workflows,
    execute_approved_run,
    load_dev_traces,
    resolve_candidate_model_config,
)


SKILL_NAME = "HealthVault-Task-Execution"
ACTIVE_SHA = hashlib.sha256(ACTIVE_SKILL_PATH.read_bytes()).hexdigest()


def usage_event(run_id: str, *, used: bool) -> SkillUsageEvent:
    return SkillUsageEvent(
        run_id=run_id,
        task_id=f"task-{run_id}",
        task_type="synthetic_health_summary",
        skill_name=SKILL_NAME,
        skill_sha256=ACTIVE_SHA,
        retrieved=True,
        injected=True,
        used=used,
        task_status="succeeded",
        validator_status="passed",
        hard_failure=False,
        failure_code=None,
        tool_count=3,
        step_count=4,
        retry_count=0,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        cost_usd=0.0,
    )


class WorkflowDistillationTests(unittest.TestCase):
    def test_candidate_model_is_stronger_and_separate_from_execution_model(self):
        config = resolve_candidate_model_config({
            KEY_ENV: "fake-key",
            "HEALTHVAULT_OPENAI_MODEL": "glm-5.3-flash",
        })

        self.assertEqual("glm-5.3", config["model"])
        self.assertEqual(CANDIDATE_MODEL_ENV, config["model_environment"])
        self.assertFalse(config["do_sample"])
        self.assertEqual("high", config["reasoning_effort"])
        self.assertEqual(4096, config["max_output_tokens"])
        self.assertEqual(
            "glm-5.3-custom",
            resolve_candidate_model_config({
                KEY_ENV: "fake-key",
                CANDIDATE_MODEL_ENV: "glm-5.3-custom",
            })["model"],
        )

    def test_candidate_request_uses_deterministic_high_reasoning_config(self):
        transport = ScriptedFakeTransport([{
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }])
        query = CandidateModelSideQuery(
            transport=transport,
            api_key="fake-key",
            model_config=resolve_candidate_model_config({KEY_ENV: "fake-key"}),
            active_skill_content="active",
        )

        result = asyncio.run(query("{}", {"type": "object"}))
        payload = transport.requests[0]

        self.assertEqual("{}", result)
        self.assertEqual("glm-5.3", payload["model"])
        self.assertFalse(payload["do_sample"])
        self.assertNotIn("temperature", payload)
        self.assertEqual("high", payload["reasoning_effort"])
        self.assertEqual(4096, payload["max_tokens"])

    def test_repeated_success_is_aggregated_and_noise_is_excluded(self):
        result = distill_workflows(load_dev_traces())

        self.assertEqual(10, result.trace_count)
        self.assertEqual(1, len(result.workflow_signals))
        self.assertEqual(2, len(result.recovery_signals))
        self.assertEqual(
            ("TRACE-DEV-001", "TRACE-DEV-002", "TRACE-DEV-003"),
            result.workflow_signals[0].source_trace_ids,
        )
        self.assertEqual("verified_reusable_workflow", result.workflow_signals[0].signal_type)
        self.assertEqual(1, result.one_off_workflow_count)
        self.assertEqual(1, result.duplicate_success_count)
        self.assertEqual(1, result.excluded_patient_fact_count)
        self.assertEqual(1, result.excluded_hard_failure_count)
        self.assertEqual(1, result.excluded_validator_failure_count)

    def test_below_threshold_does_not_trigger_and_schema_is_fail_closed(self):
        dataset = copy.deepcopy(load_dev_traces())
        dataset["traces"] = [
            trace for trace in dataset["traces"] if trace["trace_id"] != "TRACE-DEV-003"
        ]
        result = distill_workflows(dataset)
        self.assertEqual(0, len(result.workflow_signals))

        malformed = load_dev_traces()
        malformed["traces"][0]["unexpected"] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "traces.json"
            path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaises(WorkflowTraceError):
                load_dev_traces(path)

    def test_aggregated_signal_generates_one_verified_fake_candidate(self):
        result = distill_workflows(load_dev_traces())
        captured: list[dict[str, object]] = []

        async def fake_side_query(prompt: str, _schema: dict[str, object]) -> dict[str, object]:
            request = json.loads(prompt)
            captured.append(request)
            trace_ids = list(dict.fromkeys(
                trace_id
                for evidence in request["evidence"]
                for trace_id in evidence["trace_ids"]
            ))
            return {
                "action": "evolve",
                "skill_name": SKILL_NAME,
                "content": FAKE_CANDIDATE_CONTENT,
                "rationale": "Fixed fake candidate for the offline verification path.",
                "trace_ids": trace_ids,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generator = CandidateGenerator(root, fake_side_query)
            processor = TaskEndProcessor(generator)
            self.assertEqual(
                ["eligible", "eligible", "eligible"],
                [processor.record(signal) for signal in result.candidate_signals],
            )
            candidate = asyncio.run(processor.process("evolve"))
            self.assertIsNotNone(candidate)
            assert candidate is not None
            verification = CandidateVerifier(SKILL_NAME).verify_content(
                candidate.content,
                action=candidate.action,
                active_content=ACTIVE_SKILL_PATH.read_text(encoding="utf-8"),
            )
            model_config = {"provider": "zhipu", "model": "glm-5.3-flash"}
            artifact = build_candidate_artifact(
                candidate,
                verification,
                model_config=model_config,
                trace_artifact_sha256=hashlib.sha256(DEFAULT_TRACE_PATH.read_bytes()).hexdigest(),
                generation_mode="fake_side_query",
            )
            persisted = list((root / "candidates").glob("*.json"))

        self.assertEqual(1, generator.call_count)
        self.assertEqual(1, len(captured))
        self.assertEqual(1, len(persisted))
        self.assertEqual("verified", verification.status)
        self.assertEqual(5, len(candidate.trace_ids))
        self.assertEqual(
            {"trace_ids", "signal_type", "lesson"},
            set(captured[0]["evidence"][0]),
        )
        prompt_text = json.dumps(captured[0], ensure_ascii=False, sort_keys=True)
        self.assertNotIn("task_id", prompt_text)
        self.assertNotIn("tool_calls", prompt_text)
        self.assertNotIn("contains_patient_or_medical_fact", prompt_text)
        self.assertEqual("verified", artifact["verification"]["status"])
        self.assertEqual(
            hashlib.sha256(json.dumps(model_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            artifact["model_config_sha256"],
        )

    def test_five_unused_injections_watch_but_observed_use_can_keep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = SkillUsageEvaluator(Path(temp_dir))
            unused = [usage_event(f"unused-{index}", used=False) for index in range(5)]
            unused_result = evaluator.evaluate(
                unused, SKILL_NAME, ACTIVE_SHA, "synthetic_health_summary"
            )
            used = [replace(item, used=index == 0) for index, item in enumerate(unused)]
            used_result = evaluator.evaluate(
                used, SKILL_NAME, ACTIVE_SHA, "synthetic_health_summary"
            )

        self.assertEqual(5, unused_result.effective_samples)
        self.assertEqual(0, unused_result.used_count)
        self.assertEqual("watch", unused_result.decision)
        self.assertEqual(1, used_result.used_count)
        self.assertEqual("keep", used_result.decision)

    def test_default_preflight_is_zero_api_and_gate_is_fail_closed(self):
        active_before = ACTIVE_SKILL_PATH.read_bytes()
        preflight = build_preflight(environ={})
        active_after = ACTIVE_SKILL_PATH.read_bytes()

        self.assertEqual(active_before, active_after)
        self.assertEqual("verified", preflight["fake_candidate"]["verification"])
        self.assertFalse(preflight["fake_candidate"]["published"])
        self.assertFalse(preflight["dev_trace"]["heldout_content_read"])
        self.assertEqual(
            {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            preflight["observed"],
        )
        configured = build_preflight(environ={KEY_ENV: "fake-key"})
        self.assertEqual("glm-5.3-flash", configured["model_config"]["model"])
        self.assertEqual("glm-5.3", configured["candidate_model_config"]["model"])
        transport = ScriptedFakeTransport([])
        blocked = (
            {"execute": False, "approved": True, "api_key": "fake-key", "current": configured},
            {"execute": True, "approved": False, "api_key": "fake-key", "current": configured},
            {"execute": True, "approved": True, "api_key": "", "current": configured},
        )
        for item in blocked:
            with self.assertRaises(ApiGateError):
                asyncio.run(execute_approved_run(
                    configured,
                    execute=item["execute"],
                    approved=item["approved"],
                    api_key=item["api_key"],
                    current_preflight=item["current"],
                    runtime_root="unused",
                    candidate_transport=transport,
                    benchmark_transport=transport,
                    environ={KEY_ENV: "fake-key"},
                ))
        tampered = copy.deepcopy(configured)
        tampered["plan"]["benchmark_arm_cases"] = 61
        with self.assertRaises(ApiGateError):
            asyncio.run(execute_approved_run(
                tampered,
                execute=True,
                approved=True,
                api_key="fake-key",
                current_preflight=configured,
                runtime_root="unused",
                candidate_transport=transport,
                benchmark_transport=transport,
                environ={KEY_ENV: "fake-key"},
            ))
        self.assertEqual([], transport.requests)
        self.assertEqual(0, preflight["observed"]["api_calls"])

    def test_approved_fake_transport_runs_candidate_and_generated_arm_end_to_end(self):
        generated_content = FAKE_CANDIDATE_CONTENT.replace(
            "Use for HealthVault timeline, indicator comparison, knowledge search, and visit-brief tasks.",
            "Use for repeated verified HealthVault workflows and deterministic recovery tasks.",
        )
        candidate_response = {
            "action": "evolve",
            "skill_name": SKILL_NAME,
            "content": generated_content,
            "rationale": "Distill the repeated successful workflow and recovery evidence.",
            "trace_ids": [
                "TRACE-DEV-001", "TRACE-DEV-002", "TRACE-DEV-003",
                "TRACE-DEV-005", "TRACE-DEV-006",
            ],
        }
        candidate_transport = ScriptedFakeTransport([
            TransportError("synthetic transient", retryable=True),
            {
                "choices": [{"message": {"content": json.dumps(candidate_response)}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        ])
        final = json.dumps({
            "final_status": "ok",
            "citations": ["REC:T021-COB-01:hba1c"],
            "refusal": False,
        })
        benchmark_responses = []
        for index in range(3):
            benchmark_responses.extend([
                {
                    "choices": [{"message": {
                        "role": "assistant", "content": None,
                        "tool_calls": [{
                            "id": f"call-{index}", "type": "function",
                            "function": {"name": "query_health_timeline", "arguments": "{}"},
                        }],
                    }}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
                {
                    "choices": [{"message": {"role": "assistant", "content": final}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                },
            ])
        benchmark_transport = ScriptedFakeTransport(benchmark_responses)
        environ = {KEY_ENV: "fake-key"}
        preflight = build_preflight(environ=environ)
        active_before = ACTIVE_SKILL_PATH.read_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact, summary = asyncio.run(execute_approved_run(
                preflight,
                execute=True,
                approved=True,
                api_key="fake-key",
                current_preflight=build_preflight(environ=environ),
                runtime_root=root / "runtime",
                candidate_transport=candidate_transport,
                benchmark_transport=benchmark_transport,
                output_dir=root / "outputs",
                case_order=["SHO-003"],
                environ=environ,
            ))
            matrix = json.loads(
                (root / "outputs" / "task-023-case-matrix.json").read_text(encoding="utf-8")
            )
            injected = (
                root / "runtime" / "benchmark" / "cases" / "generated_candidate"
                / "SHO-003" / "candidate" / ".bear" / "skills"
                / SKILL_NAME / "SKILL.md"
            ).read_text(encoding="utf-8")

        self.assertEqual(active_before, ACTIVE_SKILL_PATH.read_bytes())
        self.assertEqual("frozen", artifact["status"])
        self.assertEqual(generated_content, artifact["content"])
        self.assertEqual(generated_content, injected)
        candidate_payload = candidate_transport.requests[-1]
        candidate_request = json.loads(candidate_payload["messages"][1]["content"])
        self.assertEqual(
            ACTIVE_SKILL_PATH.read_text(encoding="utf-8"),
            candidate_request["active_skill_content"],
        )
        self.assertEqual(
            [
                "When to Use", "Prerequisites", "Steps", "Failure Handling",
                "Verification", "Safety Boundaries",
            ],
            candidate_request["content_contract"]["required_h2_sections"],
        )
        self.assertIn(
            "complete replacement SKILL.md",
            candidate_payload["messages"][0]["content"],
        )
        self.assertEqual(2, summary["candidate_generation"]["transport_attempts"])
        self.assertEqual(1, summary["candidate_generation"]["logical_calls"])
        self.assertEqual(0, summary["api_calls"])
        self.assertEqual(1.0, summary["arms"]["no_skill"]["task_success_rate"])
        self.assertEqual(1.0, summary["arms"]["active_v1"]["task_success_rate"])
        self.assertEqual(1.0, summary["arms"]["generated_candidate"]["task_success_rate"])
        self.assertEqual(3, len(matrix["cases"]))
        self.assertEqual(
            {"no_skill", "active_v1", "generated_candidate"},
            {item["arm_id"] for item in matrix["cases"]},
        )


if __name__ == "__main__":
    unittest.main()
