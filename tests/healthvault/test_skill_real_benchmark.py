import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.evaluation.skill.benchmark import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MANIFEST_PATH,
    FakeArmRunner,
    load_dataset,
    load_manifest,
    run_benchmark,
    validate_attempt,
)
from healthvault.evaluation.skill.benchmark.task_022 import (
    API_BASE,
    BASE_URL_ENV,
    KEY_ENV,
    MODEL,
    MODEL_ENV,
    PROVIDER,
    REASONING_EFFORT,
    CallLedger,
    ExecutionGateError,
    ResumeStore,
    ScriptedFakeTransport,
    assert_execution_gate,
    build_preflight,
    main,
    run_agent_case,
    summarize_real_attempts,
)


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_SKILL = ROOT / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"


class SkillRealBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DEFAULT_DATASET_PATH)
        cls.task021_manifest = load_manifest(
            DEFAULT_MANIFEST_PATH, cls.dataset, workspace_root=ROOT,
        )

    def test_preflight_binds_zhipu_defaults_and_environment_overrides(self):
        default = build_preflight(workspace_root=ROOT, environ={})
        self.assertEqual(PROVIDER, default["model_config"]["provider"])
        self.assertEqual(MODEL, default["model_config"]["model"])
        self.assertEqual(API_BASE, default["model_config"]["api_base"])
        self.assertEqual(KEY_ENV, default["model_config"]["key_environment"])
        self.assertEqual(
            {"type": "enabled", "clear_thinking": False},
            default["model_config"]["thinking"],
        )
        self.assertEqual(REASONING_EFFORT, default["model_config"]["reasoning_effort"])
        self.assertEqual(2048, default["model_config"]["max_output_tokens"])
        self.assertEqual(0.0, default["model_config"]["temperature"])
        self.assertFalse(default["model_config"]["key_configured"])
        self.assertEqual(60, default["plan"]["arm_cases"])
        self.assertEqual(240, default["limits"]["max_model_calls"])
        self.assertEqual(480, default["limits"]["max_transport_attempts"])
        encoded = json.dumps(default, ensure_ascii=False).lower()
        self.assertNotIn("deepseek", encoded)
        self.assertNotIn("cost_usd", encoded)
        self.assertNotIn("price", encoded)

        overrides = {
            KEY_ENV: "secret-not-for-output",
            BASE_URL_ENV: "https://example.invalid/v4/",
            MODEL_ENV: "glm-test-override",
        }
        overridden = build_preflight(workspace_root=ROOT, environ=overrides)
        self.assertEqual("https://example.invalid/v4", overridden["model_config"]["api_base"])
        self.assertEqual("glm-test-override", overridden["model_config"]["model"])
        self.assertTrue(overridden["model_config"]["key_configured"])
        self.assertNotEqual(default["run_config_sha256"], overridden["run_config_sha256"])
        self.assertNotIn(overrides[KEY_ENV], json.dumps(overridden))

    async def test_fake_transport_uses_glm_payload_and_real_agent_trace(self):
        case = next(item for item in self.dataset["cases"] if item["case_id"] == "SHO-003")
        final = json.dumps({
            "final_status": "ok",
            "citations": case["expected"]["citations"],
            "refusal": False,
        })
        transport = ScriptedFakeTransport([
            {
                "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-1", "type": "function",
                    "function": {"name": "query_health_timeline", "arguments": "{}"},
                }]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "choices": [{"message": {"role": "assistant", "content": final, "tool_calls": None}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            },
        ])
        preflight = build_preflight(workspace_root=ROOT, environ={})
        before = hashlib.sha256(ACTIVE_SKILL.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp_dir:
            attempt = await run_agent_case(
                preflight["arms"][2], case, preflight,
                transport=transport, runtime_root=Path(temp_dir), workspace_root=ROOT,
            )
        self.assertTrue(validate_attempt(case, attempt)["task_success"])
        self.assertEqual(before, hashlib.sha256(ACTIVE_SKILL.read_bytes()).hexdigest())
        self.assertEqual(0, attempt["api_calls"])
        self.assertNotIn("cost_usd", attempt)
        for request in transport.requests:
            self.assertEqual(MODEL, request["model"])
            self.assertEqual(0.0, request["temperature"])
            self.assertEqual(
                {"type": "enabled", "clear_thinking": False}, request["thinking"],
            )
            self.assertEqual("low", request["reasoning_effort"])
            self.assertEqual(2048, request["max_tokens"])
            self.assertFalse(request["stream"])

    async def test_reasoning_content_is_preserved_after_tool_call(self):
        case = next(item for item in self.dataset["cases"] if item["case_id"] == "SHO-003")
        reasoning = "inspect the requested synthetic timeline and preserve citations"
        final = json.dumps({
            "final_status": "ok",
            "citations": case["expected"]["citations"],
            "refusal": False,
        })
        transport = ScriptedFakeTransport([
            {
                "choices": [{"message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": reasoning,
                    "tool_calls": [{
                        "id": "reasoning-call-1", "type": "function",
                        "function": {"name": "query_health_timeline", "arguments": "{}"},
                    }],
                }, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "choices": [{"message": {
                    "role": "assistant", "content": final, "tool_calls": None,
                }, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            },
        ])
        preflight = build_preflight(workspace_root=ROOT, environ={})
        with tempfile.TemporaryDirectory() as temp_dir:
            attempt = await run_agent_case(
                preflight["arms"][0], case, preflight,
                transport=transport, runtime_root=Path(temp_dir), workspace_root=ROOT,
            )
        self.assertTrue(validate_attempt(case, attempt)["task_success"])
        assistant_messages = [
            message for message in transport.requests[1]["messages"]
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        self.assertEqual(1, len(assistant_messages))
        self.assertEqual(reasoning, assistant_messages[0]["reasoning_content"])

    def test_usage_ledger_gate_and_resume_are_config_bound(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = CallLedger(
                Path(temp_dir) / "ledger.jsonl",
                run_config_sha256="a" * 64,
                max_transport_attempts=2,
            )
            first = ledger.begin()
            known = ledger.commit(
                first,
                usage={
                    "prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12,
                    "ignored_provider_field": "ignored",
                },
                latency_ms=2.0, status="ok",
            )
            self.assertEqual((9, 3, 12), (
                known["actual_input_tokens"], known["actual_output_tokens"], known["actual_total_tokens"],
            ))
            second = ledger.begin()
            unknown = ledger.commit(second, usage=None, latency_ms=3.0, status="transport_error")
            self.assertIsNone(unknown["actual_total_tokens"])
            self.assertNotIn("cost_usd", json.dumps(ledger.records))
            with self.assertRaisesRegex(RuntimeError, "attempt"):
                ledger.begin()

            resume_path = Path(temp_dir) / "resume.jsonl"
            ResumeStore(resume_path, "b" * 64).append({
                "arm_id": "active_v1", "case_id": "SHO-001", "complete": True,
            })
            with self.assertRaisesRegex(ValueError, "incompatible"):
                ResumeStore(resume_path, "c" * 64)

        with patch.dict(os.environ, {KEY_ENV: "configured"}, clear=True):
            preflight = build_preflight(workspace_root=ROOT)
            assert_execution_gate(
                preflight, execute=True, approved=True,
                api_key="configured", workspace_root=ROOT,
            )
            with self.assertRaises(ExecutionGateError):
                assert_execution_gate(
                    preflight, execute=False, approved=True,
                    api_key="configured", workspace_root=ROOT,
                )
            os.environ[MODEL_ENV] = "changed-after-preflight"
            with self.assertRaisesRegex(ExecutionGateError, "hash"):
                assert_execution_gate(
                    preflight, execute=True, approved=True,
                    api_key="configured", workspace_root=ROOT,
                )

    def test_summary_uses_active_v1_as_primary_baseline(self):
        preflight = build_preflight(workspace_root=ROOT, environ={})
        attempts = run_benchmark(self.dataset, self.task021_manifest, FakeArmRunner())
        for attempt in attempts:
            attempt["run_config_sha256"] = preflight["run_config_sha256"]
            attempt.pop("cost_usd", None)
            attempt["total_tokens"] = attempt["input_tokens"] + attempt["output_tokens"]
        active = next(item for item in attempts if item["arm_id"] == "active_v1")
        candidate = next(
            item for item in attempts
            if item["arm_id"] == "candidate_v2" and item["case_id"] != active["case_id"]
        )
        active["invalid_calls"] = 1
        candidate["hard_failure"] = True
        summary = summarize_real_attempts(self.dataset, preflight, attempts)
        delta = summary["active_v1_to_candidate_v2"]
        self.assertEqual("active_v1", delta["primary_baseline"])
        self.assertEqual(0.0, delta["task_success_delta_pp"])
        self.assertEqual(5.0, delta["hard_failure_delta_pp"])
        self.assertEqual(1, delta["improved_case_count"])
        self.assertEqual(1 / 20, delta["improved_rate"])
        self.assertEqual(1, delta["regressed_case_count"])
        self.assertEqual(1 / 20, delta["regressed_rate"])
        self.assertEqual("optional_reference", summary["arms"]["no_skill"]["role"])

    def test_default_cli_is_zero_api_preflight(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {}, clear=True):
            output = Path(temp_dir) / "preflight.json"
            self.assertEqual(0, main(["--preflight-output", str(output)]))
            written = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            written["observed"],
        )
        self.assertNotIn("cost", json.dumps(written).lower())


if __name__ == "__main__":
    unittest.main()
