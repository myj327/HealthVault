from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.evaluation.skill.benchmark.task_024 import (
    ARM_IDS,
    DATASET_PATH,
    FAKE_CANDIDATE_V2,
    INTENTS,
    MAX_MODEL_TURNS,
    ROOT,
    ApiGateError,
    CandidateVerificationError,
    _fake_candidate,
    assert_api_gate,
    build_preflight,
    generate_real_candidate_once,
    load_r2_dataset,
    load_r2_fixture,
    load_degraded_candidate,
    materialize_r2_fixture,
    run_r2_agent_case,
    validate_protocol_attempt,
)
from healthvault.evaluation.skill.benchmark.task_022 import (
    BASE_URL_ENV,
    KEY_ENV,
    MODEL_ENV,
    CallLedger,
    ResumeStore,
    ScriptedFakeTransport,
)
from healthvault.evaluation.used_cases import normalized_query_sha256
from healthvault.evolution.verifier import CandidateVerifier


class Task024CandidateV2BenchmarkTests(unittest.TestCase):
    def test_new_heldout_and_fixture_are_strict_and_isolated(self) -> None:
        dataset = load_r2_dataset()
        fixture = load_r2_fixture()
        self.assertEqual(20, len(dataset["cases"]))
        self.assertEqual({intent: 4 for intent in INTENTS}, {
            intent: sum(case["intent"] == intent for case in dataset["cases"])
            for intent in INTENTS
        })
        self.assertEqual(10, fixture["report_count"])
        self.assertEqual(
            "4c0aadb0c91d93204258535ee2c7bdac896a94f8449c5e20a91f4773cde0461e",
            load_degraded_candidate()["candidate_content_sha256"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            loaded = materialize_r2_fixture(Path(temporary) / "task024.sqlite3")
            self.assertEqual(10, loaded["report_count"])
        old = json.loads((ROOT / "healthvault/evaluation/skill/task-021-heldout-dataset.json").read_text(encoding="utf-8"))
        old_hashes = {normalized_query_sha256(case["prompt"]) for case in old["cases"]}
        self.assertFalse(old_hashes & {normalized_query_sha256(case["prompt"]) for case in dataset["cases"]})
        with tempfile.TemporaryDirectory() as temporary:
            bad = copy.deepcopy(dataset)
            bad["cases"][0]["unexpected"] = True
            path = Path(temporary) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact schema"):
                load_r2_dataset(path)

    def test_protocol_validator_enforces_observed_citations_and_refusal(self) -> None:
        case = load_r2_dataset()["cases"][0]
        allowed = case["expected"]["allowed_citation_ids"]
        valid = {
            "tool_calls": [{
                "name": "query_health_timeline",
                "status": "ok",
                "available_citation_ids": allowed,
            }],
            "final_status": "ok",
            "citations": [allowed[0]],
            "refusal": False,
            "model_turns": 2,
            "hard_failure": False,
            "invalid_calls": 0,
            "complete": True,
        }
        self.assertTrue(validate_protocol_attempt(case, valid)["task_success"])
        forged = copy.deepcopy(valid); forged["citations"] = ["REC:FORGED:value"]
        self.assertFalse(validate_protocol_attempt(case, forged)["citation_correct"])
        wrong_refusal = copy.deepcopy(valid); wrong_refusal["refusal"] = True
        self.assertFalse(validate_protocol_attempt(case, wrong_refusal)["task_success"])

    def test_protocol_validator_enforces_exact_tools_call_limit_and_four_turns(self) -> None:
        case = load_r2_dataset()["cases"][0]
        attempt = {
            "tool_calls": [
                {"name": "query_health_timeline", "status": "ok", "available_citation_ids": case["expected"]["allowed_citation_ids"]},
                {"name": "preview_health_report", "status": "ok", "available_citation_ids": []},
            ],
            "final_status": "ok", "citations": [case["expected"]["allowed_citation_ids"][0]],
            "refusal": False, "model_turns": MAX_MODEL_TURNS + 1,
            "hard_failure": False, "invalid_calls": 0, "complete": True,
        }
        result = validate_protocol_attempt(case, attempt)
        self.assertFalse(result["tool_policy_correct"])
        self.assertTrue(result["hard_failure"])
        self.assertFalse(result["task_success"])

    def test_candidate_verifier_requires_five_routes_and_rejects_unconditional_chain(self) -> None:
        active = (ROOT / ".bear/skills/HealthVault-Task-Execution/SKILL.md").read_text(encoding="utf-8")
        verifier = CandidateVerifier("HealthVault-Task-Execution")
        valid = verifier.verify_content(FAKE_CANDIDATE_V2, action="evolve", active_content=active, policy="task024_v2")
        self.assertEqual("verified", valid.status)
        unsafe = FAKE_CANDIDATE_V2.replace(
            "1. Timeline: call `query_health_timeline` only.",
            "1. Always for every task call query_health_timeline, compare_health_indicators, generate_visit_brief.",
        )
        rejected = verifier.verify_content(unsafe, action="evolve", active_content=active, policy="task024_v2")
        self.assertIn("unconditional_multi_tool_chain", rejected.reasons)
        sneaky = FAKE_CANDIDATE_V2.replace(
            "1. Timeline: call `query_health_timeline` only.",
            "1. Timeline: call `query_health_timeline`, `compare_health_indicators`, and `generate_visit_brief`.",
        )
        rejected_route = verifier.verify_content(
            sneaky, action="evolve", active_content=active, policy="task024_v2"
        )
        self.assertIn("route_tool_mismatch:timeline", rejected_route.reasons)

    def test_candidate_generator_uses_dev_evidence_without_heldout_leakage(self) -> None:
        smoke, captured = asyncio.run(_fake_candidate(ROOT))
        dataset = load_r2_dataset()
        self.assertEqual(1, smoke["candidate_generator_calls"])
        self.assertEqual("verified", smoke["verification_status"])
        self.assertTrue(smoke["support_trace_ids"])
        for case in dataset["cases"]:
            self.assertNotIn(case["case_id"], captured)
            self.assertNotIn(case["prompt"], captured)
        self.assertNotIn("task-023-case-matrix", captured)
        self.assertNotIn("task-023-real-summary", captured)

    def test_preflight_is_zero_api_two_arm_and_gate_fails_closed(self) -> None:
        preflight = build_preflight(environ={})
        self.assertEqual(list(ARM_IDS), preflight["benchmark"]["arms"])
        self.assertEqual(40, preflight["benchmark"]["arm_cases"])
        self.assertEqual(4, preflight["limits"]["max_model_turns_per_case"])
        self.assertEqual(92, preflight["limits"]["expected_model_calls"])
        self.assertEqual(160, preflight["limits"]["max_model_calls"])
        self.assertEqual(320, preflight["limits"]["max_transport_attempts"])
        self.assertEqual(0, preflight["observed"]["api_calls"])
        self.assertFalse(preflight["heldout_isolation"]["task_023_twelve_round_experiment_loaded"])
        for execute, approved, key, digest in (
            (False, False, "", ""),
            (True, False, "key", preflight["run_config_sha256"]),
            (True, True, "", preflight["run_config_sha256"]),
            (True, True, "key", "wrong"),
        ):
            with self.assertRaises(ApiGateError):
                assert_api_gate(preflight, execute=execute, approved=approved, api_key=key, expected_hash=digest)

    def test_default_preflight_reads_process_environment(self) -> None:
        values = {
            KEY_ENV: "synthetic-test-key",
            BASE_URL_ENV: "https://open.bigmodel.cn/api/paas/v4",
            MODEL_ENV: "glm-5.3-flash",
        }
        with patch.dict(os.environ, values, clear=False):
            preflight = build_preflight()
        self.assertTrue(preflight["model_config"]["key_configured"])
        self.assertEqual("glm-5.3-flash", preflight["model_config"]["model"])
        self.assertEqual(
            "https://open.bigmodel.cn/api/paas/v4",
            preflight["model_config"]["api_base"],
        )

    def test_full_fake_agent_tool_final_validator_and_interrupted_sqlite_resume(self) -> None:
        case = load_r2_dataset()["cases"][0]
        citation = case["expected"]["allowed_citation_ids"][0]
        transport = ScriptedFakeTransport([
            {
                "choices": [{"message": {
                    "role": "assistant", "content": None,
                    "reasoning_content": "Use the minimal timeline route.",
                    "tool_calls": [{
                        "id": "task024-call-1", "type": "function",
                        "function": {"name": "query_health_timeline", "arguments": "{}"},
                    }],
                }, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "choices": [{"message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "final_status": "ok", "citations": [citation], "refusal": False,
                    }),
                    "tool_calls": None,
                }, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            },
        ])
        preflight = build_preflight(environ={})
        digest = "d" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_root = root / "cases" / "active_v1" / case["case_id"]
            case_root.mkdir(parents=True)
            database = case_root / "healthvault.sqlite3"
            database.write_bytes(b"interrupted-sqlite")
            Path(f"{database}-wal").write_bytes(b"interrupted-wal")
            Path(f"{database}-shm").write_bytes(b"interrupted-shm")
            resume = ResumeStore(root / "completed-attempts.jsonl", digest)
            self.assertFalse(resume.is_complete("active_v1", case["case_id"]))
            ledger = CallLedger(root / "call-ledger.jsonl", run_config_sha256=digest, max_transport_attempts=8)
            attempt = asyncio.run(run_r2_agent_case(
                arm_id="active_v1",
                skill_sha256=preflight["frozen_inputs"]["active_skill_sha256"],
                skill_content="",
                case=case,
                execution_config_sha256=digest,
                dataset_sha256=preflight["frozen_inputs"]["dataset_semantic_sha256"],
                model_config=preflight["model_config"],
                transport=transport,
                ledger=ledger,
                api_key="fake",
                runtime_root=root,
            ))
            resume.append(attempt)
            self.assertTrue(resume.is_complete("active_v1", case["case_id"]))
        self.assertTrue(attempt["validation"]["task_success"])
        self.assertEqual(["query_health_timeline"], [item["name"] for item in attempt["tool_calls"]])
        self.assertEqual([citation], attempt["citations"])
        self.assertEqual(0, attempt["api_calls"])

    def test_rejected_real_candidate_is_frozen_and_never_redrawn_across_resume(self) -> None:
        preflight = build_preflight(environ={})
        rejected_content = load_degraded_candidate()["content"]
        response = json.dumps({
            "action": "evolve",
            "skill_name": "HealthVault-Task-Execution",
            "content": rejected_content,
            "rationale": "Synthetic rejected candidate for one-shot freeze testing.",
            "trace_ids": ["TRACE-DEV-001"],
        })
        transport = ScriptedFakeTransport([{
            "choices": [{"message": {
                "role": "assistant", "content": response, "tool_calls": None,
            }, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }])
        with tempfile.TemporaryDirectory() as temporary:
            runtime_root = Path(temporary)
            with self.assertRaises(CandidateVerificationError):
                asyncio.run(generate_real_candidate_once(
                    preflight, transport=transport, api_key="fake", runtime_root=runtime_root
                ))
            artifact = json.loads((runtime_root / "generated-candidate-v2.json").read_text(encoding="utf-8"))
            self.assertEqual("frozen_rejected_not_published", artifact["status"])
            self.assertEqual(preflight["run_config_sha256"], artifact["approved_preflight_sha256"])
            self.assertEqual("rejected", artifact["verification"]["status"])
            self.assertEqual(1, len(transport.requests))
            candidate_request = transport.requests[0]["messages"][1]["content"]
            self.assertIn("exact_route_declaration_templates", candidate_request)

            no_redraw = ScriptedFakeTransport([])
            with self.assertRaises(CandidateVerificationError):
                asyncio.run(generate_real_candidate_once(
                    preflight, transport=no_redraw, api_key="fake", runtime_root=runtime_root
                ))
            self.assertEqual([], no_redraw.requests)

            changed = copy.deepcopy(preflight)
            changed["run_config_sha256"] = "e" * 64
            with self.assertRaisesRegex(ValueError, "preflight hash"):
                asyncio.run(generate_real_candidate_once(
                    changed, transport=no_redraw, api_key="fake", runtime_root=runtime_root
                ))
            self.assertEqual([], no_redraw.requests)


if __name__ == "__main__":
    unittest.main()
