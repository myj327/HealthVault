import json
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from healthvault.evaluation.rag.deepseek import (
    DEEPSEEK_GENERATION_CONFIG,
    DEEPSEEK_JUDGE_CONFIG,
    DeepSeekChatCompletionsClient,
)
from healthvault.evaluation.rag.generation import (
    GenerationJudgeRunner,
    MockModelClient,
    ModelConfig,
    ModelResponse,
)
from healthvault.evaluation.rag.generation_run import (
    DEFAULT_RETRIEVAL_RESULT_PATH,
    capture_implementation_provenance,
    execute_generation_run,
    prepare_generation_preflight,
)
from healthvault.evaluation.rag.dataset import DEFAULT_DATASET_PATH
from healthvault.evaluation.rag.run_control import (
    AttemptLedger,
    BudgetExceededError,
    DEFAULT_REAL_RUN_LIMITS,
    RunLimits,
    build_run_manifest,
)
from healthvault.evaluation.rag.__main__ import main as rag_eval_main


class CapturingTransport:
    def __init__(self):
        self.requests = []

    def post_json(self, url, headers, payload, timeout_seconds):
        self.requests.append((url, dict(headers), payload, timeout_seconds))
        return {
            "choices": [{"message": {"content": '{"answer":"ok"}'}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }, 7


class EndToEndFakeTransport:
    def __init__(self):
        self.requests = []

    def post_json(self, url, headers, payload, timeout_seconds):
        del headers, timeout_seconds
        self.requests.append((url, payload))
        if payload["model"] == "deepseek-v4-flash":
            citations = re.findall(r"\[(HVK:[^\]]+)\]", payload["messages"][0]["content"])
            content = json.dumps(
                {
                    "status": "answered" if citations else "refused",
                    "answer": "Synthetic evidence-bound answer." if citations else "Insufficient evidence.",
                    "claims": ([{"claim_id": "C1", "text": "Synthetic evidence-bound answer.", "citation_ids": citations[:1]}] if citations else []),
                    "refusal_reason": None if citations else "The supplied evidence is insufficient.",
                }
            )
        else:
            content = json.dumps(
                {"faithfulness": 4, "relevance": 4, "completeness": 4,
                 "expression": 4, "reason": "synthetic offline judge",
                 "evidence_citation_ids": []}
            )
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        }, 1


def config(model, max_tokens=128):
    return ModelConfig(
        provider="deepseek",
        model=model,
        api_format="openai-chat-completions",
        endpoint_path="/chat/completions",
        thinking_type="disabled",
        response_format="json_object",
        temperature=0.0,
        max_output_tokens=max_tokens,
        input_cost_per_million=0.14 if model.endswith("flash") else 0.435,
        output_cost_per_million=0.28 if model.endswith("flash") else 0.87,
    )


def synthetic_provenance(source_hash="1" * 64):
    return {
        "schema_version": 1,
        "current_head": "320254658709b05e285d7d547e52881cccb0c679",
        "worktree_dirty": True,
        "evaluation_source_sha256": source_hash,
        "evaluation_source_files": {
            "healthvault/evaluation/rag/generation.py": source_hash,
        },
        "prompt_sha256": "2" * 64,
        "validator_sha256": "3" * 64,
    }


class DeepSeekApiGateTests(unittest.TestCase):
    def test_formal_preflight_and_fake_transport_execute_both_strategies_and_resume(self):
        citation = "HVK:r1-2026-08-24-v1:niddk-diabetes-tests:003:9a52fd667ccc"
        queries = tuple(
            SimpleNamespace(query_id=f"Q{index:02d}", split="test")
            for index in range(40)
        )
        dataset = SimpleNamespace(
            eval_version="rag-eval-r1-v2",
            dataset_sha256="a" * 64,
            frozen_test_sha256="b" * 64,
            corpus_sha256="c" * 64,
            artifact_sha256={"natural_section": "d" * 64, "hybrid_v1": "e" * 64},
            aliases_sha256="f" * 64,
            queries=queries,
        )
        cases = tuple(
            {
                "query_id": query.query_id, "query": "synthetic query", "split": "test",
                "answerable": True, "reference_claims": [],
                "expected_citations": [citation],
            }
            for query in queries
        )
        contexts = {
            query.query_id: [{"citation_id": citation, "body": "synthetic evidence"}]
            for query in queries
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "healthvault.evaluation.rag.generation_run._load_bound_inputs",
                return_value=(dataset, {}),
            ), patch(
                "healthvault.evaluation.rag.generation_run.prepare_test_generation_inputs",
                return_value=(cases, contexts),
            ):
                plan = prepare_generation_preflight(
                    dataset_path=DEFAULT_DATASET_PATH,
                    retrieval_result_path=DEFAULT_RETRIEVAL_RESULT_PATH,
                    output_dir=root,
                    implementation_provenance=synthetic_provenance(),
                    key_configured=False,
                    created_at="2026-08-26T13:00:00+08:00",
                )
                self.assertEqual("not_configured", plan["api_key_status"])
                self.assertEqual(80, plan["logical_generation_requests"])
                self.assertEqual(80, plan["logical_judge_requests"])
                self.assertEqual(0, plan["api_calls_executed"])
                self.assertTrue((root / "generation-run-manifest.json").exists())
                self.assertTrue((root / "generation-preflight.json").exists())
                transport = EndToEndFakeTransport()
                first = execute_generation_run(
                    dataset_path=DEFAULT_DATASET_PATH,
                    retrieval_result_path=DEFAULT_RETRIEVAL_RESULT_PATH,
                    output_dir=root,
                    implementation_provenance=synthetic_provenance(),
                    api_key="synthetic-not-a-real-key",
                    transport=transport,
                    created_at="2026-08-26T13:00:00+08:00",
                )
                self.assertEqual(160, len(transport.requests))
                self.assertEqual(40, first["strategies"]["natural_section"]["records"])
                self.assertEqual(40, first["strategies"]["hybrid_v1"]["records"])
                self.assertEqual(160, first["ledger"]["attempts"])
                self.assertTrue((root / "natural_section-results.jsonl").exists())
                self.assertTrue((root / "hybrid_v1-results.jsonl").exists())
                second = execute_generation_run(
                    dataset_path=DEFAULT_DATASET_PATH,
                    retrieval_result_path=DEFAULT_RETRIEVAL_RESULT_PATH,
                    output_dir=root,
                    implementation_provenance=synthetic_provenance(),
                    api_key="synthetic-not-a-real-key",
                    transport=transport,
                    created_at="2026-08-26T13:00:00+08:00",
                )
                self.assertEqual(first, second)
                self.assertEqual(160, len(transport.requests))

    def test_generation_cli_defaults_to_preflight_and_never_executes(self):
        plan = {
            "api_key_status": "not_configured",
            "maximum_attempts_with_retries": 320,
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "healthvault.evaluation.rag.__main__.prepare_generation_preflight",
            return_value=plan,
        ) as preflight, patch(
            "healthvault.evaluation.rag.__main__.execute_generation_run"
        ) as execute:
            exit_code = rag_eval_main(["generation", "--output", directory])
        self.assertEqual(0, exit_code)
        preflight.assert_called_once()
        execute.assert_not_called()

    def test_current_implementation_provenance_is_deterministic_and_bound_to_git(self):
        first = capture_implementation_provenance()
        second = capture_implementation_provenance()
        self.assertEqual(first, second)
        self.assertRegex(first["current_head"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(first["worktree_dirty"], bool)
        for field in (
            "evaluation_source_sha256",
            "prompt_sha256",
            "validator_sha256",
        ):
            self.assertRegex(first[field], r"^[0-9a-f]{64}$")
        self.assertIn(
            "healthvault/evaluation/rag/generation.py",
            first["evaluation_source_files"],
        )

    def test_manifest_rejects_resume_after_evaluation_source_changes(self):
        limits = RunLimits(1, 1_000, 256, 1.0)
        common = dict(
            eval_version="rag-eval-r1-v2",
            dataset_sha256="a" * 64,
            frozen_test_sha256="b" * 64,
            corpus_sha256="c" * 64,
            artifact_sha256={"natural_section": "d" * 64, "hybrid_v1": "e" * 64},
            aliases_sha256="f" * 64,
            query_ids=("q1",),
            generation_config=config("deepseek-v4-flash"),
            judge_config=config("deepseek-v4-pro"),
            max_retries=1,
            limits=limits,
            created_at="2026-08-27T12:00:00+08:00",
        )
        first = build_run_manifest(
            **common, implementation_provenance=synthetic_provenance("1" * 64)
        )
        changed = build_run_manifest(
            **common, implementation_provenance=synthetic_provenance("4" * 64)
        )
        self.assertNotEqual(first["run_config_sha256"], changed["run_config_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            from healthvault.evaluation.rag.run_control import ensure_run_manifest

            ensure_run_manifest(manifest_path, first)
            with self.assertRaisesRegex(ValueError, "incompatible run manifest"):
                ensure_run_manifest(manifest_path, changed)

    def test_preflight_cost_plan_is_prompt_derived_and_has_three_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = prepare_generation_preflight(
                output_dir=directory,
                key_configured=False,
                created_at="2026-08-27T12:00:00+08:00",
                implementation_provenance=synthetic_provenance(),
            )
        pricing = plan["pricing_snapshot"]
        self.assertEqual("deepseek", pricing["provider"])
        self.assertEqual("2026-08-27T12:00:00+08:00", pricing["verified_at"])
        self.assertEqual(2.0, pricing["peak_price_multiplier"])
        self.assertIn("api-docs.deepseek.com", pricing["source_url"])
        scenarios = plan["token_cost_scenarios"]
        self.assertEqual(
            {"normal", "near_output_limits", "all_once_retry"}, set(scenarios)
        )
        self.assertEqual(
            {
                "natural_section": {
                    "generation_input_tokens": 19_803,
                    "judge_input_tokens_with_fixed_refusal": 20_989,
                },
                "hybrid_v1": {
                    "generation_input_tokens": 18_769,
                    "judge_input_tokens_with_fixed_refusal": 19_956,
                },
            },
            plan["prompt_input_estimates"],
        )
        for scenario in scenarios.values():
            self.assertEqual(set(plan["strategies"]), set(scenario["strategies"]))
            for strategy in plan["strategies"]:
                estimate = scenario["strategies"][strategy]
                self.assertGreater(estimate["generation_input_tokens"], 0)
                self.assertGreater(estimate["judge_input_tokens"], 0)
                self.assertGreaterEqual(estimate["peak_2x_cost_usd"], estimate["cost_usd"])
        normal = scenarios["normal"]["totals"]
        near = scenarios["near_output_limits"]["totals"]
        retried = scenarios["all_once_retry"]["totals"]
        self.assertGreaterEqual(near["output_tokens"], normal["output_tokens"])
        self.assertEqual(near["input_tokens"] * 2, retried["input_tokens"])
        self.assertEqual(near["output_tokens"] * 2, retried["output_tokens"])
        self.assertAlmostEqual(near["cost_usd"] * 2, retried["cost_usd"], places=9)

    def test_fixed_real_configs_and_global_retry_ceilings_match_the_gate_plan(self):
        self.assertEqual("deepseek-v4-flash", DEEPSEEK_GENERATION_CONFIG.model)
        self.assertEqual("deepseek-v4-pro", DEEPSEEK_JUDGE_CONFIG.model)
        self.assertEqual("disabled", DEEPSEEK_GENERATION_CONFIG.thinking_type)
        self.assertEqual("disabled", DEEPSEEK_JUDGE_CONFIG.thinking_type)
        self.assertEqual(320, DEFAULT_REAL_RUN_LIMITS.max_attempts)
        self.assertEqual(200_000, DEFAULT_REAL_RUN_LIMITS.max_input_tokens)
        self.assertEqual(61_440, DEFAULT_REAL_RUN_LIMITS.max_output_tokens)
        self.assertEqual(0.30, DEFAULT_REAL_RUN_LIMITS.max_cost_usd)

    def test_chat_completions_payload_explicitly_disables_thinking(self):
        transport = CapturingTransport()
        client = DeepSeekChatCompletionsClient(
            api_key="synthetic-not-a-real-key",
            base_url="https://api.deepseek.com",
            transport=transport,
        )

        response = client.complete(
            "prompt", call_type="generation", request_id="RAG-001", config=config("deepseek-v4-flash")
        )
        client.complete(
            "prompt",
            call_type="generation",
            request_id="RAG-001-hybrid",
            config=config("deepseek-v4-flash"),
        )

        self.assertEqual(11, response.input_tokens)
        url, headers, payload, _timeout = transport.requests[0]
        self.assertEqual("https://api.deepseek.com/chat/completions", url)
        self.assertEqual("Bearer synthetic-not-a-real-key", headers["Authorization"])
        self.assertEqual({"type": "disabled"}, payload["thinking"])
        self.assertEqual({"type": "json_object"}, payload["response_format"])
        self.assertEqual(0.0, payload["temperature"])
        self.assertEqual(128, payload["max_tokens"])
        self.assertEqual(transport.requests[0][2], transport.requests[1][2])

    def test_attempt_ledger_manifest_resume_binding_and_hard_budget_stop_are_offline(self):
        citation = "HVK:r1-2026-08-24-v1:niddk-diabetes-tests:003:9a52fd667ccc"
        cases = ({"query_id": "q1", "query": "q", "split": "test", "answerable": True,
                  "reference_claims": [], "expected_citations": [citation]},)
        contexts = {"q1": [{"citation_id": citation, "body": "evidence"}]}
        generator = MockModelClient({"q1": [TimeoutError("first"), ModelResponse(
            '{"status":"answered","answer":"ok","claims":[{"claim_id":"C1","text":"ok","citation_ids":["' + citation + '"]}],"refusal_reason":null}', 10, 5, 2
        )]})
        judge = MockModelClient({"q1": [ConnectionError("judge first"), ModelResponse(
            '{"faithfulness":4,"relevance":4,"completeness":4,"expression":4,"reason":"mock","evidence_citation_ids":[]}', 8, 4, 1
        )]})
        limits = RunLimits(
            max_attempts=4,
            max_input_tokens=4_000,
            max_output_tokens=512,
            max_cost_usd=1.0,
        )
        manifest = build_run_manifest(
            eval_version="rag-eval-r1-v2",
            dataset_sha256="a" * 64,
            frozen_test_sha256="b" * 64,
            corpus_sha256="c" * 64,
            artifact_sha256={"natural_section": "d" * 64, "hybrid_v1": "e" * 64},
            aliases_sha256="f" * 64,
            query_ids=("q1",),
            generation_config=config("deepseek-v4-flash"),
            judge_config=config("deepseek-v4-pro"),
            max_retries=1,
            limits=limits,
            implementation_provenance=synthetic_provenance(),
            created_at="2026-08-25T12:00:00+08:00",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = GenerationJudgeRunner(
                generator, judge, config("deepseek-v4-flash"), config("deepseek-v4-pro"),
                max_retries=1, limits=limits, run_manifest=manifest,
                manifest_path=root / "manifest.json", attempt_ledger_path=root / "attempts.jsonl",
            )
            first = runner.run(cases, contexts, "hybrid_v1", root / "results.jsonl")
            second = runner.run(cases, contexts, "hybrid_v1", root / "results.jsonl")
            attempts = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]
            self.assertEqual(first, second)
            self.assertEqual(4, len(attempts))
            self.assertEqual([1, 2], [item["attempt_number"] for item in attempts[:2]])
            self.assertFalse(attempts[0]["success"])
            self.assertTrue(attempts[1]["success"])
            self.assertEqual([1, 2], [item["attempt_number"] for item in attempts[2:]])
            self.assertFalse(attempts[2]["success"])
            self.assertTrue(attempts[3]["success"])
            self.assertTrue(all(item["run_config_sha256"] == manifest["run_config_sha256"] for item in attempts))

            incompatible = dict(manifest)
            incompatible["run_config_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "incompatible run manifest"):
                GenerationJudgeRunner(
                    MockModelClient({}), MockModelClient({}), config("deepseek-v4-flash"),
                    config("deepseek-v4-pro"), max_retries=1, limits=limits,
                    run_manifest=incompatible, manifest_path=root / "manifest.json",
                    attempt_ledger_path=root / "attempts.jsonl",
                )

        tiny_limits = RunLimits(max_attempts=0, max_input_tokens=0, max_output_tokens=0, max_cost_usd=0.0)
        tiny_manifest = build_run_manifest(
            eval_version="rag-eval-r1-v2", dataset_sha256="a" * 64,
            frozen_test_sha256="b" * 64, corpus_sha256="c" * 64,
            artifact_sha256={"natural_section": "d" * 64, "hybrid_v1": "e" * 64},
            aliases_sha256="f" * 64, query_ids=("q1",),
            generation_config=config("deepseek-v4-flash"), judge_config=config("deepseek-v4-pro"),
            max_retries=1, limits=tiny_limits, implementation_provenance=synthetic_provenance(),
            created_at="2026-08-25T12:00:00+08:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = GenerationJudgeRunner(
                MockModelClient({}), MockModelClient({}), config("deepseek-v4-flash"),
                config("deepseek-v4-pro"), max_retries=1, limits=tiny_limits,
                run_manifest=tiny_manifest, manifest_path=root / "manifest.json",
                attempt_ledger_path=root / "attempts.jsonl",
            )
            with self.assertRaises(BudgetExceededError):
                runner.run(cases, contexts, "hybrid_v1", root / "results.jsonl")
            self.assertFalse((root / "attempts.jsonl").exists())

    def test_actual_usage_above_reservation_is_atomically_recorded_before_stop(self):
        limits = RunLimits(
            max_attempts=1,
            max_input_tokens=300,
            max_output_tokens=300,
            max_cost_usd=1.0,
        )
        run_hash = "a" * 64
        response = ModelResponse(
            '{"status":"answered","answer":"ok","claims":[],"refusal_reason":null}',
            input_tokens=500,
            output_tokens=2,
            latency_ms=1,
            cost_usd=0.0001,
            cost_source="synthetic_actual",
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "attempts.jsonl"
            ledger = AttemptLedger(ledger_path, limits, run_hash)
            reservation = ledger.preflight("x", config("deepseek-v4-flash"))
            record = {
                "schema_version": 2,
                "run_config_sha256": run_hash,
                "sequence": 1,
                "call_id": "natural_section:q:generation",
                "call_type": "generation",
                "strategy": "natural_section",
                "query_id": "q",
                "attempt_number": 1,
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "model_config_sha256": "b" * 64,
                "prompt_sha256": "c" * 64,
                "estimated_input_tokens": reservation.estimated_input_tokens,
                "reserved_input_tokens": reservation.input_tokens,
                "reserved_output_tokens": reservation.output_tokens,
                "reserved_cost_usd": reservation.cost_usd,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "success": True,
                "failure": None,
                "cost_usd": response.cost_usd,
                "cost_source": response.cost_source,
                "usage_status": "actual",
                "budget_input_tokens": max(reservation.input_tokens, response.input_tokens),
                "budget_output_tokens": max(reservation.output_tokens, response.output_tokens),
                "budget_cost_usd": max(reservation.cost_usd, response.cost_usd),
            }
            with self.assertRaises(BudgetExceededError):
                ledger.append(record)
            stored = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            self.assertEqual(1, len(stored))
            self.assertEqual(500, stored[0]["input_tokens"])
            self.assertEqual("actual", stored[0]["usage_status"])

    def test_unknown_usage_consumes_reservation_before_retry(self):
        citation = "HVK:r1-2026-08-24-v1:niddk-diabetes-tests:003:9a52fd667ccc"
        cases = ({"query_id": "q1", "query": "q", "split": "test", "answerable": True,
                  "reference_claims": [], "expected_citations": [citation]},)
        contexts = {"q1": [{"citation_id": citation, "body": "evidence"}]}
        limits = RunLimits(3, 2_000, 768, 1.0)
        manifest = build_run_manifest(
            eval_version="rag-eval-r1-v2", dataset_sha256="a" * 64,
            frozen_test_sha256="b" * 64, corpus_sha256="c" * 64,
            artifact_sha256={"natural_section": "d" * 64, "hybrid_v1": "e" * 64},
            aliases_sha256="f" * 64, query_ids=("q1",),
            generation_config=config("deepseek-v4-flash"), judge_config=config("deepseek-v4-pro"),
            max_retries=1, limits=limits, implementation_provenance=synthetic_provenance(),
            created_at="2026-08-26T12:00:00+08:00",
        )
        generator = MockModelClient({"q1": [TimeoutError("unknown usage"), ModelResponse(
            '{"status":"answered","answer":"ok","claims":[{"claim_id":"C1","text":"ok","citation_ids":["' + citation + '"]}],"refusal_reason":null}', 10, 5, 1
        )]})
        judge = MockModelClient({"q1": [ModelResponse(
            '{"faithfulness":4,"relevance":4,"completeness":4,"expression":4,"reason":"mock","evidence_citation_ids":[]}', 8, 4, 1
        )]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = GenerationJudgeRunner(
                generator, judge, config("deepseek-v4-flash"), config("deepseek-v4-pro"),
                max_retries=1, limits=limits, run_manifest=manifest,
                manifest_path=root / "manifest.json", attempt_ledger_path=root / "attempts.jsonl",
            )
            runner.run(cases, contexts, "natural_section", root / "results.jsonl")
            attempts = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual("conservative_unknown", attempts[0]["usage_status"])
        self.assertIsNone(attempts[0]["input_tokens"])
        self.assertGreater(attempts[0]["budget_input_tokens"], 0)
        self.assertGreater(attempts[0]["budget_output_tokens"], 0)
        self.assertEqual("actual", attempts[1]["usage_status"])


if __name__ == "__main__":
    unittest.main()
