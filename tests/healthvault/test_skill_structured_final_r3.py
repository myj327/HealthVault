from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.evaluation.skill.benchmark.task_022 import (
    BASE_URL_ENV,
    KEY_ENV,
    MODEL_ENV,
    CallLedger,
    ScriptedFakeTransport,
)
from healthvault.evaluation.skill.benchmark.task_025 import (
    ARM_IDS,
    INTENTS,
    ApiGateError,
    assert_api_gate,
    build_preflight,
    load_frozen_arms,
    load_r3_dataset,
    parse_final_v3,
    run_r3_case,
    tool_citation_contract_source_hashes,
    validate_attempt_v3,
)
from healthvault.infrastructure.local_health_knowledge import LocalHealthKnowledgeRepository


class Task025StructuredFinalR3Tests(unittest.TestCase):
    def test_parser_accepts_one_bare_or_json_fenced_object(self) -> None:
        value = {"final_status": "ok", "citations": ["CIT:1"], "refusal": False}
        bare = parse_final_v3(json.dumps(value))
        fenced = parse_final_v3(f"```json\n{json.dumps(value)}\n```")
        self.assertIsNone(bare.failure_code)
        self.assertEqual(bare, fenced)
        self.assertEqual("missing_content", parse_final_v3("").failure_code)

    def test_parser_rejects_prose_multiple_objects_unknown_fields_and_status(self) -> None:
        valid = '{"final_status":"ok","citations":[],"refusal":false}'
        self.assertEqual("invalid_json", parse_final_v3(f"answer: {valid}").failure_code)
        self.assertEqual("invalid_json", parse_final_v3(f"{valid}\n{valid}").failure_code)
        self.assertEqual(
            "schema_mismatch",
            parse_final_v3('{"final_status":"ok","citations":[],"refusal":false,"extra":1}').failure_code,
        )
        self.assertEqual(
            "invalid_status",
            parse_final_v3('{"final_status":"maybe","citations":[],"refusal":false}').failure_code,
        )

    def test_validator_requires_minimal_direct_citations_and_empty_refusal(self) -> None:
        dataset = load_r3_dataset()
        answerable = next(case for case in dataset["cases"] if case["case_id"] == "R3H-009")
        allowed = answerable["expected"]["allowed_citation_ids"][0]
        base = {
            "tool_calls": [{
                "name": "search_health_knowledge", "status": "ok",
                "available_citation_ids": [allowed, "HVK:OBSERVED-BUT-IRRELEVANT"],
            }],
            "final_status": "ok", "citations": [allowed], "refusal": False,
            "failure_code": None, "model_turns": 2,
            "transport_or_runtime_failure": False, "invalid_calls": 0,
        }
        self.assertTrue(validate_attempt_v3(answerable, base)["task_success"])
        copied_all = dict(base, citations=[allowed, "HVK:OBSERVED-BUT-IRRELEVANT"])
        copied_result = validate_attempt_v3(answerable, copied_all)
        self.assertFalse(copied_result["citation_correct"])
        self.assertEqual("citation_policy", copied_result["failure_code"])

        refusal = next(case for case in dataset["cases"] if case["case_id"] == "R3H-012")
        invalid_refusal = dict(base, final_status="insufficient_evidence", refusal=True, citations=[allowed])
        invalid_refusal["tool_calls"] = [{
            "name": "search_health_knowledge", "status": "ok", "available_citation_ids": [allowed],
        }]
        self.assertEqual("citation_policy", validate_attempt_v3(refusal, invalid_refusal)["failure_code"])

    def test_r3_isolation_frozen_arms_and_zero_api_preflight(self) -> None:
        dataset = load_r3_dataset()
        arms = load_frozen_arms()
        preflight = build_preflight(environ={})
        self.assertEqual(20, len(dataset["cases"]))
        self.assertEqual({intent: 4 for intent in INTENTS}, {
            intent: sum(case["intent"] == intent for case in dataset["cases"])
            for intent in INTENTS
        })
        self.assertEqual("b1c7f20ef81c6fc857ae05f395668f969ea6e1cc20a415792a838d9aa5de26ae", arms["candidate_v2"]["skill_sha256"])
        self.assertFalse(arms["candidate_v2"]["published"])
        self.assertEqual(list(ARM_IDS), preflight["benchmark"]["arms"])
        self.assertEqual(0, preflight["isolation"]["r2_exact_prompt_overlap"])
        self.assertLess(preflight["isolation"]["r2_max_token_jaccard"], 0.70)
        self.assertFalse(preflight["isolation"]["skill_generation_performed"])
        self.assertEqual(0, preflight["observed"]["api_calls"])
        repository = LocalHealthKnowledgeRepository()
        reachability = {
            "R3H-009": "why laboratory reference ranges can differ",
            "R3H-010": "A1C average blood glucose past three months",
            "R3H-011": "cholesterol blood test fasting",
        }
        by_id = {case["case_id"]: case for case in dataset["cases"]}
        for case_id, query in reachability.items():
            retrieved = {hit.document.citation_id for hit in repository.search(query, 5)}
            self.assertLessEqual(
                set(by_id[case_id]["expected"]["allowed_citation_ids"]), retrieved,
                msg=f"{case_id} Gold citations must be reachable in Top-5",
            )
            self.assertEqual(1, by_id[case_id]["expected"]["max_citations"])
        with self.assertRaises(ApiGateError):
            assert_api_gate(
                preflight, execute=False, approved=False, api_key="",
                expected_hash=preflight["run_config_sha256"],
            )

    def test_default_preflight_reads_configured_environment(self) -> None:
        with patch.dict(os.environ, {
            KEY_ENV: "synthetic-test-key",
            MODEL_ENV: "glm-env-test",
            BASE_URL_ENV: "https://example.invalid/v1",
        }, clear=True):
            preflight = build_preflight()
        self.assertTrue(preflight["model_config"]["key_configured"])
        self.assertEqual("glm-env-test", preflight["model_config"]["model"])
        self.assertEqual("https://example.invalid/v1", preflight["model_config"]["api_base"])
        self.assertEqual(0, preflight["observed"]["api_calls"])

    def test_preflight_binds_task031_tool_citation_contract_sources(self) -> None:
        preflight = build_preflight(environ={})
        contracts = preflight["frozen_inputs"]["tool_citation_contract_source_sha256"]
        self.assertEqual(tool_citation_contract_source_hashes(), contracts)
        self.assertEqual(contracts, preflight["source_sha256"]["task_031_tool_citation_contracts"])
        self.assertEqual(
            {
                "healthvault/domain/records.py",
                "healthvault/domain/comparisons.py",
                "healthvault/domain/knowledge.py",
                "healthvault/domain/visit_briefs.py",
            },
            set(contracts),
        )

    def test_both_arms_share_prompt_parser_validator_in_full_fake_loop(self) -> None:
        case = next(case for case in load_r3_dataset()["cases"] if case["case_id"] == "R3H-002")
        citation = case["expected"]["allowed_citation_ids"][0]
        preflight = build_preflight(environ={})
        initial_prompts: list[str] = []
        tool_messages: list[str] = []
        attempts = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = CallLedger(root / "ledger.jsonl", run_config_sha256=preflight["run_config_sha256"], max_transport_attempts=16)
            with patch("agents.agent.print_tool_call"), patch("agents.agent.print_tool_result"):
                for index, arm in enumerate(ARM_IDS):
                    final = json.dumps({"final_status": "ok", "citations": [citation], "refusal": False})
                    if index == 1:
                        final = f"```json\n{final}\n```"
                    transport = ScriptedFakeTransport([
                        {
                            "choices": [{"message": {
                                "role": "assistant", "content": None,
                                "reasoning_content": "Use the one-tool timeline route.",
                                "tool_calls": [{
                                    "id": f"r3-{index}", "type": "function",
                                    "function": {"name": "query_health_timeline", "arguments": "{}"},
                                }],
                            }, "finish_reason": "tool_calls"}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        },
                        {
                            "choices": [{"message": {"role": "assistant", "content": final, "tool_calls": None}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                        },
                    ])
                    attempts.append(asyncio.run(run_r3_case(
                        arm_id=arm, case=case, preflight=preflight,
                        transport=transport, ledger=ledger, api_key="fake",
                        runtime_root=root,
                    )))
                    initial_prompts.append(next(
                        message["content"] for message in reversed(transport.requests[0]["messages"])
                        if message.get("role") == "user"
                    ))
                    tool_messages.extend(
                        message["content"] for message in transport.requests[1]["messages"]
                        if message.get("role") == "tool"
                    )
        self.assertEqual(initial_prompts[0], initial_prompts[1])
        self.assertEqual(2, len(tool_messages))
        self.assertTrue(all('"available_citation_ids"' in message for message in tool_messages))
        self.assertTrue(all(citation in message for message in tool_messages))
        self.assertTrue(all(item["validation"]["task_success"] for item in attempts))
        self.assertTrue(all(item["validation"]["failure_code"] is None for item in attempts))
        self.assertTrue(all(item["api_calls"] == 0 for item in attempts))


if __name__ == "__main__":
    unittest.main()
