import asyncio
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from healthvault.evaluation.rag.task_026 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SIGNED_WORKBOOK,
    PRIMARY_CASE_IDS,
    REWRITE_DEVELOPMENT_QUERY_IDS,
    build_generator_request_payload,
    build_generator_system_prompt,
    build_judge_system_prompt,
    build_human_annotation_package,
    build_human_reviewed_gold,
    build_task_026_draft,
    deterministic_query_rewrite,
    evaluate_signed_gold,
    write_generation_judge_preflight,
    write_human_annotation_package,
)
from healthvault.evaluation.rag.task_026_run import (
    Task026ExecutionGateError,
    execute_approved_task026_run,
)
from healthvault.evaluation.skill.benchmark.task_022 import ScriptedFakeTransport


class Task026HumanAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.draft = build_task_026_draft()
        cls.package = build_human_annotation_package(cls.draft)

    def test_fixed_24_cases_remain_unsigned_and_use_only_non_gold_rewrite_development(self):
        self.assertEqual(24, len(PRIMARY_CASE_IDS))
        self.assertEqual(list(PRIMARY_CASE_IDS), [item["query_id"] for item in self.draft["queries"]])
        self.assertEqual("draft_pending_human_annotation", self.draft["status"])
        self.assertEqual("not_signed", self.draft["gold_status"])
        self.assertFalse(self.draft["human_annotation_gate"]["executor_may_sign_gold"])
        self.assertTrue(set(REWRITE_DEVELOPMENT_QUERY_IDS).isdisjoint(PRIMARY_CASE_IDS))
        self.assertEqual([], self.draft["used_case_check"]["collisions"])

    def test_query_rewrite_is_deterministic_and_changes_only_matching_queries(self):
        query = "Which measurements are included in a lipid profile?"
        rewritten = deterministic_query_rewrite(query)

        self.assertEqual(rewritten, deterministic_query_rewrite(query))
        self.assertIn(query, rewritten)
        self.assertIn("lipid panel", rewritten)
        self.assertNotIn(
            "(are)", deterministic_query_rewrite("What is (are) High Blood Cholesterol ?")
        )
        self.assertEqual(
            "How should a white blood cell differential be interpreted?",
            deterministic_query_rewrite(
                "How should a white blood cell differential be interpreted?"
            ),
        )

    def test_two_arms_share_hybrid_bm25_and_pool_is_the_top_five_union(self):
        rankings = self.package["rankings"]
        by_query = {item["query_id"]: item for item in rankings["queries"]}
        pools = {item["query_id"]: item for item in self.package["pooled"]["queries"]}

        self.assertEqual("hybrid_v1", rankings["corpus_strategy"])
        self.assertEqual(
            rankings["arms"]["baseline"]["repository"],
            rankings["arms"]["variant"]["repository"],
        )
        for query_id, result in by_query.items():
            self.assertLessEqual(len(result["baseline"]["citation_ids"]), 5)
            self.assertLessEqual(len(result["variant"]["citation_ids"]), 5)
            expected = set(result["baseline"]["citation_ids"]) | set(
                result["variant"]["citation_ids"]
            )
            self.assertEqual(
                expected,
                {item["citation_id"] for item in pools[query_id]["candidates"]},
            )

    def test_blind_rows_hide_arm_rank_and_rewrite_while_suggestions_are_separate(self):
        blind = self.package["blind_rows"]
        suggestions = self.package["suggestion_rows"]
        forbidden = {"arm", "rank", "strategy", "rewritten_query", "baseline", "variant"}

        self.assertGreaterEqual(len(blind), 24)
        self.assertGreaterEqual(len(suggestions), len(blind))
        self.assertTrue(forbidden.isdisjoint(blind[0]))
        self.assertEqual(
            {row["pool_item_id"] for row in blind},
            {row["pool_item_id"] for row in suggestions if row["pool_item_id"]},
        )
        self.assertTrue(all(row["human_action"] == "" for row in blind))
        self.assertTrue(all("source_url" in row for row in suggestions))
        suggested_evidence = {
            (row["query_id"], row["candidate_citation_id"]) for row in suggestions
        }
        for query in self.draft["queries"]:
            for citation in query["provisional"]["allowed_citations"]:
                self.assertIn((query["query_id"], citation), suggested_evidence)

    def test_written_gate_package_is_complete_deterministic_and_zero_api(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = write_human_annotation_package(output)
            first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
            second = write_human_annotation_package(output)
            second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual("human_annotation_gate", first["status"])
            self.assertEqual("not_signed", first["gold_status"])
            self.assertEqual(0, first["api_calls"])
            self.assertEqual(0, first["input_tokens"])
            self.assertEqual(0, first["output_tokens"])
            self.assertEqual(24, first["query_count"])
            with (output / "task-026-blind-review.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(first["blind_row_count"], len(rows))
            manifest = json.loads(
                (output / "task-026-preflight.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first, manifest)

    def test_user_signed_workbook_freezes_24_complete_human_gold_queries(self):
        gold = build_human_reviewed_gold(
            DEFAULT_SIGNED_WORKBOOK, signed_on="2026-09-01"
        )

        self.assertEqual("human_reviewed_gold", gold["status"])
        self.assertEqual("signed", gold["gold_status"])
        self.assertEqual(24, gold["summary"]["query_count"])
        self.assertEqual(16, gold["summary"]["answerable_count"])
        self.assertEqual(8, gold["summary"]["unanswerable_count"])
        self.assertEqual(
            {"approve": 22, "correct": 2, "exclude": 0},
            gold["summary"]["actions"],
        )
        queries = {query["query_id"]: query for query in gold["queries"]}
        self.assertNotIn(
            "HVK:r1-2026-08-24-v1:niddk-diabetes-tests:002:46658a772415",
            queries["MQ-001"]["qrels"],
        )
        self.assertIn(
            "HVK:r1-2026-08-24-v1:medlineplus-lab-preparation:001:46d6705b79b0",
            queries["RAG-033"]["qrels"],
        )

    def test_signed_gold_recomputes_both_frozen_retrieval_arms_with_core_metrics(self):
        gold = build_human_reviewed_gold(
            DEFAULT_SIGNED_WORKBOOK, signed_on="2026-09-01"
        )
        rankings = json.loads(
            (DEFAULT_OUTPUT_DIR / "task-026-rankings.json").read_text(encoding="utf-8")
        )
        result = evaluate_signed_gold(gold, rankings)

        self.assertEqual("completed_offline", result["status"])
        self.assertEqual(0, result["api_calls"])
        for arm in ("baseline", "variant"):
            self.assertEqual(
                {
                    "query_count", "answerable_count", "unanswerable_count",
                    "recall_at_5", "mrr_at_5", "unanswerable_false_positive_rate",
                },
                set(result["arms"][arm]["metrics"]),
            )

    def test_v3_bound_model_separated_blind_preflight_stops_at_api_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            completed = write_generation_judge_preflight(output)
            preflight = completed["preflight"]
            contexts = completed["contexts"]
            calibration = completed["calibration"]

            self.assertFalse(preflight["execute_authorized"])
            self.assertEqual("glm-5.3-flash", preflight["generator"]["model"])
            self.assertFalse(preflight["generator"]["do_sample"])
            self.assertEqual(0.0, preflight["generator"]["temperature"])
            self.assertEqual(
                {"type": "enabled"}, preflight["generator"]["thinking"]
            )
            self.assertEqual("low", preflight["generator"]["reasoning_effort"])
            self.assertEqual("glm-5.3", preflight["judge"]["model"])
            self.assertFalse(preflight["judge"]["do_sample"])
            self.assertEqual(0.0, preflight["judge"]["temperature"])
            self.assertEqual({"type": "enabled"}, preflight["judge"]["thinking"])
            self.assertEqual("low", preflight["judge"]["reasoning_effort"])
            self.assertNotEqual(
                preflight["generator"]["client_id"], preflight["judge"]["client_id"]
            )
            self.assertFalse(preflight["judge"]["shared_client_with_generator"])
            self.assertFalse(preflight["judge"]["shared_context_with_generator"])
            self.assertEqual(48, preflight["logical_generation_requests"])
            self.assertEqual(48, preflight["logical_final_judge_requests"])
            self.assertEqual(8, preflight["logical_calibration_judge_requests"])
            self.assertEqual(56, preflight["logical_judge_requests"])
            self.assertEqual(104, preflight["maximum_model_calls"])
            self.assertEqual(208, preflight["maximum_transport_attempts"])
            self.assertEqual(0, preflight["api_calls"])
            self.assertEqual(0, preflight["input_tokens"])
            self.assertEqual(0, preflight["output_tokens"])
            self.assertEqual(0.0, preflight["cost_usd"])
            self.assertEqual("passed_fake_protocol_calibration", calibration["status"])
            self.assertEqual(4, calibration["pair_count"])
            self.assertTrue(all(item["passed"] for item in calibration["judgments"]))
            self.assertEqual(0, calibration["api_calls"])

            final = json.loads(
                (
                    DEFAULT_OUTPUT_DIR.parent
                    / "task-027/task-027-v3-retrieval-results.json"
                ).read_text(encoding="utf-8")
            )
            final_by_id = {item["query_id"]: item for item in final["queries"]}
            for item in contexts["queries"]:
                self.assertEqual(
                    final_by_id[item["query_id"]]["citation_ids"],
                    [evidence["citation_id"] for evidence in item["variant"]],
                )
            first = contexts["queries"][0]
            payload = build_generator_request_payload(first["query"], first["variant"])
            self.assertEqual({"query", "evidence"}, set(payload))
            generator_prompt = build_generator_system_prompt()
            for field in ("status", "answer", "claims", "refusal_reason"):
                self.assertIn(f'"{field}"', generator_prompt)
            self.assertIn('"enum":["answered","refused"]', generator_prompt)
            self.assertTrue(
                set(preflight["generator_input_contract"]["forbidden_fields"]).isdisjoint(
                    payload
                )
            )
            for schema in preflight["judge_response_schemas"].values():
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(
                    {"dimension", "winner", "reason", "evidence_citation_ids"},
                    set(schema["required"]),
                )
            for dimension in ("faithfulness", "completeness"):
                prompt = build_judge_system_prompt(dimension)
                self.assertIn(f'"const":"{dimension}"', prompt)
                for field in (
                    "dimension", "winner", "reason", "evidence_citation_ids"
                ):
                    self.assertIn(f'"{field}"', prompt)
            schedule = preflight["pairwise_order"]["schedule"]
            for dimension in ("faithfulness", "completeness"):
                rows = [item for item in schedule if item["dimension"] == dimension]
                self.assertEqual(12, sum(item["answer_a"] == "baseline" for item in rows))
                self.assertEqual(12, sum(item["answer_a"] == "variant" for item in rows))
            self.assertTrue(
                (output / "task-026-generation-judge-run-manifest.json").is_file()
            )

            preflight_path = output / "task-026-generation-judge-preflight.json"
            preflight_sha256 = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
            empty_generator = ScriptedFakeTransport([])
            empty_judge = ScriptedFakeTransport([])
            closed_gates = (
                (False, True, "fake-key", preflight_sha256),
                (True, False, "fake-key", preflight_sha256),
                (True, True, "", preflight_sha256),
                (True, True, "fake-key", "0" * 64),
            )
            for index, (execute, approved, api_key, digest) in enumerate(closed_gates):
                with self.subTest(gate=index), self.assertRaises(Task026ExecutionGateError):
                    asyncio.run(
                        execute_approved_task026_run(
                            preflight_path=preflight_path,
                            provided_preflight_sha256=digest,
                            execute=execute,
                            approved=approved,
                            api_key=api_key,
                            output_dir=output / f"closed-{index}",
                            generator_transport=empty_generator,
                            judge_transport=empty_judge,
                        )
                    )
            self.assertEqual([], empty_generator.requests)
            self.assertEqual([], empty_judge.requests)

            def response(value):
                return {
                    "choices": [{"message": {"content": json.dumps(value)}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }

            judge_responses = [
                response(item["fake_output"]) for item in calibration["judgments"]
            ]
            first_citation = next(
                evidence["citation_id"]
                for context in contexts["queries"]
                for arm in ("baseline", "variant")
                for evidence in context[arm]
            )
            generation_responses = [
                response({
                    "status": "refused",
                    "answer": "",
                    "claims": [{
                        "claim_id": "C1",
                        "text": "Invalid claim on a refusal.",
                        "citation_ids": [first_citation],
                    }],
                    "refusal_reason": "The supplied evidence is insufficient.",
                })
            ]
            for context in contexts["queries"]:
                for arm in ("baseline", "variant"):
                    evidence = context[arm]
                    if evidence:
                        citation = evidence[0]["citation_id"]
                        value = {
                            "status": "answered",
                            "answer": "Supported by the supplied evidence.",
                            "claims": [{
                                "claim_id": "C1",
                                "text": "Supported by the supplied evidence.",
                                "citation_ids": [citation],
                            }],
                            "refusal_reason": None,
                        }
                    else:
                        value = {
                            "status": "refused",
                            "answer": "",
                            "claims": [],
                            "refusal_reason": "The supplied evidence is insufficient.",
                        }
                    generation_responses.append(response(value))
            judge_responses.extend(
                response({
                    "dimension": item["dimension"],
                    "winner": "tie",
                    "reason": "The blind answers are equivalent for this dimension.",
                    "evidence_citation_ids": [],
                })
                for item in preflight["pairwise_order"]["schedule"]
            )
            generator_transport = ScriptedFakeTransport(generation_responses)
            judge_transport = ScriptedFakeTransport(judge_responses)
            run_output = output / "fake-run"
            run = asyncio.run(
                execute_approved_task026_run(
                    preflight_path=preflight_path,
                    provided_preflight_sha256=preflight_sha256,
                    execute=True,
                    approved=True,
                    api_key="fake-key",
                    output_dir=run_output,
                    generator_transport=generator_transport,
                    judge_transport=judge_transport,
                )
            )

            self.assertEqual(49, len(generator_transport.requests))
            self.assertEqual(56, len(judge_transport.requests))
            self.assertTrue(all(
                item["model"] == "glm-5.3-flash"
                for item in generator_transport.requests
            ))
            self.assertTrue(all(
                item["model"] == "glm-5.3" for item in judge_transport.requests
            ))
            self.assertTrue(all(
                '"dimension"' in item["messages"][0]["content"]
                and '"winner"' in item["messages"][0]["content"]
                and '"evidence_citation_ids"' in item["messages"][0]["content"]
                for item in judge_transport.requests
            ))
            self.assertTrue(all(
                item["response_format"] == {"type": "json_object"}
                for item in (*generator_transport.requests, *judge_transport.requests)
            ))
            self.assertTrue(all(
                item["do_sample"] is False
                and item["temperature"] == 0.0
                and item["thinking"] == {"type": "enabled"}
                and item["reasoning_effort"] == "low"
                for item in (*generator_transport.requests, *judge_transport.requests)
            ))
            for request in generator_transport.requests:
                generator_input = json.loads(request["messages"][1]["content"])
                self.assertEqual({"query", "evidence"}, set(generator_input))
                self.assertIn(
                    '"enum":["answered","refused"]',
                    request["messages"][0]["content"],
                )
            self.assertIn(
                "previous response failed the local JSON schema",
                generator_transport.requests[1]["messages"][-1]["content"],
            )
            self.assertEqual(105, run["results"]["usage"]["transport_attempts"])
            self.assertEqual(0, run["results"]["usage"]["api_calls"])
            self.assertEqual(48, run["summary"]["generation_count"])
            self.assertEqual(8, run["summary"]["judge_calibration_count"])
            self.assertEqual(48, run["summary"]["final_judge_count"])
            expected_files = {
                "task-026-attempt-ledger.jsonl",
                "task-026-generation-judge-results.json",
                "task-026-generation-judge-summary.json",
                "task-026-human-review-sample.json",
                "task-026-execution-run-manifest.json",
            }
            self.assertTrue(all((run_output / name).is_file() for name in expected_files))
            ledger_rows = [
                json.loads(line)
                for line in (run_output / "task-026-attempt-ledger.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                ["judge_calibration"] * 8
                + ["generation"] * 49
                + ["final_judge"] * 48,
                [item["phase"] for item in ledger_rows],
            )
            self.assertEqual("protocol_error", ledger_rows[8]["status"])
            self.assertEqual("ok", ledger_rows[9]["status"])
            manifest_text = (
                run_output / "task-026-execution-run-manifest.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn("fake-key", manifest_text)


if __name__ == "__main__":
    unittest.main()
