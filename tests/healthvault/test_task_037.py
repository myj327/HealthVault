import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.adapters.bearcode.final_response_normalizer import (
    FinalResponseNormalizer,
    parse_final_response,
)
from healthvault.evaluation import task_037
from healthvault.evaluation.skill.benchmark.task_022 import ScriptedFakeTransport


class Task037Tests(unittest.TestCase):
    def test_normalizer_uses_only_top_level_ids_and_deduplicates(self):
        result = FinalResponseNormalizer().normalize(
            {"final_status": "ok", "citations": ["REC:a:x", "REC:a:x", "REC:forged:x"]},
            [{"name": "query_health_timeline", "result_status": "ok", "available_citation_ids": ["REC:a:x", "REC:b:x"]}],
        )
        self.assertEqual(("REC:a:x",), result.citations)

    def test_normalizer_maps_real_terminal_statuses_without_model_inference(self):
        normalizer = FinalResponseNormalizer()
        partial = normalizer.normalize({"final_status": "ok", "citations": ["REC:a:x"]}, [{"name": "generate_visit_brief", "result_status": "partial", "available_citation_ids": ["REC:a:x"]}])
        self.assertEqual("ok", partial.final_status)
        self.assertEqual(("REC:a:x",), partial.citations)
        self.assertEqual("rejected", normalizer.normalize({"final_status": "ok", "citations": []}, [{"name": "search_health_knowledge", "result_status": "rejected", "available_citation_ids": []}]).final_status)
        unsafe = normalizer.normalize({"final_status": "ok", "citations": []}, [{"name": "search_health_knowledge", "result_status": "empty", "available_citation_ids": [], "issue_codes": ["unsupported_treatment"]}])
        self.assertEqual("rejected", unsafe.final_status)

    def test_final_parser_accepts_one_bare_or_json_fenced_object(self):
        value = '{"final_status":"ok","citations":[],"refusal":false}'
        self.assertEqual(parse_final_response(value), parse_final_response(f"```json\n{value}\n```"))
        self.assertIsNone(parse_final_response(f"answer: {value}"))

    def test_isolated_dev_traces_produce_one_verified_card(self):
        cards = asyncio.run(task_037.extract_verified_dev_cards())
        self.assertEqual(1, len(cards)); self.assertEqual("strong", cards[0]["strength"])
        self.assertTrue(all(trace.startswith("TASK037-DEV-") for trace in cards[0]["trace_ids"]))

    def test_dev_observations_read_the_bound_trace_and_feedback_artifact(self):
        document = json.loads(task_037.DEV_TRACE_PATH.read_text(encoding="utf-8"))
        document["feedback"] = "frozen feedback read by the extractor input"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"; path.write_text(json.dumps(document), encoding="utf-8")
            with patch.object(task_037, "DEV_TRACE_PATH", path):
                observations = task_037.dev_observations()
        self.assertEqual("frozen feedback read by the extractor input", observations[0].trace["feedback"])

    def test_r4_datasets_are_twenty_each_and_collision_free(self):
        skill = json.loads(task_037.SKILL_R4_PATH.read_text(encoding="utf-8")); e2e = json.loads(task_037.E2E_R4_PATH.read_text(encoding="utf-8"))
        self.assertEqual(20, len(skill["cases"])); self.assertEqual(20, len(e2e["cases"]))
        self.assertEqual([], task_037.r4_collision_check())

    def test_gate_a_is_zero_api_hash_bound_and_candidate_only(self):
        preflight = task_037.build_gate_a_preflight({})
        self.assertEqual(0, preflight["api_calls_made"]); self.assertEqual(1, preflight["limits"]["maximum_model_calls"])
        self.assertEqual("glm-5.3", preflight["candidate_model"]["model"])
        task_037.assert_gate_a(preflight, approved=True, supplied_hash=preflight["preflight_sha256"], api_key="fake")

    def test_existing_candidate_is_rejected_before_any_transport(self):
        preflight = task_037.build_gate_a_preflight({})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "task-037-frozen-candidate.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(task_037.ApiGateError):
                asyncio.run(task_037.execute_gate_a_once(preflight, approved=True, supplied_hash=preflight["preflight_sha256"], api_key="fake", transport=object(), runtime_root=root))

    def test_r4_fixture_and_oracle_are_deterministic_and_gate_binds_candidate(self):
        skill = task_037.load_r4_dataset(task_037.SKILL_R4_PATH)
        e2e = task_037.load_r4_dataset(task_037.E2E_R4_PATH)
        self.assertEqual(7, task_037.load_r4_fixture()["report_count"])
        self.assertEqual("ok", skill["cases"][0]["expected"]["final_status"])
        self.assertEqual("rejected", e2e["cases"][-1]["expected"]["final_status"])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "fixture.sqlite3"
            fixture_result = task_037.materialize_r4_fixture(database)
            self.assertEqual(7, fixture_result["report_count"])
            candidate = Path(directory) / "candidate.json"
            content = "# verified synthetic candidate\n"
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            candidate.write_text(json.dumps({"status": "frozen_unpublished", "candidate_sha256": digest, "candidate_content": content, "preflight_sha256": "b" * 64}), encoding="utf-8")
            preflight = task_037.build_gate_b_preflight(candidate)
            fake_final = json.dumps({"final_status": "rejected", "citations": [], "refusal": True})
            fake_transport = ScriptedFakeTransport([{
                "choices": [{"message": {"role": "assistant", "content": fake_final, "tool_calls": None}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }])
            smoke = asyncio.run(task_037.run_r4_case(
                skill["cases"][16], arm_id="active",
                skill_content=task_037.ACTIVE_SKILL_PATH.read_text(encoding="utf-8"),
                model=preflight["model"], transport=fake_transport, ledger=None,
                runtime_root=Path(directory), api_key="fake",
            ))
        self.assertEqual(digest, preflight["frozen_inputs"]["candidate_sha256"])
        self.assertEqual(40, preflight["limits"]["estimated_model_calls"])
        self.assertIsNone(smoke["attempt"]["runtime_error"])
        self.assertEqual(1, len(fake_transport.requests))
        fake_results = [{"case_id": case["case_id"], "passed": True, "hard_failure": False, "oracle": {"behavior": {"passed": True}, "response": {"citation_correct": True, "refusal_correct": True}}, "attempt": {"latency_ms": float(index), "total_tokens": index * 10}} for index, case in enumerate(e2e["cases"], start=1)]
        summary = task_037.summarize_gate_c(e2e, fake_results, selected_arm="active", api_calls=20)
        self.assertEqual(1.0, summary["refusal_correctness_rate"])
        self.assertEqual(2100, summary["total_tokens"])
        self.assertEqual(105.0, summary["average_tokens_per_attempt"])
        self.assertEqual(10.5, summary["mean_latency_ms"])
        self.assertIn("p95_latency_ms", summary)


if __name__ == "__main__":
    unittest.main()
