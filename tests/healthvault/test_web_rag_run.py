from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.evaluation import web_rag_run as run
from healthvault.web.answer_validation import verify_answer
from healthvault.web.answers import answer_with_model


class FakeClient:
    is_real = False

    def __init__(self, phase: str, requests: list):
        self.phase, self.requests = phase, requests
        self.calls = 0

    def complete(self, payload):
        self.requests.append((self.phase, payload))
        data = json.loads(payload["messages"][1]["content"])
        if self.phase == "generation":
            source = data["evidence"]["sources"][0]
            value = {"status": "answered", "claims": [{"text": source["detail"], "citation_id": source["citation_id"]}]}
        elif self.phase == "verification":
            value = {"protocol": "fact_span_v1", "answer_supported": True, "answers_question": True,
                     "claims": [{"claim_id": fact["claim_id"], "supported": True,
                                 "evidence_ids": [span["evidence_id"] for span in data["evidence_spans"]
                                                  if span["citation_id"] == fact["citation_id"] and span["source_field"] == "detail"]}
                                for fact in data["facts"]]}
        else:
            system = payload["messages"][0]["content"]
            dimension = "faithfulness" if '"faithfulness"' in system else "completeness"
            winner = ("A" if (self.calls // 2 + self.calls % 2) % 2 == 0 else "B") if self.phase == "calibration" else "tie"
            value = {"dimension": dimension, "winner": winner, "reason": "Synthetic protocol response",
                     "evidence_citation_ids": [source["citation_id"] for source in data["evidence"]]}
        self.calls += 1
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}


class WebRAGRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        # Tests use a two-query synthetic package, independent of ignored run outputs.
        original = json.loads(run.prepare.__globals__["GOLD"].read_text(encoding="utf-8"))
        self.gold = {**original, "queries": [next(item for item in original["queries"] if item["answerable"]),
                                           next(item for item in original["queries"] if not item["answerable"])]}
        workbook = self.root / "synthetic-workbook.txt"
        workbook.write_text("Synthetic test confirmation, not human Gold", encoding="utf-8")
        self.gold["human_confirmation"] = {"signed_by": "synthetic_test_fixture", "source_workbook": str(workbook), "source_workbook_sha256": run.digest(workbook)}
        corpus = run.SQLiteFTSHealthKnowledgeRepository()
        identifier = next(iter(self.gold["queries"][0]["qrels"]))
        hit = corpus.hit_for_chunk_id(identifier, rank=1, score=1, matched_terms=()).to_dict()
        corpus.close()
        self.retrieval = self.root / "retrieval"
        self.retrieval.mkdir()
        run.save(self.retrieval / "generation-contexts.json", {"implementation_sha256": "synthetic", "queries": [
            {"query_id": item["query_id"], "query": item["query"], "lexical": [hit], "hybrid": [hit]} for item in self.gold["queries"]]})
        run.save(self.retrieval / "retrieval-results.json", {"implementation_sha256": "synthetic", "gold_sha256": "synthetic-gold", "queries": [
            {"query_id": item["query_id"], "lexical": {"citation_ids": [identifier]}, "hybrid": {"citation_ids": [identifier]}} for item in self.gold["queries"]]})
        def fixture_binding(output, *, gold_path=None):
            binding = {"gold_sha256": "synthetic-gold", "implementation": run.implementation_binding(),
                       "gold_review_kind": "signed", "gold_confirmation": self.gold["human_confirmation"]}
            run.save(output / "implementation-manifest.json", binding)
            return self.gold, binding
        fixture_patch = patch.object(run, "prepare", side_effect=fixture_binding)
        fixture_patch.start()
        self.addCleanup(fixture_patch.stop)

    def prepare(self):
        plan = run.prepare_run(self.root / "preflight", self.retrieval, api_base="https://open.bigmodel.cn/api/paas/v4")
        return self.root / "preflight/preflight.json", plan

    def test_missing_approval_makes_zero_calls(self):
        path, plan = self.prepare()
        with self.assertRaisesRegex(ValueError, "explicit paid approval"):
            run.execute_run(path, self.root / "run", approved_sha256="", reviewed_sha256=plan["sha256"])
        self.assertFalse((self.root / "run").exists())

    def test_changed_inputs_are_rejected_before_execution(self):
        path, plan = self.prepare()
        (path.parent / "inputs/contexts.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Bound input changed"):
            run.execute_run(path, self.root / "run", approved_sha256=plan["sha256"], reviewed_sha256=plan["sha256"])
        self.assertFalse((self.root / "run").exists())

    def test_source_changes_invalidate_the_reviewed_binding(self):
        path, plan = self.prepare()
        with patch.object(run, "implementation_binding", return_value={"sha256": "changed"}):
            with self.assertRaisesRegex(ValueError, "Implementation changed"):
                run.validate_preflight(path)

    def test_failure_is_recorded_once_without_retry(self):
        client = FakeClient("generation", [])
        client.complete = lambda _payload: (_ for _ in ()).throw(ConnectionError("SYNTHETIC_SECRET"))
        calls = run.BoundedCalls({"limits": {"phases": run.PHASE_CAPS, "maximum_calls": 152}, "parameters": {}, "models": {"generation": "demo"}}, self.root, {"generation": client})
        with self.assertRaises(ConnectionError):
            calls.call("generation", "System", {"question": "public"})
        with self.assertRaises(RuntimeError):
            calls.call("generation", "System", {"question": "public"})
        self.assertEqual(len(calls.calls), 1)
        log = (self.root / "calls.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("SYNTHETIC_SECRET", log)
        self.assertIn('"status": "failed"', log)

    def test_phase_cap_and_no_tools_are_enforced(self):
        requests = []
        client = FakeClient("judge", requests)
        plan = {"limits": {"phases": {"judge": 1}, "maximum_calls": 1}, "parameters": {"max_tokens": 2048}, "models": {"judge": "glm-5.3"}}
        calls = run.BoundedCalls(plan, self.root, {"judge": client})
        calls.call("judge", 'Schema {"const": "faithfulness"}', {"evidence": []})
        with self.assertRaises(RuntimeError):
            calls.call("judge", "System", {})
        self.assertEqual(len(requests), 1)
        self.assertNotIn("tools", requests[0][1])
        self.assertEqual(requests[0][1]["max_tokens"], 2048)

    def test_production_claim_generation_and_omission_check_are_used(self):
        source = {"citation_id": "HVK:fixture", "detail": "Left nodule 6 mm", "title": "Report"}
        generated = answer_with_model("What is recorded?", "knowledge", {"sources": [source]}, query=lambda *_args, **_kwargs: json.dumps({"status": "answered", "claims": [{"text": "Left nodule 6 mm", "citation_id": "HVK:fixture"}, {"text": "Right nodule 60 mm", "citation_id": "HVK:fixture"}]}))
        checked = verify_answer("What is recorded?", generated, [source], query=lambda *_args, **_kwargs: {"answer_supported": True, "answers_question": True, "claims": [{"text": "Left nodule 6 mm", "citation_id": "HVK:fixture", "quote": "Left nodule 6 mm", "supported": True}]})
        self.assertFalse(checked["passed"])

    def test_provider_error_is_not_a_successful_refusal(self):
        rows = [{"answerable": False, "qrels": [], "lexical": {"status": "error"}}]
        metrics = run.summary(rows)["lexical"]
        self.assertEqual(metrics["unanswerable_correct_refusal_rate"], 0)
        self.assertEqual(metrics["errors"], 1)
        client = FakeClient("verification", [])
        client.complete = lambda _payload: {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
        calls = run.BoundedCalls({"limits": {"phases": run.PHASE_CAPS, "maximum_calls": 152}, "parameters": {}, "models": {"verification": "demo"}}, self.root, {"verification": client})
        with self.assertRaisesRegex(ValueError, "Verifier protocol failure"):
            calls.call("verification", "System", {})
        self.assertTrue(calls.failed)

    def test_full_web_contract_run_is_bound_and_consumed_once(self):
        path, plan = self.prepare()
        requests = []
        clients = {phase: FakeClient(phase, requests) for phase in run.PHASE_CAPS}
        result = run.execute_run(path, self.root / "run", approved_sha256=plan["sha256"], reviewed_sha256=plan["sha256"], clients=clients, approval_root=self.root / "approvals")
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["completed_queries"], 2)
        self.assertEqual(result["mode"], "offline_contract_check")
        self.assertEqual(result["api_calls"], 0)
        self.assertLessEqual(result["transport_attempts"], 152)
        for phase, request in requests:
            data = json.loads(request["messages"][1]["content"])
            if phase in {"generation", "verification"}:
                self.assertNotIn("reference_claims", data)
                self.assertNotIn("qrels", data)
            if phase == 'verification':
                self.assertTrue(data['facts'])
                self.assertTrue(data['evidence_spans'])
            self.assertNotIn("tools", request)
        with self.assertRaises(FileExistsError):
            run.execute_run(path, self.root / "another-run", approved_sha256=plan["sha256"], reviewed_sha256=plan["sha256"], clients=clients, approval_root=self.root / "approvals")

    def test_judge_receives_full_long_evidence_from_the_application_inputs(self):
        corpus = run.SQLiteFTSHealthKnowledgeRepository()
        try:
            hit = corpus.hit_for_chunk_id(
                "HVK:r1-2026-08-24-v1:medlineplus-understand-lab-results:002:4a356f113e37",
                rank=1, score=1, matched_terms=(),
            ).to_dict()
        finally:
            corpus.close()
        self.assertGreater(len(hit["excerpt"]), 1200)
        # Replace only this test's temporary package; the signed Gold stays intact.
        for name in ("generation-contexts.json", "retrieval-results.json"):
            path = self.retrieval / name
            package = json.loads(path.read_text(encoding="utf-8"))
            for row in package["queries"]:
                for arm in ("lexical", "hybrid"):
                    row[arm] = [hit] if name == "generation-contexts.json" else {"citation_ids": [hit["citation_id"]]}
            path.write_text(json.dumps(package), encoding="utf-8")
        path, plan = self.prepare()
        requests = []
        result = run.execute_run(path, self.root / "run", approved_sha256=plan["sha256"],
                     reviewed_sha256=plan["sha256"], clients={phase: FakeClient(phase, requests) for phase in run.PHASE_CAPS},
                     approval_root=self.root / "approvals")
        self.assertEqual("completed", result["status"])
        judgments = [json.loads(payload["messages"][1]["content"]) for phase, payload in requests if phase == "judge"]
        self.assertEqual(4, len(judgments))
        for data in judgments:
            self.assertEqual(hit["citation_id"], data["evidence"][0]["citation_id"])
            self.assertEqual(len(hit["excerpt"]), len(data["evidence"][0]["evidence_text"]))
            self.assertEqual(hit["excerpt"], data["evidence"][0]["evidence_text"])
        self.assertEqual(0, result["api_calls"])


if __name__ == "__main__":
    unittest.main()
