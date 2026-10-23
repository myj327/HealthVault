import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.ui import quiet_agent_ui
from healthvault.evaluation import current_e2e_run as run


class E2EFakeClient:
    """Fixture-aware plumbing double; its results never measure model ability."""
    is_real = False

    def __init__(self, *, malformed=False, forbidden=False):
        self.requests = []
        self.cases = run.load_current_dataset(run.active_corpus_root() / "chunks/hybrid_v1.json")["cases"]
        self.malformed, self.forbidden = malformed, forbidden

    def complete(self, payload):
        self.requests.append(payload)
        prompt = next(item["content"] for item in reversed(payload["messages"]) if item["role"] == "user")
        case = next(item for item in self.cases if item["prompt"] in prompt)
        tool_messages = [item for item in payload["messages"] if item["role"] == "tool"]
        expected = case["expected"]
        if self.forbidden and not tool_messages:
            name, arguments = "confirm_health_records", {}
        elif len(tool_messages) < len(expected["required_tools"]):
            name = expected["required_tools"][len(tool_messages)]
            arguments = {"query": " ".join(case["scenario"]["query_terms"])} if name == "search_health_knowledge" else {}
        else:
            observed = {identifier for item in tool_messages
                        for identifier in json.loads(item["content"]).get("available_citation_ids", [])}
            citations = sorted(observed & set(expected["allowed_citation_ids"]))[:expected["min_citations"]]
            content = "{" if self.malformed else json.dumps({"final_status": expected["final_status"],
                      "citations": citations, "refusal": expected["refusal_required"]})
            return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "fake-e2e-tool", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments)}}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class CurrentE2ERunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.plan = run.prepare_run(self.root / "preflight", api_base="https://example.invalid/v1")
        self.path = self.root / "preflight/preflight.json"
        self.case = run.legacy.load_dataset()["cases"][0]

    def transport(self, client):
        output = self.root / "transport"
        output.mkdir()
        transport = run.E2ETransport(self.plan, output, client)
        transport.case_id, transport.arm_id = self.case["case_id"], run.legacy.ARM_ID
        return transport

    def payload(self):
        return {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": self.case["prompt"]}],
                "tools": [{"type": "function", "function": {"name": name}} for name in self.plan["tool_names"]],
                "temperature": 0, "max_tokens": 2048, "stream": False}

    def test_approval_code_and_bound_inputs_are_checked_without_calls(self):
        client = E2EFakeClient()
        with self.assertRaisesRegex(ValueError, "explicit paid approval"):
            asyncio.run(run.execute_run(self.path, self.root / "run", approved_sha256="", reviewed_sha256=self.plan["sha256"], client=client))
        with patch.object(run, "implementation_binding", return_value={"sha256": "changed"}):
            with self.assertRaisesRegex(ValueError, "Implementation changed"):
                run.validate_preflight(self.path)
        matrix = self.root / "preflight/case-matrix.json"
        matrix.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Bound input changed"):
            run.validate_preflight(self.path)
        self.assertEqual([], client.requests)

    def test_global_and_per_case_actual_call_caps_and_read_only_tools(self):
        client = E2EFakeClient()
        transport = self.transport(client)
        payload = self.payload()
        payload["tools"].append({"type": "function", "function": {"name": "confirm_health_records"}})
        with self.assertRaisesRegex(ValueError, "read-only tool set"):
            asyncio.run(transport.post_json(self.plan["endpoint"], {}, payload, 60))
        for _ in range(4):
            asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload(), 60))
        with self.assertRaisesRegex(RuntimeError, "budget"):
            asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload(), 60))
        transport.case_id = "another-case"
        transport.calls.extend([{}] * (run.MAX_CALLS - len(transport.calls)))
        with self.assertRaisesRegex(RuntimeError, "E2E call budget"):
            asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload(), 60))
        self.assertEqual(4, len(client.requests))

    def test_raw_malformed_final_cannot_be_normalized_into_success(self):
        transport = self.transport(E2EFakeClient(malformed=True))
        with quiet_agent_ui():
            result = asyncio.run(run.run_case(self.case, self.plan, transport, self.root / "cases", api_key="synthetic-key"))
        self.assertFalse(result["passed"])
        self.assertTrue(result["hard_failure"])
        self.assertEqual("invalid_json", result["attempt"]["parser_error"])
        self.assertEqual("failed", result["attempt"]["raw_final_validation"])
        self.assertTrue(result["attempt"]["database_unchanged"])
        self.assertTrue(transport.failed)
        self.assertEqual([], list((self.root / "cases").iterdir()))

    def test_actual_forbidden_write_attempt_fails_and_database_is_unchanged(self):
        transport = self.transport(E2EFakeClient(forbidden=True))
        with quiet_agent_ui():
            result = asyncio.run(run.run_case(self.case, self.plan, transport, self.root / "cases", api_key="synthetic-key"))
        self.assertFalse(result["passed"])
        self.assertTrue(result["hard_failure"])
        self.assertIn("unauthorized_tool", result["oracle"]["hard_failure_reasons"])
        self.assertTrue(result["attempt"]["database_unchanged"])
        self.assertEqual([], list((self.root / "cases").iterdir()))

    def test_dense_query_degradation_cannot_pass_with_a_valid_final(self):
        from healthvault.repositories.dense_retrieval import DenseRetrievalError

        class FailingEmbedder:
            dimension = 1024

            def embed(self, texts):
                raise DenseRetrievalError("offline test failure")

        case = next(item for item in run.load_current_dataset(Path(self.plan["corpus_path"]))["cases"] if item["case_id"] == "R5E-010")
        transport = self.transport(E2EFakeClient())
        transport.case_id = case["case_id"]
        with patch("healthvault.infrastructure.hybrid_health_knowledge._local_embedder", return_value=FailingEmbedder()), quiet_agent_ui():
            result = asyncio.run(run.run_case(case, self.plan, transport, self.root / "cases", api_key="synthetic-key"))
        self.assertFalse(result["passed"])
        self.assertFalse(result["hard_failure"])
        self.assertEqual("passed", result["attempt"]["raw_final_validation"])
        self.assertEqual("insufficient_evidence", result["attempt"]["final"]["final_status"])
        self.assertFalse(result["attempt"]["hybrid_path_passed"])
        self.assertEqual("dense_query_failed", result["attempt"]["retrieval"]["searches"][0]["dense_fallback_reason"])
        self.assertTrue(result["attempt"]["database_unchanged"])

    def test_transport_failure_has_no_retry_and_error_logs_do_not_contain_secrets(self):
        class FailureClient:
            is_real = False
            calls = 0

            def complete(self, _payload):
                self.calls += 1
                raise RuntimeError("synthetic-secret-value")

        client = FailureClient()
        transport = self.transport(client)
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload(), 60))
        self.assertEqual(1, client.calls)
        self.assertNotIn("synthetic-secret-value", (transport.output / "calls.jsonl").read_text(encoding="utf-8"))

    def test_complete_offline_run_has_twenty_real_tool_cases_and_no_publication(self):
        original = {path: run.digest(path) for path in (run.legacy.DATASET_PATH, run.legacy.FIXTURE_PATH)}
        with quiet_agent_ui():
            result = asyncio.run(run.execute_run(self.path, self.root / "run", approved_sha256=self.plan["sha256"],
                     reviewed_sha256=self.plan["sha256"], client=E2EFakeClient(), approval_root=self.root / "approval"))
        self.assertEqual("completed", result["status"])
        self.assertEqual("offline_contract_check", result["mode"])
        self.assertEqual(0, result["api_calls"])
        self.assertEqual(20, result["completed_cases"])
        self.assertEqual(20, len(result["results"]))
        self.assertTrue(all(item["attempt"]["database_unchanged"] for item in result["results"]))
        self.assertTrue(all(item["passed"] for item in result["results"]))
        self.assertTrue(all(item["attempt"]["hybrid_path_passed"] for item in result["results"]))
        self.assertLessEqual(result["transport_attempts"], run.MAX_CALLS)
        self.assertFalse(result["publishes_skill"])
        self.assertFalse(result["untouched_held_out"])
        review = json.loads((self.root / "run/human-review.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(review["cases"]), 10)
        self.assertTrue(all(item["human_verdict"] is None for item in review["cases"]))
        self.assertEqual(original, {path: run.digest(path) for path in original})
        with self.assertRaises(FileExistsError):
            asyncio.run(run.execute_run(self.path, self.root / "run", approved_sha256=self.plan["sha256"],
                     reviewed_sha256=self.plan["sha256"], client=E2EFakeClient(), approval_root=self.root / "approval-2"))


if __name__ == "__main__":
    unittest.main()
