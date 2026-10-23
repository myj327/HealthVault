from __future__ import annotations

import asyncio
import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.ui import quiet_agent_ui
from healthvault.evaluation import current_skill_run as run


class OracleFakeClient:
    """Use the frozen Oracle to exercise plumbing; never measure model ability."""
    is_real = False

    def __init__(self, cases=None):
        self.requests = []
        self.cases = {item["case_id"]: item for item in (cases or run.legacy.load_r3_dataset()["cases"])}

    def complete(self, payload):
        self.requests.append(payload)
        prompt = next(message["content"] for message in reversed(payload["messages"]) if message["role"] == "user")
        case = next(item for identifier, item in self.cases.items() if f"Case ID: {identifier}" in prompt)
        step = sum(message["role"] == "tool" for message in payload["messages"])
        expected = case["expected"]
        if step < len(expected["required_tools"]):
            name = expected["required_tools"][step]
            arguments = {"query": case["prompt"]} if name == "search_health_knowledge" else {}
            message = {"role": "assistant", "content": None, "tool_calls": [{"id": f"fake-{step}", "type": "function",
                       "function": {"name": name, "arguments": json.dumps(arguments)}}]}
            reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": json.dumps({"final_status": expected["final_status"],
                       "citations": expected["allowed_citation_ids"][:expected["min_citations"]], "refusal": expected["refusal_required"]})}
            reason = "stop"
        return {"choices": [{"message": message, "finish_reason": reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class CurrentSkillRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.plan = run.prepare_run(self.root / "preflight", api_base="https://example.invalid/v1")
        self.path = self.root / "preflight/preflight.json"

    def transport(self, client):
        output = self.root / "transport"
        output.mkdir()
        value = run.CurrentTransport(self.plan, output, client)
        value.case_id, value.arm_id = "R3H-002", "active_v1"
        return value

    def payload(self):
        case = next(item for item in run.legacy.load_r3_dataset()["cases"] if item["case_id"] == "R3H-002")
        return {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": run.legacy.PROMPT_TEMPLATE.format(case_id=case["case_id"], prompt=case["prompt"])}],
                "tools": [{"type": "function", "function": {"name": name}} for name in self.plan["tool_names"]],
                "temperature": 0, "max_tokens": 2048, "stream": False}

    def test_missing_approval_and_changed_code_make_no_calls(self):
        client = OracleFakeClient()
        with self.assertRaisesRegex(ValueError, "explicit paid approval"):
            asyncio.run(run.execute_run(self.path, self.root / "run", approved_sha256="", reviewed_sha256=self.plan["sha256"], client=client))
        with patch.object(run, "implementation_binding", return_value={"sha256": "changed"}):
            with self.assertRaisesRegex(ValueError, "Implementation changed"):
                run.validate_preflight(self.path)
        self.assertEqual([], client.requests)
        self.assertFalse((self.root / "run").exists())

    def test_bound_inputs_and_old_artifacts_are_not_overwritten(self):
        original = {path: run.digest(path) for path in (run.legacy.DATASET_PATH, run.legacy.ARMS_PATH, run.legacy.ACTIVE_SKILL_PATH)}
        added = self.root / "bound.txt"
        added.write_text("original", encoding="utf-8")
        self.plan["bindings"][str(added)] = run.digest(added)
        self.plan["sha256"] = run.value_hash({key: value for key, value in self.plan.items() if key != "sha256"})
        self.path.unlink()
        run.save(self.path, self.plan)
        added.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Bound input changed"):
            run.validate_preflight(self.path)
        with self.assertRaises(FileExistsError):
            run.prepare_run(self.root / "preflight", api_base="https://example.invalid/v1")
        self.assertEqual(original, {path: run.digest(path) for path in original})

    def test_transport_failure_has_no_retry_and_no_secret_in_log(self):
        class FailureClient:
            is_real = False
            calls = 0
            def complete(self, payload):
                self.calls += 1
                raise RuntimeError("synthetic-secret")
        client = FailureClient()
        transport = self.transport(client)
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload(), 60))
        self.assertEqual(1, client.calls)
        self.assertEqual("failed", transport.calls[0]["status"])
        self.assertNotIn("synthetic-secret", (transport.output / "calls.jsonl").read_text(encoding="utf-8"))

    def test_actual_turn_limit_and_read_only_tools_are_enforced(self):
        client = OracleFakeClient()
        transport = self.transport(client)
        forbidden = self.payload()
        forbidden["tools"].append({"type": "function", "function": {"name": "confirm_health_records"}})
        with self.assertRaisesRegex(ValueError, "read-only tool set"):
            asyncio.run(transport.post_json(self.plan["endpoint"], {}, forbidden, 60))
        for _ in range(4):
            asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload(), 60))
        with self.assertRaisesRegex(RuntimeError, "budget"):
            asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload(), 60))
        self.assertEqual(4, len(client.requests))
        self.assertTrue(all(payload["do_sample"] is False for payload in client.requests))

    def test_both_arms_use_real_native_tools_and_identical_task(self):
        client = OracleFakeClient()
        transport = self.transport(client)
        case = next(item for item in run.legacy.load_r3_dataset()["cases"] if item["case_id"] == "R3H-002")
        preflight = {**self.plan, "run_config_sha256": self.plan["sha256"], "frozen_inputs": {"dataset_semantic_sha256": "synthetic-unit-test"}}
        attempts = []
        with patch("agents.agent.print_tool_call"), patch("agents.agent.print_tool_result"):
            for arm in run.legacy.ARM_IDS:
                transport.arm_id = arm
                attempts.append(asyncio.run(run.legacy.run_r3_case(arm_id=arm, case=case, preflight=preflight,
                                transport=transport, ledger=None, api_key="fake", runtime_root=self.root / "cases")))
        self.assertTrue(all(item["validation"]["task_success"] for item in attempts))
        self.assertNotEqual(attempts[0]["skill_sha256"], attempts[1]["skill_sha256"])
        self.assertEqual(client.requests[0]["tools"], client.requests[2]["tools"])
        first = next(item["content"] for item in reversed(client.requests[0]["messages"]) if item["role"] == "user")
        second = next(item["content"] for item in reversed(client.requests[2]["messages"]) if item["role"] == "user")
        self.assertEqual(first, second)
        self.assertEqual(4, len(transport.calls))
        gc.collect()

    def test_raw_malformed_final_fails_before_runtime_normalization(self):
        class MalformedClient(OracleFakeClient):
            def complete(self, payload):
                response = super().complete(payload)
                if response['choices'][0]['finish_reason'] == 'stop':
                    response['choices'][0]['message']['content'] = '{'
                return response
        transport = self.transport(MalformedClient())
        case = next(item for item in run.legacy.load_r3_dataset()['cases'] if item['case_id'] == 'R3H-003')
        transport.case_id = case['case_id']
        with quiet_agent_ui():
            attempt = asyncio.run(run.legacy.run_r3_case(
                arm_id=transport.arm_id, case=case, transport=transport, ledger=None, api_key='synthetic-key',
                preflight={**self.plan, 'run_config_sha256':self.plan['sha256'],
                           'frozen_inputs':{'dataset_semantic_sha256':'synthetic-test'}}, runtime_root=self.root/'cases'))
        self.assertFalse(attempt['validation']['task_success'])
        self.assertTrue(attempt['validation']['hard_failure'])
        self.assertTrue(transport.failed)
        self.assertEqual('invalid_json',transport.calls[-1]['raw_final']['failure_code'])
        saved = json.loads((transport.output/'responses'/f"{len(transport.calls):04}.json").read_text(encoding='utf-8'))
        self.assertEqual('{',saved['choices'][0]['message']['content'])

    def test_raw_final_field_types_and_status_are_checked(self):
        samples = [
            ({'final_status':'ok','citations':[],'refusal':'false'},'schema_mismatch'),
            ({'final_status':'ok','citations':[1],'refusal':False},'schema_mismatch'),
            ({'final_status':'ok','citations':[],'refusal':False,'extra':True},'schema_mismatch'),
            ({'final_status':'invented','citations':[],'refusal':False},'invalid_status'),
        ]
        class FinalClient:
            is_real = False
            def complete(self, payload):
                return {'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(self.value)}}]}
        for index,(value,code) in enumerate(samples):
            with self.subTest(code=code,value=value):
                client = FinalClient(); client.value = value
                output = self.root/f'fields-{index}'; output.mkdir()
                transport = run.CurrentTransport(self.plan,output,client)
                transport.case_id,transport.arm_id = 'R3H-002','active_v1'
                with self.assertRaisesRegex(ValueError,code):
                    asyncio.run(transport.post_json(self.plan['endpoint'],{},self.payload(),60))
                self.assertEqual(code,transport.calls[-1]['raw_final']['failure_code'])
                self.assertEqual('failed',transport.calls[-1]['status'])

    def test_valid_wrong_raw_status_is_not_replaced_by_correct_refusal(self):
        class WrongStatusClient(OracleFakeClient):
            def complete(self,payload):
                response = super().complete(payload)
                if response['choices'][0]['finish_reason']=='stop' and 'Case ID: R3H-003' in str(payload['messages']):
                    response['choices'][0]['message']['content'] = json.dumps({'final_status':'ok','citations':[],'refusal':False})
                return response
        with quiet_agent_ui():
            result = asyncio.run(run.execute_run(self.path,self.root/'wrong-status-run',
                approved_sha256=self.plan['sha256'],reviewed_sha256=self.plan['sha256'],
                client=WrongStatusClient(),approval_root=self.root/'fake-approvals'))
        self.assertEqual('completed',result['status'])
        attempts = [item for item in result['attempts'] if item['case_id']=='R3H-003']
        self.assertEqual(2,len(attempts))
        for attempt in attempts:
            self.assertEqual('ok',attempt['final_status'])
            self.assertFalse(attempt['validation']['task_success'])
            self.assertFalse(attempt['validation']['status_correct'])
        self.assertEqual(0,result['api_calls'])

    def test_valid_raw_final_does_not_clear_runtime_failure(self):
        class IncompleteUsageClient(OracleFakeClient):
            def complete(self,payload):
                response = super().complete(payload)
                if response['choices'][0]['finish_reason']=='stop' and 'Case ID: R3H-003' in str(payload['messages']):
                    response['usage'] = {'prompt_tokens':100}
                return response
        with quiet_agent_ui():
            result = asyncio.run(run.execute_run(self.path,self.root/'runtime-failure-run',
                approved_sha256=self.plan['sha256'],reviewed_sha256=self.plan['sha256'],
                client=IncompleteUsageClient(),approval_root=self.root/'fake-approvals'))
        attempts = [item for item in result['attempts'] if item['case_id']=='R3H-003']
        self.assertEqual(2,len(attempts))
        for attempt in attempts:
            self.assertEqual('passed',attempt['raw_final_validation'])
            self.assertTrue(attempt['transport_or_runtime_failure'])
            self.assertIsNotNone(attempt['failure_code'])
            self.assertFalse(attempt['validation']['task_success'])
            self.assertTrue(attempt['validation']['hard_failure'])
        self.assertEqual(0,result['api_calls'])


if __name__ == "__main__":
    unittest.main()
