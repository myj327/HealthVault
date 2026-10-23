import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext

from healthvault.evolution.review import SkillEvolutionService
from healthvault.evolution.trace import TaskTraceRecorder


ACTIVE_SKILL = """---
name: HealthVault-Task-Execution
description: Execute read-only HealthVault tasks with evidence.
---

## When to Use
Use for HealthVault read-only workflows.

## Prerequisites
Use only approved tools.

## Steps
1. Identify the requested HealthVault outcome before selecting a tool.
2. Run only approved read-only domain tools and preserve returned citations.
3. Return the tool status accurately; never present a rejection as success.

## Failure Handling
If evidence is insufficient or a tool rejects input, report that result without guessing.

## Verification
Check the final status, requested fields, and every citation before answering.

## Safety Boundaries
Never infer patient facts or medical advice, expose secrets, or use write tools.
"""


def signal_context(status="passed"):
    return {
        "intent": "record_timeline",
        "preconditions": ["authenticated task", "read-only tool access"],
        "expected_outcome": "validated read-only workflow response",
        "validator_status": status,
        "contract_checks": {"tool": status, "status": status, "citation": status},
    }


class SkillReviewLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.skill_root = self.root / ".bear" / "skills"
        self.active = self.skill_root / "HealthVault-Task-Execution" / "SKILL.md"
        self.active.parent.mkdir(parents=True)
        self.active.write_text(ACTIVE_SKILL, encoding="utf-8")
        self.evolution = self.root / ".bear" / "skill-evolution" / "healthvault"
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    async def test_only_matching_route_and_valid_final_contract_promote_trace(self):
        for index, (tool, final, expected) in enumerate((
            ("query_health_timeline", {"final_status": "insufficient_evidence", "citations": [], "refusal": True}, "passed"),
            ("compare_health_indicators", {"final_status": "insufficient_evidence", "citations": [], "refusal": True}, "failed"),
            ("query_health_timeline", {"final_status": "insufficient_evidence", "citations": ["REC:invented"], "refusal": True}, "failed"),
            ("query_health_timeline", {"final_status": "ok", "citations": [], "refusal": False}, "failed"),
        )):
            with self.subTest(tool=tool, final=final):
                with patch.object(agent_module.openai, "AsyncOpenAI", return_value=MagicMock()):
                    runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                        context=RuntimeContext("synthetic-profile", f"contract-{index}"),
                        database_path=self.root / "records.sqlite3",
                        model_config=HealthVaultModelConfig("openai-compatible", "synthetic-model", "https://invalid.local/v1", "synthetic-key"),
                        skill_root=self.skill_root, skill_names=("HealthVault-Task-Execution",),
                        evolution_root=self.evolution, signal_context=signal_context(),
                    ))
                async def fake_run(_prompt):
                    await runtime.agent._execute_tool_call(tool, {})
                    return {"text": json.dumps(final), "tokens": {"input": 0, "output": 0}}
                with patch.object(runtime.agent, "run_once", new=AsyncMock(side_effect=fake_run)):
                    await runtime.run("synthetic workflow contract")
                trace = json.loads(runtime.trace_recorder.trace_path.read_text(encoding="utf-8"))
                self.assertEqual(expected, trace["signal_context"]["validator_status"])

    async def fake_patch(self, prompt, _schema):
        request = json.loads(prompt)
        self.calls.append(request)
        card = request["eligible_experience_cards"][0]
        return {"patches": [{
            "patch_id": "PATCH-RECORD-TIMELINE",
            "operation": "add",
            "source_trace_ids": card["trace_ids"],
            "verification_source": card["verification_source"],
            "verification_strength": card["verification_strength"],
            "target_section": "Steps",
            "target_paragraph": "1. Identify the requested HealthVault outcome before selecting a tool.",
            "applicability_conditions": ["The request asks for a record timeline."],
            "when_not_to_use": ["The request asks for another workflow."],
            "tool_path": ["query_health_timeline"],
            "stop_conditions": ["The timeline tool returns a terminal status."],
            "content": "1a. For a record timeline, call query_health_timeline before drafting.",
            "rationale": "Use the independently verified workflow path.",
        }]}

    async def add_trace(self, index, *, success=True):
        async def tool(_name, _arguments, _state):
            return json.dumps({"status": "ok", "validator_code": "citation_valid"})

        recorder = TaskTraceRecorder(
            self.evolution,
            request_id=f"request-{index}",
            task_id=f"task-{index}",
            skill={
                "name": "HealthVault-Task-Execution",
                "sha256": hashlib.sha256(self.active.read_bytes()).hexdigest(),
            },
            executor=tool,
            signal_context=signal_context("passed" if success else "failed"),
        )
        await recorder.execute("query_health_timeline", {})
        return recorder.finish("succeeded" if success else "failed")

    async def test_three_independent_tasks_create_gated_review_then_publish_and_rollback(self):
        for index in range(3):
            await self.add_trace(index)
        service = SkillEvolutionService(self.skill_root, self.evolution, self.fake_patch)
        before = self.active.read_text(encoding="utf-8")

        review = await service.consider()

        self.assertIsNotNone(review)
        self.assertEqual("pending_review", review["status"])
        self.assertEqual(3, len(review["source_trace_ids"]))
        self.assertTrue(review["evaluation"]["passed"])
        self.assertEqual("medium", review["risk"]["level"])
        self.assertEqual("not_run", review["evaluation"]["held_out"]["status"])
        self.assertEqual(1, len(self.calls))
        self.assertEqual(before, self.active.read_text(encoding="utf-8"))
        self.assertEqual(review["candidate_id"], (await service.consider())["candidate_id"])
        self.assertEqual(1, len(self.calls))

        published = service.store.decide(review["candidate_id"], "approve", "admin")
        self.assertEqual("published", published["status"])
        self.assertIn("query_health_timeline before drafting", self.active.read_text(encoding="utf-8"))
        rolled_back = service.store.rollback("admin")
        self.assertEqual("rolled_back", rolled_back["status"])
        self.assertEqual(before, self.active.read_text(encoding="utf-8"))

    async def test_concurrent_batches_generate_one_candidate_and_stale_base_cannot_publish(self):
        for index in range(3):
            await self.add_trace(index)
        service = SkillEvolutionService(self.skill_root, self.evolution, self.fake_patch)
        reviews = await asyncio.gather(service.consider(), service.consider())
        self.assertEqual(reviews[0]["candidate_id"], reviews[1]["candidate_id"])
        self.assertEqual(1, len(self.calls))
        self.active.write_bytes(self.active.read_bytes() + b"\nNew approved revision.\n")
        review = service.store.decide(reviews[0]["candidate_id"], "approve", "owner")
        self.assertEqual("stale", review["status"])
        self.assertNotIn(b"query_health_timeline before drafting", self.active.read_bytes())

    async def test_two_tasks_do_not_call_candidate_model(self):
        await self.add_trace(1)
        await self.add_trace(2)
        service = SkillEvolutionService(self.skill_root, self.evolution, self.fake_patch)

        self.assertIsNone(await service.consider())
        self.assertEqual([], self.calls)

    async def test_personal_preference_is_not_promoted_to_global_skill(self):
        path = await self.add_trace(1, success=False)
        trace_id = json.loads(path.read_text(encoding="utf-8"))["trace_id"]
        service = SkillEvolutionService(self.skill_root, self.evolution, self.fake_patch)
        review = await service.consider(feedback_by_trace_id={trace_id: "我喜欢简短回答，应该改变我的回答格式。"}, user_confirmed_trace_ids={trace_id}, only_trace_ids={trace_id})
        self.assertIsNone(review)
        self.assertEqual([], self.calls)

    async def test_explicit_confirmed_workflow_correction_is_strong_without_prior_pass(self):
        path = await self.add_trace(1, success=False)
        trace_id = json.loads(path.read_text(encoding="utf-8"))["trace_id"]
        service = SkillEvolutionService(self.skill_root, self.evolution, self.fake_patch)

        review = await service.consider(
            feedback_by_trace_id={
                trace_id: "This workflow was wrong; use the timeline route before drafting."
            },
            user_confirmed_trace_ids={trace_id},
            only_trace_ids={trace_id},
        )

        self.assertEqual("pending_review", review["status"])
        self.assertEqual([trace_id], review["source_trace_ids"])
        serialized = json.dumps(review, ensure_ascii=False)
        self.assertNotIn("This workflow was wrong", serialized)


if __name__ == "__main__":
    unittest.main()
