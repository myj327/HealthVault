import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.skills import load_project_skill_resources, project_skill_manifest
from healthvault.agent_runtime import (
    HealthVaultAgentRuntimeConfig,
    build_healthvault_agent_runtime,
)
from healthvault.evolution.candidates import (
    CandidateGenerator,
    EvolutionSignal,
    TaskEndProcessor,
)
from healthvault.evolution.publisher import SkillPublisher
from healthvault.evolution.synthetic_demo import SKILL_V1, SKILL_V2, run_synthetic_demo
from healthvault.evolution.verifier import CandidateVerifier
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SkillEvolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_trace_is_minimal_redacted_and_version_bound(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            agent_module.openai,
            "AsyncOpenAI",
            return_value=MagicMock(),
        ):
            root = Path(temp_dir)
            skill_root = root / ".bear" / "skills"
            skill_dir = skill_root / "HealthVault-Task-Execution"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_bytes(SKILL_V1.encode("utf-8"))
            runtime = build_healthvault_agent_runtime(
                HealthVaultAgentRuntimeConfig(
                    context=RuntimeContext("profile-a", "request-019"),
                    database_path=root / "health.sqlite3",
                    model_config=HealthVaultModelConfig(
                        "openai-compatible",
                        "fake-model",
                        "https://invalid.local/v1",
                        "fake-key-never-used",
                    ),
                    skill_root=skill_root,
                    skill_names=("HealthVault-Task-Execution",),
                    evolution_root=root / ".bear" / "skill-evolution" / "healthvault",
                    task_id="task-019-trace",
                )
            )
            calls = 0

            async def fake_model_call():
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {
                        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search_health_knowledge",
                                "arguments": json.dumps({
                                    "query": "synthetic private report text",
                                    "api_key": "SECRET-KEY-MUST-NOT-PERSIST",
                                }),
                            },
                        }]}}],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 2},
                    }
                runtime.agent._emit_text("done")
                return {
                    "choices": [{"message": {"role": "assistant", "content": "done", "tool_calls": None}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1},
                }

            runtime.agent._call_openai_stream = fake_model_call
            runtime.agent._get_current_cost_usd = MagicMock(return_value=0.0)
            runtime.components.adapter.execute = AsyncMock(
                return_value=json.dumps({"status": "ok", "validator_code": "citation_valid"})
            )
            with patch.object(agent_module, "start_spinner"), patch.object(
                agent_module, "stop_spinner"
            ), patch.object(agent_module, "print_tool_call"), patch.object(
                agent_module, "print_tool_result"
            ), patch.object(agent_module, "print_cost"), patch.object(
                agent_module, "print_divider"
            ):
                result = await runtime.run("synthetic task")

            trace_files = list((root / ".bear" / "skill-evolution" / "healthvault" / "traces").glob("*.json"))
            self.assertEqual(1, len(trace_files))
            raw = trace_files[0].read_text(encoding="utf-8")
            trace = json.loads(raw)

        self.assertEqual("done", result.text)
        self.assertEqual("succeeded", trace["result_status"])
        self.assertEqual({"input": 13, "output": 3}, trace["tokens"])
        self.assertEqual(0.0, trace["cost_usd"])
        self.assertEqual("HealthVault-Task-Execution", trace["skill"]["name"])
        self.assertEqual(sha256(SKILL_V1), trace["skill"]["sha256"])
        self.assertEqual("citation_valid", trace["tool_calls"][0]["validator_code"])
        self.assertEqual(["query"], trace["tool_calls"][0]["input_summary"]["keys"])
        self.assertEqual(1, trace["tool_calls"][0]["input_summary"]["unknown_key_count"])
        self.assertNotIn("private report", raw)
        self.assertNotIn("SECRET-KEY", raw)
        self.assertNotIn("fake-key-never-used", raw)
        self.assertNotIn("profile-a", raw)
        self.assertFalse(trace["candidate_generation_triggered"])

    async def test_task_end_batches_create_evolve_and_ignore_ineligible_signals(self):
        responses = [
            {"action": "create", "skill_name": "HealthVault-Task-Execution", "content": SKILL_V1,
             "rationale": "Reusable tool ordering correction.", "trace_ids": ["trace-create"]},
            {"action": "evolve", "skill_name": "HealthVault-Task-Execution", "content": SKILL_V2,
             "rationale": "Reusable deterministic recovery step.", "trace_ids": ["trace-evolve"]},
        ]
        schemas = []

        async def fake_side_query(_prompt, schema):
            schemas.append(schema)
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            generator = CandidateGenerator(Path(temp_dir), fake_side_query)
            processor = TaskEndProcessor(generator)
            self.assertEqual("eligible", processor.record(EvolutionSignal(
                "trace-create", "explicit_correction", "Validate citations before returning.", True
            )))
            self.assertEqual("ignored", processor.record(EvolutionSignal(
                "trace-weak", "weak_feedback", "looks good", True
            )))
            self.assertEqual("ignored", processor.record(EvolutionSignal(
                "trace-patient", "explicit_correction", "Patient glucose is 9.1.", True,
            )))
            created = await processor.process("create")
            self.assertEqual("eligible", processor.record(EvolutionSignal(
                "trace-evolve", "deterministic_failure", "Retry only after a retryable tool status.", True
            )))
            evolved = await processor.process("evolve")
            self.assertEqual("ignored", processor.record(EvolutionSignal(
                "trace-noop", "explicit_correction", "repeat current rule", True, has_increment=False
            )))

            files = sorted(Path(temp_dir, "candidates").glob("*.json"))

        self.assertEqual(["create", "evolve"], [created.action, evolved.action])
        self.assertEqual(2, len(files))
        self.assertEqual(2, len(schemas))
        self.assertTrue(all(schema["additionalProperties"] is False for schema in schemas))
        self.assertEqual(2, generator.call_count)

    def test_verifier_accepts_valid_and_rejects_missing_danger_sensitive_and_noop(self):
        verifier = CandidateVerifier("HealthVault-Task-Execution")
        valid = verifier.verify_content(SKILL_V1, action="create")
        missing = verifier.verify_content(SKILL_V1.replace("## Verification", "## Checks"), action="create")
        dangerous = verifier.verify_content(SKILL_V1.replace("Run only approved", "Run rm -rf / then approved"), action="create")
        sensitive = verifier.verify_content(SKILL_V1.replace("Run only approved", "API key sk-secret123 and patient Alice BP 180/100"), action="create")
        duplicate = verifier.verify_content(SKILL_V2, action="create", active_content=SKILL_V1)
        noop = verifier.verify_content(SKILL_V1, action="evolve", active_content=SKILL_V1)

        self.assertEqual("verified", valid.status)
        self.assertEqual("rejected", missing.status)
        self.assertIn("missing_section:Verification", missing.reasons)
        self.assertIn("dangerous_command", dangerous.reasons)
        self.assertIn("sensitive_or_patient_fact", sensitive.reasons)
        self.assertIn("duplicate_ability", duplicate.reasons)
        self.assertIn("empty_patch", noop.reasons)

    async def test_publish_create_evolve_reload_manifest_and_idempotent_rollback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_root = root / ".bear" / "skills"
            summary_path = root / "summary.json"
            summary = await run_synthetic_demo(root, summary_path=summary_path)
            publisher = SkillPublisher(skill_root, root / ".bear" / "skill-evolution" / "healthvault")
            verifier = CandidateVerifier("HealthVault-Task-Execution")
            create_again = publisher.publish_content(
                SKILL_V1, action="create", verification=verifier.verify_content(SKILL_V1, action="create"), approved=True
            )
            rollback_again = publisher.rollback()
            manifest = project_skill_manifest(load_project_skill_resources(
                skill_root, ("HealthVault-Task-Execution",)
            ))
            written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            candidate_files = list((root / ".bear" / "skill-evolution" / "healthvault" / "candidates").glob("*.json"))

        self.assertEqual("published", summary["create"]["status"])
        self.assertEqual("verified", summary["create"]["verification"])
        self.assertEqual("published", summary["evolve"]["status"])
        self.assertEqual("verified", summary["evolve"]["verification"])
        self.assertEqual("rolled_back", summary["rollback"]["status"])
        self.assertEqual("already_published", create_again.status)
        self.assertEqual("already_rolled_back", rollback_again.status)
        self.assertEqual(sha256(SKILL_V1), summary["create"]["sha256"])
        self.assertEqual(sha256(SKILL_V2), summary["evolve"]["sha256"])
        self.assertEqual(sha256(SKILL_V1), summary["rollback"]["to_sha256"])
        self.assertEqual(sha256(SKILL_V1), manifest[0]["sha256"])
        self.assertEqual(2, summary["side_query_calls"])
        self.assertEqual(2, len(candidate_files))
        self.assertEqual(summary, written_summary)

    def test_unverified_unapproved_and_failed_replace_never_damage_active_skill(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_root = root / ".bear" / "skills"
            publisher = SkillPublisher(skill_root, root / ".bear" / "skill-evolution" / "healthvault")
            verifier = CandidateVerifier("HealthVault-Task-Execution")
            publisher.publish_content(SKILL_V1, action="create", verification=verifier.verify_content(SKILL_V1, action="create"), approved=True)
            active = skill_root / "HealthVault-Task-Execution" / "SKILL.md"
            before = active.read_bytes()
            rejected = publisher.publish_content(
                SKILL_V2, action="evolve", verification=verifier.verify_content(SKILL_V2, action="evolve", active_content=SKILL_V1), approved=False
            )
            failing = SkillPublisher(
                skill_root,
                root / ".bear" / "skill-evolution" / "healthvault",
                replace_file=lambda _source, _target: (_ for _ in ()).throw(OSError("synthetic replace failure")),
            )
            failed = failing.publish_content(
                SKILL_V2, action="evolve", verification=verifier.verify_content(SKILL_V2, action="evolve", active_content=SKILL_V1), approved=True
            )
            after = active.read_bytes()
            evolved = publisher.publish_content(
                SKILL_V2, action="evolve", verification=verifier.verify_content(SKILL_V2, action="evolve", active_content=SKILL_V1), approved=True
            )
            before_failed_rollback = active.read_bytes()
            failed_rollback = failing.rollback()
            after_failed_rollback = active.read_bytes()

        self.assertEqual("approval_required", rejected.status)
        self.assertEqual("publish_failed", failed.status)
        self.assertEqual(before, after)
        self.assertEqual("published", evolved.status)
        self.assertEqual("rollback_failed", failed_rollback.status)
        self.assertEqual(before_failed_rollback, after_failed_rollback)

    async def test_candidate_response_schema_is_strict_and_trace_ids_are_required(self):
        async def invalid_side_query(_prompt, _schema):
            return {
                "action": "create",
                "skill_name": "HealthVault-Task-Execution",
                "content": SKILL_V1,
                "rationale": "synthetic",
                "trace_ids": [],
                "extra": "not allowed",
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            processor = TaskEndProcessor(CandidateGenerator(Path(temp_dir), invalid_side_query))
            processor.record(EvolutionSignal(
                "trace-schema", "explicit_correction", "Reusable deterministic correction.", True
            ))
            with self.assertRaises(ValueError):
                await processor.process("create")


if __name__ == "__main__":
    unittest.main()
