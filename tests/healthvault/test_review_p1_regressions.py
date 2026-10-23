import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents import agent as agent_module
from agents.ui import quiet_agent_ui
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.application.confirm_extracted_report import confirm_extracted_report
from healthvault.evolution.signal_extractor import SignalExtractor, eligible_cards, observation_from_persisted_trace
from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
from healthvault.model_config import HealthVaultModelConfig
from healthvault.rag.releases import KnowledgeReleaseStore
from healthvault.runtime_context import RuntimeContext
from healthvault.web.answer_validation import verify_answer
from healthvault.web.answers import answer_with_model
from healthvault.web.extraction import build_natural_preview
from tests.healthvault.test_healthvault_agent_runtime import seed_profile
from tests.healthvault.test_knowledge_releases import update_payload


SOURCE = {"citation_id": "REC:synthetic:thyroid", "title": "合成报告", "detail": "甲状腺左叶结节 6×4 mm。"}
FACT = "左叶结节尺寸为 6×4 mm。"


def verdict():
    return {"answer_supported": True, "answers_question": True, "claims": [
        {"text": FACT, "citation_id": SOURCE["citation_id"], "quote": SOURCE["detail"], "supported": True},
    ]}


class ReviewP1RegressionTests(unittest.TestCase):
    def test_historical_value_is_not_bound_to_current_report_date(self):
        text = "报告日期: 2026-10-01\n既往空腹血糖 7.2 mmol/L 3.9–6.1\n本次空腹血糖 5.2 mmol/L 3.9–6.1"
        # A shortened model quote still derives its temporal scope from the source row.
        model = {"indicators": [{"code": "fasting_glucose", "value": 7.2, "unit": "mmol/L",
                 "reference_lower": 3.9, "reference_upper": 6.1, "quote": "空腹血糖 7.2 mmol/L 3.9–6.1"}]}
        preview = build_natural_preview("synthetic-current", text, model)
        self.assertTrue(preview["can_confirm"])
        self.assertEqual([5.2], [item["value"] for item in preview["indicators"]])
        self.assertTrue(any(item["code"] == "historical_indicator_excluded" for item in preview["warnings"]))

    def test_conflicting_or_only_historical_measurements_require_review(self):
        for rows in ("空腹血糖 7.2 mmol/L 3.9–6.1\n空腹血糖 5.2 mmol/L 3.9–6.1",
                     "既往检查结果：\n空腹血糖 7.2 mmol/L 3.9–6.1"):
            with self.subTest(rows=rows):
                preview = build_natural_preview("synthetic-conflict", "报告日期: 2026-10-01\n" + rows, {})
                self.assertFalse(preview["can_confirm"])
                self.assertEqual([], preview["indicators"])

    def test_different_quotes_of_one_finding_save_once_and_keep_both_sides(self):
        text = "报告日期: 2026-10-01\n甲状腺左叶结节 6×4 mm，边界清晰。\n甲状腺右叶结节 6×4 mm。"
        model = {"imaging_findings": [{"finding_type": "nodule", "body_site": "甲状腺", "laterality": "left",
                 "location": "左叶", "dimensions": [6, 4], "unit": "mm", "quote": "甲状腺左叶结节 6×4 mm"}]}
        preview = build_natural_preview("synthetic-finding", text, model)
        self.assertEqual(2, len(preview["imaging_findings"]))
        self.assertEqual(2, len({item["finding_id"] for item in preview["imaging_findings"]}))
        self.assertEqual({"left", "right"}, {item["laterality"] for item in preview["imaging_findings"]})
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteHealthRecordRepository(Path(directory) / "records.sqlite3")
            result = confirm_extracted_report("synthetic-user", "synthetic-finding", text, preview, repository)
            replay = confirm_extracted_report("synthetic-user", "synthetic-finding", text, preview, repository)
            self.assertTrue(result["confirmed"])
            self.assertTrue(replay["idempotent_replay"])

    def test_omitted_fact_cannot_pass_a_partial_verifier(self):
        answer = FACT + " [" + SOURCE["citation_id"] + "] 另有一个右叶结节，尺寸为 60 mm。"
        result = verify_answer("所有结节尺寸是什么？", answer, [SOURCE], query=lambda *a, **k: verdict())
        self.assertFalse(result["passed"])

    def test_only_checked_facts_are_rendered_and_generation_is_structured(self):
        answer = FACT + " [" + SOURCE["citation_id"] + "]"
        result = verify_answer("左叶结节多大？", answer, [SOURCE], query=lambda *a, **k: verdict())
        self.assertTrue(result["passed"])
        self.assertEqual(answer, result["answer"])
        payload = {"status": "answered", "claims": [{"text": FACT, "citation_id": SOURCE["citation_id"]}]}
        with patch("healthvault.web.answers.run_model_task", return_value=json.dumps(payload)):
            self.assertEqual(answer, answer_with_model("左叶结节多大？", "visit", {"sources": [SOURCE]}))

    def test_used_version_names_are_immutable_but_original_rollback_works(self):
        with tempfile.TemporaryDirectory() as directory:
            releases = KnowledgeReleaseStore(Path(directory) / "releases")
            for version in ("review-version-a", "review-version-b"):
                candidate = releases.stage(update_payload(version), "synthetic-owner")
                releases.decide(candidate["candidate_id"], "approve", "synthetic-owner")
            changed = update_payload("review-version-a")
            changed["source_updates"][0]["sections"][0]["blocks"][0]["text"] += " Changed synthetic revision."
            changed["source_updates"][0]["source_sha256"] = "auto"
            with self.assertRaises(ValueError):
                releases.stage(changed, "synthetic-owner")
            releases.rollback("synthetic-owner")
            self.assertEqual("review-version-a", releases.active()["corpus_version"])
            candidate = releases.stage(update_payload("review-version-c"), "synthetic-owner")
            self.assertEqual(candidate["candidate_id"], releases.stage(update_payload("review-version-c"), "synthetic-owner")["candidate_id"])
            changed = copy.deepcopy(update_payload("review-version-c"))
            changed["source_updates"][0]["sections"][0]["blocks"][0]["text"] += " Different pending content."
            changed["source_updates"][0]["source_sha256"] = "auto"
            with self.assertRaises(ValueError):
                releases.stage(changed, "synthetic-owner")

    def test_three_natural_language_tasks_are_eligible_but_forged_citations_fail(self):
        async def run(root):
            skills = root / ".bear" / "skills"
            target = skills / "HealthVault-Task-Execution" / "SKILL.md"
            target.parent.mkdir(parents=True)
            target.write_bytes((Path(__file__).resolve().parents[2] / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md").read_bytes())
            database = root / "records.sqlite3"
            seed_profile(SQLiteHealthRecordRepository(database), "synthetic-user")
            observations = []
            with patch.object(agent_module.openai, "AsyncOpenAI", return_value=MagicMock()), quiet_agent_ui():
                for index in range(5):
                    runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
                        context=RuntimeContext("synthetic-user", f"request-{index}"), database_path=database,
                        model_config=HealthVaultModelConfig("openai-compatible", "synthetic-model", "https://invalid.local/v1", "synthetic-key"),
                        skill_root=skills, skill_names=("HealthVault-Task-Execution",), evolution_root=root / "traces",
                        task_id=f"task-{index}", signal_context={"intent": "record_timeline", "preconditions": ["authenticated task", "read-only tool access"],
                        "expected_outcome": "validated read-only workflow response", "validator_status": "failed", "contract_checks": {"status": "failed"}}))
                    calls = 0
                    async def fake_model():
                        nonlocal calls
                        calls += 1
                        if calls == 1:
                            message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "query_health_timeline", "arguments": "{}"}}]}
                        else:
                            payload = json.loads(next(item["content"] for item in runtime.agent._openai_messages if item["role"] == "tool"))
                            citation = payload["available_citation_ids"][0] if index < 3 else "REC:forged" if index == 3 else ""
                            text = "已找到已确认的报告时间线。" + (" [" + citation + "]" if citation else "")
                            runtime.agent._emit_text(text)
                            message = {"role": "assistant", "content": text, "tool_calls": None}
                        return {"choices": [{"message": message}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
                    runtime.agent._call_openai_stream = fake_model
                    await runtime.run("请用简洁中文显示已确认报告的时间线。")
                    observations.append(observation_from_persisted_trace(runtime.trace_recorder.trace_path))
            self.assertEqual(["passed"] * 3 + ["failed"] * 2, [item.validator_status for item in observations])
            cards = eligible_cards((await SignalExtractor().extract(observations)).cards)
            self.assertEqual(1, len(cards))
            self.assertEqual(3, len(cards[0].task_ids))
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(Path(directory)))


if __name__ == "__main__":
    unittest.main()
