"""Focused, zero-API regression coverage for TASK-042."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents import agent as agent_module
from healthvault.adapters.bearcode.final_response_normalizer import FinalResponseNormalizer
from healthvault.agent_runtime import (
    HealthVaultAgentRuntimeConfig,
    build_healthvault_agent_runtime,
)
from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.application.knowledge_support_gate import assess_health_knowledge_support
from healthvault.evaluation.e2e import task_042
from healthvault.evaluation.split_integrity import validate_patient_profile_splits
from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext


FIXTURE = Path(__file__).parents[1] / "fixtures" / "health_reports" / "synthetic_checkup_2024.txt"


def _confirmation_input(profile_id: str = "task042-profile") -> dict[str, object]:
    report_text = FIXTURE.read_text(encoding="utf-8")
    return {
        "profile_id": profile_id,
        "source_id": f"{profile_id}-source",
        "report_text": report_text,
        "expected_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        "user_confirmed": True,
    }


def _make_runtime(database_path: Path, profile_id: str):
    with patch.object(agent_module.openai, "AsyncOpenAI", return_value=MagicMock()):
        return build_healthvault_agent_runtime(
            HealthVaultAgentRuntimeConfig(
                context=RuntimeContext(profile_id, f"{profile_id}-request"),
                database_path=database_path,
                model_config=HealthVaultModelConfig(
                    "openai-compatible",
                    "synthetic-model",
                    "https://invalid.local/v1",
                    "synthetic-test-key",
                ),
            )
        )


def _tool_call(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": f"task042-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }


def _final(content: str) -> dict[str, object]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": None}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }


class Task042Tests(unittest.IsolatedAsyncioTestCase):
    def test_development_registry_has_no_patient_profile_split_conflict(self):
        registry = task_042.load_development_split_registry()
        self.assertEqual("development_only", registry["purpose"])
        with self.assertRaisesRegex(ValueError, "patient/profile split conflict"):
            validate_patient_profile_splits(
                [
                    {"patient_id": "patient-a", "profile_id": "profile-a", "split": "dev"},
                    {"patient_id": "patient-a", "profile_id": "profile-b", "split": "test"},
                ]
            )

    async def test_cross_profile_tool_input_is_rejected_before_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "records.sqlite3"
            repository = SQLiteHealthRecordRepository(database)
            for profile_id in ("profile-a", "profile-b"):
                outcome = confirm_health_records(
                    _confirmation_input(profile_id), repository
                )
                self.assertEqual("confirmed", outcome.status)
            runtime = _make_runtime(database, "profile-a")
            result = await runtime.components.adapter.execute(
                "query_health_timeline", {"profile_id": "profile-b"}
            )
        self.assertEqual(
            "Tool input rejected: profile_id is supplied by RuntimeContext.", result
        )

    def test_unconfirmed_report_never_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "records.sqlite3"
            repository = SQLiteHealthRecordRepository(database)
            payload = _confirmation_input()
            payload["user_confirmed"] = False
            outcome = confirm_health_records(payload, repository)
            connection = sqlite3.connect(database)
            try:
                report_count = connection.execute("SELECT COUNT(*) FROM confirmed_reports").fetchone()[0]
            finally:
                connection.close()
        self.assertEqual("confirmation_required", outcome.status)
        self.assertFalse(outcome.write_performed)
        self.assertEqual(0, report_count)

    def test_exact_confirmation_replay_has_no_new_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "records.sqlite3"
            repository = SQLiteHealthRecordRepository(database)
            first = confirm_health_records(_confirmation_input(), repository)
            replay = confirm_health_records(_confirmation_input(), repository)
            connection = sqlite3.connect(database)
            try:
                report_count = connection.execute("SELECT COUNT(*) FROM confirmed_reports").fetchone()[0]
                record_count = connection.execute("SELECT COUNT(*) FROM health_records").fetchone()[0]
            finally:
                connection.close()
        self.assertEqual("confirmed", first.status)
        self.assertEqual("already_confirmed", replay.status)
        self.assertFalse(replay.write_performed)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual((1, 3), (report_count, record_count))

    def test_diagnosis_treatment_and_dose_requests_fail_closed(self):
        for prompt, code in (
            ("Use these symptoms to diagnose me.", "unsupported_diagnosis"),
            ("Change my medication treatment.", "unsupported_treatment"),
            ("What prescription dose should I take?", "unsupported_treatment"),
        ):
            with self.subTest(code=code):
                decision = assess_health_knowledge_support(prompt)
                self.assertFalse(decision.allowed)
                self.assertEqual(code, decision.reason_code)

    async def test_task_terminal_allows_empty_recovery_and_stops_explicit_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = _make_runtime(Path(directory) / "records.sqlite3", "profile-a")
            model_calls = 0

            async def fake_model_call():
                nonlocal model_calls
                model_calls += 1
                if model_calls == 1:
                    response = _tool_call("query_health_timeline", {})
                elif model_calls == 2:
                    response = _tool_call("search_health_knowledge", {"query": "What does an A1C test measure?"})
                else:
                    tool_messages = [
                        message["content"]
                        for message in runtime.agent._openai_messages
                        if message.get("role") == "tool"
                    ]
                    citation = json.loads(tool_messages[-1])["available_citation_ids"][0]
                    response = _final(json.dumps({"final_status": "ok", "citations": [citation], "refusal": False}))
                content = response["choices"][0]["message"]["content"]
                if content:
                    runtime.agent._emit_text(content)
                return response

            runtime.agent._call_openai_stream = fake_model_call
            with patch("agents.agent.start_spinner"), patch("agents.agent.stop_spinner"), patch("agents.agent.print_tool_call"), patch("agents.agent.print_tool_result"), patch("agents.agent.print_cost"):
                result = await runtime.run("synthetic no-evidence task")
            tool_messages = [message["content"] for message in runtime.agent._openai_messages if message.get("role") == "tool"]
        self.assertEqual(3, model_calls)
        self.assertEqual(2, len(tool_messages))
        self.assertIn('"status": "empty"', tool_messages[0])
        recovered = json.loads(result.text)
        self.assertEqual("ok", recovered["final_status"])
        self.assertTrue(recovered["citations"][0].startswith("HVK:"))

        with tempfile.TemporaryDirectory() as directory:
            safety_runtime = _make_runtime(Path(directory) / "records.sqlite3", "profile-a")
            safety_model_calls = 0

            async def unsafe_model_call():
                nonlocal safety_model_calls
                safety_model_calls += 1
                return _tool_call("search_health_knowledge", {"query": "Use these symptoms to diagnose me."})

            safety_runtime.agent._call_openai_stream = unsafe_model_call
            with patch("agents.agent.start_spinner"), patch("agents.agent.stop_spinner"), patch("agents.agent.print_tool_call"), patch("agents.agent.print_tool_result"), patch("agents.agent.print_cost"):
                safety_result = await safety_runtime.run("synthetic unsafe task")
        self.assertEqual(1, safety_model_calls)
        self.assertEqual(
            {"final_status": "rejected", "citations": [], "refusal": True},
            json.loads(safety_result.text),
        )

    async def test_malformed_structured_final_after_tool_is_safe_json(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "records.sqlite3"
            repository = SQLiteHealthRecordRepository(database)
            self.assertEqual(
                "confirmed",
                confirm_health_records(_confirmation_input("profile-a"), repository).status,
            )
            runtime = _make_runtime(database, "profile-a")
            responses = iter((_tool_call("query_health_timeline", {}), _final('{"final_status":')))

            async def fake_model_call():
                response = next(responses)
                content = response["choices"][0]["message"]["content"]
                if content:
                    runtime.agent._emit_text(content)
                return response

            runtime.agent._call_openai_stream = fake_model_call
            with patch("agents.agent.start_spinner"), patch("agents.agent.stop_spinner"), patch("agents.agent.print_tool_call"), patch("agents.agent.print_tool_result"), patch("agents.agent.print_cost"):
                result = await runtime.run("synthetic record lookup")
        self.assertEqual(
            {"final_status": "rejected", "citations": [], "refusal": True},
            json.loads(result.text),
        )

    def test_normalizer_preserves_safe_refusal_and_minimizes_citations(self):
        normalizer = FinalResponseNormalizer()
        minimized = normalizer.normalize(
            {"final_status": "ok", "citations": ["REC:a:one", "REC:a:one", "REC:a:two"], "refusal": False},
            [{"name": "query_health_timeline", "result_status": "ok", "available_citation_ids": ["REC:a:one", "REC:a:two"], "recommended_citation_ids": ["REC:a:one"]}],
        )
        safe_refusal = normalizer.normalize(
            {"final_status": "rejected", "citations": [], "refusal": True},
            [{"name": "search_health_knowledge", "result_status": "empty", "available_citation_ids": []}],
        )
        missing = normalizer.normalize(
            {"final_status": "ok", "citations": [], "refusal": False},
            [{"name": "search_health_knowledge", "result_status": "ok", "available_citation_ids": ["HVK:r2:source:001:abc"], "recommended_citation_ids": ["HVK:r2:source:001:abc"]}],
        )
        self.assertEqual(("REC:a:one",), minimized.citations)
        self.assertEqual(("rejected", True), (safe_refusal.final_status, safe_refusal.refusal))
        self.assertEqual(("rejected", (), True), (missing.final_status, missing.citations, missing.refusal))


if __name__ == "__main__":
    unittest.main()
