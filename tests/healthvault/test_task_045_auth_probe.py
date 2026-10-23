from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.evaluation.skill import task_045
from healthvault.evaluation.skill.benchmark.task_022 import TransportError


class FakeTransport:
    is_real = False

    def __init__(self, outcome: dict[str, object] | Exception) -> None:
        self.outcome = outcome
        self.requests: list[dict[str, object]] = []

    async def post_json(
        self,
        _url: str,
        _headers: dict[str, str],
        payload: dict[str, object],
        _timeout: float,
    ) -> dict[str, object]:
        self.requests.append(copy.deepcopy(payload))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return copy.deepcopy(self.outcome)


class Task045AuthProbeTests(unittest.TestCase):
    def test_fake_probe_is_single_call_candidate_free_and_binds_later_gate(self) -> None:
        preflight = task_045.build_candidate_gate_a_auth_probe_preflight({})
        response = {
            "id": "probe-response",
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        transport = FakeTransport(response)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            task_045, "CANDIDATE_GATE_A_AUTH_PROBE_ROOT", Path(directory) / "probes"
        ), patch.object(
            task_045,
            "CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH",
            Path(directory) / "auth-probe-preflight.json",
        ):
            task_045._atomic_json(task_045.CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH, preflight)
            with self.assertRaises(task_045.ApiGateError):
                asyncio.run(task_045.execute_candidate_gate_a_auth_probe_once(
                    preflight,
                    approved=False,
                    supplied_hash=preflight["preflight_sha256"],
                    api_key="fake-key",
                    transport=transport,
                ))
            self.assertEqual([], transport.requests)

            result = asyncio.run(task_045.execute_candidate_gate_a_auth_probe_once(
                preflight,
                approved=True,
                supplied_hash=preflight["preflight_sha256"],
                api_key="fake-key",
                transport=transport,
            ))
            self.assertEqual("passed", result["status"])
            self.assertFalse(result["candidate_generated"])
            self.assertEqual(0, result["api_calls"])
            self.assertEqual(1, result["transport_attempts"])
            self.assertEqual(1, transport.requests[0]["max_tokens"])
            self.assertEqual({"type": "disabled"}, transport.requests[0]["thinking"])
            record = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
            self.assertFalse(record["candidate_generated"])
            self.assertFalse(record["response_content_persisted"])
            self.assertNotIn("choices", record)
            gate = task_045.build_candidate_gate_a_preflight(
                {},
                auth_probe_hash=preflight["preflight_sha256"],
            )
            self.assertEqual(
                preflight["preflight_sha256"],
                gate["auth_probe"]["preflight_sha256"],
            )
            self.assertEqual(
                task_045._sha(task_045.CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH),
                gate["auth_probe"]["preflight_artifact_sha256"],
            )
            self.assertEqual(
                task_045._sha(Path(result["result_path"])),
                gate["auth_probe"]["result_artifact_sha256"],
            )

    def test_fake_401_diagnostics_are_persisted_without_credentials(self) -> None:
        preflight = task_045.build_candidate_gate_a_auth_probe_preflight({})
        error = TransportError(
            "OpenAI-compatible HTTP 401",
            retryable=False,
            http_status=401,
            provider_error_code="AUTH-401",
            provider_error_message="API key rejected",
            request_id="req-probe-401",
        )
        transport = FakeTransport(error)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            task_045, "CANDIDATE_GATE_A_AUTH_PROBE_ROOT", Path(directory) / "probes"
        ), patch.object(
            task_045,
            "CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH",
            Path(directory) / "auth-probe-preflight.json",
        ):
            task_045._atomic_json(task_045.CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH, preflight)
            with self.assertRaisesRegex(TransportError, "HTTP 401"):
                asyncio.run(task_045.execute_candidate_gate_a_auth_probe_once(
                    preflight,
                    approved=True,
                    supplied_hash=preflight["preflight_sha256"],
                    api_key="never-persist-this-key",
                    transport=transport,
                ))
            path = task_045._candidate_gate_a_auth_probe_path(preflight["preflight_sha256"])
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("failed", record["status"])
            self.assertEqual("rejected_consumed", record["disposition"])
            self.assertEqual({
                "http_status": 401,
                "provider_error_code": "AUTH-401",
                "provider_error_message": "API key rejected",
                "request_id": "req-probe-401",
            }, record["provider_error"])
            serialized = path.read_text(encoding="utf-8")
            self.assertNotIn("never-persist-this-key", serialized)
            self.assertNotIn("Authorization", serialized)
            with self.assertRaisesRegex(task_045.CandidateGateError, "did not pass"):
                task_045.build_candidate_gate_a_preflight(
                    {},
                    auth_probe_hash=preflight["preflight_sha256"],
                )
            with self.assertRaisesRegex(task_045.ApiGateError, "already consumed"):
                asyncio.run(task_045.execute_candidate_gate_a_auth_probe_once(
                    preflight,
                    approved=True,
                    supplied_hash=preflight["preflight_sha256"],
                    api_key="another-key",
                    transport=FakeTransport({}),
                ))

    def test_forged_or_stale_probe_chain_is_rejected_before_gate_preflight(self) -> None:
        base = task_045.build_candidate_gate_a_auth_probe_preflight({})
        with tempfile.TemporaryDirectory() as directory, patch.object(
            task_045, "CANDIDATE_GATE_A_AUTH_PROBE_ROOT", Path(directory) / "probes"
        ), patch.object(
            task_045,
            "CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH",
            Path(directory) / "auth-probe-preflight.json",
        ):
            fake_hash = "a" * 64
            fake_path = task_045._candidate_gate_a_auth_probe_path(fake_hash)
            task_045._atomic_json(fake_path, {
                "schema_version": "healthvault-task-045-candidate-gate-a-auth-probe-result-v1",
                "preflight_sha256": fake_hash,
                "preflight_artifact_sha256": "b" * 64,
                "status": "passed",
                "disposition": "authentication_verified",
                "started_at_utc": "2026-09-26T00:00:00Z",
                "finished_at_utc": "2026-09-26T00:00:01Z",
                "api_calls": 1,
                "transport_attempts": 1,
                "total_tokens": 1,
                "response_sha256": "c" * 64,
                "request_id": None,
                "provider_error": None,
                "failure_class": None,
                "failure_reason": None,
                "candidate_generated": False,
                "response_content_persisted": False,
            })
            with self.assertRaisesRegex(task_045.CandidateGateError, "preflight is missing"):
                task_045.build_candidate_gate_a_preflight({}, auth_probe_hash=fake_hash)

            tampered = copy.deepcopy(base)
            tampered["model"]["api_base"] = "https://tampered.invalid/v4/"
            task_045._atomic_json(task_045.CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH, tampered)
            with self.assertRaisesRegex(task_045.CandidateGateError, "self-hash is invalid"):
                task_045.build_candidate_gate_a_preflight(
                    {}, auth_probe_hash=base["preflight_sha256"]
                )

            for field, value in (
                (("model", "model"), "stale-model"),
                (("bindings", "probe_runner_sha256"), "d" * 64),
            ):
                stale = copy.deepcopy(base)
                stale[field[0]][field[1]] = value
                stale["preflight_sha256"] = task_045.preflight_sha256(stale)
                task_045._atomic_json(
                    task_045.CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH,
                    stale,
                )
                with self.assertRaisesRegex(task_045.CandidateGateError, "bindings are stale"):
                    task_045.build_candidate_gate_a_preflight(
                        {}, auth_probe_hash=stale["preflight_sha256"]
                    )

            task_045._atomic_json(task_045.CANDIDATE_GATE_A_AUTH_PROBE_PREFLIGHT_PATH, base)
            result = asyncio.run(task_045.execute_candidate_gate_a_auth_probe_once(
                base,
                approved=True,
                supplied_hash=base["preflight_sha256"],
                api_key="fake-key",
                transport=FakeTransport({
                    "id": "probe-response",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                    "usage": {"total_tokens": 2},
                }),
            ))
            result_path = Path(result["result_path"])
            original = json.loads(result_path.read_text(encoding="utf-8"))

            changed_field = copy.deepcopy(original)
            changed_field["candidate_generated"] = True
            task_045._atomic_json(result_path, changed_field)
            with self.assertRaisesRegex(task_045.CandidateGateError, "did not pass"):
                task_045.build_candidate_gate_a_preflight(
                    {}, auth_probe_hash=base["preflight_sha256"]
                )

            changed_hash = copy.deepcopy(original)
            changed_hash["response_sha256"] = "e" * 64
            task_045._atomic_json(result_path, changed_hash)
            with self.assertRaisesRegex(task_045.CandidateGateError, "result hash is invalid"):
                task_045.build_candidate_gate_a_preflight(
                    {}, auth_probe_hash=base["preflight_sha256"]
                )


if __name__ == "__main__":
    unittest.main()
