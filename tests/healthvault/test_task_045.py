from __future__ import annotations

import copy
import asyncio
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from healthvault.evaluation.skill import task_045
from healthvault.evolution import skill_patch as patching


REAL_GATE_A_CONSUMPTION_ROOT = (
    task_045.ROOT / ".task_outputs" / "TASK-045" / "candidate-gate-a-consumptions"
)
CONSUMED_GATE_A_HASHES = (
    "5aa31505342965feecef5f2a15021793e2afdf2661f5202e22c80e0aaebb5d2f",
    "370d9f15a943b9dacbd95cd989787dba42326b5c35391c11e2921fae16ec09ae",
)


def _candidate(path: Path, *, trace_ids: list[str] | None = None, content: str | None = None) -> Path:
    active = task_045.ACTIVE_SKILL_PATH.read_text(encoding="utf-8")
    source_trace_ids = trace_ids or ["TASK037-DEV-1"]
    replacement = content or "Organize the response, then check the final status, requested fields, and citations."
    raw_patch = {
        "patch_id": "PATCH-TASK045-TEST",
        "operation": "replace",
        "source_trace_ids": source_trace_ids,
        "verification_source": "user_confirmed",
        "verification_strength": "strong",
        "target_section": "Verification",
        "target_paragraph": "Check the final status, requested fields, and every citation before answering.",
        "applicability_conditions": ["A final response is ready to validate."],
        "when_not_to_use": ["A terminal tool result still needs handling."],
        "tool_path": ["query_health_timeline"],
        "stop_conditions": ["Status and citations have been checked."],
        "content": replacement,
        "rationale": "Apply a verified local workflow correction.",
    }
    patches = patching._parse_patch_response({"patches": [raw_patch]})
    candidate_content, unified_diff = patching.merge_patches(active, patches)
    candidate_sha = hashlib.sha256(candidate_content.encode("utf-8")).hexdigest()
    patch_artifact = {
        "schema_version": "healthvault-skill-local-patch-artifact-v1",
        "status": "pending_manual_review",
        "reasons": [],
        "patches": [raw_patch],
        "active_skill_sha256": hashlib.sha256(active.encode("utf-8")).hexdigest(),
        "candidate_skill_sha256": candidate_sha,
        "unified_diff": unified_diff,
        "generator_calls": 1,
        "published": False,
        "api_calls": 0,
    }
    patch_path = path.with_name(path.stem + "-patch.json")
    patch_path.write_text(json.dumps(patch_artifact, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    value = {
        "schema_version": "healthvault-task-045-candidate-v1",
        "candidate_id": "TASK045-FAKE-CANDIDATE",
        "status": "frozen_unpublished",
        "active_skill_sha256": hashlib.sha256(active.encode("utf-8")).hexdigest(),
        "candidate_content": candidate_content,
        "candidate_sha256": candidate_sha,
        "patch_artifact_path": str(patch_path),
        "patch_artifact_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        "source_trace_ids": source_trace_ids,
        "source_splits": ["dev"],
        "verifier_status": "verified",
        "gate_a_preflight_sha256": json.loads(
            task_045.CANDIDATE_GATE_A_PREFLIGHT_PATH.read_text(encoding="utf-8")
        )["preflight_sha256"],
        "frozen_at_utc": "2026-09-24T00:00:00Z",
    }
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    consumption_path = task_045._candidate_gate_a_consumption_path(
        value["gate_a_preflight_sha256"]
    )
    consumption_path.parent.mkdir(parents=True, exist_ok=True)
    consumption_path.write_text(json.dumps({
        "schema_version": "healthvault-task-045-candidate-gate-a-consumption-v1",
        "status": "completed",
        "disposition": "frozen_unpublished",
        "preflight_sha256": value["gate_a_preflight_sha256"],
        "candidate_sha256": candidate_sha,
        "patch_artifact_sha256": value["patch_artifact_sha256"],
        "candidate_artifact_path": str(path.resolve()),
        "patch_artifact_path": str(patch_path.resolve()),
        "candidate_artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return path


def _result(
    arm: str,
    case_id: str,
    *,
    passed: bool,
    environment_incident: bool = False,
) -> dict[str, object]:
    return {
        "arm_id": arm,
        "case_id": case_id,
        "category": "timeline",
        "passed": passed,
        "hard_failure": not passed and not environment_incident,
        "oracle": {
            "outcome": {"passed": passed},
            "behavior": {"passed": passed},
            "response": {"passed": passed},
        },
        "attempt": {
            "calls": [] if not passed else [{"name": "query_health_timeline"}],
            "environment_incident": environment_incident,
            "transport_exhausted": environment_incident,
            "incident_class": "provider_or_transport" if environment_incident else None,
            "runtime_error": "TransportError" if environment_incident else None,
            "parser_error": None,
            "latency_ms": 10.0,
            "total_tokens": 100,
            "api_calls": 1,
        },
    }


def _matrix(active_passes: set[int], candidate_passes: set[int]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index in range(1, 21):
        case_id = f"S5-{index:02d}"
        results.append(_result("active", case_id, passed=index in active_passes))
        results.append(_result("candidate", case_id, passed=index in candidate_passes))
    return results


class Task045Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._gate_directory = tempfile.TemporaryDirectory()
        root = Path(cls._gate_directory.name)
        cls._preflight_patch = patch.object(
            task_045,
            "CANDIDATE_GATE_A_PREFLIGHT_PATH",
            root / "candidate-gate-a-preflight.json",
        )
        cls._consumption_patch = patch.object(
            task_045,
            "CANDIDATE_GATE_A_CONSUMPTION_ROOT",
            root / "consumptions",
        )
        cls._preflight_patch.start()
        cls._consumption_patch.start()
        task_045._atomic_json(
            task_045.CANDIDATE_GATE_A_PREFLIGHT_PATH,
            task_045.build_candidate_gate_a_preflight({}),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._consumption_patch.stop()
        cls._preflight_patch.stop()
        cls._gate_directory.cleanup()

    def test_dataset_fixture_and_three_layer_oracles_are_frozen(self) -> None:
        dataset = task_045.load_dataset()
        fixture = task_045.load_fixture()
        self.assertEqual(20, len(dataset["cases"]))
        self.assertEqual(task_045.CATEGORY_COUNTS, dict(__import__("collections").Counter(
            case["category"] for case in dataset["cases"]
        )))
        self.assertEqual(5, len({case["patient_id"] for case in dataset["cases"]}))
        self.assertTrue(all(set(case["expected"]) >= {
            "required_tools", "allowed_tools", "required_tool_result_statuses",
            "final_status", "refusal_required", "allowed_citation_ids",
        } for case in dataset["cases"]))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "fixture.sqlite3"
            first = task_045.materialize_fixture(target)
            self.assertEqual(fixture["report_count"], first["report_count"])
            with self.assertRaises(ValueError):
                task_045.materialize_fixture(target)
        audit = task_045.build_leakage_audit(dataset)
        self.assertTrue(audit["baseline_isolation_passed"])
        self.assertEqual(0, audit["patient_profile_split_conflict_count"])
        self.assertEqual(0, audit["text_collision_count"])
        self.assertEqual("pending", audit["candidate_audit_status"])
        self.assertFalse(audit["passed"])
        self.assertIn("candidate_not_frozen", audit["candidate_audit_reasons"])

    def test_candidate_gate_a_isolated_fake_generation_freezes_replayable_unpublished_candidate(self) -> None:
        class FakeTransport:
            is_real = False

            def __init__(self, response: dict[str, object]) -> None:
                self.response = response
                self.requests: list[dict[str, object]] = []

            async def post_json(
                self,
                _url: str,
                _headers: dict[str, str],
                payload: dict[str, object],
                _timeout: float,
            ) -> dict[str, object]:
                self.requests.append(copy.deepcopy(payload))
                return copy.deepcopy(self.response)

        with patch.object(task_045, "load_dataset", side_effect=AssertionError("held-out loaded by preflight")):
            preflight = task_045.build_candidate_gate_a_preflight({})
        self.assertEqual("API_GATE_PENDING", preflight["status"])
        self.assertEqual(0, preflight["api_calls_made"])
        self.assertEqual(1, preflight["limits"]["maximum_model_calls"])
        self.assertEqual(list(task_045.CANDIDATE_SOURCE_TRACE_IDS), preflight["generator_inputs"]["source_trace_ids"])
        self.assertFalse(preflight["generator_inputs"]["heldout_or_oracle_exposed"])

        raw_patch = {
            "patch_id": "PATCH-TASK045-RETRY-CONTRACT",
            "operation": "add",
            "source_trace_ids": list(task_045.CANDIDATE_SOURCE_TRACE_IDS),
            "verification_source": "deterministic",
            "verification_strength": "strong",
            "target_anchor_id": "steps.run_read_only_tools",
            "applicability_conditions": ["A permitted read-only tool explicitly returns retryable_failure."],
            "when_not_to_use": ["The result is rejected, empty, or otherwise non-retryable."],
            "tool_path": ["search_health_knowledge"],
            "stop_conditions": ["Stop after the retried call returns its terminal result."],
            "content": (
                "After a retryable result, retry only the same permitted call once, then "
                "require the validator contract before reporting its terminal result."
            ),
            "rationale": "Preserve the verified retry and validation workflow from all source Traces.",
        }
        response = {
            "choices": [{"message": {"role": "assistant", "content": json.dumps({"patches": [raw_patch]})}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        active_before = hashlib.sha256(task_045.ACTIVE_SKILL_PATH.read_bytes()).hexdigest()
        active_content = task_045.ACTIVE_SKILL_PATH.read_text(encoding="utf-8")
        anchor_table = task_045._candidate_gate_a_anchor_table(active_content=active_content)
        schema = task_045._candidate_gate_a_patch_schema(active_content=active_content)
        patches_contract = schema["properties"]["patches"]
        required = schema["properties"]["patches"]["items"]["required"]
        schema_operations = schema["properties"]["patches"]["items"]["properties"][
            "operation"
        ]["enum"]
        content_contract = schema["properties"]["patches"]["items"]["properties"]["content"]
        self.assertIn("target_anchor_id", required)
        self.assertNotIn("target_section", required)
        self.assertNotIn("target_paragraph", required)
        self.assertEqual(["add"], schema_operations)
        self.assertEqual(1, patches_contract["minItems"])
        self.assertEqual(1, patches_contract["maxItems"])
        self.assertEqual(240, content_contract["maxLength"])
        self.assertEqual(task_045.CANDIDATE_GATE_A_CONTENT_PATTERN, content_contract["pattern"])
        for anchor in anchor_table["anchors"]:
            for schema_operation in schema_operations:
                resolved = task_045.resolve_candidate_gate_a_anchor(
                    anchor["target_anchor_id"],
                    schema_operation,
                    active_content=active_content,
                    anchor_table=anchor_table,
                )
                self.assertEqual(anchor["target_section"], resolved["target_section"])
                self.assertEqual(anchor["target_paragraph"], resolved["target_paragraph"])
        self.assertEqual(
            {"target_section": "Steps", "target_paragraph": (
                "2. Run only approved read-only domain tools and preserve returned citations."
            )},
            task_045.resolve_candidate_gate_a_anchor(
                "steps.run_read_only_tools",
                "add",
                active_content=active_content,
                anchor_table=anchor_table,
            ),
        )
        with self.assertRaisesRegex(task_045.CandidateGateError, "unknown target_anchor_id"):
            task_045.resolve_candidate_gate_a_anchor(
                "steps.unknown",
                "add",
                active_content=active_content,
                anchor_table=anchor_table,
            )
        with self.assertRaisesRegex(task_045.CandidateGateError, "operation is not allowed"):
            task_045.resolve_candidate_gate_a_anchor(
                "steps.run_read_only_tools",
                "replace",
                active_content=active_content,
                anchor_table=anchor_table,
            )
        anchor_text = anchor_table["anchors"][1]["target_paragraph"]
        for altered_content in (
            active_content.replace(anchor_text, "Anchor intentionally removed."),
            active_content.replace(anchor_text, anchor_text + "\n" + anchor_text),
        ):
            altered_table = copy.deepcopy(anchor_table)
            altered_table["active_skill_sha256"] = hashlib.sha256(
                altered_content.encode("utf-8")
            ).hexdigest()
            with self.assertRaisesRegex(task_045.CandidateGateError, "missing or ambiguous"):
                task_045.resolve_candidate_gate_a_anchor(
                    "steps.run_read_only_tools",
                    "add",
                    active_content=altered_content,
                    anchor_table=altered_table,
                )
        with self.assertRaisesRegex(task_045.CandidateGateError, "active Skill hash mismatch"):
            task_045.resolve_candidate_gate_a_anchor(
                "steps.run_read_only_tools",
                "add",
                active_content=active_content + "\n",
                anchor_table=anchor_table,
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rejected_patch = copy.deepcopy(raw_patch)
            rejected_patch["tool_path"] = [
                "search_health_knowledge:retryable_failure",
                "search_health_knowledge:ok",
            ]
            rejected_transport = FakeTransport({
                "choices": [{"message": {
                    "role": "assistant",
                    "content": json.dumps({"patches": [rejected_patch]}),
                }}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })
            rejected_consumptions = root / "rejected-consumptions"
            with patch.object(
                task_045, "CANDIDATE_GATE_A_CONSUMPTION_ROOT", rejected_consumptions
            ):
                with self.assertRaises(task_045.CandidateGateError):
                    asyncio.run(task_045.execute_candidate_gate_a_once(
                        preflight,
                        approved=True,
                        supplied_hash=preflight["preflight_sha256"],
                        api_key="fake",
                        transport=rejected_transport,
                        runtime_root=root / "rejected-first",
                    ))
                self.assertEqual(1, len(rejected_transport.requests))
                consumption = json.loads(
                    task_045._candidate_gate_a_consumption_path(
                        preflight["preflight_sha256"]
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual("failed", consumption["status"])
                self.assertEqual("rejected_consumed", consumption["disposition"])
                self.assertEqual("model_response", consumption["transport_attempts"][0]["status"])
                failure_path = Path(consumption["validation_failure_artifact_path"])
                self.assertTrue(failure_path.is_file())
                self.assertEqual(
                    hashlib.sha256(failure_path.read_bytes()).hexdigest(),
                    consumption["validation_failure_artifact_sha256"],
                )
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
                self.assertEqual("rejected_consumed", failure["disposition"])
                self.assertIn("tool_path_out_of_scope", failure["validation_reasons"])
                self.assertEqual([
                    "search_health_knowledge:retryable_failure",
                    "search_health_knowledge:ok",
                ], [item["value"] for item in failure["rejected_fields"][0]["values"]])
                self.assertFalse(failure["contains_api_key"])
                self.assertFalse(failure["contains_authorization_header"])
                self.assertFalse(failure["contains_raw_model_response"])
                self.assertEqual("PATCH-TASK045-RETRY-CONTRACT", failure["patch_controls"][0]["patch_id"])
                self.assertEqual("add", failure["patch_controls"][0]["operation"])
                self.assertEqual(
                    "steps.run_read_only_tools",
                    failure["patch_controls"][0]["target_anchor_id"],
                )
                self.assertEqual("Steps", failure["patch_controls"][0]["resolved_target_section"])
                second_transport = FakeTransport(response)
                with self.assertRaisesRegex(task_045.ApiGateError, "already consumed"):
                    asyncio.run(task_045.execute_candidate_gate_a_once(
                        preflight,
                        approved=True,
                        supplied_hash=preflight["preflight_sha256"],
                        api_key="fake",
                        transport=second_transport,
                        runtime_root=root / "rejected-second",
                    ))
                self.assertEqual([], second_transport.requests)

            free_text_patch = copy.deepcopy(raw_patch)
            free_text_patch.pop("target_anchor_id")
            free_text_patch["target_section"] = "Steps"
            free_text_patch["target_paragraph"] = (
                "2. Run only approved read-only domain tools and preserve returned citations."
            )
            free_text_transport = FakeTransport({
                "choices": [{"message": {
                    "role": "assistant",
                    "content": json.dumps({"patches": [free_text_patch]}),
                }}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })
            with patch.object(
                task_045,
                "CANDIDATE_GATE_A_CONSUMPTION_ROOT",
                root / "free-text-consumptions",
            ):
                with self.assertRaisesRegex(task_045.CandidateGateError, "free-text target"):
                    asyncio.run(task_045.execute_candidate_gate_a_once(
                        preflight,
                        approved=True,
                        supplied_hash=preflight["preflight_sha256"],
                        api_key="fake",
                        transport=free_text_transport,
                        runtime_root=root / "free-text",
                    ))
                self.assertEqual(1, len(free_text_transport.requests))
                free_text_consumption = json.loads(
                    task_045._candidate_gate_a_consumption_path(
                        preflight["preflight_sha256"]
                    ).read_text(encoding="utf-8")
                )
                free_text_failure = json.loads(Path(
                    free_text_consumption["validation_failure_artifact_path"]
                ).read_text(encoding="utf-8"))
                self.assertEqual(
                    "Candidate Gate A free-text target fields are forbidden",
                    free_text_failure["validation_reasons"][0],
                )
                self.assertEqual("[REDACTED]", free_text_failure["patch_controls"][0]["target_anchor_id"])

            unknown_anchor_patch = copy.deepcopy(raw_patch)
            unknown_anchor_patch["target_anchor_id"] = "steps.not_registered"
            unknown_anchor_transport = FakeTransport({
                "choices": [{"message": {
                    "role": "assistant",
                    "content": json.dumps({"patches": [unknown_anchor_patch]}),
                }}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })
            with patch.object(
                task_045,
                "CANDIDATE_GATE_A_CONSUMPTION_ROOT",
                root / "unknown-anchor-consumptions",
            ):
                with self.assertRaisesRegex(task_045.CandidateGateError, "unknown target_anchor_id"):
                    asyncio.run(task_045.execute_candidate_gate_a_once(
                        preflight,
                        approved=True,
                        supplied_hash=preflight["preflight_sha256"],
                        api_key="fake",
                        transport=unknown_anchor_transport,
                        runtime_root=root / "unknown-anchor",
                    ))
                self.assertEqual(1, len(unknown_anchor_transport.requests))
                unknown_consumption = json.loads(
                    task_045._candidate_gate_a_consumption_path(
                        preflight["preflight_sha256"]
                    ).read_text(encoding="utf-8")
                )
                unknown_failure = json.loads(Path(
                    unknown_consumption["validation_failure_artifact_path"]
                ).read_text(encoding="utf-8"))
                self.assertEqual(
                    "Candidate Gate A unknown target_anchor_id",
                    unknown_failure["validation_reasons"][0],
                )
                self.assertEqual("[REDACTED]", unknown_failure["patch_controls"][0]["target_anchor_id"])

            long_content_patch = copy.deepcopy(raw_patch)
            long_content_patch["content"] = (
                "Retry the same permitted read-only knowledge lookup once after a retryable result "
                * 4
            ).strip() + "."
            self.assertGreater(len(long_content_patch["content"]), 240)
            long_content_transport = FakeTransport({
                "choices": [{"message": {
                    "role": "assistant",
                    "content": json.dumps({"patches": [long_content_patch]}),
                }}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })
            with patch.object(
                task_045,
                "CANDIDATE_GATE_A_CONSUMPTION_ROOT",
                root / "long-content-consumptions",
            ):
                with self.assertRaisesRegex(task_045.CandidateGateError, "one sentence.*240"):
                    asyncio.run(task_045.execute_candidate_gate_a_once(
                        preflight,
                        approved=True,
                        supplied_hash=preflight["preflight_sha256"],
                        api_key="fake",
                        transport=long_content_transport,
                        runtime_root=root / "long-content",
                    ))
                self.assertEqual(1, len(long_content_transport.requests))
                long_consumption = json.loads(
                    task_045._candidate_gate_a_consumption_path(
                        preflight["preflight_sha256"]
                    ).read_text(encoding="utf-8")
                )
                long_failure = json.loads(Path(
                    long_consumption["validation_failure_artifact_path"]
                ).read_text(encoding="utf-8"))
                self.assertEqual(
                    "Candidate Gate A Patch content must be one sentence of at most 240 characters",
                    long_failure["validation_reasons"][0],
                )

            multi_sentence_patch = copy.deepcopy(raw_patch)
            multi_sentence_patch["content"] = "Retry the permitted call once. Then validate it."
            with self.assertRaisesRegex(task_045.CandidateGateError, "one sentence.*240"):
                task_045._resolve_candidate_gate_a_model_response(
                    {"patches": [multi_sentence_patch]},
                    active_content=active_content,
                )

            maximum_content_patch = copy.deepcopy(raw_patch)
            maximum_content_patch["content"] = "R" * 239 + "."
            maximum_response = task_045._resolve_candidate_gate_a_model_response(
                {"patches": [maximum_content_patch]},
                active_content=active_content,
            )
            maximum_parsed = patching._parse_patch_response(maximum_response)
            maximum_candidate, maximum_diff = patching.merge_patches(
                active_content,
                maximum_parsed,
            )
            self.assertIn(maximum_content_patch["content"], maximum_candidate)
            self.assertTrue(maximum_diff)

            second_patch = copy.deepcopy(raw_patch)
            second_patch["patch_id"] = "PATCH-TASK045-SECOND-RULE"
            second_patch["target_anchor_id"] = "verification.check_final_response"
            raw_patch["content"] = "A" * 199 + "."
            second_patch["content"] = "B" * 199 + "."
            with self.assertRaisesRegex(task_045.CandidateGateError, "Patch count is invalid"):
                task_045._resolve_candidate_gate_a_model_response(
                    {"patches": [raw_patch, second_patch]},
                    active_content=active_content,
                )

            transport = FakeTransport(response)
            with self.assertRaises(task_045.ApiGateError):
                asyncio.run(task_045.execute_candidate_gate_a_once(
                    preflight,
                    approved=False,
                    supplied_hash=preflight["preflight_sha256"],
                    api_key="fake",
                    transport=transport,
                    runtime_root=root,
                ))
            self.assertEqual([], transport.requests)

            original_load_dataset = task_045.load_dataset

            def guarded_load_dataset(*args: object, **kwargs: object) -> dict[str, object]:
                self.assertEqual(1, len(transport.requests), "held-out opened before Patch generation finished")
                return original_load_dataset(*args, **kwargs)

            with patch.object(task_045, "load_dataset", side_effect=guarded_load_dataset):
                result = asyncio.run(task_045.execute_candidate_gate_a_once(
                    preflight,
                    approved=True,
                    supplied_hash=preflight["preflight_sha256"],
                    api_key="fake",
                    transport=transport,
                    runtime_root=root,
                ))
            candidate_path = Path(result["candidate_path"])
            candidate = task_045.load_frozen_candidate(candidate_path)
            artifact_path = Path(candidate["patch_artifact_path"])
            self.assertEqual(hashlib.sha256(artifact_path.read_bytes()).hexdigest(), candidate["patch_artifact_sha256"])
            self.assertEqual(preflight["preflight_sha256"], candidate["gate_a_preflight_sha256"])
            self.assertEqual("frozen_unpublished", candidate["status"])
            self.assertFalse(result["published"])
            self.assertFalse(result["formal_ab_executed"])
            self.assertEqual(0, result["api_calls"])
            self.assertEqual(1, result["generator_calls"])

        user_request = json.loads(transport.requests[0]["messages"][1]["content"])
        self.assertEqual(task_045.ACTIVE_SKILL_PATH.read_text(encoding="utf-8"), user_request["active_skill"])
        self.assertEqual(list(task_045.CANDIDATE_SOURCE_TRACE_IDS), [
            item["trace_id"] for item in user_request["authorized_dev_traces"]
        ])
        generator_card = user_request["eligible_experience_cards"][0]
        self.assertEqual(["search_health_knowledge"], generator_card["tool_path"])
        self.assertEqual([
            {"sequence": 1, "tool_name": "search_health_knowledge", "result_status": "retryable_failure"},
            {"sequence": 2, "tool_name": "search_health_knowledge", "result_status": "ok"},
        ], generator_card["observed_transitions"])
        system_prompt = transport.requests[0]["messages"][0]["content"]
        self.assertIn("Every Patch tool_path item must be exactly one of", system_prompt)
        self.assertIn("search_official_health_web", system_prompt)
        self.assertIn("Never put a result-status suffix", system_prompt)
        self.assertIn("Choose only target_anchor_id", system_prompt)
        self.assertIn("Generate exactly one sourced", system_prompt)
        self.assertIn("one short workflow sentence of at most 240 characters", system_prompt)
        self.assertNotIn('"target_paragraph"', system_prompt)
        self.assertNotIn('"target_section"', system_prompt)
        self.assertEqual(
            "steps.run_read_only_tools",
            user_request["candidate_gate_a_anchor_catalog"][1]["target_anchor_id"],
        )
        serialized_input = json.dumps(user_request, ensure_ascii=False)
        self.assertNotIn("S5-", serialized_input)
        self.assertNotIn("s5-cobalt", serialized_input)
        self.assertNotIn("S45-COB", serialized_input)
        self.assertEqual(active_before, hashlib.sha256(task_045.ACTIVE_SKILL_PATH.read_bytes()).hexdigest())

        for consumed_hash in CONSUMED_GATE_A_HASHES:
            record_path = REAL_GATE_A_CONSUMPTION_ROOT / f"{consumed_hash}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual("failed", record["status"])
            self.assertEqual("rejected_consumed", record["disposition"])
            old_transport = FakeTransport(response)
            with patch.object(
                task_045,
                "CANDIDATE_GATE_A_CONSUMPTION_ROOT",
                REAL_GATE_A_CONSUMPTION_ROOT,
            ):
                with self.assertRaises(task_045.ApiGateError):
                    asyncio.run(task_045.execute_candidate_gate_a_once(
                        preflight,
                        approved=True,
                        supplied_hash=consumed_hash,
                        api_key="fake",
                        transport=old_transport,
                        runtime_root=Path(directory) / f"old-{consumed_hash[:8]}",
                    ))
            self.assertEqual([], old_transport.requests)

        with tempfile.TemporaryDirectory() as anchor_directory:
            changed_anchor_path = Path(anchor_directory) / "anchors.json"
            changed_anchor_table = copy.deepcopy(anchor_table)
            first_paragraph = changed_anchor_table["anchors"][0]["target_paragraph"]
            changed_anchor_table["anchors"][0]["target_paragraph"] = (
                changed_anchor_table["anchors"][1]["target_paragraph"]
            )
            changed_anchor_table["anchors"][1]["target_paragraph"] = first_paragraph
            changed_anchor_path.write_text(
                json.dumps(changed_anchor_table, ensure_ascii=False),
                encoding="utf-8",
            )
            with patch.object(
                task_045,
                "CANDIDATE_GATE_A_ANCHOR_TABLE_PATH",
                changed_anchor_path,
            ):
                changed_preflight = task_045.build_candidate_gate_a_preflight({})
            self.assertNotEqual(preflight["preflight_sha256"], changed_preflight["preflight_sha256"])

    def test_missing_or_hash_mismatched_patch_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = _candidate(root / "missing.json")
            Path(json.loads(missing.read_text(encoding="utf-8"))["patch_artifact_path"]).unlink()
            with self.assertRaisesRegex(task_045.CandidateGateError, "Patch artifact is missing"):
                task_045.build_preflight(missing, environ={})

            mismatched = _candidate(root / "mismatched.json")
            value = json.loads(mismatched.read_text(encoding="utf-8"))
            value["patch_artifact_sha256"] = "a" * 64
            mismatched.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(task_045.CandidateGateError, "Patch artifact hash mismatch"):
                task_045.build_preflight(mismatched, environ={})

    def test_unknown_trace_is_rejected_against_authoritative_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate(root / "unknown.json", trace_ids=["TRACE-NOT-AUTHORIZED"])
            with self.assertRaisesRegex(task_045.CandidateGateError, "absent from the authorized"):
                task_045.build_preflight(candidate, environ={})

            forged = _candidate(root / "forged-gate.json")
            value = json.loads(forged.read_text(encoding="utf-8"))
            value["gate_a_preflight_sha256"] = "f" * 64
            forged.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(task_045.CandidateGateError, "approved Candidate Gate A"):
                task_045.build_preflight(forged, environ={})

    def test_heldout_case_id_and_oracle_targeting_are_rejected(self) -> None:
        dataset = task_045.load_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targeted = _candidate(
                root / "targeted.json",
                content="For S5-01 use query_health_timeline and one citation.",
            )
            audit = task_045.build_leakage_audit(dataset, candidate_path=targeted)
            self.assertIn(
                "candidate_contains_heldout_identifier:case_id:S5-01",
                audit["candidate_audit_reasons"],
            )
            with self.assertRaisesRegex(task_045.CandidateGateError, "leakage audit failed"):
                task_045.build_preflight(targeted, environ={})

            oracle_targeted = _candidate(
                root / "oracle.json",
                content="For the expected answer, use query_health_timeline and one citation.",
            )
            audit = task_045.build_leakage_audit(dataset, candidate_path=oracle_targeted)
            self.assertIn(
                "candidate_contains_oracle_or_expected_answer_targeting",
                audit["candidate_audit_reasons"],
            )

            fixture_targeted = _candidate(
                root / "fixture.json",
                content=(
                    "When profile s5-cobalt asks about filing S45-COB-02, use "
                    "query_health_timeline and one citation."
                ),
            )
            audit = task_045.build_leakage_audit(dataset, candidate_path=fixture_targeted)
            self.assertIn(
                "candidate_contains_heldout_identifier:profile_id:s5-cobalt",
                audit["candidate_audit_reasons"],
            )
            self.assertIn(
                "candidate_contains_heldout_identifier:report_id:S45-COB-02",
                audit["candidate_audit_reasons"],
            )
            with self.assertRaisesRegex(task_045.CandidateGateError, "leakage audit failed"):
                task_045.build_preflight(fixture_targeted, environ={})

    def test_preflight_binds_candidate_and_requires_exact_hash_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = _candidate(Path(directory) / "candidate.json")
            preflight = task_045.build_preflight(candidate, environ={})
            self.assertEqual("API_GATE_PENDING", preflight["status"])
            self.assertEqual(0, preflight["api_calls_made"])
            self.assertEqual(40, preflight["limits"]["attempt_count"])
            self.assertEqual("skill_content", preflight["arms"]["only_variable"])
            self.assertEqual(preflight["preflight_sha256"], task_045.preflight_sha256(preflight))
            with self.assertRaises(task_045.ApiGateError):
                task_045.assert_api_gate(
                    preflight,
                    approved=False,
                    supplied_hash=preflight["preflight_sha256"],
                    api_key="fake",
                )
            task_045.assert_api_gate(
                preflight,
                approved=True,
                supplied_hash=preflight["preflight_sha256"],
                api_key="fake",
            )
            changed = copy.deepcopy(preflight)
            changed["publish_thresholds"]["minimum_task_success_delta"] = 0.05
            self.assertNotEqual(preflight["preflight_sha256"], task_045.preflight_sha256(changed))

    def test_paired_statistics_regressions_and_incidents_never_auto_publish(self) -> None:
        summary = task_045.summarize(_matrix(set(range(1, 9)), set(range(1, 12))))
        self.assertAlmostEqual(0.15, summary["paired"]["task_success_delta"])
        self.assertEqual([], summary["paired"]["regressed_cases"])
        self.assertEqual(["S5-09", "S5-10", "S5-11"], summary["paired"]["gained_cases"])
        self.assertTrue(summary["publish_eligible"])
        self.assertEqual("publish_eligible_requires_human_signoff", summary["candidate_disposition"])
        self.assertTrue(summary["active_skill_unchanged"])

        active = set(range(1, 11))
        candidate = set(range(3, 16))
        summary = task_045.summarize(_matrix(active, candidate))
        self.assertEqual(["S5-01", "S5-02"], summary["paired"]["regressed_cases"])
        self.assertEqual(0.10, summary["paired"]["regression_rate"])
        self.assertFalse(summary["gate_checks"]["regression_rate"])
        self.assertFalse(summary["publish_eligible"])
        self.assertEqual("candidate_remains_unpublished", summary["candidate_disposition"])

        results = _matrix(set(range(1, 21)), set(range(1, 21)))
        results[0] = _result("active", "S5-01", passed=False, environment_incident=True)
        summary = task_045.summarize(results)
        self.assertEqual("invalid_environment_run", summary["run_status"])
        self.assertFalse(summary["formal_metrics_valid"])
        self.assertIsNone(summary["arms"])
        self.assertFalse(summary["publish_eligible"])
        self.assertEqual("candidate_remains_unpublished_invalid_environment", summary["candidate_disposition"])
        self.assertEqual("environment_incident", summary["badcases"][0]["classification"])

    def test_both_arms_use_production_loop_with_fake_transport_and_same_user_prompt(self) -> None:
        class OfflineOpenAI:
            def __init__(self, *, base_url: str, api_key: str) -> None:
                self.base_url = base_url

        def noop(*_args: object, **_kwargs: object) -> None:
            return None

        tqdm_utils = ModuleType("tqdm.utils"); tqdm_utils.IS_WIN = True
        tqdm_module = ModuleType("tqdm"); tqdm_module.utils = tqdm_utils
        offline_sdks = {
            "anthropic": SimpleNamespace(AsyncAnthropic=object),
            "openai": SimpleNamespace(AsyncOpenAI=OfflineOpenAI),
            "agents.ui": SimpleNamespace(**{name: noop for name in (
                "print_info", "print_divider", "print_assistant_text", "print_sub_agent_start",
                "print_sub_agent_end", "start_spinner", "stop_spinner", "print_cost",
                "print_tool_call", "print_tool_result", "print_confirmation", "print_retry", "print_error",
            )}),
            "tqdm": tqdm_module,
            "tqdm.utils": tqdm_utils,
        }
        final = json.dumps({"final_status": "rejected", "citations": [], "refusal": True})
        response = {
            "choices": [{"message": {"role": "assistant", "content": final, "tool_calls": None}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        case = {item["case_id"]: item for item in task_045.load_dataset()["cases"]}["S5-16"]
        active = task_045.ACTIVE_SKILL_PATH.read_text(encoding="utf-8")
        candidate = active.replace(
            "Check the final status, requested fields, and every citation before answering.",
            "Organize the response, then check the final status, requested fields, and citations.",
        )
        with patch.dict(sys.modules, offline_sdks):
            from healthvault.evaluation.skill.benchmark.task_022 import ScriptedFakeTransport
            transports = {arm: ScriptedFakeTransport([copy.deepcopy(response)]) for arm in task_045.ARM_IDS}
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
                results = {
                    arm: __import__("asyncio").run(task_045.run_case(
                        case,
                        arm_id=arm,
                        skill_content=active if arm == "active" else candidate,
                        model_config=task_045._model_config({}),
                        transport=transports[arm],
                        ledger=None,
                        runtime_root=Path(directory),
                        api_key="fake",
                    ))
                    for arm in task_045.ARM_IDS
                }
        self.assertTrue(all(result["passed"] for result in results.values()), results)
        self.assertTrue(all(result["attempt"]["api_calls"] == 0 for result in results.values()))
        user_prompts = []
        for arm in task_045.ARM_IDS:
            user_prompts.append(next(
                message["content"] for message in reversed(transports[arm].requests[0]["messages"])
                if message.get("role") == "user"
            ))
        self.assertEqual(user_prompts[0], user_prompts[1])
        self.assertEqual(task_045.MODEL_CONFIG["model"], transports["active"].requests[0]["model"])


if __name__ == "__main__":
    unittest.main()
