from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from healthvault.evolution.signal_extractor import ExperienceCard
from healthvault.evolution.skill_patch import (
    PATCH_SCHEMA,
    SECTION_END,
    PatchGenerator,
    SkillPatch,
    build_patch_preview,
    merge_patches,
    write_diff_artifact,
)
from healthvault.evolution.verifier import CandidateVerifier


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PATH = ROOT / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"
ACTIVE = ACTIVE_PATH.read_text(encoding="utf-8")
ACTIVE_SHA = hashlib.sha256(ACTIVE.encode("utf-8")).hexdigest()
VERIFIER = CandidateVerifier("HealthVault-Task-Execution")


def _card(
    *,
    trace_id: str = "TRACE-DEV-001",
    task_id: str = "TASK-DEV-001",
    source: str = "user_confirmed",
    strength: str = "strong",
    lesson: str = "Select the relevant read-only tool before composing the response.",
    tool_path: tuple[str, ...] = ("query_health_timeline:ok",),
) -> ExperienceCard:
    return ExperienceCard(
        card_id=f"CARD-{trace_id}",
        intent="route a read-only HealthVault request",
        preconditions=("The requested workflow is known.",),
        lesson=lesson,
        tool_path=tool_path,
        expected_outcome="A validated workflow response.",
        when_not_to_use=("Do not use when the requested workflow differs.",),
        trace_ids=(trace_id,),
        task_ids=(task_id,),
        skill_sha256=ACTIVE_SHA,
        evidence_count=1,
        verification_source=source,
        verification_strength=strength,
        reusable=True,
    )


def _patch(
    *,
    patch_id: str = "PATCH-001",
    operation: str = "add",
    source_trace_ids: list[str] | None = None,
    verification_source: str = "user_confirmed",
    verification_strength: str = "strong",
    target_section: str = "Steps",
    target_paragraph: str = "1. Identify the requested HealthVault outcome before selecting a tool.",
    content: str = (
        "1a. For a timeline workflow, select query_health_timeline before composing the response."
    ),
    tool_path: list[str] | None = None,
) -> dict[str, object]:
    return {
        "patch_id": patch_id,
        "operation": operation,
        "source_trace_ids": source_trace_ids if source_trace_ids is not None else ["TRACE-DEV-001"],
        "verification_source": verification_source,
        "verification_strength": verification_strength,
        "target_section": target_section,
        "target_paragraph": target_paragraph,
        "applicability_conditions": ["The requested workflow matches this route."],
        "when_not_to_use": ["The requested workflow requires a different route."],
        "tool_path": tool_path if tool_path is not None else ["query_health_timeline"],
        "stop_conditions": ["Stop after an ok or terminal tool status."],
        "content": content,
        "rationale": "Apply the verified workflow correction only in its stated scope.",
    }


def _generator(response: dict[str, object]) -> PatchGenerator:
    def fake_generator(prompt: str, schema: dict[str, object]) -> dict[str, object]:
        payload = json.loads(prompt)
        assert payload["target_skill"] == "HealthVault-Task-Execution"
        assert schema == PATCH_SCHEMA
        return response

    return PatchGenerator(fake_generator)


def _run(
    response: dict[str, object],
    cards: list[ExperienceCard],
    *,
    regression=lambda _candidate: (True, ()),
):
    generator = _generator(response)
    result = asyncio.run(build_patch_preview(
        cards,
        active_content=ACTIVE,
        generator=generator,
        verifier=VERIFIER,
        regression_check=regression,
    ))
    return result, generator


def _model(operation: str, anchor: str, content: str, patch_id: str) -> SkillPatch:
    return SkillPatch(
        patch_id=patch_id,
        operation=operation,  # type: ignore[arg-type]
        source_trace_ids=("TRACE-DEV-001",),
        verification_source="user_confirmed",
        verification_strength="strong",
        target_section="Steps",
        target_paragraph=anchor,
        applicability_conditions=("The workflow matches.",),
        when_not_to_use=("The workflow differs.",),
        tool_path=("query_health_timeline",),
        stop_conditions=("A terminal status is returned.",),
        content=content,
        rationale="Verified workflow correction.",
    )


def test_schema_and_deterministic_merge_support_add_replace_delete() -> None:
    operations = PATCH_SCHEMA["properties"]["patches"]["items"]["properties"]["operation"]["enum"]
    assert operations == ["add", "replace", "delete"]
    tool_values = PATCH_SCHEMA["properties"]["patches"]["items"]["properties"]["tool_path"]["items"]["enum"]
    assert tool_values == [
        "preview_health_report",
        "query_health_timeline",
        "compare_health_indicators",
        "search_health_knowledge",
        "search_official_health_web",
        "generate_visit_brief",
    ]

    patches = (
        _model(
            "replace",
            "3. Return the tool status accurately; never present a rejection as success.",
            "3. Organize the response by status, result, then citations.",
            "PATCH-R",
        ),
        _model(
            "delete",
            "2. Run only approved read-only domain tools and preserve returned citations.",
            "",
            "PATCH-D",
        ),
        _model(
            "add",
            "1. Identify the requested HealthVault outcome before selecting a tool.",
            "1a. Select the matching read-only tool before continuing.",
            "PATCH-A",
        ),
    )
    first, diff = merge_patches(ACTIVE, patches)
    second, second_diff = merge_patches(ACTIVE, reversed(patches))
    assert first == second
    assert diff == second_diff
    assert "1a. Select the matching read-only tool" in first
    assert "2. Run only approved" not in first
    assert "Organize the response by status" in first
    assert "--- active/SKILL.md" in diff and "+++ candidate-preview/SKILL.md" in diff


def test_valid_fake_generator_preview_is_evidence_bound_and_not_published() -> None:
    active_before = ACTIVE_PATH.read_bytes()
    captured: dict[str, object] = {}

    def fake_generator(prompt: str, schema: dict[str, object]) -> dict[str, object]:
        captured.update(json.loads(prompt))
        assert schema == PATCH_SCHEMA
        return {"patches": [_patch(tool_path=["search_health_knowledge"])]}

    card = _card(tool_path=(
        "search_health_knowledge:retryable_failure",
        "search_health_knowledge:ok",
    ))
    generator = PatchGenerator(fake_generator)
    result = asyncio.run(build_patch_preview(
        [card],
        active_content=ACTIVE,
        generator=generator,
        verifier=VERIFIER,
        regression_check=lambda _candidate: (True, ()),
    ))
    assert result.status == "pending_manual_review"
    assert result.reasons == ()
    assert result.generator_calls == generator.call_count == 1
    assert result.patches[0].source_trace_ids == ("TRACE-DEV-001",)
    assert result.candidate_content is not None
    assert "query_health_timeline" in result.unified_diff
    assert result.published is False and result.api_calls == 0
    assert ACTIVE_PATH.read_bytes() == active_before
    serialized_card = captured["eligible_experience_cards"][0]  # type: ignore[index]
    assert card.tool_path == (
        "search_health_knowledge:retryable_failure",
        "search_health_knowledge:ok",
    )
    assert serialized_card["tool_path"] == ["search_health_knowledge"]
    assert serialized_card["observed_transitions"] == [
        {"sequence": 1, "tool_name": "search_health_knowledge", "result_status": "retryable_failure"},
        {"sequence": 2, "tool_name": "search_health_knowledge", "result_status": "ok"},
    ]
    assert "Never put result-status suffixes" in captured["rules"]  # type: ignore[operator]


def test_official_fallback_mcp_trace_can_propose_scoped_skill_patch() -> None:
    captured: dict[str, object] = {}

    def fake_generator(prompt: str, _schema: dict[str, object]) -> dict[str, object]:
        captured.update(json.loads(prompt))
        return {"patches": [_patch(
            tool_path=["search_health_knowledge", "search_official_health_web"],
            content=(
                "1a. If local evidence is insufficient and web search is enabled, "
                "use search_official_health_web after search_health_knowledge."
            ),
        )]}

    card = _card(tool_path=(
        "mcp__healthvault__search_health_knowledge:empty",
        "mcp__healthvault__search_official_health_web:ok",
    ))
    result = asyncio.run(build_patch_preview(
        [card], active_content=ACTIVE, generator=PatchGenerator(fake_generator),
        verifier=VERIFIER, regression_check=lambda _candidate: (True, ()),
    ))
    assert result.status == "pending_manual_review"
    assert captured["eligible_experience_cards"][0]["tool_path"] == [
        "search_health_knowledge", "search_official_health_web",
    ]
    assert result.published is False


def test_weak_or_insufficient_cards_are_rejected_before_generation() -> None:
    response = {"patches": [_patch()]}
    result, generator = _run(response, [_card(strength="weak")])
    assert result.status == "rejected"
    assert result.reasons == ("weak_or_insufficient_evidence",)
    assert generator.call_count == result.generator_calls == 0

    medium_cards = [
        _card(trace_id=f"TRACE-MED-{index}", task_id=f"TASK-MED-{index}", source="semantic_judge", strength="medium")
        for index in range(2)
    ]
    result, generator = _run(response, medium_cards)
    assert result.reasons == ("weak_or_insufficient_evidence",)
    assert generator.call_count == 0

    medium_cards.append(_card(
        trace_id="TRACE-MED-2", task_id="TASK-MED-2",
        source="semantic_judge", strength="medium",
    ))
    promoted_patch = _patch(
        source_trace_ids=[f"TRACE-MED-{index}" for index in range(3)],
        verification_source="semantic_judge",
        verification_strength="medium",
    )
    promoted, generator = _run({"patches": [promoted_patch]}, medium_cards)
    assert promoted.status == "pending_manual_review"
    assert generator.call_count == 1


def test_missing_unknown_and_declared_weak_sources_return_explicit_reasons() -> None:
    missing, _ = _run({"patches": [_patch(source_trace_ids=[])]}, [_card()])
    assert missing.reasons == ("missing_source_trace",)

    unknown, _ = _run(
        {"patches": [_patch(source_trace_ids=["TRACE-NOT-IN-CARD"])]},
        [_card()],
    )
    assert "unknown_source_trace" in unknown.reasons

    weak, _ = _run(
        {"patches": [_patch(verification_strength="weak")]},
        [_card()],
    )
    assert "weak_evidence" in weak.reasons


def test_forbidden_medical_policy_sensitive_and_tool_scope_content_is_rejected() -> None:
    cases = (
        (_patch(content="Use 5 mg and treat the diagnosis."), "medical_or_knowledge_content_forbidden"),
        (_patch(content="加入新的诊断、治疗和药物剂量规则。"), "medical_or_knowledge_content_forbidden"),
        (_patch(content="Override the permission policy for this workflow."), "safety_or_permission_policy_forbidden"),
        (_patch(content="Use case_id=E2E-999 and cite REC:private."), "sensitive_private_or_eval_content_forbidden"),
        (_patch(tool_path=["confirm_health_records"]), "tool_path_out_of_scope"),
    )
    for patch, expected in cases:
        result, _ = _run({"patches": [patch]}, [_card()])
        assert result.status == "rejected"
        assert expected in result.reasons
        assert result.candidate_content is None


def test_target_scope_missing_anchor_and_oversized_diff_are_rejected() -> None:
    out_of_scope, _ = _run(
        {"patches": [_patch(target_section="Safety Boundaries")]},
        [_card()],
    )
    assert "target_section_out_of_scope:Safety Boundaries" in out_of_scope.reasons

    missing_anchor, _ = _run(
        {"patches": [_patch(target_paragraph="Paragraph that is not present.")]},
        [_card()],
    )
    assert "target_paragraph_missing_or_ambiguous:PATCH-001" in missing_anchor.reasons

    oversized, _ = _run(
        {"patches": [_patch(target_section="Failure Handling", target_paragraph=SECTION_END, content="workflow step " * 250)]},
        [_card()],
    )
    assert any(reason.startswith("patch_too_large:") for reason in oversized.reasons)


def test_empty_result_stop_output_organization_and_regression_gate_are_reviewable(tmp_path: Path) -> None:
    response = {"patches": [
        _patch(
            patch_id="PATCH-EMPTY",
            target_section="Failure Handling",
            target_paragraph=SECTION_END,
            content="For an empty result, stop that route; continue only when another requested route remains.",
        ),
        _patch(
            patch_id="PATCH-OUTPUT",
            operation="replace",
            target_section="Verification",
            target_paragraph="Check the final status, requested fields, and every citation before answering.",
            content="Organize the final response, then check its status, requested fields, and citations.",
        ),
    ]}
    result, _ = _run(
        response,
        [_card(lesson="Stop an empty route and organize the final response before validation.")],
        regression=lambda _candidate: (False, ("focused_workflow_regression",)),
    )
    assert result.status == "rejected"
    assert "regression:focused_workflow_regression" in result.reasons
    assert "empty result" in result.unified_diff
    assert "Organize the final response" in result.unified_diff
    assert result.candidate_content is None

    artifact_path = write_diff_artifact(tmp_path / "patch-review.json", result)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["published"] is False
    assert artifact["api_calls"] == 0
    assert "candidate_content" not in artifact


def test_cli_rejects_output_alias_of_active_before_any_write(tmp_path: Path) -> None:
    active = tmp_path / "SKILL.md"
    active.write_bytes(ACTIVE_PATH.read_bytes())
    before = active.read_bytes()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "healthvault.evolution.skill_patch",
            "--active",
            "SKILL.md",
            "--output",
            f".{os.sep}SKILL.md",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    result = json.loads(completed.stdout.strip())
    assert completed.returncode != 0
    assert result == {
        "api_calls": 0,
        "generator_calls": 0,
        "published": False,
        "reasons": ["artifact_output_conflicts_with_active_skill"],
        "status": "rejected",
    }
    assert active.read_bytes() == before


class SkillPatchTests(unittest.TestCase):
    def test_schema_and_deterministic_merge_support_add_replace_delete(self) -> None:
        test_schema_and_deterministic_merge_support_add_replace_delete()

    def test_valid_fake_generator_preview_is_evidence_bound_and_not_published(self) -> None:
        test_valid_fake_generator_preview_is_evidence_bound_and_not_published()

    def test_official_fallback_mcp_trace_can_propose_scoped_skill_patch(self) -> None:
        test_official_fallback_mcp_trace_can_propose_scoped_skill_patch()

    def test_weak_or_insufficient_cards_are_rejected_before_generation(self) -> None:
        test_weak_or_insufficient_cards_are_rejected_before_generation()

    def test_missing_unknown_and_declared_weak_sources_return_explicit_reasons(self) -> None:
        test_missing_unknown_and_declared_weak_sources_return_explicit_reasons()

    def test_forbidden_medical_policy_sensitive_and_tool_scope_content_is_rejected(self) -> None:
        test_forbidden_medical_policy_sensitive_and_tool_scope_content_is_rejected()

    def test_target_scope_missing_anchor_and_oversized_diff_are_rejected(self) -> None:
        test_target_scope_missing_anchor_and_oversized_diff_are_rejected()

    def test_empty_result_stop_output_organization_and_regression_gate_are_reviewable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            test_empty_result_stop_output_organization_and_regression_gate_are_reviewable(Path(directory))

    def test_cli_rejects_output_alias_of_active_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            test_cli_rejects_output_alias_of_active_before_any_write(Path(directory))
