"""Evidence-bound local Patch generation for the shared HealthVault Skill.

This module produces review previews only.  It never writes the active Skill,
publishes a candidate, or constructs a real model transport.
"""

from __future__ import annotations

import difflib
import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Literal, Mapping

from agents.frontmatter import parse_frontmatter
from healthvault.evolution.signal_extractor import ExperienceCard, eligible_cards
from healthvault.evolution.verifier import CandidateVerifier
from healthvault.tools.directory import SKILL_PATCH_TOOL_NAMES


PATCH_SCHEMA_PATH = Path(__file__).with_name("skill-patch-schema.json")
PATCH_SCHEMA: dict[str, Any] = json.loads(PATCH_SCHEMA_PATH.read_text(encoding="utf-8"))
PATCH_OPERATIONS = frozenset({"add", "replace", "delete"})
VERIFICATION_SOURCES = frozenset({"deterministic", "user_confirmed", "semantic_judge"})
VERIFICATION_STRENGTHS = frozenset({"strong", "medium", "weak"})
ALLOWED_TARGET_SECTIONS = frozenset({
    "When to Use", "Prerequisites", "Steps", "Failure Handling", "Verification",
})
ALLOWED_TOOL_PATH_VALUES = SKILL_PATCH_TOOL_NAMES
ALLOWED_TOOL_PATH = frozenset(ALLOWED_TOOL_PATH_VALUES)
PATCH_SCHEMA["properties"]["patches"]["items"]["properties"]["tool_path"]["items"] = {
    "type": "string",
    "enum": list(ALLOWED_TOOL_PATH_VALUES),
}
MAX_PATCHES = 3
MAX_CHANGED_LINES = 24
MAX_CHANGED_CHARACTERS = 1600
MAX_CHANGE_RATIO = 0.35
SECTION_END = "__section_end__"

_MEDICAL_OR_KNOWLEDGE_PATTERN = re.compile(
    r"(?i)(\b(?:diagnos(?:e|is)|treat(?:ment)?|prescrib\w*|dos(?:e|age)|medication|drug)\b|"
    r"\b(?:a1c|hba1c|glucose|cholesterol|tsh|ferritin|blood pressure)\b|"
    r"reference\s+(?:range|value)|normal\s+(?:range|value)|knowledge\s+(?:base|corpus)|"
    r"\b\d+(?:\.\d+)?\s*(?:mg|mg/dl|mmol/l|miu/l|mmhg|%)\b|"
    r"诊断|治疗|药物|用药|剂量|参考值|正常范围|知识库|医学事实)"
)
_POLICY_OR_PERMISSION_PATTERN = re.compile(
    r"(?i)(safety\s+boundar|permission\s+polic|bypass\s+(?:permission|safety)|"
    r"enable\s+(?:write|shell|network)|cross[- ]profile\s+access|override\s+runtime|"
    r"安全边界|权限策略|绕过(?:权限|安全)|启用(?:写入|网络|shell))"
)
_SENSITIVE_OR_CASE_PATTERN = re.compile(
    r"(?i)(api[_ -]?key|(?<![a-z0-9-])sk-[a-z0-9_-]{6,}|"
    r"\b(?:patient|profile|report)[_ -]?id\s*[:=]\s*\S+|"
    r"\b(?:REC|HVK):\S+|\b(?:case[_ -]?id|gold|expected answer|held[- ]out)\b|"
    r"题号|标准答案|期望(?:答案|工具)|"
    r"患者.{0,24}(?:血压|血糖|诊断|用药))"
)
_DANGEROUS_PATTERN = re.compile(
    r"(?i)(rm\s+-rf|del\s+/[sq]|format\s+[a-z]:|git\s+reset\s+--hard|"
    r"git\s+clean\s+-[a-z]*f|powershell|cmd\.exe|subprocess)"
)
_HEADING = re.compile(r"(?m)^##\s+(.+?)\s*$")


PatchOperation = Literal["add", "replace", "delete"]
PatchSideQuery = Callable[
    [str, dict[str, Any]],
    Awaitable[dict[str, Any] | str] | dict[str, Any] | str,
]
RegressionCheck = Callable[[str], tuple[bool, tuple[str, ...]]]


class ArtifactPathConflictError(ValueError):
    """Raised before generation when an artifact could overwrite the active Skill."""


@dataclass(frozen=True)
class SkillPatch:
    patch_id: str
    operation: PatchOperation
    source_trace_ids: tuple[str, ...]
    verification_source: str
    verification_strength: str
    target_section: str
    target_paragraph: str
    applicability_conditions: tuple[str, ...]
    when_not_to_use: tuple[str, ...]
    tool_path: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    content: str
    rationale: str


@dataclass(frozen=True)
class PatchPreview:
    status: str
    reasons: tuple[str, ...]
    patches: tuple[SkillPatch, ...]
    active_skill_sha256: str
    candidate_skill_sha256: str | None
    candidate_content: str | None
    unified_diff: str
    generator_calls: int
    published: bool = False
    api_calls: int = 0


class PatchGenerator:
    """One bounded side query; tests inject a fake callable."""

    def __init__(self, side_query: PatchSideQuery) -> None:
        self._side_query = side_query
        self.call_count = 0

    async def generate(
        self,
        cards: Iterable[ExperienceCard],
        *,
        active_content: str,
    ) -> tuple[SkillPatch, ...]:
        evidence = tuple(cards)
        if not evidence:
            raise ValueError("weak_or_insufficient_evidence")
        card_payloads: list[dict[str, Any]] = []
        for card in evidence:
            tool_path: list[str] = []
            observed_transitions: list[dict[str, Any]] = []
            for sequence, step in enumerate(card.tool_path, start=1):
                tool_name, separator, result_status = step.partition(":")
                tool_name = tool_name.strip()
                if tool_name.startswith("mcp__healthvault__"):
                    tool_name = tool_name.removeprefix("mcp__healthvault__")
                result_status = result_status.strip()
                if tool_name not in ALLOWED_TOOL_PATH or (separator and not result_status):
                    raise ValueError("experience_card_tool_path_out_of_scope")
                if tool_name not in tool_path:
                    tool_path.append(tool_name)
                if separator:
                    observed_transitions.append({
                        "sequence": sequence,
                        "tool_name": tool_name,
                        "result_status": result_status,
                    })
            card_payloads.append({
                "card_id": card.card_id,
                "intent": card.intent,
                "preconditions": list(card.preconditions),
                "lesson": card.lesson,
                "tool_path": tool_path,
                "observed_transitions": observed_transitions,
                "expected_outcome": card.expected_outcome,
                "when_not_to_use": list(card.when_not_to_use),
                "trace_ids": list(card.trace_ids),
                "verification_source": card.verification_source,
                "verification_strength": card.verification_strength,
            })
        prompt = json.dumps({
            "target_skill": "HealthVault-Task-Execution",
            "active_skill_sha256": _sha256(active_content),
            "active_skill": active_content,
            "eligible_experience_cards": card_payloads,
            "rules": (
                "Return only local workflow patches. Do not add medical facts, diagnosis or "
                "treatment rules, doses, reference values, knowledge content, safety boundaries, "
                "permission policy, patient facts, case IDs, Gold, or expected answers. "
                "Patch tool_path values must be exact tool names from: "
                + ", ".join(ALLOWED_TOOL_PATH_VALUES)
                + ". Never put result-status suffixes, natural-language steps, or nonexistent "
                "tool names in tool_path; use observed_transitions only as evidence when writing "
                "applicability or stop conditions. For an add at the end of a section, "
                "target_paragraph must be exactly __section_end__. Otherwise copy the exact "
                "existing paragraph from that section of active_skill; never invent an anchor."
            ),
        }, ensure_ascii=False, sort_keys=True)
        self.call_count += 1
        response = self._side_query(prompt, PATCH_SCHEMA)
        if inspect.isawaitable(response):
            response = await response
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as exc:
                raise ValueError("patch_response_invalid_json") from exc
        return _parse_patch_response(response)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _text_tuple(value: Any) -> tuple[str, ...] | None:
    if (
        not isinstance(value, list)
        or not value
        or any(not _nonempty_text(item) for item in value)
    ):
        return None
    return tuple(str(item).strip() for item in value)


def _parse_patch_response(value: object) -> tuple[SkillPatch, ...]:
    if not isinstance(value, dict) or set(value) != {"patches"}:
        raise ValueError("patch_schema_mismatch")
    raw_patches = value["patches"]
    if not isinstance(raw_patches, list) or not 1 <= len(raw_patches) <= MAX_PATCHES:
        raise ValueError("patch_count_out_of_bounds")
    required = set(PATCH_SCHEMA["properties"]["patches"]["items"]["required"])
    parsed: list[SkillPatch] = []
    ids: set[str] = set()
    for raw in raw_patches:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("patch_schema_mismatch")
        if not _nonempty_text(raw["patch_id"]) or raw["patch_id"] in ids:
            raise ValueError("patch_id_invalid_or_duplicate")
        ids.add(raw["patch_id"])
        if raw["operation"] not in PATCH_OPERATIONS:
            raise ValueError("patch_operation_invalid")
        if raw["verification_source"] not in VERIFICATION_SOURCES:
            raise ValueError("verification_source_invalid")
        if raw["verification_strength"] not in VERIFICATION_STRENGTHS:
            raise ValueError("verification_strength_invalid")
        source_ids = _text_tuple(raw["source_trace_ids"])
        if source_ids is None:
            raise ValueError("missing_source_trace")
        conditions = _text_tuple(raw["applicability_conditions"])
        exclusions = _text_tuple(raw["when_not_to_use"])
        tool_path = _text_tuple(raw["tool_path"])
        stops = _text_tuple(raw["stop_conditions"])
        if any(item is None for item in (conditions, exclusions, tool_path, stops)):
            raise ValueError("patch_required_list_missing")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_trace_ids_duplicate")
        for field in ("target_section", "target_paragraph", "rationale"):
            if not _nonempty_text(raw[field]):
                raise ValueError(f"patch_required_text_missing:{field}")
        if not isinstance(raw["content"], str):
            raise ValueError("patch_content_invalid")
        if raw["operation"] == "delete" and raw["content"].strip():
            raise ValueError("delete_patch_content_must_be_empty")
        if raw["operation"] in {"add", "replace"} and not raw["content"].strip():
            raise ValueError("patch_content_required")
        parsed.append(SkillPatch(
            patch_id=raw["patch_id"].strip(),
            operation=raw["operation"],
            source_trace_ids=source_ids,
            verification_source=raw["verification_source"],
            verification_strength=raw["verification_strength"],
            target_section=raw["target_section"].strip(),
            target_paragraph=raw["target_paragraph"].strip(),
            applicability_conditions=conditions,
            when_not_to_use=exclusions,
            tool_path=tool_path,
            stop_conditions=stops,
            content=raw["content"].strip(),
            rationale=raw["rationale"].strip(),
        ))
    return tuple(parsed)


def _section_spans(content: str) -> dict[str, tuple[int, int]]:
    matches = list(_HEADING.finditer(content))
    spans: dict[str, tuple[int, int]] = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        if name in spans:
            raise ValueError(f"duplicate_section:{name}")
        spans[name] = (match.end(), matches[index + 1].start() if index + 1 < len(matches) else len(content))
    return spans


def _workflow_text(patch: SkillPatch) -> str:
    return "\n".join((
        patch.patch_id,
        *patch.source_trace_ids,
        patch.target_paragraph,
        *patch.applicability_conditions,
        *patch.when_not_to_use,
        *patch.tool_path,
        *patch.stop_conditions,
        patch.content,
        patch.rationale,
    ))


def _verify_patch_scope_and_content(patch: SkillPatch) -> list[str]:
    reasons: list[str] = []
    if patch.target_section not in ALLOWED_TARGET_SECTIONS:
        reasons.append(f"target_section_out_of_scope:{patch.target_section}")
    if not set(patch.tool_path) <= ALLOWED_TOOL_PATH:
        reasons.append("tool_path_out_of_scope")
    text = _workflow_text(patch)
    if _MEDICAL_OR_KNOWLEDGE_PATTERN.search(text):
        reasons.append("medical_or_knowledge_content_forbidden")
    if _POLICY_OR_PERMISSION_PATTERN.search(text):
        reasons.append("safety_or_permission_policy_forbidden")
    if _SENSITIVE_OR_CASE_PATTERN.search(text):
        reasons.append("sensitive_private_or_eval_content_forbidden")
    if _DANGEROUS_PATTERN.search(text):
        reasons.append("dangerous_content_forbidden")
    return reasons


def _verify_evidence(
    patch: SkillPatch,
    cards: tuple[ExperienceCard, ...],
    active_sha256: str,
) -> list[str]:
    reasons: list[str] = []
    if not patch.source_trace_ids:
        return ["missing_source_trace"]
    if any(card.skill_sha256 != active_sha256 for card in cards):
        reasons.append("source_version_mismatch")
    known_ids = {trace_id for card in cards for trace_id in card.trace_ids}
    if not set(patch.source_trace_ids) <= known_ids:
        reasons.append("unknown_source_trace")
    supporting = [
        card for card in cards
        if set(card.trace_ids) <= set(patch.source_trace_ids)
    ]
    if patch.verification_strength == "weak":
        reasons.append("weak_evidence")
    elif not any(card.verification_strength == patch.verification_strength for card in supporting):
        reasons.append("evidence_strength_mismatch")
    if not any(card.verification_source == patch.verification_source for card in supporting):
        reasons.append("verification_source_mismatch")
    if patch.verification_strength == "medium":
        task_ids = {task_id for card in supporting for task_id in card.task_ids}
        if len(task_ids) < 3:
            reasons.append("medium_evidence_requires_three_independent_tasks")
    return reasons


def merge_patches(active_content: str, patches: Iterable[SkillPatch]) -> tuple[str, str]:
    """Apply non-overlapping patches in a stable order and return a unified diff."""
    normalized = active_content.replace("\r\n", "\n")
    ordered = tuple(sorted(
        patches,
        key=lambda item: (item.target_section, item.target_paragraph, item.operation, item.patch_id),
    ))
    targets = [(item.target_section, item.target_paragraph) for item in ordered]
    if len(targets) != len(set(targets)):
        raise ValueError("conflicting_patch_target")
    candidate = normalized
    for patch in ordered:
        spans = _section_spans(candidate)
        if patch.target_section not in spans:
            raise ValueError(f"target_section_missing:{patch.target_section}")
        start, end = spans[patch.target_section]
        section = candidate[start:end]
        if patch.target_paragraph == SECTION_END:
            if patch.operation != "add":
                raise ValueError("section_end_only_supports_add")
            updated = section.rstrip() + "\n" + patch.content.strip() + "\n\n"
        else:
            if section.count(patch.target_paragraph) != 1:
                raise ValueError(f"target_paragraph_missing_or_ambiguous:{patch.patch_id}")
            if patch.operation == "add":
                replacement = patch.target_paragraph + "\n" + patch.content.strip()
            elif patch.operation == "replace":
                replacement = patch.content.strip()
            else:
                replacement = ""
            updated = section.replace(patch.target_paragraph, replacement, 1)
            updated = re.sub(r"\n{3,}", "\n\n", updated)
        candidate = candidate[:start] + updated + candidate[end:]
    if not candidate.endswith("\n"):
        candidate += "\n"
    diff = "".join(difflib.unified_diff(
        normalized.splitlines(keepends=True),
        candidate.splitlines(keepends=True),
        fromfile="active/SKILL.md",
        tofile="candidate-preview/SKILL.md",
        lineterm="\n",
    ))
    return candidate, diff


def _diff_size_reasons(active: str, diff: str) -> list[str]:
    changed = [
        line for line in diff.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]
    changed_characters = sum(max(0, len(line) - 1) for line in changed)
    reasons: list[str] = []
    if len(changed) > MAX_CHANGED_LINES:
        reasons.append("patch_too_large:changed_lines")
    if changed_characters > MAX_CHANGED_CHARACTERS:
        reasons.append("patch_too_large:changed_characters")
    if changed_characters / max(1, len(active)) > MAX_CHANGE_RATIO:
        reasons.append("patch_too_large:change_ratio")
    return reasons


async def build_patch_preview(
    cards: Iterable[ExperienceCard],
    *,
    active_content: str,
    generator: PatchGenerator,
    verifier: CandidateVerifier,
    regression_check: RegressionCheck,
) -> PatchPreview:
    """Generate, merge, and verify one fake/local patch batch without publication."""
    active_sha = _sha256(active_content)
    promoted = eligible_cards(cards)
    if not promoted:
        return PatchPreview(
            "rejected", ("weak_or_insufficient_evidence",), (), active_sha,
            None, None, "", 0,
        )
    calls_before = generator.call_count
    try:
        patches = await generator.generate(promoted, active_content=active_content)
    except ValueError as exc:
        return PatchPreview(
            "rejected", (str(exc),), (), active_sha, None, None, "",
            generator.call_count - calls_before,
        )
    reasons: list[str] = []
    for patch in patches:
        reasons.extend(_verify_evidence(patch, promoted, active_sha))
        reasons.extend(_verify_patch_scope_and_content(patch))
    if reasons:
        return PatchPreview(
            "rejected", tuple(dict.fromkeys(reasons)), patches, active_sha,
            None, None, "", generator.call_count - calls_before,
        )
    try:
        candidate, diff = merge_patches(active_content, patches)
    except ValueError as exc:
        return PatchPreview(
            "rejected", (str(exc),), patches, active_sha, None, None, "",
            generator.call_count - calls_before,
        )
    reasons.extend(_diff_size_reasons(active_content, diff))
    try:
        parsed = parse_frontmatter(candidate)
        spans = _section_spans(candidate)
        if parsed.meta.get("name") != "HealthVault-Task-Execution":
            reasons.append("target_skill_mismatch")
        for section in ALLOWED_TARGET_SECTIONS:
            if section in spans:
                start, end = spans[section]
                if not candidate[start:end].strip():
                    reasons.append(f"empty_required_workflow_section:{section}")
    except Exception:
        reasons.append("invalid_candidate_format")
    verification = verifier.verify_content(
        candidate,
        action="evolve",
        active_content=active_content,
    )
    reasons.extend(f"candidate_verifier:{reason}" for reason in verification.reasons)
    regression_passed, regression_reasons = regression_check(candidate)
    if not regression_passed:
        reasons.extend(f"regression:{reason}" for reason in regression_reasons)
    candidate_sha = _sha256(candidate)
    if verification.content_sha256 != candidate_sha:
        reasons.append("candidate_hash_mismatch")
    if reasons:
        return PatchPreview(
            "rejected", tuple(dict.fromkeys(reasons)), patches, active_sha,
            candidate_sha, None, diff, generator.call_count - calls_before,
        )
    return PatchPreview(
        "pending_manual_review", (), patches, active_sha, candidate_sha,
        candidate, diff, generator.call_count - calls_before,
    )


def diff_artifact(preview: PatchPreview) -> dict[str, Any]:
    """Return the review artifact; candidate content stays in memory only."""
    return {
        "schema_version": "healthvault-skill-local-patch-artifact-v1",
        "status": preview.status,
        "reasons": list(preview.reasons),
        "patches": [asdict(patch) for patch in preview.patches],
        "active_skill_sha256": preview.active_skill_sha256,
        "candidate_skill_sha256": preview.candidate_skill_sha256,
        "unified_diff": preview.unified_diff,
        "generator_calls": preview.generator_calls,
        "published": False,
        "api_calls": 0,
    }


def write_diff_artifact(path: str | Path, preview: PatchPreview) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(diff_artifact(preview), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _paths_refer_to_same_file(left: str | Path, right: str | Path) -> bool:
    """Resolve aliases and hard links before allowing an artifact write."""
    left_path = Path(left)
    right_path = Path(right)
    try:
        if left_path.exists() and right_path.exists() and left_path.samefile(right_path):
            return True
    except OSError:
        pass
    try:
        return left_path.resolve(strict=False) == right_path.resolve(strict=False)
    except OSError:
        return left_path.absolute() == right_path.absolute()


async def run_fixed_fake_demo(
    active_skill_path: str | Path,
    artifact_path: str | Path,
) -> dict[str, Any]:
    """Exercise the bounded path with one synthetic user-confirmed dev correction."""
    active_path = Path(active_skill_path)
    if _paths_refer_to_same_file(active_path, artifact_path):
        raise ArtifactPathConflictError("artifact_output_conflicts_with_active_skill")
    active_content = active_path.read_text(encoding="utf-8")
    active_sha = _sha256(active_content)
    card = ExperienceCard(
        card_id="CARD-TASK044-FAKE-WORKFLOW",
        intent="route and organize a read-only HealthVault workflow",
        preconditions=("A user-confirmed workflow correction is available.",),
        lesson=(
            "Select the matching read-only tool before drafting, stop an empty route, "
            "and organize the final response before verification."
        ),
        tool_path=("query_health_timeline:ok",),
        expected_outcome="A correctly ordered and validated workflow response.",
        when_not_to_use=("Do not use when the requested workflow differs.",),
        trace_ids=("TRACE-TASK044-FAKE-DEV-001",),
        task_ids=("TASK044-FAKE-DEV-001",),
        skill_sha256=active_sha,
        evidence_count=1,
        verification_source="user_confirmed",
        verification_strength="strong",
        reusable=True,
    )
    fake_response = {"patches": [
        {
            "patch_id": "PATCH-TASK044-TOOL-ORDER",
            "operation": "add",
            "source_trace_ids": ["TRACE-TASK044-FAKE-DEV-001"],
            "verification_source": "user_confirmed",
            "verification_strength": "strong",
            "target_section": "Steps",
            "target_paragraph": "1. Identify the requested HealthVault outcome before selecting a tool.",
            "applicability_conditions": ["The requested workflow is a timeline."],
            "when_not_to_use": ["The requested workflow is not a timeline."],
            "tool_path": ["query_health_timeline"],
            "stop_conditions": ["Stop after a terminal tool status."],
            "content": "1a. For timeline requests, call query_health_timeline before drafting.",
            "rationale": "Preserve the user-confirmed tool selection and step order.",
        },
        {
            "patch_id": "PATCH-TASK044-EMPTY-STOP",
            "operation": "add",
            "source_trace_ids": ["TRACE-TASK044-FAKE-DEV-001"],
            "verification_source": "user_confirmed",
            "verification_strength": "strong",
            "target_section": "Failure Handling",
            "target_paragraph": (
                "If evidence is insufficient or a tool rejects input, report that result without guessing."
            ),
            "applicability_conditions": ["A requested route returns an empty result."],
            "when_not_to_use": ["Another requested route still requires execution."],
            "tool_path": ["query_health_timeline"],
            "stop_conditions": ["The empty route has been reported."],
            "content": "When a route is empty, stop it unless another requested route remains.",
            "rationale": "Apply the confirmed empty-result stopping workflow.",
        },
        {
            "patch_id": "PATCH-TASK044-OUTPUT",
            "operation": "replace",
            "source_trace_ids": ["TRACE-TASK044-FAKE-DEV-001"],
            "verification_source": "user_confirmed",
            "verification_strength": "strong",
            "target_section": "Verification",
            "target_paragraph": (
                "Check the final status, requested fields, and every citation before answering."
            ),
            "applicability_conditions": ["A final response is ready to validate."],
            "when_not_to_use": ["A terminal tool result still needs handling."],
            "tool_path": ["query_health_timeline"],
            "stop_conditions": ["Status and citations have been checked."],
            "content": "Organize the response, then verify its status and citations.",
            "rationale": "Apply the confirmed output-organization workflow.",
        },
    ]}

    def fake_side_query(_prompt: str, _schema: dict[str, Any]) -> dict[str, Any]:
        return fake_response

    safety_section = active_content.split("## Safety Boundaries", 1)[1]

    def focused_regression(candidate: str) -> tuple[bool, tuple[str, ...]]:
        checks = (
            "query_health_timeline before drafting" in candidate,
            "When a route is empty" in candidate,
            "Organize the response" in candidate,
            candidate.split("## Safety Boundaries", 1)[1] == safety_section,
        )
        return (all(checks), () if all(checks) else ("fake_workflow_contract_failed",))

    generator = PatchGenerator(fake_side_query)
    preview = await build_patch_preview(
        (card,),
        active_content=active_content,
        generator=generator,
        verifier=CandidateVerifier("HealthVault-Task-Execution"),
        regression_check=focused_regression,
    )
    write_diff_artifact(artifact_path, preview)
    return {
        "status": preview.status,
        "reasons": list(preview.reasons),
        "active_skill_sha256": preview.active_skill_sha256,
        "candidate_skill_sha256": preview.candidate_skill_sha256,
        "generator_calls": preview.generator_calls,
        "published": False,
        "api_calls": 0,
    }


def main() -> int:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        result = asyncio.run(run_fixed_fake_demo(arguments.active, arguments.output))
    except ArtifactPathConflictError as exc:
        result = {
            "status": "rejected",
            "reasons": [str(exc)],
            "generator_calls": 0,
            "published": False,
            "api_calls": 0,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pending_manual_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
