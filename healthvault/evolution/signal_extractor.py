"""Rule-first, batch-only Trace to ExperienceCard adaptation for HealthVault.

The runtime already persists redacted tool traces.  This module consumes those
traces only after a task/session ends (or from an explicit manual batch); it
does not run in the Agent turn loop and it never publishes a Skill.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Literal, Mapping

from healthvault.evolution.candidates import (
    CandidateGenerator,
    EvolutionSignal,
    SkillCandidate,
    TaskEndProcessor,
)
from healthvault.evolution.verifier import CandidateVerifier, VerificationResult


VerificationSource = Literal["deterministic", "user_confirmed", "semantic_judge"]
VerificationStrength = Literal["strong", "medium", "weak"]
SemanticExtractor = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]],
    Awaitable[dict[str, Any] | str] | dict[str, Any] | str,
]

SEMANTIC_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["corrections"],
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "trace_id", "intent", "preconditions", "lesson",
                    "when_not_to_use", "reusable",
                ],
                "properties": {
                    "trace_id": {"type": "string", "minLength": 1},
                    "intent": {"type": "string", "minLength": 1},
                    "preconditions": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "lesson": {"type": "string", "minLength": 1},
                    "when_not_to_use": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "reusable": {"type": "boolean"},
                },
            },
        },
    },
}
SEMANTIC_EXTRACTION_CONFIG = {
    "temperature": 0.0,
    "max_output_tokens": 256,
    "response_format": "strict_json",
}

_SUSPECTED_CORRECTION = re.compile(
    r"(?i)(\bshould\b|\binstead\b|\bwrong\b|\bcorrect(?:ion)?\b|"
    r"不要|不应|应该|改成|纠正|有误|错了)"
)
_PATIENT_OR_MEDICAL_FACT = re.compile(
    r"(?i)(患者|病人|诊断|用药|血压|血糖|glucose|\bBP\b|\bdiagnosis\b|"
    r"\bmedication\b|\b\d+(?:\.\d+)?\s*(?:mg|g|mmol|cm|mm|%|次|天)\b|"
    r"\b\d{2,3}/\d{2,3}\b)"
)
_ENGINEERING_FAILURE = re.compile(
    r"(?i)(tool|data|permission|environment|database|sqlite|unicodeencode|"
    r"gbk|network|timeout)"
)
_PERSONAL_PREFERENCE = re.compile(
    r"(?i)(我的|我(?:是|叫|今年|喜欢|习惯|偏好)|称呼我|只(?:对我|影响我)|个人偏好|"
    r"\bmy preference\b|\bI (?:prefer|am)\b|\bfor me only\b|"
    r"\b\d{7,}\b|[\w.+-]+@[\w.-]+\.[a-z]{2,})"
)
_PASSING_VALIDATION = frozenset({"passed", "pass", "ok", "valid"})
_KNOWN_CONTRACT_CHECKS = frozenset({
    "tool", "status", "permission", "side_effect", "sqlite", "citation", "refusal",
})


@dataclass(frozen=True)
class TaskObservation:
    """Redacted task-end material used by the extractor, never a patient record."""

    trace: Mapping[str, Any]
    intent: str
    preconditions: tuple[str, ...]
    expected_outcome: str
    validator_status: str
    feedback: str | None = None
    user_confirmed: bool = False
    contract_checks: Mapping[str, str] = field(default_factory=dict)


def observation_from_persisted_trace(
    trace_path: str | Path,
    *,
    feedback: str | None = None,
    user_confirmed: bool = False,
) -> TaskObservation:
    """Adapt one redacted Runtime Trace plus later feedback for a manual batch."""
    target = Path(trace_path)
    try:
        trace = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("persisted task trace is unreadable") from exc
    if not isinstance(trace, Mapping) or trace.get("schema_version") != "healthvault-task-trace-v1":
        raise ValueError("persisted task trace has an unsupported schema")
    context = trace.get("signal_context")
    if not isinstance(context, Mapping) or set(context) != {
        "intent", "preconditions", "expected_outcome", "validator_status", "contract_checks",
    }:
        raise ValueError("persisted task trace has no signal context")
    intent = context["intent"]
    expected = context["expected_outcome"]
    validator = context["validator_status"]
    preconditions = context["preconditions"]
    contracts = context["contract_checks"]
    if (
        not all(isinstance(value, str) and value.strip() for value in (intent, expected, validator))
        or not isinstance(preconditions, list)
        or not all(isinstance(value, str) and value.strip() for value in preconditions)
        or not isinstance(contracts, Mapping)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in contracts.items())
        or (feedback is not None and not isinstance(feedback, str))
    ):
        raise ValueError("persisted signal context is invalid")
    return TaskObservation(
        trace=trace,
        intent=intent,
        preconditions=tuple(preconditions),
        expected_outcome=expected,
        validator_status=validator,
        feedback=feedback,
        user_confirmed=user_confirmed,
        contract_checks={str(key): str(value) for key, value in contracts.items()},
    )


@dataclass(frozen=True)
class ExperienceCard:
    card_id: str
    intent: str
    preconditions: tuple[str, ...]
    lesson: str
    tool_path: tuple[str, ...]
    expected_outcome: str
    when_not_to_use: tuple[str, ...]
    trace_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    skill_sha256: str
    evidence_count: int
    verification_source: VerificationSource
    verification_strength: VerificationStrength
    reusable: bool

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.intent,
            self.preconditions,
            self.tool_path,
            self.expected_outcome,
            self.lesson,
            self.when_not_to_use,
            self.skill_sha256,
        )

    def as_signal(self) -> EvolutionSignal:
        return EvolutionSignal(
            trace_id=self.card_id,
            signal_type="verified_reusable_workflow",
            lesson=self.lesson,
            reusable=True,
            source_trace_ids=self.trace_ids,
        )


@dataclass(frozen=True)
class ExtractionReport:
    cards: tuple[ExperienceCard, ...]
    engineering_issues: tuple[str, ...]
    ignored_trace_ids: tuple[str, ...]
    semantic_extractor_calls: int


@dataclass(frozen=True)
class CandidateBatchResult:
    candidate: SkillCandidate | None
    verification: VerificationResult | None
    supporting_cards: tuple[ExperienceCard, ...]
    status: str
    generator_calls: int


class SignalExtractor:
    """Make only deterministic cards unless a bounded correction batch needs help."""

    def __init__(self, semantic_extractor: SemanticExtractor | None = None) -> None:
        self._semantic_extractor = semantic_extractor
        self.semantic_extractor_calls = 0

    @staticmethod
    def _trace_id(trace: Mapping[str, Any]) -> str:
        value = trace.get("trace_id")
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _task_id(trace: Mapping[str, Any]) -> str:
        value = trace.get("task_id")
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _skill_sha(trace: Mapping[str, Any]) -> str:
        skill = trace.get("skill")
        if isinstance(skill, Mapping):
            return str(skill.get("sha256", "")).strip()
        return ""

    @staticmethod
    def _tool_path(trace: Mapping[str, Any]) -> tuple[str, ...]:
        calls = trace.get("tool_calls")
        if not isinstance(calls, list):
            return ()
        path: list[str] = []
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("tool_name", call.get("name", ""))).strip()
            status = str(call.get("result_status", call.get("status", ""))).strip()
            if name and status:
                path.append(f"{name}:{status}")
        return tuple(path)

    @staticmethod
    def _known_contract_passed(observation: TaskObservation) -> bool:
        trace = observation.trace
        status = str(observation.validator_status).strip().lower()
        if status not in _PASSING_VALIDATION:
            return False
        if not set(observation.contract_checks).issubset(_KNOWN_CONTRACT_CHECKS):
            return False
        if any(
            str(value).strip().lower() not in _PASSING_VALIDATION
            for value in observation.contract_checks.values()
        ):
            return False
        if bool(trace.get("hard_failure", False)):
            return False
        if str(trace.get("result_status", "")).strip().lower() not in {"succeeded", "ok"}:
            return False
        for call in trace.get("tool_calls", []):
            if not isinstance(call, Mapping):
                return False
            validator = call.get("validator_code")
            if validator is not None and str(validator).strip().lower() in {
                "failed", "invalid", "rejected",
            }:
                return False
        return True

    @staticmethod
    def _engineering_reason(observation: TaskObservation) -> str | None:
        trace = observation.trace
        failure_values = [trace.get("failure_code")]
        for call in trace.get("tool_calls", []):
            if isinstance(call, Mapping):
                failure_values.append(call.get("failure_code"))
        for value in failure_values:
            if value is not None and _ENGINEERING_FAILURE.search(str(value)):
                return "engineering_defect"
        return None

    @staticmethod
    def _contains_forbidden_fact(observation: TaskObservation) -> bool:
        text = "\n".join(
            item for item in (
                observation.intent,
                " ".join(observation.preconditions),
                observation.expected_outcome,
                observation.feedback or "",
            ) if item
        )
        return bool(_PATIENT_OR_MEDICAL_FACT.search(text))

    @staticmethod
    def _card_id(prefix: str, identity: Iterable[object]) -> str:
        encoded = json.dumps(list(identity), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"

    def _deterministic_recovery_card(
        self, observation: TaskObservation
    ) -> ExperienceCard | None:
        if not self._known_contract_passed(observation):
            return None
        path = self._tool_path(observation.trace)
        recovered = any(
            item.endswith(":retryable_failure")
            and any(later == item.removesuffix("retryable_failure") + "ok" for later in path[index + 1:])
            for index, item in enumerate(path)
        )
        if not recovered or any(
            not item.endswith((":retryable_failure", ":ok")) for item in path
        ):
            return None
        trace_id = self._trace_id(observation.trace)
        task_id = self._task_id(observation.trace)
        skill_sha = self._skill_sha(observation.trace)
        if not all((trace_id, task_id, skill_sha, observation.intent.strip())):
            return None
        lesson = (
            "Retry only the tool call explicitly marked retryable, then require the "
            "known validator contract before reporting the expected outcome."
        )
        return ExperienceCard(
            card_id=self._card_id("RECOVERY", (observation.intent, path, observation.expected_outcome)),
            intent=observation.intent.strip(),
            preconditions=tuple(observation.preconditions),
            lesson=lesson,
            tool_path=path,
            expected_outcome=observation.expected_outcome.strip(),
            when_not_to_use=("Do not retry rejected or non-retryable tool results.",),
            trace_ids=(trace_id,),
            task_ids=(task_id,),
            skill_sha256=skill_sha,
            evidence_count=1,
            verification_source="deterministic",
            verification_strength="strong",
            reusable=True,
        )

    def _user_confirmed_card(self, observation: TaskObservation) -> ExperienceCard | None:
        # Explicit non-medical user confirmation is independent strong evidence;
        # the preceding task may be exactly what the user is correcting.
        if not observation.user_confirmed or not observation.feedback:
            return None
        trace_id = self._trace_id(observation.trace)
        task_id = self._task_id(observation.trace)
        skill_sha = self._skill_sha(observation.trace)
        path = self._tool_path(observation.trace)
        if not all((trace_id, task_id, skill_sha, observation.intent.strip(), path)):
            return None
        return ExperienceCard(
            card_id=self._card_id("CONFIRMED", (observation.intent, path, observation.feedback)),
            intent=observation.intent.strip(),
            preconditions=tuple(observation.preconditions),
            lesson=observation.feedback.strip(),
            tool_path=path,
            expected_outcome=observation.expected_outcome.strip(),
            when_not_to_use=("Do not use for patient facts or medical conclusions.",),
            trace_ids=(trace_id,),
            task_ids=(task_id,),
            skill_sha256=skill_sha,
            evidence_count=1,
            verification_source="user_confirmed",
            verification_strength="strong",
            reusable=True,
        )

    def _aggregate_successes(
        self, observations: Iterable[TaskObservation]
    ) -> list[ExperienceCard]:
        groups: dict[tuple[object, ...], list[TaskObservation]] = {}
        for observation in observations:
            if not self._known_contract_passed(observation):
                continue
            path = self._tool_path(observation.trace)
            trace_id = self._trace_id(observation.trace)
            task_id = self._task_id(observation.trace)
            skill_sha = self._skill_sha(observation.trace)
            if not all((path, trace_id, task_id, skill_sha, observation.intent.strip())):
                continue
            if any(item.endswith(":retryable_failure") for item in path):
                continue
            if not all(item.endswith(":ok") for item in path):
                continue
            key = (
                observation.intent.strip(), tuple(observation.preconditions), path,
                observation.expected_outcome.strip(),
            )
            groups.setdefault(key, []).append(observation)

        cards: list[ExperienceCard] = []
        for key, grouped in sorted(groups.items(), key=lambda item: repr(item[0])):
            unique_tasks = {self._task_id(item.trace) for item in grouped}
            if len(unique_tasks) < 3:
                continue
            intent, preconditions, path, expected = key
            skill_hashes = {self._skill_sha(item.trace) for item in grouped}
            if len(skill_hashes) != 1:
                continue
            skill_sha = next(iter(skill_hashes))
            trace_ids = tuple(sorted({self._trace_id(item.trace) for item in grouped}))
            task_ids = tuple(sorted(unique_tasks))
            tools = " -> ".join(path)
            cards.append(ExperienceCard(
                card_id=self._card_id("PATTERN", key),
                intent=str(intent),
                preconditions=tuple(preconditions),
                lesson=f"For this verified repeated task, follow {tools} and validate {expected}.",
                tool_path=tuple(path),
                expected_outcome=str(expected),
                when_not_to_use=("Do not use when the preconditions or expected outcome differ.",),
                trace_ids=trace_ids,
                task_ids=task_ids,
                skill_sha256=str(skill_sha),
                evidence_count=len(trace_ids),
                verification_source="deterministic",
                verification_strength="strong",
                reusable=True,
            ))
        return cards

    async def _semantic_cards(
        self, observations: list[TaskObservation]
    ) -> list[ExperienceCard]:
        suspected = [
            item for item in observations
            if item.feedback
            and not item.user_confirmed
            and _SUSPECTED_CORRECTION.search(item.feedback)
        ]
        if not suspected or self._semantic_extractor is None:
            return []
        payload = {
            "kind": "healthvault_correction_batch",
            "observations": [
                {
                    "trace_id": self._trace_id(item.trace),
                    "intent": item.intent,
                    "preconditions": list(item.preconditions),
                    "expected_outcome": item.expected_outcome,
                    "tool_path": list(self._tool_path(item.trace)),
                    "feedback": item.feedback,
                }
                for item in suspected
            ],
            "rules": (
                "Return reusable non-medical workflow corrections only. Exclude patient "
                "facts, medical conclusions, tool/data/permission/environment defects, "
                "and unsupported semantic claims."
            ),
        }
        self.semantic_extractor_calls += 1
        response = self._semantic_extractor(
            payload, SEMANTIC_EXTRACTION_SCHEMA, dict(SEMANTIC_EXTRACTION_CONFIG)
        )
        if inspect.isawaitable(response):
            response = await response
        if isinstance(response, str):
            response = json.loads(response)
        if not isinstance(response, Mapping) or set(response) != {"corrections"}:
            raise ValueError("semantic extractor response does not match strict schema")
        corrections = response["corrections"]
        if not isinstance(corrections, list):
            raise ValueError("semantic extractor corrections must be a list")
        lookup = {self._trace_id(item.trace): item for item in suspected}
        cards: list[ExperienceCard] = []
        for correction in corrections:
            if not isinstance(correction, Mapping) or set(correction) != {
                "trace_id", "intent", "preconditions", "lesson", "when_not_to_use", "reusable",
            }:
                raise ValueError("semantic correction does not match strict schema")
            trace_id = str(correction["trace_id"]).strip()
            source = lookup.get(trace_id)
            if source is None:
                raise ValueError("semantic correction must cite a batch trace")
            fields = ("intent", "lesson")
            if not isinstance(correction["reusable"], bool) or not correction["reusable"] or any(
                not isinstance(correction[field], str) or not correction[field].strip()
                for field in fields
            ):
                continue
            preconditions = correction["preconditions"]
            exclusions = correction["when_not_to_use"]
            if not all(
                isinstance(value, list) and all(isinstance(item, str) for item in value)
                for value in (preconditions, exclusions)
            ):
                raise ValueError("semantic correction lists must contain strings")
            content = "\n".join([
                str(correction["intent"]), str(correction["lesson"]),
                " ".join(preconditions), " ".join(exclusions),
            ])
            if _PATIENT_OR_MEDICAL_FACT.search(content):
                continue
            path = self._tool_path(source.trace)
            cards.append(ExperienceCard(
                card_id=self._card_id("SEMANTIC", (trace_id, correction["lesson"])),
                intent=str(correction["intent"]).strip(),
                preconditions=tuple(preconditions),
                lesson=str(correction["lesson"]).strip(),
                tool_path=path,
                expected_outcome=source.expected_outcome.strip(),
                when_not_to_use=tuple(exclusions),
                trace_ids=(trace_id,),
                task_ids=(self._task_id(source.trace),),
                skill_sha256=self._skill_sha(source.trace),
                evidence_count=1,
                verification_source="semantic_judge",
                verification_strength="medium",
                reusable=True,
            ))
        return cards

    async def extract(self, observations: Iterable[TaskObservation]) -> ExtractionReport:
        """Extract one task-end/manual batch without invoking a candidate model."""
        calls_before = self.semantic_extractor_calls
        items = list(observations)
        issues: list[str] = []
        ignored: list[str] = []
        usable: list[TaskObservation] = []
        for observation in items:
            trace_id = self._trace_id(observation.trace)
            if not trace_id or self._contains_forbidden_fact(observation) or (
                observation.feedback and _PERSONAL_PREFERENCE.search(observation.feedback)
            ):
                ignored.append(trace_id)
                continue
            engineering = self._engineering_reason(observation)
            if engineering:
                issues.append(f"{trace_id}:{engineering}")
                continue
            usable.append(observation)

        cards = self._aggregate_successes(usable)
        for observation in usable:
            card = self._deterministic_recovery_card(observation)
            if card is not None:
                cards.append(card)
            card = self._user_confirmed_card(observation)
            if card is not None:
                cards.append(card)
        cards.extend(await self._semantic_cards(usable))

        deduped: dict[tuple[object, ...], ExperienceCard] = {}
        for card in cards:
            existing = deduped.get(card.identity)
            if existing is None:
                deduped[card.identity] = card
                continue
            merged_trace_ids = tuple(sorted(set(existing.trace_ids) | set(card.trace_ids)))
            merged_task_ids = tuple(sorted(set(existing.task_ids) | set(card.task_ids)))
            deduped[card.identity] = ExperienceCard(
                **{**existing.__dict__, "trace_ids": merged_trace_ids,
                   "task_ids": merged_task_ids, "evidence_count": len(merged_trace_ids)}
            )
        return ExtractionReport(
            cards=tuple(sorted(deduped.values(), key=lambda item: item.card_id)),
            engineering_issues=tuple(sorted(set(issues))),
            ignored_trace_ids=tuple(sorted(set(ignored))),
            semantic_extractor_calls=self.semantic_extractor_calls - calls_before,
        )


def eligible_cards(cards: Iterable[ExperienceCard]) -> tuple[ExperienceCard, ...]:
    """Promote a reusable strong card, or an exactly matching 3-task medium pattern."""
    strong = [
        card for card in cards
        if card.reusable and card.verification_strength == "strong"
    ]
    medium_groups: dict[tuple[object, ...], list[ExperienceCard]] = {}
    for card in cards:
        if card.reusable and card.verification_strength == "medium":
            medium_groups.setdefault(card.identity, []).append(card)
    promoted: list[ExperienceCard] = list(strong)
    for identity, grouped in medium_groups.items():
        task_ids = tuple(sorted({task_id for card in grouped for task_id in card.task_ids}))
        if len(task_ids) < 3:
            continue
        trace_ids = tuple(sorted({trace_id for card in grouped for trace_id in card.trace_ids}))
        base = grouped[0]
        promoted.append(ExperienceCard(
            **{**base.__dict__, "card_id": SignalExtractor._card_id("MEDIUM", identity),
               "trace_ids": trace_ids, "task_ids": task_ids,
               "evidence_count": len(trace_ids)}
        ))
    unique: dict[tuple[object, ...], ExperienceCard] = {}
    for card in promoted:
        unique.setdefault(card.identity, card)
    return tuple(sorted(unique.values(), key=lambda item: item.card_id))


async def generate_candidate_once(
    report: ExtractionReport,
    *,
    generator: CandidateGenerator,
    action: Literal["create", "evolve"],
    verifier: CandidateVerifier,
    active_content: str | None = None,
) -> CandidateBatchResult:
    """Use the existing candidate path once, leaving release approval to a human."""
    calls_before = generator.call_count
    cards = eligible_cards(report.cards)
    if not cards:
        return CandidateBatchResult(None, None, (), "no_eligible_cards", 0)
    processor = TaskEndProcessor(generator)
    queued = [processor.record(card.as_signal()) for card in cards]
    if not any(item == "eligible" for item in queued):
        return CandidateBatchResult(None, None, (), "no_eligible_cards", 0)
    candidate = await processor.process(action)
    if candidate is None:
        return CandidateBatchResult(None, None, (), "no_candidate", generator.call_count - calls_before)
    supported_ids = {trace_id for card in cards for trace_id in card.trace_ids}
    if set(candidate.trace_ids) != supported_ids:
        raise ValueError("candidate must bind every supporting Trace ID")
    verification = verifier.verify_content(
        candidate.content, action=candidate.action, active_content=active_content
    )
    return CandidateBatchResult(
        candidate=candidate,
        verification=verification,
        supporting_cards=cards,
        status="pending_manual_review" if verification.status == "verified" else "verifier_rejected",
        generator_calls=generator.call_count - calls_before,
    )


async def process_task_end_batch(
    observations: Iterable[TaskObservation],
    *,
    extractor: SignalExtractor,
    generator: CandidateGenerator,
    action: Literal["create", "evolve"],
    verifier: CandidateVerifier,
    active_content: str | None = None,
) -> tuple[ExtractionReport, CandidateBatchResult]:
    """The explicit task-end/manual-batch entrypoint; no per-turn invocation exists."""
    report = await extractor.extract(observations)
    return report, await generate_candidate_once(
        report,
        generator=generator,
        action=action,
        verifier=verifier,
        active_content=active_content,
    )


async def process_persisted_trace_batch(
    trace_paths: Iterable[str | Path],
    *,
    feedback_by_trace_id: Mapping[str, str] | None,
    user_confirmed_trace_ids: Iterable[str] = (),
    extractor: SignalExtractor,
    generator: CandidateGenerator,
    action: Literal["create", "evolve"],
    verifier: CandidateVerifier,
    active_content: str | None = None,
) -> tuple[ExtractionReport, CandidateBatchResult]:
    """Explicit manual batch entrypoint for persisted Runtime Trace and feedback."""
    feedbacks = dict(feedback_by_trace_id or {})
    confirmed = {str(trace_id) for trace_id in user_confirmed_trace_ids}
    observations: list[TaskObservation] = []
    for path in trace_paths:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        trace_id = raw.get("trace_id") if isinstance(raw, Mapping) else None
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise ValueError("persisted task trace has no trace_id")
        observations.append(observation_from_persisted_trace(
            path,
            feedback=feedbacks.get(trace_id),
            user_confirmed=trace_id in confirmed,
        ))
    return await process_task_end_batch(
        observations,
        extractor=extractor,
        generator=generator,
        action=action,
        verifier=verifier,
        active_content=active_content,
    )


async def run_fixed_demo(workspace_root: str | Path) -> dict[str, Any]:
    """A zero-API persisted Trace/feedback -> card -> candidate review demonstration."""
    from healthvault.evolution.synthetic_demo import SKILL_V1, SKILL_V2
    from healthvault.evolution.trace import TaskTraceRecorder

    root = Path(workspace_root)
    active = root / ".bear" / "skills" / "HealthVault-Task-Execution" / "SKILL.md"
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_bytes(SKILL_V1.encode("utf-8"))
    skill_sha = hashlib.sha256(SKILL_V1.encode("utf-8")).hexdigest()
    evolution_root = root / ".bear" / "skill-evolution" / "healthvault"
    context = {
        "intent": "retrieve a chronological record timeline",
        "preconditions": ["read-only record access"],
        "expected_outcome": "validated timeline result",
        "validator_status": "passed",
        "contract_checks": {"tool": "passed", "status": "passed", "citation": "passed"},
    }

    async def fake_tool(_name: str, _input: dict[str, Any], _state: dict[str, float] | None) -> str:
        return json.dumps({"status": "ok", "validator_code": "citation_valid"})

    trace_paths: list[Path] = []
    for index in range(1, 4):
        recorder = TaskTraceRecorder(
            evolution_root,
            request_id=f"request-demo-{index}", task_id=f"TASK-DEMO-{index}",
            skill={"name": "HealthVault-Task-Execution", "sha256": skill_sha},
            executor=fake_tool, signal_context=context,
        )
        await recorder.execute("query_health_timeline", {})
        trace_paths.append(recorder.finish("succeeded"))

    failed_context = {**context, "validator_status": "failed", "contract_checks": {"status": "failed"}}
    failed = TaskTraceRecorder(
        evolution_root,
        request_id="request-demo-feedback", task_id="TASK-DEMO-FEEDBACK",
        skill={"name": "HealthVault-Task-Execution", "sha256": skill_sha},
        executor=fake_tool, signal_context=failed_context,
    )
    failed_trace = failed.finish("failed", failure_code="response_contract_mismatch")
    trace_paths.append(failed_trace)

    async def fake_candidate(prompt: str, _schema: dict[str, Any]) -> dict[str, Any]:
        trace_ids = [
            trace_id for evidence in json.loads(prompt)["evidence"]
            for trace_id in evidence["trace_ids"]
        ]
        return {
            "action": "evolve", "skill_name": "HealthVault-Task-Execution",
            "content": SKILL_V2,
            "rationale": "Use the verified repeated read-only workflow.",
            "trace_ids": trace_ids,
        }

    async def fake_semantic(payload: dict[str, Any], _schema: dict[str, Any], _config: dict[str, Any]) -> dict[str, Any]:
        trace_id = payload["observations"][0]["trace_id"]
        return {"corrections": [{
            "trace_id": trace_id,
            "intent": "route a timeline request before answering",
            "preconditions": ["a rejected result exists"],
            "lesson": "Use the relevant read-only route instead of claiming a rejected result succeeded.",
            "when_not_to_use": ["when no correction feedback exists"],
            "reusable": True,
        }]}

    generator = CandidateGenerator(evolution_root, fake_candidate)
    report, result = await process_persisted_trace_batch(
        trace_paths,
        feedback_by_trace_id={
            json.loads(failed_trace.read_text(encoding="utf-8"))["trace_id"]:
                "This is wrong; use the relevant route instead.",
        },
        extractor=SignalExtractor(fake_semantic), generator=generator, action="evolve",
        verifier=CandidateVerifier("HealthVault-Task-Execution"),
        active_content=SKILL_V1,
    )
    active_after = hashlib.sha256(active.read_bytes()).hexdigest()
    return {
        "scenario": "fixed-trace-feedback-card-candidate-manual-review",
        "experience_cards": [card.card_id for card in report.cards],
        "candidate_id": result.candidate.candidate_id if result.candidate else None,
        "candidate_trace_ids": list(result.candidate.trace_ids) if result.candidate else [],
        "verification": result.verification.status if result.verification else None,
        "status": result.status,
        "generator_calls": result.generator_calls,
        "semantic_extractor_calls": report.semantic_extractor_calls,
        "active_skill_unchanged": active_after == skill_sha,
        "published": False,
        "api_calls": 0,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Ignored runtime demo JSON path.")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    summary = asyncio.run(run_fixed_demo(output.parent))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
