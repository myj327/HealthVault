"""Deterministic active-Skill usage evaluation and lifecycle actions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from healthvault.evolution.publisher import SkillPublisher
from healthvault.evolution.failures import FAILURE_CATEGORIES


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TARGET_SKILL = "HealthVault-Task-Execution"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
            + b"\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class SkillUsageEvent:
    run_id: str
    task_id: str
    task_type: str
    skill_name: str
    skill_sha256: str
    retrieved: bool
    injected: bool
    used: bool
    task_status: str
    validator_status: str
    hard_failure: bool
    failure_code: str | None
    tool_count: int
    step_count: int
    retry_count: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    failure_category: str = 'none'
    excluded_from_skill_rate: bool = False

    def __post_init__(self) -> None:
        if self.failure_category not in FAILURE_CATEGORIES or type(self.excluded_from_skill_rate) is not bool:
            raise ValueError('Invalid usage failure attribution.')
        if self.excluded_from_skill_rate and (self.failure_category != 'external_service' or self.hard_failure):
            raise ValueError('Only explicit external service failures may be excluded.')
        for name in ("run_id", "task_id", "task_type", "skill_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Skill usage {name} must be non-empty.")
        if SHA256_PATTERN.fullmatch(self.skill_sha256) is None:
            raise ValueError("Skill usage SHA-256 must be lowercase hexadecimal.")
        if self.task_status not in {"succeeded", "failed", "rejected"}:
            raise ValueError("Skill usage task status is invalid.")
        if self.validator_status not in {"passed", "failed", "not_applicable"}:
            raise ValueError("Skill usage validator status is invalid.")
        for name in (
            "tool_count", "step_count", "retry_count", "input_tokens", "output_tokens"
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Skill usage {name} must be a non-negative integer.")
        for name in ("latency_ms", "cost_usd"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Skill usage {name} must be finite and non-negative.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UsageEvaluation:
    evaluation_id: str
    skill_name: str
    skill_sha256: str
    task_type: str
    decision: str
    total_events: int
    effective_samples: int
    success_count: int
    success_rate: float
    hard_failures: int
    safety_hard_failures: int
    validator_failures: int
    retrieved_not_injected: int
    excluded_other_version: int
    duplicate_runs: int
    used_count: int
    total_tool_count: int
    total_step_count: int
    total_retry_count: int
    total_tokens: int
    total_latency_ms: float
    total_cost_usd: float
    observed_tasks: int = 0
    task_failures: int = 0
    observed_task_success_rate: float = 0.0
    external_service_failures: int = 0
    tool_failures: int = 0
    workflow_failures: int = 0
    answer_quality_failures: int = 0
    unknown_failures: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LifecycleResult:
    status: str
    actions: tuple[str, ...]
    problem_sha256: str
    final_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["actions"] = list(self.actions)
        return data


class SkillUsageEvaluator:
    def __init__(self, evolution_root: str | Path) -> None:
        self._root = Path(evolution_root)

    def evaluate(
        self,
        events: Iterable[SkillUsageEvent],
        skill_name: str,
        skill_sha256: str,
        task_type: str,
    ) -> UsageEvaluation:
        if not skill_name.strip() or not task_type.strip() or SHA256_PATTERN.fullmatch(skill_sha256) is None:
            raise ValueError("Usage evaluation target is invalid.")
        supplied = list(events)
        seen_target_runs: set[str] = set()
        effective: list[SkillUsageEvent] = []
        observed: list[SkillUsageEvent] = []
        retrieved_not_injected = 0
        excluded_other_version = 0
        duplicate_runs = 0
        for item in supplied:
            event_scope = (
                f"{item.skill_name}\0{item.skill_sha256}\0"
                f"{item.task_type}\0{item.run_id}"
            )
            event_id = hashlib.sha256(event_scope.encode("utf-8")).hexdigest()[:20]
            path = self._root / "usage-eval" / "events" / f"{event_id}.json"
            if not path.exists():
                _atomic_json(path, item.to_dict())
            if (
                item.skill_name != skill_name
                or item.skill_sha256 != skill_sha256
                or item.task_type != task_type
            ):
                excluded_other_version += 1
                continue
            if item.run_id in seen_target_runs:
                duplicate_runs += 1
                continue
            seen_target_runs.add(item.run_id)
            if not item.injected:
                if item.retrieved:
                    retrieved_not_injected += 1
                continue
            observed.append(item)
            if not item.excluded_from_skill_rate:
                effective.append(item)

        success_count = sum(
            item.task_status == "succeeded"
            and item.validator_status == "passed"
            and not item.hard_failure
            for item in effective
        )
        hard_failures = sum(item.hard_failure for item in effective)
        safety_hard_failures = sum(
            item.hard_failure
            and bool(item.failure_code)
            and str(item.failure_code).startswith("safety_")
            for item in effective
        )
        validator_failures = sum(item.validator_status == "failed" for item in effective)
        used_count = sum(item.used for item in effective)
        sample_count = len(effective)
        success_rate = success_count / sample_count if sample_count else 0.0
        if safety_hard_failures:
            decision = "disable"
        elif sample_count >= 10 and success_rate < 0.5 and validator_failures >= 5:
            decision = "disable"
        elif (
            sample_count >= 5
            and success_rate >= 0.8
            and hard_failures == 0
            and validator_failures == 0
            and used_count > 0
        ):
            decision = "keep"
        else:
            decision = "watch"
        identity = {
            "skill_name": skill_name,
            "skill_sha256": skill_sha256,
            "task_type": task_type,
            "run_ids": [item.run_id for item in effective],
            "decision": decision,
        }
        evaluation = UsageEvaluation(
            evaluation_id=hashlib.sha256(_canonical_json(identity)).hexdigest()[:20],
            skill_name=skill_name,
            skill_sha256=skill_sha256,
            task_type=task_type,
            decision=decision,
            total_events=len(supplied),
            effective_samples=sample_count,
            success_count=success_count,
            success_rate=round(success_rate, 6),
            hard_failures=hard_failures,
            safety_hard_failures=safety_hard_failures,
            validator_failures=validator_failures,
            retrieved_not_injected=retrieved_not_injected,
            excluded_other_version=excluded_other_version,
            duplicate_runs=duplicate_runs,
            used_count=used_count,
            total_tool_count=sum(item.tool_count for item in observed),
            total_step_count=sum(item.step_count for item in observed),
            total_retry_count=sum(item.retry_count for item in observed),
            total_tokens=sum(item.input_tokens + item.output_tokens for item in observed),
            total_latency_ms=round(sum(item.latency_ms for item in observed), 3),
            total_cost_usd=round(sum(item.cost_usd for item in observed), 8),
            observed_tasks=len(observed), task_failures=len(observed) - success_count,
            observed_task_success_rate=round(success_count / len(observed), 6) if observed else 0.0,
            external_service_failures=sum(item.failure_category == 'external_service' for item in observed),
            tool_failures=sum(item.failure_category == 'tool' for item in observed),
            workflow_failures=sum(item.failure_category == 'workflow' for item in observed),
            answer_quality_failures=sum(item.failure_category == 'answer_quality' for item in observed),
            unknown_failures=sum(item.failure_category == 'unknown' for item in observed),
        )
        _atomic_json(
            self._root / "usage-eval" / "evaluations" / f"{evaluation.evaluation_id}.json",
            evaluation.to_dict(),
        )
        return evaluation


class SkillLifecycleManager:
    def __init__(self, skill_root: str | Path, evolution_root: str | Path) -> None:
        self._skill_root = Path(skill_root)
        self._root = Path(evolution_root)
        self._state_path = self._root / "usage-eval" / "lifecycle-state.json"
        self._publisher = SkillPublisher(self._skill_root, self._root)

    def _read_state(self) -> dict[str, object]:
        if not self._state_path.is_file():
            return {"versions": {}, "applied_evaluations": {}}
        try:
            value = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"versions": {}, "applied_evaluations": {}}
        if not isinstance(value, dict):
            return {"versions": {}, "applied_evaluations": {}}
        value.setdefault("versions", {})
        value.setdefault("applied_evaluations", {})
        return value

    def is_injectable(self, skill_sha256: str) -> bool:
        state = self._read_state()
        version = state["versions"].get(skill_sha256, {})
        return version.get("status") not in {"disabled", "archived"}

    def disable_active(self, reason: str = "user_disabled") -> dict[str, object]:
        digest = self._active_sha256()
        if digest is None:
            return {"status": "active_skill_missing"}
        state = self._read_state()
        state["versions"][digest] = {"status": "disabled", "failure": reason}
        _atomic_json(self._state_path, state)
        return {"status": "disabled", "skill_sha256": digest}

    def apply(self, evaluation: UsageEvaluation) -> LifecycleResult:
        state = self._read_state()
        applied = state["applied_evaluations"]
        if evaluation.evaluation_id in applied:
            previous = applied[evaluation.evaluation_id]
            return LifecycleResult(
                "already_applied",
                tuple(previous["actions"]),
                evaluation.skill_sha256,
                previous.get("final_sha256"),
            )
        if evaluation.decision in {"keep", "watch"}:
            result = LifecycleResult(
                "applied",
                (evaluation.decision,),
                evaluation.skill_sha256,
                self._active_sha256(),
            )
            applied[evaluation.evaluation_id] = result.to_dict()
            _atomic_json(self._state_path, state)
            return result

        failure = "safety_hard_failure" if evaluation.safety_hard_failures else "sustained_contract_failure"
        versions = state["versions"]
        versions[evaluation.skill_sha256] = {
            "status": "disabled",
            "evaluation_id": evaluation.evaluation_id,
            "failure": failure,
        }
        _atomic_json(self._state_path, state)
        if self._active_sha256() != evaluation.skill_sha256:
            result = LifecycleResult("inactive_version_disabled", ("disable",), evaluation.skill_sha256, self._active_sha256())
            state["applied_evaluations"][evaluation.evaluation_id] = result.to_dict()
            _atomic_json(self._state_path, state)
            return result
        rollback = self._publisher.rollback()
        if rollback.status != "rolled_back":
            result = LifecycleResult(
                "rollback_failed",
                ("disable",),
                evaluation.skill_sha256,
                rollback.active_sha256,
            )
            state = self._read_state()
            state["applied_evaluations"][evaluation.evaluation_id] = result.to_dict()
            _atomic_json(self._state_path, state)
            return result

        archive_record = {
            "schema_version": "healthvault-skill-archive-v1",
            "skill_name": evaluation.skill_name,
            "skill_sha256": evaluation.skill_sha256,
            "status": "archived",
            "evaluation_id": evaluation.evaluation_id,
            "effective_samples": evaluation.effective_samples,
            "hard_failures": evaluation.hard_failures,
            "validator_failures": evaluation.validator_failures,
            "version_evidence": f"versions/{evaluation.skill_sha256}/SKILL.md",
            "physical_delete": False,
        }
        _atomic_json(
            self._root / "archive" / f"{evaluation.skill_sha256}.json",
            archive_record,
        )
        state = self._read_state()
        state["versions"][evaluation.skill_sha256] = {
            "status": "archived",
            "evaluation_id": evaluation.evaluation_id,
            "failure": failure,
        }
        result = LifecycleResult(
            "applied",
            ("disable", "rollback", "archive"),
            evaluation.skill_sha256,
            rollback.active_sha256,
        )
        state["applied_evaluations"][evaluation.evaluation_id] = result.to_dict()
        _atomic_json(self._state_path, state)
        _atomic_json(
            self._root / "usage-eval" / "lifecycle" / f"{evaluation.evaluation_id}.json",
            result.to_dict(),
        )
        return result

    def _active_sha256(self) -> str | None:
        active = self._skill_root / TARGET_SKILL / "SKILL.md"
        if not active.is_file():
            return None
        return hashlib.sha256(active.read_bytes()).hexdigest()
