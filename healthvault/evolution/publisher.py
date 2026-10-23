"""Atomic publication and rollback for the single active HealthVault Skill."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agents.skills import load_project_skill_resources, project_skill_manifest
from healthvault.evolution.candidates import SkillCandidate, TARGET_SKILL
from healthvault.evolution.verifier import VerificationResult


@dataclass(frozen=True)
class ReleaseResult:
    status: str
    active_sha256: str | None
    previous_sha256: str | None = None
    manifest: tuple[dict[str, str], ...] = ()


class SkillPublisher:
    def __init__(
        self,
        skill_root: str | Path,
        evolution_root: str | Path,
        *,
        replace_file: Callable[[str | Path, str | Path], None] = os.replace,
    ) -> None:
        self._skill_root = Path(skill_root)
        self._evolution_root = Path(evolution_root)
        self._replace = replace_file
        self._state_path = self._evolution_root / "release-state.json"

    @property
    def _active_path(self) -> Path:
        return self._skill_root / TARGET_SKILL / "SKILL.md"

    @staticmethod
    def _digest(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _manifest(self) -> tuple[dict[str, str], ...]:
        resources = load_project_skill_resources(self._skill_root, (TARGET_SKILL,))
        return tuple(project_skill_manifest(resources))

    def _read_state(self) -> dict[str, object]:
        if not self._state_path.is_file():
            return {"history": [], "rolled_back_from": None}
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"history": [], "rolled_back_from": None}
        if not isinstance(state, dict) or not isinstance(state.get("history"), list):
            return {"history": [], "rolled_back_from": None}
        return state

    def _write_state(self, state: dict[str, object]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_name(f".{self._state_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self._state_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _save_version(self, content: bytes) -> str:
        digest = self._digest(content)
        version = self._evolution_root / "versions" / digest / "SKILL.md"
        version.parent.mkdir(parents=True, exist_ok=True)
        if not version.exists():
            version.write_bytes(content)
        return digest

    def _restore_active(self, content: bytes | None) -> None:
        if content is None:
            self._active_path.unlink(missing_ok=True)
            return
        restore = self._active_path.with_name(
            f".{self._active_path.name}.{uuid.uuid4().hex}.restore"
        )
        try:
            restore.write_bytes(content)
            os.replace(restore, self._active_path)
        finally:
            restore.unlink(missing_ok=True)

    def publish_content(
        self,
        content: str,
        *,
        action: str,
        verification: VerificationResult,
        approved: bool,
    ) -> ReleaseResult:
        active_before = self._active_path.read_bytes() if self._active_path.is_file() else None
        previous_sha = self._digest(active_before) if active_before is not None else None
        candidate = content.encode("utf-8")
        candidate_sha = self._digest(candidate)
        if verification.status != "verified" or verification.content_sha256 != candidate_sha:
            return ReleaseResult("verification_required", previous_sha)
        if not approved:
            return ReleaseResult("approval_required", previous_sha)
        if active_before == candidate:
            return ReleaseResult("already_published", candidate_sha, previous_sha, self._manifest())
        if action == "create" and active_before is not None:
            return ReleaseResult("active_skill_exists", previous_sha)
        if action == "evolve" and active_before is None:
            return ReleaseResult("active_skill_missing", None)

        self._active_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._active_path.with_name(f".{self._active_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(candidate)
            self._replace(temporary, self._active_path)
            manifest = self._manifest()
            if manifest[0]["sha256"] != candidate_sha:
                raise RuntimeError("Reloaded Skill manifest hash does not match publication.")
        except Exception:
            temporary.unlink(missing_ok=True)
            if self._active_path.is_file() or active_before is not None:
                self._restore_active(active_before)
            return ReleaseResult("publish_failed", previous_sha, previous_sha)
        finally:
            temporary.unlink(missing_ok=True)

        try:
            if active_before is not None:
                self._save_version(active_before)
            self._save_version(candidate)
            state = self._read_state()
            history = [str(item) for item in state["history"]]
            if previous_sha is not None and previous_sha not in history:
                history.append(previous_sha)
            if not history or history[-1] != candidate_sha:
                history.append(candidate_sha)
            self._write_state({"history": history, "rolled_back_from": None})
        except Exception:
            self._restore_active(active_before)
            return ReleaseResult("publish_failed", previous_sha, previous_sha)
        return ReleaseResult("published", candidate_sha, previous_sha, manifest)

    def publish_candidate(
        self,
        candidate: SkillCandidate,
        *,
        verification: VerificationResult,
        approved: bool,
    ) -> ReleaseResult:
        if candidate.skill_name != TARGET_SKILL:
            return ReleaseResult("candidate_target_mismatch", None)
        return self.publish_content(
            candidate.content,
            action=candidate.action,
            verification=verification,
            approved=approved,
        )

    def rollback(self) -> ReleaseResult:
        if not self._active_path.is_file():
            return ReleaseResult("active_skill_missing", None)
        current = self._digest(self._active_path.read_bytes())
        state = self._read_state()
        history = [str(item) for item in state["history"]]
        if len(history) < 2 or history[-1] != current:
            if state.get("rolled_back_from"):
                return ReleaseResult("already_rolled_back", current, str(state["rolled_back_from"]), self._manifest())
            return ReleaseResult("rollback_unavailable", current)
        target_sha = history[-2]
        target_path = self._evolution_root / "versions" / target_sha / "SKILL.md"
        if not target_path.is_file() or self._digest(target_path.read_bytes()) != target_sha:
            return ReleaseResult("rollback_unavailable", current)
        temporary = self._active_path.with_name(f".{self._active_path.name}.{uuid.uuid4().hex}.tmp")
        current_content = self._active_path.read_bytes()
        try:
            shutil.copyfile(target_path, temporary)
            self._replace(temporary, self._active_path)
            manifest = self._manifest()
            if manifest[0]["sha256"] != target_sha:
                raise RuntimeError("Reloaded rollback manifest hash does not match target.")
            self._write_state({"history": history[:-1], "rolled_back_from": current})
        except Exception:
            temporary.unlink(missing_ok=True)
            if self._active_path.is_file() and self._active_path.read_bytes() != current_content:
                self._restore_active(current_content)
            return ReleaseResult("rollback_failed", current, target_sha)
        finally:
            temporary.unlink(missing_ok=True)
        return ReleaseResult("rolled_back", target_sha, current, manifest)
