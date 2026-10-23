"""Structured, profile-bound workflow checkpoints; never a medical fact store."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from healthvault.bootstrap import READ_ONLY_TOOL_NAMES
from healthvault.repositories.health_records import HealthRecordRepository
from healthvault.runtime_context import RuntimeContext

STEPS = frozenset(("start", *READ_ONLY_TOOL_NAMES, "done"))
PREFERENCES = {
    "language": frozenset({"zh-CN", "en-US"}),
    "unit_system": frozenset({"original", "metric"}),
    "answer_style": frozenset({"concise", "detailed"}),
}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class TaskMemoryConflict(RuntimeError):
    """A checkpoint was changed by another session."""


@dataclass(frozen=True)
class TaskCheckpoint:
    profile_id: str
    task_id: str
    version: int
    created_at: str
    updated_at: str
    expires_at: str
    source_refs: tuple[str, ...]
    artifact_refs: tuple[str, ...]
    completed_steps: tuple[str, ...]
    next_step: str
    status: str
    preferences: dict[str, str]

    def prompt_section(self) -> str:
        """Generate a process summary from constrained fields, not stored prose."""
        return (
            "# HealthVault task checkpoint\n"
            f"Task: {self.task_id}; status: {self.status}; next step: {self.next_step}.\n"
            f"Completed steps: {', '.join(self.completed_steps) or 'none'}.\n"
            f"Confirmed source references: {', '.join(self.source_refs) or 'none'}.\n"
            f"Artifact references: {', '.join(self.artifact_refs) or 'none'}.\n"
            f"Display preferences: {json.dumps(self.preferences, sort_keys=True)}.\n"
            "Re-read confirmed records for all medical facts. This checkpoint is workflow state only."
        )


class TaskMemoryStore:
    """SQLite store keyed by application-supplied RuntimeContext.profile_id."""

    def __init__(self, path: str | Path, records: HealthRecordRepository) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = records
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS task_checkpoints (
                profile_id TEXT NOT NULL, task_id TEXT NOT NULL, version INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, expires_at TEXT NOT NULL,
                source_refs TEXT NOT NULL, artifact_refs TEXT NOT NULL,
                completed_steps TEXT NOT NULL, next_step TEXT NOT NULL,
                status TEXT NOT NULL, preferences TEXT NOT NULL,
                PRIMARY KEY (profile_id, task_id))""")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _check_id(value: str) -> None:
        if not isinstance(value, str) or not _ID.fullmatch(value):
            raise ValueError("Expected a short opaque identifier.")

    @staticmethod
    def _decode(row: sqlite3.Row) -> TaskCheckpoint:
        return TaskCheckpoint(
            row["profile_id"], row["task_id"], row["version"], row["created_at"],
            row["updated_at"], row["expires_at"],
            tuple(json.loads(row["source_refs"])), tuple(json.loads(row["artifact_refs"])),
            tuple(json.loads(row["completed_steps"])), row["next_step"], row["status"],
            json.loads(row["preferences"]),
        )

    def get(self, context: RuntimeContext, task_id: str) -> TaskCheckpoint | None:
        self._check_id(task_id)
        with closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT * FROM task_checkpoints WHERE profile_id=? AND task_id=?",
                (context.profile_id, task_id),
            ).fetchone()
            if row is None:
                return None
            if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
                db.execute("DELETE FROM task_checkpoints WHERE profile_id=? AND task_id=?",
                           (context.profile_id, task_id))
                return None
            return self._decode(row)

    def save(self, context: RuntimeContext, task_id: str, *, expected_version: int,
             completed_steps: tuple[str, ...] = (), next_step: str = "start",
             status: str = "in_progress", source_refs: tuple[str, ...] = (),
             artifact_refs: tuple[str, ...] = (), preferences: Mapping[str, str] | None = None,
             ttl: timedelta = timedelta(days=30)) -> TaskCheckpoint:
        self._check_id(task_id)
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("expected_version must be nonnegative.")
        if (status not in {"in_progress", "completed"} or next_step not in STEPS
                or (status == "completed") != (next_step == "done")):
            raise ValueError("Invalid workflow state.")
        if any(step not in STEPS or step in {"start", "done"} for step in completed_steps):
            raise ValueError("Invalid completed step.")
        if ttl <= timedelta(0):
            raise ValueError("TTL must be positive.")
        prefs = dict(preferences or {})
        if any(key not in PREFERENCES or value not in PREFERENCES[key]
               for key, value in prefs.items()):
            raise ValueError("Only enumerated display preferences may be stored.")
        if len(source_refs) > 20 or len(artifact_refs) > 20:
            raise ValueError("Too many references.")
        for ref in (*source_refs, *artifact_refs):
            self._check_id(ref)
        confirmed = {report.source_id for report in self.records.list_timeline(context.profile_id)}
        if not set(source_refs) <= confirmed:
            raise ValueError("Source reference is not confirmed for this profile.")
        now = datetime.now(timezone.utc)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM task_checkpoints WHERE profile_id=? AND task_id=?",
                             (context.profile_id, task_id)).fetchone()
            if row is not None and datetime.fromisoformat(row["expires_at"]) <= now:
                db.execute("DELETE FROM task_checkpoints WHERE profile_id=? AND task_id=?",
                           (context.profile_id, task_id))
                row = None
            version = row["version"] if row is not None else 0
            if version != expected_version:
                raise TaskMemoryConflict("Task checkpoint version changed.")
            created = row["created_at"] if row is not None else now.isoformat()
            db.execute("""INSERT OR REPLACE INTO task_checkpoints VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (context.profile_id, task_id, version + 1, created, now.isoformat(),
                 (now + ttl).isoformat(), json.dumps(sorted(set(source_refs))),
                 json.dumps(sorted(set(artifact_refs))), json.dumps(list(dict.fromkeys(completed_steps))),
                 next_step, status, json.dumps(prefs, sort_keys=True)))
            saved = self._decode(db.execute(
                "SELECT * FROM task_checkpoints WHERE profile_id=? AND task_id=?",
                (context.profile_id, task_id),
            ).fetchone())
        return saved

    def delete(self, context: RuntimeContext, task_id: str) -> bool:
        self._check_id(task_id)
        with closing(self._connect()) as db, db:
            return bool(db.execute("DELETE FROM task_checkpoints WHERE profile_id=? AND task_id=?",
                                   (context.profile_id, task_id)).rowcount)
