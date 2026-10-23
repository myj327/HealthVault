"""Profile-bound persistent Agent sessions for the local HealthVault web app."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict
from contextlib import closing
from pathlib import Path
from typing import Any

TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
MAX_SESSION_BYTES = 750_000


class AgentTaskConflict(RuntimeError):
    pass


def conversation_title(dialogue: list[dict[str, str]]) -> str:
    first = next((item.get("text", "") for item in dialogue if item.get("role") == "user"), "")
    title = " ".join(first.split())
    return title[:46] + ("…" if len(title) > 46 else "") if title else "新对话"


class WebAgentTaskStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS web_agent_tasks (
                    profile_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    messages_json TEXT NOT NULL,
                    folded_json TEXT NOT NULL,
                    dialogue_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (profile_id, task_id)
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _check_task_id(task_id: str) -> None:
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError("任务 ID 必须是 1–128 位字母、数字或 . _ : -。")

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"],
            "version": row["version"],
            "messages": json.loads(row["messages_json"]),
            "folded": json.loads(row["folded_json"]),
            "dialogue": json.loads(row["dialogue_json"]),
            "updated_at": row["updated_at"],
        }

    def get(self, profile_id: str, task_id: str) -> dict[str, Any] | None:
        self._check_task_id(task_id)
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT * FROM web_agent_tasks WHERE profile_id=? AND task_id=?",
                (profile_id, task_id),
            ).fetchone()
        return self._decode(row) if row else None

    def list_tasks(self, profile_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT task_id, version, dialogue_json, folded_json, updated_at "
                "FROM web_agent_tasks WHERE profile_id=? ORDER BY updated_at DESC LIMIT 30",
                (profile_id,),
            ).fetchall()
        return [
            {
                "task_id": row["task_id"],
                "version": row["version"],
                "title": conversation_title(json.loads(row["dialogue_json"])),
                "message_count": len(json.loads(row["dialogue_json"])),
                "fold_count": len(json.loads(row["folded_json"])),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def save(
        self,
        profile_id: str,
        task_id: str,
        *,
        expected_version: int,
        messages: list[dict[str, Any]],
        folded: list[dict[str, Any]],
        dialogue: list[dict[str, str]],
    ) -> dict[str, Any]:
        self._check_task_id(task_id)
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("Invalid task version.")
        encoded = [json.dumps(value, ensure_ascii=False) for value in (messages, folded, dialogue)]
        if sum(len(value.encode("utf-8")) for value in encoded) > MAX_SESSION_BYTES:
            raise ValueError("会话内容过长，请先折叠会话。")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT version FROM web_agent_tasks WHERE profile_id=? AND task_id=?",
                (profile_id, task_id),
            ).fetchone()
            version = row["version"] if row else 0
            if version != expected_version:
                raise AgentTaskConflict("任务已在另一页面更新，请刷新后继续。")
            db.execute(
                "INSERT INTO web_agent_tasks "
                "(profile_id, task_id, version, messages_json, folded_json, dialogue_json) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(profile_id, task_id) DO UPDATE SET "
                "version=excluded.version, messages_json=excluded.messages_json, "
                "folded_json=excluded.folded_json, dialogue_json=excluded.dialogue_json, "
                "updated_at=CURRENT_TIMESTAMP",
                (profile_id, task_id, version + 1, *encoded),
            )
        saved = self.get(profile_id, task_id)
        assert saved is not None
        return saved

    def delete(self, profile_id: str, task_id: str) -> bool:
        self._check_task_id(task_id)
        with closing(self._connect()) as db, db:
            return bool(db.execute(
                "DELETE FROM web_agent_tasks WHERE profile_id=? AND task_id=?",
                (profile_id, task_id),
            ).rowcount)


def public_task(saved: dict[str, Any] | None, task_id: str, checkpoint: Any = None) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "version": saved["version"] if saved else 0,
        "dialogue": saved["dialogue"] if saved else [],
        "title": conversation_title(saved["dialogue"]) if saved else "新对话",
        "fold_count": len(saved["folded"]) if saved else 0,
        "updated_at": saved["updated_at"] if saved else None,
        "checkpoint": asdict(checkpoint) if checkpoint is not None else None,
    }
