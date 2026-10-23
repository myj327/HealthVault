"""Persist generated answers with the exact evidence citations shown to users."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class WebAnswerHistory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS web_answer_history (
                id INTEGER PRIMARY KEY,
                profile_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            columns = {row['name'] for row in db.execute('PRAGMA table_info(web_answer_history)')}
            if 'resolved_question' not in columns:
                db.execute('ALTER TABLE web_answer_history ADD COLUMN resolved_question TEXT')
            db.execute("CREATE INDEX IF NOT EXISTS web_answer_profile ON web_answer_history(profile_id, id)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def save(self, profile_id: str, mode: str, question: str, answer: str,
             sources: list[dict[str, Any]], *, resolved_question: str | None = None) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO web_answer_history (profile_id, mode, question, answer, sources_json, resolved_question) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (profile_id, mode, question, answer, json.dumps(sources, ensure_ascii=False), resolved_question or question),
            )

    def list(self, profile_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT mode, question, answer, sources_json, created_at, COALESCE(resolved_question, question) AS resolved_question "
                "FROM web_answer_history WHERE profile_id = ? ORDER BY id DESC LIMIT 50",
                (profile_id,),
            ).fetchall()
        return [
            {"status": "answered", "mode": row["mode"], "question": row["question"], "answer": row["answer"],
             "sources": json.loads(row["sources_json"]), "created_at": row["created_at"]}
            | {"resolved_question": row["resolved_question"]}
            for row in rows
        ]
