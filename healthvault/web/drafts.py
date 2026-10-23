"""Durable, profile-owned report intake; independent of structured confirmation."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Draft:
    token: str
    profile_id: str
    source_id: str
    report_text: str
    input_kind: str
    preview_state: dict[str, Any] | None = None
    preview_sha256: str | None = None
    image_bytes: bytes | None = None
    image_filename: str | None = None
    image_media_type: str | None = None
    image_sha256: str | None = None
    ocr_text_sha256: str | None = None
    ocr_blocks: list[dict[str, Any]] | None = None
    ocr_confidence: dict[str, Any] | None = None
    ocr_reviewed: bool = False
    auto_extract: bool = False
    extraction_payload: dict[str, Any] | None = None
    upload_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    original_text: str = ""
    processing_status: str = "received"
    processing_error: str | None = None
    document_artifact: dict[str, Any] | None = None
    archived_at: str | None = None
    duplicate_of: dict[str, Any] | None = None


class DraftConflict(ValueError):
    pass


class DraftStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS report_uploads (
                upload_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
                token TEXT NOT NULL UNIQUE, payload TEXT NOT NULL, original BLOB,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def save(self, draft: Draft, *, previous_token: str | None = None) -> None:
        payload = {key: value for key, value in asdict(draft).items() if key != "image_bytes"}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT token, profile_id, payload FROM report_uploads WHERE upload_id = ?", (draft.upload_id,)).fetchone()
            if row is not None and (row["profile_id"] != draft.profile_id or row["token"] != (previous_token or draft.token)):
                raise DraftConflict("草稿已更新，请重新打开。")
            if row is not None and previous_token and json.loads(row['payload']).get('archived_at') != draft.archived_at:
                raise DraftConflict('草稿归档状态已变化，请重新打开。')
            if row is None and (fingerprint := self._fingerprint(payload)):
                existing = db.execute('SELECT payload FROM report_uploads WHERE profile_id = ? ORDER BY created_at', (draft.profile_id,)).fetchall()
                for item in existing:
                    candidate = json.loads(item['payload'])
                    if self._fingerprint(candidate) == fingerprint:
                        draft.duplicate_of = {key: candidate[key] for key in ('upload_id', 'source_id', 'image_filename')}
                        payload['duplicate_of'] = draft.duplicate_of
                        break
            db.execute("""INSERT INTO report_uploads VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(upload_id) DO UPDATE SET token=excluded.token, payload=excluded.payload,
                    original=excluded.original, updated_at=excluded.updated_at""",
                (draft.upload_id, draft.profile_id, draft.token, json.dumps(payload, ensure_ascii=False), draft.image_bytes, now, now))

    @staticmethod
    def _fingerprint(payload: dict[str, Any]) -> tuple[str, str] | None:
        if payload.get('image_sha256'):
            return ('file', payload['image_sha256'])
        text = payload.get('original_text') or payload.get('report_text')
        return ('text', hashlib.sha256(text.encode('utf-8')).hexdigest()) if text else None

    def get(self, token: str) -> Draft | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT payload, original FROM report_uploads WHERE token = ?", (token,)).fetchone()
        if row is None:
            return None
        return Draft(**json.loads(row["payload"]), image_bytes=row["original"])

    def confirmed_original(self, profile_id: str, source_id: str, report_text: str) -> str | None:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT payload FROM report_uploads WHERE profile_id = ? ORDER BY updated_at DESC",
                (profile_id,),
            ).fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            if (payload["processing_status"] == "confirmed"
                    and payload["source_id"] == source_id
                    and payload["report_text"] == report_text):
                return payload["original_text"]
        return None

    def list(self, profile_id: str, *, status: str = 'active') -> list[dict[str, Any]]:
        if status not in {'active', 'pending', 'failed', 'archived', 'all'}:
            raise ValueError('Unsupported draft filter.')
        with closing(self._connect()) as db:
            rows = db.execute("SELECT payload, created_at, updated_at FROM report_uploads WHERE profile_id = ? ORDER BY updated_at DESC", (profile_id,)).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row['payload'])
            if payload['processing_status'] == 'confirmed':
                continue
            archived = bool(payload.get('archived_at'))
            failed = payload['processing_status'] == 'extraction_failed'
            if ((status == 'archived' and not archived) or (status in {'active', 'pending', 'failed'} and archived)
                    or (status == 'pending' and failed) or (status == 'failed' and not failed)):
                continue
            result.append({**{key: payload[key] for key in (
                "token", "source_id", "input_kind", "image_filename", "processing_status", "processing_error",
            )}, 'archived_at': payload.get('archived_at'), 'duplicate_of': payload.get('duplicate_of'),
                "created_at": row["created_at"], "updated_at": row["updated_at"]})
        return result
