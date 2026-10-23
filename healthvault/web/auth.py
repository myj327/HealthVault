"""Local account and server-side session store for the web application."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


USERNAME = re.compile(r"[a-z0-9][a-z0-9_.-]{2,39}\Z")
SESSION_SECONDS = 7 * 24 * 60 * 60


class DuplicateUsername(ValueError):
    pass


@dataclass(frozen=True)
class AuthUser:
    user_id: str
    username: str
    is_admin: bool = False

    @property
    def profile_id(self) -> str:
        return self.user_id


@dataclass(frozen=True)
class AuthSession:
    user: AuthUser
    csrf_token: str


class AuthStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS web_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS web_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES web_users(user_id),
                    csrf_token TEXT NOT NULL,
                    expires_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS web_sessions_user ON web_sessions(user_id);
            """)
            if "is_admin" not in {row["name"] for row in db.execute("PRAGMA table_info(web_users)")}:
                db.execute("ALTER TABLE web_users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> bytes:
        return hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=2**15, r=8, p=3,
            maxmem=64 * 1024 * 1024, dklen=32,
        )

    @staticmethod
    def _username(value: str) -> str:
        username = value.strip().lower()
        if not USERNAME.fullmatch(username):
            raise ValueError("用户名需为 3–40 位字母、数字、点、下划线或连字符。")
        return username

    @staticmethod
    def _password(value: str) -> None:
        if not 12 <= len(value) <= 128 or len(value.encode("utf-8")) > 256:
            raise ValueError("密码需为 12–128 个字符。")

    def register(self, username: str, password: str, *, user_id: str | None = None) -> AuthUser:
        normalized = self._username(username)
        self._password(password)
        identifier = user_id or "user-" + uuid.uuid4().hex
        salt = secrets.token_bytes(16)
        password_hash = self._password_hash(password, salt)
        try:
            with closing(self._connect()) as db, db:
                db.execute(
                    "INSERT INTO web_users (user_id, username, password_salt, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (identifier, normalized, salt, password_hash, int(time.time())),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateUsername("用户名已存在。") from exc
        return AuthUser(identifier, normalized)

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        try:
            normalized = self._username(username)
        except ValueError:
            normalized = ""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT user_id, username, password_salt, password_hash, is_admin FROM web_users WHERE username = ?",
                (normalized,),
            ).fetchone()
        salt = bytes(row["password_salt"]) if row else bytes(16)
        expected = bytes(row["password_hash"]) if row else bytes(32)
        actual = self._password_hash(password, salt)
        if row is None or not hmac.compare_digest(actual, expected):
            return None
        return AuthUser(row["user_id"], row["username"], bool(row["is_admin"]))

    def set_admin(self, username: str, *, enabled: bool) -> None:
        """Server-owner operation; never exposed through public registration or HTTP."""
        with closing(self._connect()) as db, db:
            updated = db.execute(
                "UPDATE web_users SET is_admin = ? WHERE username = ?",
                (int(enabled), self._username(username)),
            )
            if updated.rowcount != 1:
                raise ValueError("账户不存在，请先创建账户再授予管理权限。")

    def create_session(self, user: AuthUser) -> tuple[str, str]:
        token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO web_sessions (token_hash, user_id, csrf_token, expires_at) VALUES (?, ?, ?, ?)",
                (hashlib.sha256(token.encode()).hexdigest(), user.user_id, csrf_token,
                 int(time.time()) + SESSION_SECONDS),
            )
        return token, csrf_token

    def session(self, token: str | None) -> AuthSession | None:
        if not token:
            return None
        with closing(self._connect()) as db:
            row = db.execute(
                """SELECT users.user_id, users.username, users.is_admin, sessions.csrf_token, sessions.expires_at
                   FROM web_sessions AS sessions JOIN web_users AS users
                   ON users.user_id = sessions.user_id WHERE sessions.token_hash = ?""",
                (hashlib.sha256(token.encode()).hexdigest(),),
            ).fetchone()
        if row is None or row["expires_at"] <= int(time.time()):
            return None
        return AuthSession(AuthUser(row["user_id"], row["username"], bool(row["is_admin"])), row["csrf_token"])

    def logout(self, token: str | None) -> None:
        if token:
            with closing(self._connect()) as db, db:
                db.execute(
                    "DELETE FROM web_sessions WHERE token_hash = ?",
                    (hashlib.sha256(token.encode()).hexdigest(),),
                )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Provision a HealthVault administrator locally.")
    parser.add_argument("--database", required=True, type=Path, help="Existing *-auth.sqlite3 database")
    parser.add_argument("--username", required=True)
    parser.add_argument("--revoke", action="store_true")
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error("Authentication database does not exist.")
    try:
        AuthStore(args.database).set_admin(args.username, enabled=not args.revoke)
    except ValueError as exc:
        parser.error(str(exc))
    print("Administrator permission revoked." if args.revoke else "Administrator permission granted.")


if __name__ == "__main__":
    main()
