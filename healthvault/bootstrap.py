"""Safe, model-free assembly of HealthVault application components."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from healthvault.adapters.bearcode.tools import BearCodeToolAdapter
from healthvault.infrastructure.local_health_knowledge import (
    LocalHealthKnowledgeRepository,
)
from healthvault.infrastructure.hybrid_health_knowledge import (
    HybridHealthKnowledgeRepository,
)
from healthvault.infrastructure.sqlite_health_records import (
    ReadOnlySQLiteHealthRecordRepository,
    SQLiteHealthRecordRepository,
)
from healthvault.prompt import build_healthvault_system_prompt
from healthvault.repositories.health_knowledge import (
    HealthKnowledgeQueryError,
    HealthKnowledgeRepository,
)
from healthvault.runtime_context import RuntimeContext
from healthvault.tools.catalog import build_healthvault_tool_registry
from healthvault.tools.directory import HARNESS_READ_ONLY_TOOL_NAMES
from healthvault.tools.registry import ToolRegistry
from healthvault.rag.releases import active_corpus_root


BLOCKED_BY = (
    "runtime_context",
    "approved_preview_boundary",
    "skill_isolation",
)

READ_ONLY_TOOL_NAMES = HARNESS_READ_ONLY_TOOL_NAMES


class HealthVaultBootstrapError(RuntimeError):
    """Raised when safe offline HealthVault assembly cannot complete."""


@dataclass(frozen=True)
class HealthVaultBootstrapConfig:
    mode: str
    database_path: str | Path
    knowledge_path: str | Path | None = None
    knowledge_mode: str = "lexical"
    hybrid_index_path: str | Path | None = None
    hybrid_model_path: str | Path | None = None
    database_mode: str = "explicit"
    runtime_context: RuntimeContext | None = None


@dataclass(frozen=True)
class HealthVaultComponents:
    record_repository: SQLiteHealthRecordRepository
    knowledge_repository: HealthKnowledgeRepository
    registry: ToolRegistry
    adapter: BearCodeToolAdapter
    system_prompt: str
    manifest: dict[str, Any]


def build_healthvault_components(
    config: HealthVaultBootstrapConfig,
) -> HealthVaultComponents:
    """Build isolated repositories, tools, prompt, adapter, and manifest."""
    if config.mode not in {"read_only", "development"}:
        raise HealthVaultBootstrapError(
            "HealthVault mode must be read_only or development."
        )
    if config.database_mode not in {"ephemeral", "explicit"}:
        raise HealthVaultBootstrapError(
            "HealthVault database_mode must be ephemeral or explicit."
        )

    knowledge_path = (
        Path(config.knowledge_path)
        if config.knowledge_path is not None
        else active_corpus_root() / "chunks" / "hybrid_v1.json"
    )
    if config.knowledge_mode not in {"lexical", "hybrid"}:
        raise HealthVaultBootstrapError("Health knowledge mode must be lexical or hybrid.")
    try:
        lexical_repository = LocalHealthKnowledgeRepository(knowledge_path)
    except HealthKnowledgeQueryError as exc:
        raise HealthVaultBootstrapError(str(exc)) from exc
    knowledge_repository: HealthKnowledgeRepository
    if config.knowledge_mode == "hybrid":
        knowledge_repository = HybridHealthKnowledgeRepository.from_local_artifacts(
            lexical_repository,
            artifact_path=knowledge_path,
            index_directory=config.hybrid_index_path,
            model_path=config.hybrid_model_path,
        )
    else:
        knowledge_repository = lexical_repository
    try:
        database_path = Path(config.database_path)
        if config.mode == "read_only":
            # Database creation/migration belongs to application bootstrap.  The
            # Repository handed to the Agent is always a SQLite mode=ro handle.
            if not database_path.is_file():
                SQLiteHealthRecordRepository(database_path)
            record_repository = ReadOnlySQLiteHealthRecordRepository(database_path)
        else:
            record_repository = SQLiteHealthRecordRepository(database_path)
    except (OSError, sqlite3.Error) as exc:
        raise HealthVaultBootstrapError(
            "Health record database could not be initialized."
        ) from exc

    registry = build_healthvault_tool_registry(
        record_repository=record_repository,
        knowledge_repository=knowledge_repository,
    )
    allowed_tools = (
        None if config.mode == "development" else READ_ONLY_TOOL_NAMES
    )
    adapter = BearCodeToolAdapter(
        registry,
        allowed_domain_tools=allowed_tools,
        runtime_context=config.runtime_context,
    )
    system_prompt = build_healthvault_system_prompt()
    tool_names = [
        definition["name"] for definition in adapter.combined_definitions
    ]
    confirmation_enabled = config.mode == "development"
    knowledge_metadata = knowledge_repository.index_metadata
    manifest = {
        "status": "ok",
        "mode": config.mode,
        "tool_names": tool_names,
        "native_tools_enabled": False,
        "confirmation_tool_enabled": confirmation_enabled,
        "confirmation_boundary": (
            "stage1_legacy_unsafe" if confirmation_enabled else "disabled"
        ),
        "live_chat_ready": False,
        "agent_constructed": False,
        "api_config_loaded": False,
        "blocked_by": list(BLOCKED_BY),
        "prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "knowledge_corpus": knowledge_metadata["corpus_version"],
        "knowledge_strategy": knowledge_metadata["strategy"],
        "knowledge_engine": knowledge_metadata["engine_id"],
        "knowledge_mode": knowledge_metadata.get("retrieval_mode", "lexical"),
        "dense_fallback_reason": knowledge_metadata.get("dense_fallback_reason"),
        "database_mode": config.database_mode,
    }
    return HealthVaultComponents(
        record_repository=record_repository,
        knowledge_repository=knowledge_repository,
        registry=registry,
        adapter=adapter,
        system_prompt=system_prompt,
        manifest=manifest,
    )
