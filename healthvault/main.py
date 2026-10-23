"""Offline-only HealthVault assembly check command."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from healthvault.bootstrap import (
    HealthVaultBootstrapConfig,
    HealthVaultBootstrapError,
    build_healthvault_components,
)
from healthvault.agent_runtime import (
    HealthVaultAgentRuntimeConfig,
    HealthVaultAgentRuntimeError,
    build_healthvault_agent_runtime,
)
from healthvault.model_config import (
    SUPPORTED_PROVIDERS,
    HealthVaultModelConfigError,
    load_healthvault_model_config,
)
from healthvault.runtime_context import RuntimeContext


_TOOLS_OUTPUT_FIELDS = (
    "status",
    "mode",
    "tool_names",
    "native_tools_enabled",
    "confirmation_tool_enabled",
    "confirmation_boundary",
    "live_chat_ready",
    "agent_constructed",
    "api_config_loaded",
    "blocked_by",
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=("read_only", "development"),
        default="read_only",
    )
    parser.add_argument("--database")
    parser.add_argument("--knowledge")
    parser.add_argument("--knowledge-mode", choices=("lexical", "hybrid"), default="lexical")
    parser.add_argument("--hybrid-index")
    parser.add_argument("--hybrid-model-path")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="healthvault",
        description="HealthVault offline self-check and bounded read-only chat.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser(
        "check",
        help="Print the safe offline assembly manifest.",
    )
    tools_parser = subparsers.add_parser(
        "tools",
        help="List exposed domain tools and safety state.",
    )
    chat_parser = subparsers.add_parser(
        "chat",
        help="Run one bounded read-only HealthVault Agent request.",
    )
    _add_common_arguments(check_parser)
    _add_common_arguments(tools_parser)
    chat_parser.add_argument(
        "--mode",
        choices=("read_only", "development"),
        default="read_only",
    )
    chat_parser.add_argument("--database", required=True)
    chat_parser.add_argument("--profile-id", required=True)
    chat_parser.add_argument("--request-id", required=True)
    chat_parser.add_argument("--task-id", help="Stable task ID for cross-session workflow memory.")
    chat_parser.add_argument("--task-memory", help="Checkpoint SQLite path; defaults under ~/.healthvault/task-memory when --task-id is set.")
    chat_parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        required=True,
    )
    chat_parser.add_argument("--model", required=True)
    chat_parser.add_argument("--api-base", required=True)
    chat_parser.add_argument("--prompt", required=True)
    chat_parser.add_argument("--knowledge")
    chat_parser.add_argument("--knowledge-mode", choices=("lexical", "hybrid"), default="lexical")
    chat_parser.add_argument("--hybrid-index")
    chat_parser.add_argument("--hybrid-model-path")
    chat_parser.add_argument("--skill-root")
    chat_parser.add_argument("--skill", action="append", default=[])
    chat_parser.add_argument(
        "--mcp-config",
        help="Explicit HealthVault read-only MCP config; no global MCP discovery.",
    )
    return parser


def _build_manifest(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.database is not None:
        components = build_healthvault_components(
            HealthVaultBootstrapConfig(
                mode=arguments.mode,
                database_path=Path(arguments.database),
                knowledge_path=arguments.knowledge,
                knowledge_mode=arguments.knowledge_mode,
                hybrid_index_path=arguments.hybrid_index,
                hybrid_model_path=arguments.hybrid_model_path,
                database_mode="explicit",
            )
        )
        return components.manifest

    with tempfile.TemporaryDirectory() as temp_dir:
        components = build_healthvault_components(
            HealthVaultBootstrapConfig(
                mode=arguments.mode,
                database_path=Path(temp_dir) / "healthvault.sqlite3",
                knowledge_path=arguments.knowledge,
                knowledge_mode=arguments.knowledge_mode,
                hybrid_index_path=arguments.hybrid_index,
                hybrid_model_path=arguments.hybrid_model_path,
                database_mode="ephemeral",
            )
        )
        return components.manifest


def _default_task_memory_path(database_path: Path) -> Path:
    database_key = hashlib.sha256(str(database_path.resolve()).encode("utf-8")).hexdigest()[:20]
    return Path.home() / ".healthvault" / "task-memory" / f"{database_key}.sqlite3"


def _run_chat(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.mode != "read_only":
        raise HealthVaultAgentRuntimeError(
            "HealthVault chat only supports read_only mode."
        )
    database_path = Path(arguments.database)
    if not database_path.is_file():
        raise HealthVaultAgentRuntimeError(
            "HealthVault chat database must already exist."
        )
    if arguments.task_memory and not arguments.task_id:
        raise HealthVaultAgentRuntimeError("--task-memory requires --task-id.")
    task_memory_path = (
        arguments.task_memory or _default_task_memory_path(database_path)
        if arguments.task_id else None
    )
    context = RuntimeContext(
        profile_id=arguments.profile_id,
        request_id=arguments.request_id,
    )
    model_config = load_healthvault_model_config(
        provider=arguments.provider,
        model=arguments.model,
        api_base=arguments.api_base,
    )
    runtime = build_healthvault_agent_runtime(
        HealthVaultAgentRuntimeConfig(
            context=context,
            database_path=database_path,
            knowledge_path=arguments.knowledge,
            knowledge_mode=arguments.knowledge_mode,
            hybrid_index_path=arguments.hybrid_index,
            hybrid_model_path=arguments.hybrid_model_path,
            skill_root=arguments.skill_root,
            skill_names=tuple(arguments.skill),
            model_config=model_config,
            task_id=arguments.task_id,
            task_memory_path=task_memory_path,
        )
    )
    async def run_chat():
        if not arguments.mcp_config:
            return await runtime.run(arguments.prompt)
        from healthvault.mcp.runtime_bridge import attach_approved_mcp

        async with await attach_approved_mcp(
            runtime, config_path=arguments.mcp_config
        ):
            return await runtime.run(arguments.prompt)

    with redirect_stdout(io.StringIO()):
        result = asyncio.run(run_chat())
    return result.to_dict()

def main(argv: list[str] | None = None) -> int:
    """Run an offline check or one explicitly configured read-only chat."""
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        if arguments.command == "chat":
            output = _run_chat(arguments)
        else:
            manifest = _build_manifest(arguments)
            output = (
                manifest
                if arguments.command == "check"
                else {key: manifest[key] for key in _TOOLS_OUTPUT_FIELDS}
            )
    except (
        HealthVaultAgentRuntimeError,
        HealthVaultBootstrapError,
        HealthVaultModelConfigError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Error: HealthVault chat failed.", file=sys.stderr)
        return 1

    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
