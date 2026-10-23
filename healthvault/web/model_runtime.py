"""Web entry point to the policy-bound HealthVault Harness model runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agents.ui import quiet_agent_ui
from healthvault.agent_runtime import build_healthvault_model_only_runtime
from healthvault.model_config import HealthVaultModelConfig, OPENAI_COMPATIBLE
from healthvault.runtime_context import RuntimeContext


MODEL_ENV_NAMES = {
    "HEALTHVAULT_OPENAI_API_KEY", "HEALTHVAULT_OPENAI_BASE_URL", "HEALTHVAULT_OPENAI_MODEL",
    "HEALTHVAULT_OPENAI_REASONING_EFFORT",
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
}


class ModelServiceError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


def _model_setting(name: str) -> str:
    """Read the existing local config without changing process environment."""
    values: dict[str, str] = {}
    path = Path(__file__).resolve().parents[2] / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            key = key.strip()
            if separator and key in MODEL_ENV_NAMES:
                values[key] = value.strip().strip('"').strip("'")
    value = (os.getenv(name) or values.get(name) or "").strip()
    if value.startswith("${") and value.endswith("}"):
        reference = value[2:-1]
        if reference in MODEL_ENV_NAMES:
            return (os.getenv(reference) or values.get(reference) or "").strip()
    return value


def configured_web_model() -> HealthVaultModelConfig:
    key = (_model_setting("HEALTHVAULT_OPENAI_API_KEY") or _model_setting("OPENAI_API_KEY")).strip()
    base = (_model_setting("HEALTHVAULT_OPENAI_BASE_URL") or _model_setting("OPENAI_BASE_URL")).strip().rstrip("/")
    model = (_model_setting("HEALTHVAULT_OPENAI_MODEL") or _model_setting("OPENAI_MODEL")).strip()
    reasoning_effort = _model_setting("HEALTHVAULT_OPENAI_REASONING_EFFORT") or None
    parsed = urlparse(base)
    if not key or not model or parsed.scheme != "https" or not parsed.hostname:
        raise ModelServiceError(503, "模型服务未正确配置，请检查本地 .env。")
    try:
        return HealthVaultModelConfig(
            provider=OPENAI_COMPATIBLE, model=model, api_base=base, api_key=key,
            reasoning_effort=reasoning_effort,
        )
    except ValueError as exc:
        raise ModelServiceError(503, "模型推理强度配置无效，请检查本地 .env。") from exc


def run_model_task(
    system_prompt: str,
    payload: dict[str, Any],
    *,
    profile_id: str,
    failure_message: str,
) -> str:
    """Run a single no-tool Agent turn under an authenticated profile context."""
    config = configured_web_model()
    context = RuntimeContext(profile_id, uuid.uuid4().hex)
    runtime = build_healthvault_model_only_runtime(
        context=context, model_config=config, system_prompt=system_prompt,
    )
    try:
        with quiet_agent_ui():
            result = asyncio.run(runtime.run(json.dumps(payload, ensure_ascii=False)))
        if not result.text:
            raise ValueError("empty model output")
        logging.getLogger(__name__).info(
            "HealthVault Harness model request %s completed (input=%d, output=%d)",
            context.request_id, result.tokens["input"], result.tokens["output"],
        )
        return result.text
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "HealthVault Harness model request %s failed: %s",
            context.request_id, type(exc).__name__,
        )
        raise ModelServiceError(502, failure_message) from exc
