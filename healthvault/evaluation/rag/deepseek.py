"""Minimal DeepSeek OpenAI-compatible Chat Completions client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from healthvault.evaluation.rag.generation import ModelConfig, ModelResponse


DEEPSEEK_PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing/"
DEEPSEEK_GENERATION_CONFIG = ModelConfig(
    provider="deepseek",
    model="deepseek-v4-flash",
    temperature=0.0,
    max_output_tokens=256,
    api_format="openai-chat-completions",
    endpoint_path="/chat/completions",
    thinking_type="disabled",
    response_format="json_object",
    input_cost_per_million=0.14,
    output_cost_per_million=0.28,
)
DEEPSEEK_JUDGE_CONFIG = ModelConfig(
    provider="deepseek",
    model="deepseek-v4-pro",
    temperature=0.0,
    max_output_tokens=128,
    api_format="openai-chat-completions",
    endpoint_path="/chat/completions",
    thinking_type="disabled",
    response_format="json_object",
    input_cost_per_million=0.435,
    output_cost_per_million=0.87,
)


class JsonTransport(Protocol):
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], int]:
        ...


class UrllibJsonTransport:
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], int]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ConnectionError("DeepSeek Chat Completions request failed") from exc
        latency_ms = round((time.monotonic() - started) * 1000)
        try:
            return json.loads(body.decode("utf-8")), latency_ms
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("DeepSeek returned invalid JSON") from exc


class DeepSeekChatCompletionsClient:
    """Send fixed non-thinking JSON requests without exposing credentials in results."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        transport: JsonTransport | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is not configured")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport or UrllibJsonTransport()
        self._timeout_seconds = timeout_seconds

    def complete(
        self,
        prompt: str,
        *,
        call_type: str,
        request_id: str,
        config: ModelConfig,
    ) -> ModelResponse:
        del call_type, request_id
        if (
            config.provider != "deepseek"
            or config.api_format != "openai-chat-completions"
            or config.endpoint_path != "/chat/completions"
            or config.thinking_type != "disabled"
            or config.response_format != "json_object"
            or config.temperature != 0.0
        ):
            raise ValueError("DeepSeek evaluation config must be fixed non-thinking JSON")
        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
            "response_format": {"type": config.response_format},
            "thinking": {"type": config.thinking_type},
        }
        response, latency_ms = self._transport.post_json(
            self._base_url + config.endpoint_path,
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            payload,
            self._timeout_seconds,
        )
        try:
            text = response["choices"][0]["message"]["content"]
            input_tokens = int(response["usage"]["prompt_tokens"])
            output_tokens = int(response["usage"]["completion_tokens"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("DeepSeek response schema is invalid") from exc
        cost = (
            input_tokens * config.input_cost_per_million
            + output_tokens * config.output_cost_per_million
        ) / 1_000_000
        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=round(cost, 10),
            cost_source="configured_official_rate",
        )
