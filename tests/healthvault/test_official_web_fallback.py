import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from healthvault.agent_runtime import build_healthvault_model_only_runtime
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext
from healthvault.tools.search_official_health_web import (
    _cached_xml,
    _fetch_official,
    _parse_result,
    search_official_health_web,
)
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app
from healthvault.web.answers import INSUFFICIENT_EVIDENCE_MARKER


OFFICIAL_XML = b"""<nlmSearchResult><list>
<document url="https://medlineplus.gov/thyroidnodules.html">
<content name="title">Thyroid Nodules</content>
<content name="FullSummary"><p>Thyroid nodules are growths in the thyroid gland.</p></content>
</document>
<document url="https://evil.example/forged">
<content name="title">Forged</content><content name="FullSummary">Bad source</content>
</document>
</list></nlmSearchResult>"""


def local_result(status):
    return SimpleNamespace(to_dict=lambda: {
        "status": status,
        "results": [{"citation_id": "HVK:local"}] if status == "ok" else [],
    })


class OfficialWebFallbackTests(unittest.TestCase):
    def tearDown(self):
        _cached_xml.cache_clear()

    def test_local_evidence_blocks_web_even_when_web_tool_is_called(self):
        with patch("healthvault.tools.search_official_health_web.search_health_knowledge", return_value=local_result("ok")), patch(
            "healthvault.tools.search_official_health_web._fetch_official"
        ) as fetch:
            result = search_official_health_web({"question": "甲状腺结节是什么？"}, object())
        self.assertEqual("local_evidence_sufficient", result["status"])
        fetch.assert_not_called()

    def test_explicitly_insufficient_local_answer_allows_official_fallback(self):
        with patch("healthvault.tools.search_official_health_web.search_health_knowledge", return_value=local_result("ok")), patch(
            "healthvault.tools.search_official_health_web._fetch_official", return_value=OFFICIAL_XML
        ) as fetch:
            result = search_official_health_web({
                "question": "甲状腺结节是什么？", "local_answer_insufficient": True,
            }, object())
        self.assertEqual("ok", result["status"])
        fetch.assert_called_once_with("thyroid nodules")

    def test_fallback_sends_only_generic_topic_and_filters_unapproved_source(self):
        with patch("healthvault.tools.search_official_health_web.search_health_knowledge", return_value=local_result("empty")), patch(
            "healthvault.tools.search_official_health_web._fetch_official", return_value=OFFICIAL_XML
        ) as fetch:
            result = search_official_health_web({"question": "张三 13800138000 的甲状腺结节是什么？"}, object())
        self.assertEqual("ok", result["status"])
        self.assertEqual(1, len(result["results"]))
        self.assertEqual("https://medlineplus.gov/thyroidnodules.html", result["results"][0]["source"]["url"])
        self.assertEqual([result["results"][0]["citation_id"]], result["available_citation_ids"])
        self.assertEqual("thyroid nodules", fetch.call_args.args[0])
        self.assertNotIn("张三", fetch.call_args.args[0])
        self.assertNotIn("13800138000", fetch.call_args.args[0])

    def test_current_guideline_and_unsafe_intent_never_use_web(self):
        with patch("healthvault.tools.search_official_health_web._fetch_official") as fetch:
            latest = search_official_health_web({"question": "甲状腺结节最新指南是什么？"}, object())
            unsafe = search_official_health_web({"question": "甲状腺结节该吃什么药？"}, object())
        self.assertEqual("unsupported_current_guideline", latest["status"])
        self.assertEqual("unsupported_intent", unsafe["status"])
        fetch.assert_not_called()

    def test_xml_parser_rejects_entities_and_unapproved_urls(self):
        with self.assertRaises(ValueError):
            _parse_result(b'<!DOCTYPE x [<!ENTITY e "bad">]><nlmSearchResult/>', 3)
        self.assertEqual(1, len(_parse_result(OFFICIAL_XML, 3)))
        highlighted = OFFICIAL_XML.replace(b"Thyroid Nodules", b"&lt;span class='qt0'&gt;Thyroid&lt;/span&gt; Nodules")
        self.assertEqual("Thyroid Nodules", _parse_result(highlighted, 3, "thyroid nodules")[0]["title"])

    def test_official_fetch_rejects_private_dns_before_http(self):
        with patch("healthvault.tools.search_official_health_web.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("127.0.0.1", 443)),
        ]), patch("healthvault.tools.search_official_health_web.httpx.Client") as client:
            with self.assertRaises(ValueError):
                _fetch_official("thyroid nodules")
        client.assert_not_called()

    def test_web_question_uses_official_citation_after_local_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "records.sqlite3")
            captured = []

            def model(_question, _mode, evidence):
                captured.append(evidence)
                return f"结节是甲状腺中的生长物。[{evidence['sources'][0]['citation_id']}]"

            app = create_app(controller, model_answerer=model)
            with TestClient(app) as client, patch(
                "healthvault.tools.search_official_health_web._fetch_official", return_value=OFFICIAL_XML
            ):
                user = app.state.auth_store.register("synthetic-official", "synthetic-password-123", user_id="demo-official")
                token, csrf = app.state.auth_store.create_session(user)
                client.cookies.set("hv_session", token)
                client.headers.update({"X-CSRF-Token": csrf})
                response = client.post("/api/answers/knowledge", json={
                    "question": "甲状腺结节是什么？", "model_consent": True, "allow_web_search": True,
                })
                current = client.post("/api/answers/knowledge", json={
                    "question": "甲状腺结节最新指南是什么？", "model_consent": True, "allow_web_search": True,
                })
            self.assertEqual(200, response.status_code, response.text)
            self.assertEqual("answered", response.json()["status"])
            self.assertEqual("official_web", response.json()["sources"][0]["kind"])
            self.assertEqual("MedlinePlus / U.S. National Library of Medicine", captured[0]["sources"][0]["publisher"])
            self.assertEqual("insufficient_evidence", current.json()["status"])
            self.assertEqual(1, len(captured))

    def test_web_retries_official_when_local_answer_is_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "records.sqlite3")
            calls = []

            def model(_question, _mode, evidence):
                source = evidence["sources"][0]
                calls.append(source["kind"])
                if source["kind"] == "knowledge":
                    return INSUFFICIENT_EVIDENCE_MARKER
                return f"结节是组织中的生长物。[{source['citation_id']}]"

            app = create_app(controller, model_answerer=model)
            local = {"status": "ok", "results": [{
                "citation_id": "HVK:synthetic", "title": "合成知识", "excerpt": "甲状腺结节概述。",
                "source": {"url": ""},
            }]}
            with TestClient(app) as client, patch(
                "healthvault.web.api.retrieve_question_knowledge", return_value=local
            ), patch("healthvault.tools.search_official_health_web._fetch_official", return_value=OFFICIAL_XML):
                user = app.state.auth_store.register("synthetic-retry", "synthetic-password-123", user_id="demo-retry")
                token, csrf = app.state.auth_store.create_session(user)
                client.cookies.set("hv_session", token)
                client.headers.update({"X-CSRF-Token": csrf})
                offline = client.post("/api/answers/knowledge", json={
                    "question": "甲状腺结节是什么？", "model_consent": True,
                })
                response = client.post("/api/answers/knowledge", json={
                    "question": "甲状腺结节是什么？", "model_consent": True, "allow_web_search": True,
                })
            self.assertEqual("insufficient_evidence", offline.json()["status"])
            self.assertEqual(200, response.status_code, response.text)
            self.assertEqual("answered", response.json()["status"])
            self.assertEqual("official_web", response.json()["sources"][0]["kind"])
            self.assertEqual(["knowledge", "knowledge", "official_web"], calls)

    def test_web_fallback_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "records.sqlite3")
            app = create_app(controller, model_answerer=lambda *_: "unused")
            with TestClient(app) as client, patch(
                "healthvault.tools.search_official_health_web._fetch_official"
            ) as fetch:
                user = app.state.auth_store.register("synthetic-offline", "synthetic-password-123", user_id="demo-offline")
                token, csrf = app.state.auth_store.create_session(user)
                client.cookies.set("hv_session", token)
                client.headers.update({"X-CSRF-Token": csrf})
                response = client.post("/api/answers/knowledge", json={
                    "question": "甲状腺结节是什么？", "model_consent": True,
                })
            self.assertEqual(200, response.status_code)
            self.assertEqual("insufficient_evidence", response.json()["status"])
            fetch.assert_not_called()


class ModelOnlyRuntimeTests(unittest.TestCase):
    def test_model_only_agent_advertises_no_tools(self):
        async def scenario():
            with patch("agents.agent.openai.AsyncOpenAI", return_value=MagicMock()):
                runtime = build_healthvault_model_only_runtime(
                    context=RuntimeContext("demo-a", "request-model-only"),
                    model_config=HealthVaultModelConfig(
                        provider="openai-compatible", model="synthetic-model",
                        api_base="https://example.org/v1", api_key="synthetic-key",
                    ),
                    system_prompt="Extract candidates only.",
                )
            self.assertEqual([], runtime.agent.tools)
            self.assertEqual(frozenset(), runtime.agent.runtime_policy.allowed_tool_names)

            class EmptyStream:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    raise StopAsyncIteration

            create = AsyncMock(return_value=EmptyStream())
            runtime.agent._openai_client.chat.completions.create = create
            await runtime.agent._call_openai_stream()
            self.assertNotIn("tools", create.call_args.kwargs)

            class AnswerStream:
                async def __aiter__(self):
                    yield SimpleNamespace(
                        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5),
                        choices=[SimpleNamespace(
                            delta=SimpleNamespace(content="candidate JSON", tool_calls=None),
                            finish_reason="stop",
                        )],
                    )

            runtime.agent._openai_client.chat.completions.create = AsyncMock(return_value=AnswerStream())
            result = await runtime.run("synthetic report text")
            self.assertEqual("candidate JSON", result.text)
            self.assertEqual({"input": 12, "output": 5}, result.tokens)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
