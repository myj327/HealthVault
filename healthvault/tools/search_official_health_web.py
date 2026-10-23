"""Read-only fallback search against the fixed MedlinePlus/NLM endpoint."""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import httpx

from healthvault.application.knowledge_support_gate import assess_health_knowledge_support
from healthvault.domain.reports import INDICATOR_ALIASES, INDICATOR_CATALOG
from healthvault.repositories.health_knowledge import HealthKnowledgeRepository
from healthvault.application.search_health_knowledge import search_health_knowledge
from healthvault.tools.directory import tool_definition


SEARCH_OFFICIAL_HEALTH_WEB_TOOL = tool_definition("search_official_health_web")

_ENDPOINT = "https://wsearch.nlm.nih.gov/ws/query"
_ALLOWED_RESULT_HOST = "medlineplus.gov"
_MAX_XML_BYTES = 256_000
_TOPIC_HINTS = (
    ("甲状腺结节", "thyroid nodules"),
    ("thyroid nodule", "thyroid nodules"),
    ("子宫肌瘤", "uterine fibroids"),
    ("uterine fibroid", "uterine fibroids"),
    ("乳腺结节", "breast diseases"),
    ("肺结节", "lung diseases"),
    ("血常规", "complete blood count"),
    ("肝功能", "liver function tests"),
    ("肾功能", "kidney function tests"),
    ("血脂", "cholesterol"),
    ("血糖", "blood glucose"),
    ("体检", "health screening"),
    ("结节", "nodules"),
    ("肌瘤", "fibroids"),
)
_CURRENT_GUIDELINE = re.compile(r"最新|现行|目前.*指南|20\d{2}.*指南|\b(?:latest|current)\b.*\bguideline\b|\bguidelines?\b", re.I)


def _topics(question: str) -> tuple[str, ...]:
    """Emit only catalogued generic topics; never send the user's question."""
    lowered = question.casefold()
    topics: list[str] = []
    matched_spans: list[str] = []
    for phrase, topic in _TOPIC_HINTS:
        if phrase.casefold() in lowered and not any(phrase in prior for prior in matched_spans):
            topics.append(topic)
            matched_spans.append(phrase)
    for code, spec in INDICATOR_CATALOG.items():
        aliases = (spec.display_name, code, *(alias for alias, target in INDICATOR_ALIASES.items() if target == code))
        if any(len(name) >= 3 and name.casefold() in lowered for name in aliases):
            topics.append(code.replace("_", " "))
    return tuple(dict.fromkeys(topics))[:2]


def _approved_result_url(raw: str) -> bool:
    parsed = urlparse(raw)
    try:
        return (
            parsed.scheme == "https"
            and parsed.hostname == _ALLOWED_RESULT_HOST
            and parsed.port is None
            and not parsed.username
            and not parsed.password
        )
    except ValueError:
        return False


def _fetch_official(topic: str) -> bytes:
    """Fetch a bounded XML response from a fixed HTTPS host without redirects or proxies."""
    addresses = socket.getaddrinfo("wsearch.nlm.nih.gov", 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("official service resolved to a non-public address")
    with httpx.Client(follow_redirects=False, trust_env=False, timeout=8.0) as client:
        with client.stream(
            "GET", _ENDPOINT,
            params={"db": "healthTopics", "term": topic, "retmax": 3, "rettype": "brief", "tool": "healthvault"},
        ) as response:
            if response.status_code != 200:
                raise ValueError("official service did not return an OK response")
            if urlparse(str(response.url)).hostname != "wsearch.nlm.nih.gov":
                raise ValueError("official service host changed")
            if "xml" not in response.headers.get("content-type", "").casefold():
                raise ValueError("official service returned non-XML content")
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > _MAX_XML_BYTES:
                    raise ValueError("official response exceeded size limit")
            return bytes(body)


@lru_cache(maxsize=256)
def _cached_xml(topic: str, half_day: int) -> bytes:
    return _fetch_official(topic)


def _clean_content(element: ET.Element) -> str:
    value = html.unescape(" ".join(element.itertext()))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", value)).strip()


def _relevant_to_topic(topic: str, title: str, excerpt: str) -> bool:
    text = f"{title} {excerpt}".casefold()
    terms = [
        term[:-1] if term.endswith("s") and not term.endswith("ss") else term
        for term in re.findall(r"[a-z]+", topic.casefold())
        if term not in {"blood", "health", "test", "tests", "disease", "diseases"}
    ]
    return all(re.search(rf"\b{re.escape(term)}[a-z]*\b", text) for term in terms)


def _parse_result(xml_bytes: bytes, limit: int, topic: str | None = None) -> list[dict[str, Any]]:
    if b"<!DOCTYPE" in xml_bytes.upper() or b"<!ENTITY" in xml_bytes.upper():
        raise ValueError("XML declarations are not accepted")
    root = ET.fromstring(xml_bytes)
    if root.tag != "nlmSearchResult":
        raise ValueError("unexpected official response")
    retrieved_on = datetime.now(timezone.utc).date().isoformat()
    results: list[dict[str, Any]] = []
    for document in root.findall("./list/document"):
        url = document.get("url", "")
        if not _approved_result_url(url):
            continue
        fields = {
            element.get("name", "").casefold(): _clean_content(element)
            for element in document.findall("content")
        }
        title = fields.get("title", "").strip()
        excerpt = re.sub(r"\s+", " ", fields.get("fullsummary") or fields.get("snippet", ""))[:1200]
        if not title or not excerpt or (topic and not _relevant_to_topic(topic, title, excerpt)):
            continue
        digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        results.append({
            "citation_id": f"HVW:{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}",
            "title": title,
            "excerpt": excerpt,
            "source": {
                "publisher": "MedlinePlus / U.S. National Library of Medicine",
                "url": url,
                "retrieved_on": retrieved_on,
                "updated_on": None,
                "snapshot_sha256": digest,
            },
        })
        if len(results) >= limit:
            break
    return results


@lru_cache(maxsize=128)
def _cached_official_document(url: str, half_day: int) -> dict[str, Any]:
    from healthvault.rag.sources import fetch_official, official_text
    snapshot = fetch_official(url)
    return {'text':official_text(snapshot),'snapshot_sha256':hashlib.sha256(snapshot['body']).hexdigest(),'url':snapshot['url']}


def _curated_official_results(topics: tuple[str,...], limit: int, repository: Any) -> list[dict[str, Any]]:
    """Refresh related known official URLs; user text never enters a remote query."""
    from healthvault.rag.releases import OFFICIAL_PUBLISHERS, _snapshots
    lexical = getattr(repository,'_lexical',repository)
    artifact = getattr(lexical,'_artifact_path',None)
    if artifact is None: return []
    candidates = []
    for source in _snapshots(Path(artifact).parent.parent).values():
        if urlparse(source['url']).hostname == 'medlineplus.gov': continue
        content = '\n'.join(block['text'] for section in source['sections'] for block in section['blocks'])
        if any(_relevant_to_topic(topic,source['title'],content) for topic in topics):
            candidates.append(source)
    results = []
    for source in candidates[:2]:
        try:
            snapshot = _cached_official_document(source['url'],int(time.time()//43_200))
            paragraphs = snapshot['text'].splitlines()
            selected = [text for text in paragraphs if any(_relevant_to_topic(topic,source['title'],text) for topic in topics)]
            excerpt = '\n'.join(selected)[:1200]
            if not excerpt: continue
            results.append({'citation_id':f"HVW:{hashlib.sha256(source['url'].encode()).hexdigest()[:16]}",
                            'title':source['title'],'excerpt':excerpt,
                            'source':{'publisher':OFFICIAL_PUBLISHERS[urlparse(snapshot['url']).hostname],'url':snapshot['url'],
                                      'retrieved_on':datetime.now(timezone.utc).date().isoformat(),'updated_on':None,
                                      'snapshot_sha256':snapshot['snapshot_sha256']}})
            if len(results)>=limit: break
        except (OSError,httpx.HTTPError,ValueError):
            continue
    return results


def search_official_health_web(tool_input: Any, repository: HealthKnowledgeRepository) -> dict[str, Any]:
    """Enforce intent, local-first fallback, and de-identified official lookup."""
    if not isinstance(tool_input, dict):
        return {"status": "invalid_request", "results": []}
    question = tool_input.get("question")
    limit = tool_input.get("limit", 3)
    local_answer_insufficient = tool_input.get("local_answer_insufficient", False)
    if (not isinstance(question, str) or not 3 <= len(question.strip()) <= 1000
            or type(limit) is not int or not 1 <= limit <= 3
            or type(local_answer_insufficient) is not bool):
        return {"status": "invalid_request", "results": []}
    if _CURRENT_GUIDELINE.search(question):
        return {"status": "unsupported_current_guideline", "results": []}
    if not assess_health_knowledge_support(question).allowed:
        return {"status": "unsupported_intent", "results": []}
    local = search_health_knowledge({"query": question, "limit": 3}, repository).to_dict()
    if local["status"] == "ok" and local["results"] and not local_answer_insufficient:
        return {"status": "local_evidence_sufficient", "results": []}
    topics = _topics(question)
    if not topics:
        return {"status": "unsupported_topic", "results": []}
    results = _curated_official_results(topics,limit,repository)
    seen: set[str] = {item['citation_id'] for item in results}
    try:
        for topic in topics:
            if len(results) >= limit: break
            for item in _parse_result(_cached_xml(topic, int(time.time() // 43_200)), limit, topic):
                if item["citation_id"] not in seen:
                    results.append(item)
                    seen.add(item["citation_id"])
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
    except (OSError, httpx.HTTPError, ValueError, ET.ParseError):
        if not results:
            return {"status": "unavailable", "results": []}
    citation_ids = [item["citation_id"] for item in results]
    return {
        "status": "ok" if results else "empty",
        "results": results,
        "available_citation_ids": citation_ids,
        "recommended_citation_ids": citation_ids[:1],
    }


def build_search_official_health_web_handler(repository: HealthKnowledgeRepository):
    async def handler(tool_input: Any) -> str:
        return json.dumps(search_official_health_web(tool_input, repository), ensure_ascii=False)
    return handler
