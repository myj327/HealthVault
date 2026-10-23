"""Bounded official-source fetching and persistent change observation."""
from __future__ import annotations

import hashlib
import json
import threading
import re
import difflib
import ipaddress
import socket
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

from healthvault.rag.releases import OFFICIAL_PUBLISHERS, _write_json, _snapshots


def fetch_official(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    import httpx
    # Every redirect is separately checked; environment proxies are disabled.
    with httpx.Client(timeout=15, follow_redirects=False, trust_env=False) as client:
        for _ in range(4):
            parsed = urlparse(url)
            if (parsed.scheme != 'https' or parsed.hostname not in OFFICIAL_PUBLISHERS
                    or parsed.username or parsed.password or parsed.port not in {None,443}):
                raise ValueError('Official source URL is not allowed.')
            addresses = socket.getaddrinfo(parsed.hostname,443,type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses):
                raise ValueError('Official source resolved to a non-public address.')
            with client.stream('GET', url, headers=headers or {}) as response:
                if response.status_code in {301,302,303,307,308}:
                    url = urljoin(url,response.headers.get('location','')); continue
                if response.status_code == 304:
                    return {'status':304, 'url':url}
                response.raise_for_status(); body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > 8_000_000:
                        raise ValueError('Official source document exceeds 8 MB.')
                return {'status':200, 'url':url, 'body':bytes(body), 'etag':response.headers.get('etag'),
                        'last_modified':response.headers.get('last-modified'), 'content_type':response.headers.get('content-type','')}
    raise ValueError('Too many official source redirects.')


class _OfficialTextParser(HTMLParser):
    """Extract review text without importing optional ingestion dependencies."""
    ignored = {'script','style','nav','header','footer','form','noscript'}
    boundaries = {'p','li','tr','td','th','br','div','section','main','article','h1','h2','h3','h4','h5','h6'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.suppressed = 0
        self.main_depth = 0
        self.all_text = []
        self.main_text = []

    def handle_starttag(self, tag, attrs):
        if tag in self.ignored:
            self.suppressed += 1
        if not self.suppressed:
            if tag in {'main','article'}:
                self.main_depth += 1
            if tag in self.boundaries:
                self.handle_data('\n')

    def handle_endtag(self, tag):
        if tag in self.ignored:
            self.suppressed = max(0,self.suppressed-1)
        elif not self.suppressed:
            if tag in self.boundaries:
                self.handle_data('\n')
            if tag in {'main','article'}:
                self.main_depth = max(0,self.main_depth-1)

    def handle_data(self, text):
        if not self.suppressed:
            self.all_text.append(text)
            if self.main_depth:
                self.main_text.append(text)

    def text(self):
        content = ''.join(self.main_text).strip() or ''.join(self.all_text)
        return '\n'.join(line.strip() for line in re.sub(r'[ \t]+',' ',content).splitlines() if line.strip())


def official_text(snapshot: dict[str, Any]) -> str:
    body = snapshot['body']
    if 'pdf' in snapshot.get('content_type','') or body.startswith(b'%PDF-'):
        import pymupdf
        with pymupdf.open(stream=body,filetype='pdf') as document:
            return '\n'.join(page.get_text() for page in document)
    parser = _OfficialTextParser()
    parser.feed(body.decode('utf-8',errors='replace'))
    return parser.text()


class OfficialSourceWatcher:
    def __init__(self, releases: Any, fetcher: Callable = fetch_official):
        self.releases = releases
        self.root = releases.root / 'source-watch'
        self.fetcher = fetcher
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def list(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding='utf-8')) for path in sorted(self.root.glob('*.json')) if path.name != 'last-run.json']

    def pending(self, identifier: str) -> dict[str, Any]:
        if not re.fullmatch(r'[0-9a-f]{20}', identifier):
            raise ValueError('Invalid source observation ID.')
        record = json.loads((self.root / f'{identifier}.json').read_text(encoding='utf-8'))
        directory = (self.root / record['pending_snapshot']).resolve()
        if not directory.is_relative_to((self.root / 'pending').resolve()):
            raise ValueError('Invalid pending source path.')
        source = _snapshots(self.releases.active_root)[record['source_id']]
        text = (directory / 'text.txt').read_text(encoding='utf-8')
        before = '\n'.join(block.get('text','') for section in source['sections'] for block in section['blocks'])
        return {**record, 'text':text, 'unified_diff':''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='active',tofile='official-pending'))}

    def check(self) -> dict[str, Any]:
        with self._lock:
            changed = 0; errors = []
            for source in _snapshots(self.releases.active_root).values():
                identifier = hashlib.sha256(source['source_id'].encode()).hexdigest()[:20]
                path = self.root / f'{identifier}.json'
                previous = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
                headers = {}
                if previous.get('etag'): headers['If-None-Match'] = previous['etag']
                if previous.get('last_modified'): headers['If-Modified-Since'] = previous['last_modified']
                now = datetime.now(timezone.utc).isoformat(timespec='seconds')
                try:
                    fetched = self.fetcher(source['url'],headers)
                    if fetched['status'] == 304:
                        record = {**previous,'checked_at':now,'fetch_status':'not_modified'}
                    else:
                        body = fetched['body']; raw_hash = hashlib.sha256(body).hexdigest()
                        text = official_text(fetched); text_hash = hashlib.sha256(text.encode()).hexdigest()
                        if not text.strip():
                            raise ValueError('Official page has no extractable document text.')
                        content_changed = bool(previous.get('text_sha256') and previous['text_sha256'] != text_hash)
                        first_observation = not previous.get('text_sha256')
                        status = 'needs_review' if content_changed or first_observation else previous.get('status','needs_review')
                        record = {'observation_id':identifier, 'source_id':source['source_id'], 'publisher':source['publisher'], 'url':fetched['url'],
                                  'checked_at':now,'status':status,'fetch_status':'fetched','raw_sha256':raw_hash,'text_sha256':text_hash,
                                  'etag':fetched.get('etag'),'last_modified':fetched.get('last_modified'),
                                  'active_source_sha256':source['source_sha256'], 'active_corpus_version':json.loads((self.releases.active_root/'manifest.json').read_text(encoding='utf-8'))['corpus_version']}
                        if content_changed or first_observation:
                            directory = self.root / 'pending' / identifier / raw_hash
                            directory.mkdir(parents=True,exist_ok=True)
                            (directory / 'original.bin').write_bytes(body)
                            (directory / 'text.txt').write_text(text,encoding='utf-8')
                            record['pending_snapshot'] = str(directory.relative_to(self.root))
                            record['reason'] = 'source_content_changed' if content_changed else 'initial_snapshot_needs_comparison'
                            changed += 1
                        elif previous.get('pending_snapshot'):
                            record['pending_snapshot'] = previous['pending_snapshot']
                    _write_json(path,record)
                except Exception as exc:
                    # One unavailable site must not stop checking the other sources.
                    errors.append({'source_id':source['source_id'],'error':type(exc).__name__})
            result = {'checked_at':datetime.now(timezone.utc).isoformat(timespec='seconds'), 'changed':changed,'errors':errors}
            _write_json(self.root / 'last-run.json',result)
            return result

    def start(self, interval_seconds: int = 86400) -> None:
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        def loop():
            while not self._stop.wait(max(60,interval_seconds)):
                self.check()
        self._thread = threading.Thread(target=loop,name='healthvault-source-watch',daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(timeout=1)
