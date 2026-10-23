"""Small offline corpus updates with explicit administrator publication."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT, load_chunk_artifact, render_corpus_artifacts


DEFAULT_RELEASE_ROOT = Path(__file__).parents[2] / ".bear" / "knowledge-releases"
OFFICIAL_PUBLISHERS = {
    "medlineplus.gov": "MedlinePlus", "www.niddk.nih.gov": "NIDDK",
    "www.nhlbi.nih.gov": "NHLBI", "odphp.health.gov": "ODPHP",
    "www.cdc.gov": "CDC", "www.who.int": "WHO", "www.nice.org.uk": "NICE",
}
_ID = re.compile(r"[0-9a-f]{20}\Z")
_VERSION = re.compile(r"[a-z0-9][a-z0-9.-]{2,79}\Z")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_json_bytes(value))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configured_release_root() -> Path:
    return Path(os.environ.get("HEALTHVAULT_KNOWLEDGE_RELEASE_ROOT", DEFAULT_RELEASE_ROOT))


def active_corpus_root(release_root: Path | None = None) -> Path:
    root = release_root or configured_release_root()
    pointer = root / "active.json"
    if not pointer.is_file():
        return DEFAULT_CORPUS_ROOT
    identifier = json.loads(pointer.read_text(encoding="utf-8")).get("active_candidate_id")
    if identifier is None:
        return DEFAULT_CORPUS_ROOT
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
        raise ValueError("Invalid active knowledge release.")
    return root / "candidates" / identifier / "corpus"


def _bundle_hash(root: Path) -> str:
    artifact_path = root / "chunks" / "hybrid_v1.json"
    artifact = load_chunk_artifact(artifact_path, expected_corpus_version=None)
    provenance = (root / "provenance.json").read_bytes() if (root / "provenance.json").is_file() else b""
    vectors = b''
    manifest_path = root / 'vectors' / 'manifest.json'
    if manifest_path.is_file():
        from healthvault.infrastructure.hybrid_health_knowledge import VectorIndexManifest, _sha256_file, _chunk_ids_sha256
        value = VectorIndexManifest.from_dict(json.loads(manifest_path.read_text(encoding='utf-8')))
        ids = tuple(json.loads((root / 'vectors' / 'chunk_ids.json').read_text(encoding='utf-8')))
        if (value.corpus_sha256 != _sha256_file(artifact_path) or value.corpus_version != artifact.corpus_version
                or tuple(chunk.chunk_id for chunk in artifact.chunks) != ids or value.chunk_ids_sha256 != _chunk_ids_sha256(ids)
                or not value.index_sha256 or value.index_sha256 != _sha256_file(root / 'vectors' / 'index.faiss')):
            raise ValueError('Candidate vector index does not match its corpus.')
        vectors = manifest_path.read_bytes()
    return hashlib.sha256((root / "manifest.json").read_bytes() + provenance + vectors).hexdigest()


def _snapshots(root: Path) -> dict[str, dict[str, Any]]:
    return {value["source_id"]: value for path in sorted((root / "sources").glob("*.json"))
            if (value := json.loads(path.read_text(encoding="utf-8")))}


def _comparable(source: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key != "corpus_version"}


def _provenance(source: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    if set(metadata) - {"document_kind", "published_on", "updated_on"}:
        raise ValueError("Unknown document metadata fields.")
    kind = metadata.get("document_kind", "general_health")
    if kind not in {"general_health", "formal_guideline"}:
        raise ValueError("Document kind must be general_health or formal_guideline.")
    for field in ("published_on", "updated_on"):
        if metadata.get(field) is not None:
            date.fromisoformat(metadata[field])
    if kind == "formal_guideline" and (
        not metadata.get("published_on")
        or not source.get("version")
        or str(source["version"]).startswith("retrieved-")
    ):
        raise ValueError("Formal guidelines require a publication date and source version.")
    if metadata.get("published_on") and metadata.get("updated_on") and metadata["updated_on"] < metadata["published_on"]:
        raise ValueError("Update date precedes publication date.")
    return {
        **{key: source[key] for key in ("source_id", "publisher", "title", "version", "url", "retrieved_on", "source_sha256", "corpus_version")},
        "normalized_content_sha256": hashlib.sha256(_json_bytes(source["sections"])).hexdigest(),
        "document_kind": kind, "published_on": metadata.get("published_on"), "updated_on": metadata.get("updated_on"),
    }


def _probe_corpus(root: Path, probes: list[dict[str, str]]) -> list[dict[str, Any]]:
    from healthvault.infrastructure.sqlite_fts_health_knowledge import SQLiteFTSHealthKnowledgeRepository

    repository = SQLiteFTSHealthKnowledgeRepository(root / "chunks" / "hybrid_v1.json",root/'aliases'/'aliases.json')
    try:
        results = []
        for probe in probes:
            hits = repository.search(probe["query"], 5)
            source_ids = [hit.document.source.source_id for hit in hits]
            results.append({**probe, "passed": probe["expected_source_id"] in source_ids,
                            "retrieved_source_ids": source_ids, "citation_ids": [hit.document.citation_id for hit in hits]})
        return results
    finally:
        repository.close()


class KnowledgeReleaseStore:
    def __init__(self, root: str | Path | None = None, *, tokenizer_path: Path | None = None) -> None:
        self.root = Path(root) if root is not None else configured_release_root()
        self.tokenizer_path = tokenizer_path
        self._lock = threading.RLock()

    @property
    def active_root(self) -> Path:
        return active_corpus_root(self.root)

    def _directory(self, candidate_id: str) -> Path:
        if not _ID.fullmatch(candidate_id):
            raise ValueError("Invalid knowledge candidate ID.")
        return self.root / "candidates" / candidate_id

    def get(self, candidate_id: str) -> dict[str, Any]:
        return json.loads((self._directory(candidate_id) / "review.json").read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        return sorted([self.get(path.name) for path in (self.root / "candidates").glob("*")
                       if path.is_dir() and _ID.fullmatch(path.name) and (path / "review.json").is_file()],
                      key=lambda item: item["created_at"], reverse=True)

    def active(self) -> dict[str, Any]:
        from healthvault.rag.retrieval_config import selection
        manifest = json.loads((self.active_root / "manifest.json").read_text(encoding="utf-8"))
        return {"corpus_version": manifest["corpus_version"], "corpus_sha256": manifest["corpus_sha256"],
                "source_count": manifest["source_count"], "release_sha256": _bundle_hash(self.active_root),
                "retrieval_mode": selection(self.active_root, self.root)['knowledge_mode']}

    def update_template(self, source_id: str) -> dict[str, Any]:
        source = _snapshots(self.active_root).get(source_id)
        if source is None: raise ValueError('Unknown knowledge source.')
        source['source_sha256'] = 'auto'
        metadata = {'document_kind':'general_health','published_on':None,'updated_on':None}
        if (self.active_root/'provenance.json').is_file():
            existing = next((item for item in json.loads((self.active_root/'provenance.json').read_text(encoding='utf-8')) if item['source_id']==source_id),None)
            if existing: metadata = {key:existing[key] for key in metadata}
        return {'corpus_version':'review-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'),
                'source_updates':[source], 'source_metadata':{source_id:metadata},
                'probes':[{'query':source['title'],'expected_source_id':source_id}]}

    def build_vectors(self, candidate_id: str, actor: str) -> dict[str, Any]:
        with self._lock:
            record = self.get(candidate_id)
            if record['status'] not in {'pending_review','watch'}:
                raise ValueError('Only unpublished candidates can build an index.')
            if _bundle_hash(self._directory(candidate_id) / 'corpus') != record['release_sha256']:
                raise ValueError('Candidate corpus changed after review.')
            from healthvault.rag.vector_cache import configured_model_path
            from healthvault.rag.build_hybrid_index import build_index
            model = configured_model_path(self.root)
            if model is None:
                raise ValueError('Configure a pre-downloaded local embedding model first.')
            corpus = self._directory(candidate_id) / 'corpus'
            build_index(artifact_path=corpus / 'chunks' / 'hybrid_v1.json', output_directory=corpus / 'vectors',
                        model_path=model, cache_directory=self.root / 'embedding-cache')
            record['vector_build'] = json.loads((corpus / 'vectors' / 'build-stats.json').read_text(encoding='utf-8'))
            record['retrieval_mode'] = 'hybrid'
            record['release_sha256'] = _bundle_hash(corpus)
            record['updated_at'] = _now()
            record['decisions'].append({'decision':'build_vectors','actor':actor,'at':_now()})
            _write_json(self._directory(candidate_id) / 'review.json',record)
            return record

    def stage(self, payload: dict[str, Any], actor: str) -> dict[str, Any]:
        with self._lock:
            return self._stage(payload, actor)

    def _stage(self, payload: dict[str, Any], actor: str) -> dict[str, Any]:
        with self._lock:
            return self._stage(payload, actor)

    def _stage(self, payload: dict[str, Any], actor: str) -> dict[str, Any]:
        if set(payload) - {"corpus_version", "source_updates", "source_metadata", "aliases", "probes"}:
            raise ValueError("Unknown update fields.")
        version = payload.get("corpus_version", "")
        if not isinstance(version, str) or not _VERSION.fullmatch(version) or version == self.active()["corpus_version"]:
            raise ValueError("Use a new corpus version with 3–80 lowercase letters, digits, dots or hyphens.")
        updates = payload.get("source_updates")
        probes = payload.get("probes")
        if not isinstance(updates, list) or not 1 <= len(updates) <= 10 or len(_json_bytes(payload)) > 2_000_000:
            raise ValueError("Provide 1–10 normalized source snapshots, at most 2 MB.")
        if not isinstance(probes, list) or not 1 <= len(probes) <= 10 or any(
            not isinstance(probe, dict) or set(probe) != {"query", "expected_source_id"}
            or any(not isinstance(value, str) or not value.strip() or len(value) > 200 for value in probe.values()) for probe in probes
        ):
            raise ValueError("Provide 1–10 retrieval probes with query and expected_source_id.")
        base = self.active_root
        base_hash = _bundle_hash(base)
        snapshots = _snapshots(base)
        previous_metadata = {}
        if (base / "provenance.json").is_file():
            previous_metadata = {item["source_id"]: {key: item[key] for key in ("document_kind", "published_on", "updated_on")} for item in json.loads((base / "provenance.json").read_text(encoding="utf-8"))}
        metadata = payload.get("source_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("source_metadata must map source IDs to date/type metadata.")
        changed, diffs, seen = [], [], set()
        for source in updates:
            if not isinstance(source, dict) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,100}", str(source.get("source_id", ""))):
                raise ValueError("Invalid source ID.")
            identifier = source["source_id"]
            if source.get('source_sha256') == 'auto':
                source = dict(source)
                source['source_sha256'] = hashlib.sha256(json.dumps(source['sections'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
            if identifier in seen:
                raise ValueError("Duplicate source update.")
            seen.add(identifier)
            parsed = urlparse(source.get("url", ""))
            if parsed.scheme != "https" or OFFICIAL_PUBLISHERS.get(parsed.hostname or "") != source.get("publisher") or parsed.username or parsed.port not in {None, 443}:
                raise ValueError("Source URL/publisher is outside the official whitelist.")
            date.fromisoformat(source["retrieved_on"])
            before = snapshots.get(identifier)
            if before and before["sections"] != source.get("sections") and before["source_sha256"] == source.get("source_sha256"):
                raise ValueError("Changed source content requires a new source hash.")
            if before is None or _comparable(before) != _comparable(source) or identifier in metadata:
                changed.append(identifier)
                diffs.extend(difflib.unified_diff(_json_bytes(_comparable(before)).decode().splitlines(True) if before else [], _json_bytes(_comparable(source)).decode().splitlines(True), fromfile=f"active/{identifier}", tofile=f"candidate/{identifier}"))
            snapshots[identifier] = dict(source)
        if not changed or not set(changed).issubset({probe["expected_source_id"] for probe in probes}):
            raise ValueError("Every changed source needs a retrieval probe.")
        if set(metadata) - seen or any(not isinstance(value, dict) for value in metadata.values()):
            raise ValueError("Metadata may only describe the updated sources.")
        candidate_id = hashlib.sha256(base_hash.encode() + _json_bytes(payload)).hexdigest()[:20]
        directory = self._directory(candidate_id)
        if (directory / "review.json").is_file():
            return self.get(candidate_id)
        initial_version = json.loads((DEFAULT_CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))["corpus_version"]
        if version == initial_version or any(record["corpus_version"] == version for record in self.list()):
            raise ValueError("Corpus version has already been used; choose a new version.")
        corpus = directory / "corpus"
        provenance = []
        for identifier, source in snapshots.items():
            source["corpus_version"] = version
            _write_json(corpus / "sources" / f"{identifier}.json", source)
            provenance.append(_provenance(source, metadata.get(identifier, previous_metadata.get(identifier, {}))))
        aliases = payload.get("aliases") or json.loads((base / "aliases" / "aliases.json").read_text(encoding="utf-8"))
        aliases = {**aliases, "corpus_version": version}
        _write_json(corpus / "aliases" / "aliases.json", aliases)
        from healthvault.rag.budget import TokenBudget
        TokenBudget.install(corpus, self.tokenizer_path)
        for relative_path, content in render_corpus_artifacts(corpus, corpus_version=version).items():
            path = corpus / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        _write_json(corpus / "provenance.json", provenance)
        bundle_hash = _bundle_hash(corpus)
        checks = _probe_corpus(corpus, probes)
        controls = [
            {"query": "HbA1c", "expected_source_id": "niddk-a1c-test"},
            {"query": "prepare laboratory", "expected_source_id": "medlineplus-lab-preparation"},
        ]
        baseline = _probe_corpus(base, controls)
        controls = [probe for probe, result in zip(controls, baseline) if result["passed"]]
        checks += _probe_corpus(corpus, controls)
        record = {"candidate_id": candidate_id, "status": "pending_review" if all(check["passed"] for check in checks) else "gate_rejected",
                  "corpus_version": version, "created_at": _now(), "updated_at": _now(), "created_by": actor,
                  "base_release_sha256": base_hash, "release_sha256": bundle_hash, "changed_source_ids": changed,
                  "unified_diff": "".join(diffs), "provenance": [item for item in provenance if item["source_id"] in changed],
                  "retrieval_mode": "lexical", "evaluation": {"passed": all(check["passed"] for check in checks), "retrieval_probes": checks, "answer_held_out": "not_run"},
                  "risk": ["Human source-content review required.", "Retrieval probes do not measure medical correctness or answer accuracy.", "Previous corpus vector indexes are never reused."], "decisions": []}
        _write_json(directory / "review.json", record)
        return record

    def decide(self, candidate_id: str, decision: str, actor: str) -> dict[str, Any]:
        if decision not in {"approve", "reject", "watch"}:
            raise ValueError("Unknown knowledge review decision.")
        with self._lock:
            record = self.get(candidate_id)
            if record["status"] not in {"pending_review", "watch"}:
                raise ValueError("Knowledge candidate is not reviewable.")
            if decision == "approve":
                if any(other["corpus_version"] == record["corpus_version"] and other["candidate_id"] != candidate_id for other in self.list()):
                    raise ValueError("Corpus version is bound to another candidate.")
                corpus = self._directory(candidate_id) / "corpus"
                if _bundle_hash(corpus) != record["release_sha256"]:
                    raise ValueError("Candidate corpus changed after review.")
                if _bundle_hash(self.active_root) != record["base_release_sha256"]:
                    record["status"] = "stale"
                else:
                    if not all(item["passed"] for item in _probe_corpus(corpus, [{key: item[key] for key in ("query", "expected_source_id")} for item in record["evaluation"]["retrieval_probes"]])):
                        raise ValueError("Candidate retrieval probes failed.")
                    pointer = self.root / "active.json"
                    previous = json.loads(pointer.read_text(encoding="utf-8")).get("active_candidate_id") if pointer.is_file() else None
                    _write_json(pointer, {"active_candidate_id": candidate_id, "previous_candidate_id": previous, "actor": actor, "at": _now()})
                    record["status"] = "published"
            else:
                record["status"] = "rejected" if decision == "reject" else "watch"
            record["updated_at"] = _now()
            record["decisions"].append({"decision": decision, "actor": actor, "at": _now()})
            _write_json(self._directory(candidate_id) / "review.json", record)
            return record

    def rollback(self, actor: str) -> dict[str, Any]:
        with self._lock:
            path = self.root / "active.json"
            if not path.is_file():
                raise ValueError("No knowledge release is available to roll back.")
            pointer = json.loads(path.read_text(encoding="utf-8"))
            if "previous_candidate_id" not in pointer:
                raise ValueError("Knowledge release has already been rolled back.")
            previous = pointer["previous_candidate_id"]
            target = self._directory(previous) / "corpus" if previous else DEFAULT_CORPUS_ROOT
            _bundle_hash(target)
            _write_json(path, {"active_candidate_id": previous, "rolled_back_candidate_id": pointer["active_candidate_id"], "actor": actor, "at": _now()})
            return {"status": "rolled_back", **self.active()}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage an offline knowledge source update for administrator review.")
    parser.add_argument("--file", required=True, type=Path, help="JSON update payload")
    parser.add_argument("--release-root", type=Path, default=configured_release_root())
    args = parser.parse_args()
    result = KnowledgeReleaseStore(args.release_root).stage(json.loads(args.file.read_text(encoding="utf-8")), "local-owner")
    print(json.dumps({key: result[key] for key in ("candidate_id", "status", "corpus_version", "changed_source_ids")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
