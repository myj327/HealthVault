"""Shared server retrieval selection, including vectors for the immutable baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from healthvault.infrastructure.bge_faiss import FaissVectorIndex
from healthvault.infrastructure.hybrid_health_knowledge import INDEX_VERSION, VectorIndexManifest, _chunk_ids_sha256
from healthvault.rag.corpus import load_chunk_artifact
from healthvault.rag.releases import DEFAULT_CORPUS_ROOT, _write_json, configured_release_root
from healthvault.rag.vector_cache import configured_model_path, model_fingerprint


def selection(corpus: Path, release_root: Path) -> dict:
    model = configured_model_path(release_root)
    if corpus != DEFAULT_CORPUS_ROOT:
        directory = corpus / "vectors"
        mode = "hybrid" if (directory / "manifest.json").is_file() else "lexical"
    else:
        config_path = release_root / "baseline-retrieval.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        directory = os.environ.get("HEALTHVAULT_HYBRID_INDEX_PATH") or config.get("index_directory")
        mode = os.environ.get("HEALTHVAULT_KNOWLEDGE_MODE") or ("hybrid" if directory else "lexical")
    return {"knowledge_mode": mode, "hybrid_index_path": str(directory) if directory else None,
            "hybrid_model_path": str(model) if model else None}


def activate_baseline(index_directory: Path, release_root: Path) -> dict:
    artifact_path = DEFAULT_CORPUS_ROOT / "chunks/hybrid_v1.json"
    artifact = load_chunk_artifact(artifact_path)
    manifest = VectorIndexManifest.from_dict(json.loads((index_directory / "manifest.json").read_text(encoding="utf-8")))
    ids = tuple(json.loads((index_directory / "chunk_ids.json").read_text(encoding="utf-8")))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    model = configured_model_path(release_root)
    if (manifest.index_version != INDEX_VERSION or manifest.corpus_sha256 != digest(artifact_path) or manifest.corpus_version != artifact.corpus_version
            or ids != tuple(chunk.chunk_id for chunk in artifact.chunks) or manifest.chunk_ids_sha256 != _chunk_ids_sha256(ids)
            or not manifest.index_sha256 or manifest.index_sha256 != digest(index_directory / "index.faiss")
            or manifest.model_name != "BAAI/bge-m3" or model is None or not manifest.model_sha256
            or manifest.model_sha256 != model_fingerprint(model)):
        raise ValueError("The baseline corpus, vector index and configured model must match exactly.")
    FaissVectorIndex.load(index_directory / "index.faiss", index_directory / "chunk_ids.json", manifest.dimension)
    target = release_root / "baseline-vectors" / manifest.index_sha256
    target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "index.faiss", "chunk_ids.json"):
        source, destination = index_directory / name, target / name
        if destination.is_file():
            if digest(destination) != digest(source):
                raise ValueError("Existing immutable baseline index differs.")
        else:
            shutil.copyfile(source, destination)
    value = {"corpus_version": manifest.corpus_version, "corpus_sha256": manifest.corpus_sha256,
             "model_sha256": manifest.model_sha256, "index_directory": str(target.resolve()),
             "index_sha256": manifest.index_sha256, "retrieval_mode": "hybrid"}
    _write_json(release_root / "baseline-retrieval.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-directory", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=configured_release_root())
    args = parser.parse_args()
    result = activate_baseline(args.index_directory, args.release_root)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
