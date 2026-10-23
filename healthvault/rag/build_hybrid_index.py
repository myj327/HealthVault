"""Build a local BGE-M3/FAISS index from the frozen chunk artifact only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from healthvault.infrastructure.bge_faiss import BGEM3Embedder, FaissVectorIndex
from healthvault.infrastructure.hybrid_health_knowledge import (
    INDEX_VERSION,
    VectorIndexManifest,
    _chunk_ids_sha256,
    _sha256_file,
)
from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    DEFAULT_CHUNK_ARTIFACT_PATH,
)
from healthvault.rag.corpus import load_chunk_artifact


def build_index(
    *, artifact_path: str | Path, output_directory: str | Path,
    model_path: str | Path, model_name: str = "BAAI/bge-m3",
    cache_directory: str | Path | None = None, embedder: object | None = None,
) -> VectorIndexManifest:
    artifact_file = Path(artifact_path); output = Path(output_directory)
    artifact = load_chunk_artifact(artifact_file, expected_corpus_version=None)
    from healthvault.rag.budget import TokenBudget
    budget = TokenBudget.load(artifact_file.parent.parent)
    if budget and budget.tokenizer and budget.identity != _sha256_file(Path(model_path)/'tokenizer.json'):
        raise ValueError('Chunk tokenizer does not match the embedding model tokenizer.')
    chunk_ids = tuple(chunk.chunk_id for chunk in artifact.chunks)
    texts = tuple("\n".join((" / ".join(chunk.title_path), " ".join(chunk.topic_tags), chunk.body)) for chunk in artifact.chunks)
    from healthvault.rag.vector_cache import cached_vectors, model_fingerprint
    model_sha = model_fingerprint(Path(model_path))
    vectors, stats = cached_vectors(texts, model_sha, Path(cache_directory) if cache_directory else output.parent / 'embedding-cache',
                                    lambda: embedder or BGEM3Embedder(model_path, model_name=model_name))
    index = FaissVectorIndex.build(chunk_ids, vectors)
    output.mkdir(parents=True, exist_ok=True); index.save(output / "index.faiss", output / "chunk_ids.json")
    manifest = VectorIndexManifest(INDEX_VERSION, _sha256_file(artifact_file), artifact.corpus_version, model_name, index.dimension, _chunk_ids_sha256(chunk_ids), model_sha, _sha256_file(output / 'index.faiss'))
    (output / "manifest.json").write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "build-stats.json").write_text(json.dumps(stats,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an offline BGE-M3 + FAISS HealthVault index")
    parser.add_argument("--model-path", required=True, help="Pre-downloaded local BGE-M3 directory; never a remote name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", default=str(DEFAULT_CHUNK_ARTIFACT_PATH))
    args = parser.parse_args(argv)
    print(json.dumps(build_index(artifact_path=args.artifact, output_directory=args.output, model_path=args.model_path).to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
