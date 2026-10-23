"""Reuse vectors only for identical embedding text and identical local model files."""
from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from healthvault.rag.releases import _write_json


def configured_model_path(release_root: Path) -> Path | None:
    configured = os.environ.get('HEALTHVAULT_HYBRID_MODEL_PATH')
    config = release_root / 'local-model.json'
    if not configured and config.is_file():
        configured = json.loads(config.read_text(encoding='utf-8')).get('model_path')
    return Path(configured) if configured else None


def model_fingerprint(root: Path) -> str:
    root = root.resolve()
    files = tuple((str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in sorted(root.rglob('*')) if path.is_file() and path.suffix in {'.json','.bin','.safetensors','.model'})
    if not files:
        raise ValueError('No local embedding model files found.')
    return _fingerprint(str(root), files)


@lru_cache(maxsize=4)
def _fingerprint(root: str, files: tuple) -> str:
    digest = hashlib.sha256()
    for name, _, _ in files:
        digest.update(name.encode()); digest.update(b'\0')
        with (Path(root) / name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
    return digest.hexdigest()


def _vector_hash(vector: list[float]) -> str:
    return hashlib.sha256(json.dumps(vector,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def cached_vectors(texts: tuple[str, ...], model_sha: str, root: Path, make_embedder: Any) -> tuple[tuple, dict]:
    keys = [hashlib.sha256((model_sha+'\n'+text).encode()).hexdigest() for text in texts]
    vectors: list[Any] = [None]*len(texts); hits = 0
    for index, key in enumerate(keys):
        path = root / f'{key}.json'
        if not path.is_file(): continue
        try:
            record = json.loads(path.read_text(encoding='utf-8')); vector = record['vector']
            if record['model_sha256'] != model_sha or record['key'] != key or not vector or not all(isinstance(value,(float,int)) and math.isfinite(value) for value in vector) or _vector_hash(vector) != record['vector_sha256']:
                continue
            vectors[index] = tuple(vector); hits += 1
        except (ValueError, KeyError, TypeError, OSError):
            continue
    missing = [index for index, vector in enumerate(vectors) if vector is None]
    if missing:
        produced = make_embedder().embed(tuple(texts[index] for index in missing))
        if len(produced) != len(missing):
            raise ValueError('Embedding result count mismatch.')
        for index, vector in zip(missing, produced):
            vector = [float(value) for value in vector]
            if not vector or not all(math.isfinite(value) for value in vector):
                raise ValueError('Embedding contains non-finite values.')
            _write_json(root / f'{keys[index]}.json', {'key':keys[index],'model_sha256':model_sha,'vector':vector,'vector_sha256':_vector_hash(vector)})
            vectors[index] = tuple(vector)
    if vectors and any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ValueError('Cached embedding dimensions are inconsistent.')
    return tuple(vectors), {'total':len(texts),'reused':hits,'embedded':len(missing),'model_sha256':model_sha}
