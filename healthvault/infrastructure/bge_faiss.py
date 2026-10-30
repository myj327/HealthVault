"""Optional BGE-M3 and FAISS adapters, imported only by an explicit hybrid setup."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from healthvault.repositories.dense_retrieval import DenseRetrievalError


class BGEM3Embedder:
    """Use a pre-downloaded BGE-M3 directory; this adapter never downloads models."""

    def __init__(self, model_path: str | Path, *, model_name: str = "BAAI/bge-m3") -> None:
        path = Path(model_path)
        if not path.is_dir():
            raise DenseRetrievalError(
                "dense_model_missing: configure a pre-downloaded local BGE-M3 directory"
            )
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise DenseRetrievalError(
                "dense_dependency_missing: install optional hybrid retrieval dependencies"
            ) from exc
        # BGE-M3 dense retrieval uses the normalized CLS vector. Loading only
        # that encoder avoids FlagEmbedding's training/reranking imports and its
        # extra trial forward pass on every encode call.
        self._tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self._model = AutoModel.from_pretrained(str(path), local_files_only=True)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device=self._device, dtype=torch.float32).eval()
        self._model_name = model_name
        self._dimension = int(self._model.config.hidden_size)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        import torch
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                inputs = self._tokenizer(list(texts[start:start + 32]), padding=True,
                                         truncation=True, max_length=512, return_tensors="pt")
                output = self._model(**inputs.to(self._device)).last_hidden_state[:, 0]
                vectors.extend(torch.nn.functional.normalize(output, p=2, dim=1).cpu().tolist())
        result = tuple(tuple(float(value) for value in vector) for vector in vectors)
        if not result or any(len(vector) != len(result[0]) for vector in result):
            raise DenseRetrievalError("BGE-M3 returned inconsistent vector dimensions")
        if self._dimension is not None and len(result[0]) != self._dimension:
            raise DenseRetrievalError("BGE-M3 vector dimension changed unexpectedly")
        self._dimension = len(result[0])
        return result


class FaissVectorIndex:
    """Small persistent FAISS inner-product index with stable external chunk IDs."""

    def __init__(self, index: object, chunk_ids: tuple[str, ...], dimension: int) -> None:
        self._index = index
        self._chunk_ids = chunk_ids
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @classmethod
    def build(
        cls, chunk_ids: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> "FaissVectorIndex":
        if not chunk_ids or len(chunk_ids) != len(vectors) or len(set(chunk_ids)) != len(chunk_ids):
            raise DenseRetrievalError("FAISS requires unique chunk IDs and aligned vectors")
        dimension = len(vectors[0]) if vectors else 0
        if not dimension or any(len(vector) != dimension for vector in vectors):
            raise DenseRetrievalError("FAISS vectors must share a non-zero dimension")
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise DenseRetrievalError(
                "dense_dependency_missing: install optional faiss-cpu before building hybrid"
            ) from exc
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(dimension); index.add(matrix)
        return cls(index, tuple(chunk_ids), dimension)

    @classmethod
    def load(cls, index_path: str | Path, ids_path: str | Path, dimension: int) -> "FaissVectorIndex":
        try:
            import faiss
            import json
        except ImportError as exc:
            raise DenseRetrievalError(
                "dense_dependency_missing: install optional faiss-cpu before enabling hybrid"
            ) from exc
        index = faiss.read_index(str(index_path))
        chunk_ids = tuple(json.loads(Path(ids_path).read_text(encoding="utf-8")))
        if index.d != dimension or index.ntotal != len(chunk_ids) or len(set(chunk_ids)) != len(chunk_ids):
            raise DenseRetrievalError("dense_index_invalid: FAISS dimension or chunk mapping mismatch")
        return cls(index, chunk_ids, dimension)

    def save(self, index_path: str | Path, ids_path: str | Path) -> None:
        try:
            import faiss
            import json
        except ImportError as exc:
            raise DenseRetrievalError("dense_dependency_missing: faiss-cpu is required") from exc
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        Path(ids_path).write_text(json.dumps(list(self._chunk_ids), ensure_ascii=False), encoding="utf-8")

    def search(self, vector: Sequence[float], limit: int) -> tuple[tuple[str, float], ...]:
        if len(vector) != self._dimension or limit <= 0:
            return ()
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise DenseRetrievalError("dense_dependency_missing: faiss-cpu is required") from exc
        query = np.asarray([vector], dtype="float32"); faiss.normalize_L2(query)
        scores, positions = self._index.search(query, min(limit, len(self._chunk_ids)))
        return tuple(
            (self._chunk_ids[position], float(score))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        )
