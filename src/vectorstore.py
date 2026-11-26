"""A minimal vector store: in-memory cosine search with JSON/npz persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunking import Chunk


class NumpyVectorStore:
    """Stores L2-normalized vectors and searches by dot product (= cosine)."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        vectors = np.asarray(vectors, dtype=np.float32)
        self._vectors = (
            vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        )
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if self._vectors is None or len(self._chunks) == 0:
            return []
        k = max(1, min(k, len(self._chunks)))
        scores = self._vectors @ query_vector.astype(np.float32)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[i], float(scores[i])) for i in top]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors is None:
            raise ValueError("nothing to save: the store is empty")
        np.save(directory / "vectors.npy", self._vectors)
        meta = [
            {"text": c.text, "doc_id": c.doc_id, "index": c.index, "metadata": c.metadata}
            for c in self._chunks
        ]
        (directory / "chunks.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "NumpyVectorStore":
        directory = Path(directory)
        store = cls()
        store._vectors = np.load(directory / "vectors.npy")
        meta = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        store._chunks = [Chunk(**m) for m in meta]
        return store
