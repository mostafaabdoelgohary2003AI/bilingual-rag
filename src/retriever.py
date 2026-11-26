"""Retrieval: embed a query and return the top-k chunks with scores."""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk
from .vectorstore import NumpyVectorStore


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> dict:
        return {
            "doc_id": self.chunk.doc_id,
            "chunk_index": self.chunk.index,
            "score": round(self.score, 4),
            "snippet": self.chunk.text[:200],
        }


class Retriever:
    def __init__(self, embedder, store: NumpyVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, k: int = 4) -> list[ScoredChunk]:
        query_vector = self.embedder.transform([question])[0]
        return [
            ScoredChunk(chunk=chunk, score=score)
            for chunk, score in self.store.search(query_vector, k=k)
            if score > 0.0  # no lexical/semantic overlap means no evidence
        ]
