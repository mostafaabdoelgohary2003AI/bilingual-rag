"""The RAG pipeline tying chunking, embedding, retrieval and generation together."""

from __future__ import annotations

from pathlib import Path

from .chunking import Chunk, chunk_text
from .embeddings import TfidfEmbedder
from .generator import ExtractiveGenerator
from .retriever import Retriever
from .vectorstore import NumpyVectorStore


class RAGPipeline:
    """Ingest documents, then answer questions with citations.

    The index is rebuilt on every ingest. That keeps the code honest and
    simple; at the scale of a personal document set (thousands of chunks)
    a rebuild takes well under a second with the TF-IDF backend.
    """

    def __init__(self, embedder=None, generator=None, chunk_size: int = 500, overlap: int = 80) -> None:
        self.embedder = embedder or TfidfEmbedder()
        self.generator = generator or ExtractiveGenerator()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = NumpyVectorStore()
        self.retriever: Retriever | None = None
        self._chunks: list[Chunk] = []

    def ingest_paths(self, paths: list[str | Path]) -> int:
        """Chunk and index every file; returns the number of new chunks."""
        new_chunks: list[Chunk] = []
        for path in paths:
            path = Path(path)
            text = path.read_text(encoding="utf-8")
            new_chunks.extend(
                chunk_text(
                    text,
                    doc_id=path.name,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                    metadata={"source": str(path)},
                )
            )
        if not new_chunks:
            return 0
        self._chunks.extend(new_chunks)
        self._rebuild_index()
        return len(new_chunks)

    def _rebuild_index(self) -> None:
        corpus = [c.text for c in self._chunks]
        self.embedder.fit(corpus)
        self.store = NumpyVectorStore()
        self.store.add(self.embedder.transform(corpus), self._chunks)
        self.retriever = Retriever(self.embedder, self.store)

    def query(self, question: str, k: int = 4) -> dict:
        if self.retriever is None:
            raise RuntimeError("ingest documents before querying")
        passages = self.retriever.retrieve(question, k=k)
        answer = self.generator.generate(question, passages)
        return {
            "question": question,
            "answer": answer,
            "citations": [p.citation for p in passages],
        }
