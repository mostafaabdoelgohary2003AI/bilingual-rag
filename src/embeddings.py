"""Embedding backends.

Two interchangeable backends:

* ``TfidfEmbedder`` - default, CPU-only, no downloads. Works surprisingly
  well for exact-term and bilingual retrieval when combined with
  character n-grams.
* ``SentenceTransformerEmbedder`` - optional dense backend (e.g.
  ``intfloat/multilingual-e5-small``) for semantic retrieval. Imported
  lazily so the project runs without it installed.

Both expose ``fit(texts)`` / ``transform(texts) -> np.ndarray`` with
L2-normalized rows, so cosine similarity is a dot product.
"""

from __future__ import annotations

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TfidfEmbedder:
    """Word + character n-gram TF-IDF embedder (scikit-learn)."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
        )

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self._vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return _l2_normalize(matrix)


class SentenceTransformerEmbedder:
    """Dense multilingual embeddings via sentence-transformers (optional)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is not installed. "
                "pip install sentence-transformers to use the dense backend."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self  # pretrained; nothing to fit

    def transform(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
