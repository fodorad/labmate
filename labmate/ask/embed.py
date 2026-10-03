"""Embeddings of queries and documents, L2-normalised so a dot product is the cosine."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from langchain_core.embeddings import Embeddings
from numpy.typing import NDArray

PREFIXES = {
    "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
    "nomic-embed-text": ("search_query: ", "search_document: "),
}
"""Query and document prefixes the model families were trained with (none for others)."""

Vectors = NDArray[np.float32]
"""A ``(n, dim)`` matrix of L2-normalised embeddings."""


class Embedder:
    """Embeds queries and documents with an embedding model.

    Args:
        embeddings: The embedding model (e.g. Ollama's).
        model: Its tag, which selects the prefixes.
        batch: Texts per request.
    """

    def __init__(self, embeddings: Embeddings, model: str, batch: int = 32) -> None:
        self.embeddings = embeddings
        self.model = model
        self.batch = batch
        family = model.split(":", 1)[0].rsplit("/", 1)[-1]
        self.query_prefix, self.doc_prefix = PREFIXES.get(family, ("", ""))

    def _embed(self, texts: Sequence[str]) -> Vectors:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            rows += self.embeddings.embed_documents(list(texts[i : i + self.batch]))
        matrix = np.asarray(rows, dtype=np.float32).reshape(len(rows), -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalised: Vectors = (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)
        return normalised

    def documents(self, texts: Sequence[str]) -> Vectors:
        """Embed documents (chunks).

        Args:
            texts: Chunk texts.

        Returns:
            One normalised vector per text.
        """
        return self._embed([self.doc_prefix + t for t in texts])

    def query(self, text: str) -> NDArray[np.float32]:
        """Embed a search query.

        Args:
            text: The query.

        Returns:
            The normalised query vector.
        """
        vector: NDArray[np.float32] = self._embed([self.query_prefix + text])[0]
        return vector
