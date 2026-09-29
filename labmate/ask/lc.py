"""LangChain embeddings and a retriever over the library index.

Like :class:`~labmate.core.lc.RecordedChatModel`, they are LangChain *interfaces* over
labmate's recorded backend, so anything built on them (the prebuilt ``create_agent``
baseline, for one) replays as exactly as the rest of labmate.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from labmate.ask.embed import Embedder
from labmate.ask.index import Index, Method


class RecordedEmbeddings(Embeddings):
    """LangChain embeddings served by :class:`~labmate.ask.embed.Embedder`.

    Args:
        embedder: The embedder.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents.

        Args:
            texts: Texts.

        Returns:
            Normalised vectors.
        """
        return [list(map(float, v)) for v in self.embedder.documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        Args:
            text: Query.

        Returns:
            Normalised vector.
        """
        return list(map(float, self.embedder.query(text)))


class LibraryRetriever(BaseRetriever):
    """A LangChain retriever over the library index.

    Attributes:
        index: The index.
        embedder: Query embeddings.
        tiers: Source tiers to search.
        k: Hits per query.
        method: ``bm25``, ``dense`` or ``hybrid``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Index
    embedder: Embedder
    tiers: tuple[int, ...] = (1, 2)
    k: int = 6
    method: Method = "hybrid"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector = None if self.method == "bm25" else self.embedder.query(query)
        hits = self.index.search(query, vector, self.tiers, self.k, self.method)
        return [
            Document(
                page_content=h.chunk.text,
                id=h.chunk.id,
                metadata={**h.chunk.model_dump(exclude={"text"}), "score": h.score},
            )
            for h in hits
        ]
