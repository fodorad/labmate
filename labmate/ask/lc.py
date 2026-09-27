"""LangChain interfaces over labmate's recorded backend.

The recorded backend (cassettes, tracing, pinned model digests) is what makes every run
replayable byte-for-byte, so LangChain is used for its *interfaces*, not its Ollama
integration: a chat model, an embeddings class and a retriever that all go through the
same backend. Anything built on LangChain (the prebuilt ``create_agent`` baseline, for
one) is then as reproducible as the rest of labmate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from labmate.ask.embed import Embedder
from labmate.ask.index import Index, Method
from labmate.core.llm.types import ChatRequest, Message, ToolCall, ToolFunction
from labmate.core.model import LLM

_ROLES = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}


def to_messages(messages: Sequence[BaseMessage]) -> list[Message]:
    """Convert LangChain messages to labmate's request messages.

    Args:
        messages: System, human, AI (with tool calls) and tool messages.

    Returns:
        The equivalent messages.

    Raises:
        ValueError: For a message type labmate can't send.
    """
    out: list[Message] = []
    for m in messages:
        role = _ROLES.get(m.type)
        if role is None:
            raise ValueError(f"unsupported message type {m.type!r}")
        content = m.content if isinstance(m.content, str) else str(m.content)
        calls = None
        if isinstance(m, AIMessage) and m.tool_calls:
            calls = [ToolCall(function=ToolFunction(name=c["name"], arguments=c["args"]))
                     for c in m.tool_calls]  # fmt: skip
        name = m.name if isinstance(m, ToolMessage) else None
        out.append(Message(role=role, content=content, tool_calls=calls, tool_name=name))
    return out


class RecordedChatModel(BaseChatModel):
    """A LangChain chat model served by labmate's recorded, traced backend.

    Tool calls get deterministic ids (``call_<turn>_<i>``), because Ollama returns none
    and random ids would make replayed conversations differ.

    Attributes:
        llm: Model settings and backend.
        tools: Tools bound with :meth:`bind_tools` (OpenAI function format).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LLM
    tools: list[dict[str, Any]] | None = None

    @property
    def _llm_type(self) -> str:
        return "labmate-recorded"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Return a copy of the model that offers ``tools`` on every call.

        Args:
            tools: LangChain tools, functions or schemas.
            tool_choice: Ignored (the local models choose freely).
            kwargs: Ignored.

        Returns:
            The bound model.
        """
        return self.model_copy(update={"tools": [convert_to_openai_tool(t) for t in tools]})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = ChatRequest(
            model=self.llm.model,
            messages=to_messages(messages),
            tools=self.tools,
            seed=self.llm.seed,
            temperature=self.llm.temperature,
            num_ctx=self.llm.num_ctx,
            think=False,
        )
        response = self.llm.backend.chat(request)
        turn = sum(isinstance(m, AIMessage) for m in messages)
        message = AIMessage(
            content=response.content,
            tool_calls=[
                {
                    "name": c.function.name,
                    "args": c.function.arguments,
                    "id": f"call_{turn}_{i}",
                    "type": "tool_call",
                }
                for i, c in enumerate(response.tool_calls)
            ],  # fmt: skip
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.prompt_tokens + response.usage.completion_tokens,
            },
            response_metadata={"model": response.model, "cached": response.cached},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


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
