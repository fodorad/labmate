"""LangChain building blocks over labmate's recorded backend.

LangChain provides the *composition*: prompts, chains (``|``), batching, run names and
callbacks, which feed LangSmith or Phoenix when tracing is switched on. labmate
provides the *model access*: every call goes through the cassette store, so every chain
and graph replays byte for byte without Ollama.

- :class:`RecordedChatModel` is a LangChain chat model served by the recorded backend.
- :func:`structured` turns a prompt into a validated Pydantic object. It puts the JSON
  schema in the system prompt (the MLX model builds ignore Ollama's ``format=``), then
  validates the reply and retries with the error shown to the model.
- :func:`step` names a function as a pipeline step, so it shows up by name in the trace.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field

from labmate.core.llm.structured import (
    StructuredOutputError,
    parse_structured,
    schema_instruction,
    validation_error,
)
from labmate.core.llm.types import ChatRequest, Message, ToolCall, ToolFunction
from labmate.core.model import LLM
from labmate.core.phases import ModelSwitcher

log = logging.getLogger(__name__)

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
    """A LangChain chat model served by labmate's recorded backend.

    Each reply carries its cassette key and whether it was replayed
    (``response_metadata["cache_key"]`` and ``["cached"]``), so a trace row points at the
    exact cassette that reproduces it. Tool calls get deterministic ids
    (``call_<turn>_<i>``), because Ollama returns none and random ids would make replayed
    conversations differ.

    Attributes:
        llm: Model settings and backend.
        switcher: Keeps one large model in memory; switched before every call (optional).
        tools: Tools bound with :meth:`bind_tools` (OpenAI function format).
        json_schema: JSON schema sent as Ollama's ``format=`` constraint (see
            :meth:`with_json_schema`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LLM
    switcher: ModelSwitcher | None = Field(default=None, exclude=True)
    tools: list[dict[str, Any]] | None = None
    json_schema: dict[str, Any] | None = None

    @property
    def _llm_type(self) -> str:
        return "labmate-recorded"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.llm.model, "seed": self.llm.seed}

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

    def with_json_schema(self, schema: dict[str, Any]) -> RecordedChatModel:
        """Return a copy of the model that sends ``schema`` as the ``format=`` constraint.

        Args:
            schema: JSON schema of the expected reply.

        Returns:
            The constrained model.
        """
        return self.model_copy(update={"json_schema": schema})

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
            format=self.json_schema,
            tools=self.tools,
            seed=self.llm.seed,
            temperature=self.llm.temperature,
            num_ctx=self.llm.num_ctx,
            think=False,
        )
        if self.switcher is not None:
            self.switcher.use(self.llm.model)
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
            response_metadata={
                "model": self.llm.model,
                "cache_key": request.cache_key(self.llm.digest),
                "cached": response.cached,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def prompt(template: str) -> ChatPromptTemplate:
    """A single-message chat prompt from a template with ``{placeholders}``.

    Args:
        template: Prompt text, e.g. read with a feature's ``load_prompt``.

    Returns:
        The prompt template.
    """
    return ChatPromptTemplate.from_messages([("human", template)])


def structured[M: BaseModel](
    model: BaseChatModel,
    schema: type[M],
    check: Callable[[M], list[str]] | None = None,
    max_retries: int = 2,
) -> Runnable[PromptValue, M]:
    """A Runnable that answers a prompt with a validated ``schema`` instance.

    The schema goes both ways: as ``format=`` (honoured by some backends) and as a system
    instruction (followed by the rest). An invalid reply is sent back to the model with the
    validation error, and so is a reply that ``check`` finds problems with: the cheapest
    self-correction loop. Every attempt is a separate model run in the trace.

    Args:
        model: The chat model.
        schema: Target Pydantic model.
        check: Optional semantic rules on top of the schema, returning problems (empty = OK).
        max_retries: Extra attempts after the first.

    Returns:
        ``prompt value -> schema instance``; raises
        :class:`~labmate.core.llm.structured.StructuredOutputError` when no attempt validates.
    """
    json_schema = schema.model_json_schema()
    constrained: Runnable[Any, AIMessage] = (
        model.with_json_schema(json_schema)
        if isinstance(model, RecordedChatModel)
        else model.bind(format=json_schema)
    )
    system = SystemMessage(content=schema_instruction(schema))

    def answer(value: PromptValue, config: RunnableConfig) -> M:
        messages: list[BaseMessage] = [system, *value.to_messages()]
        last = ""
        for _ in range(max_retries + 1):
            last = str(constrained.invoke(messages, config).content)
            parsed, _ = parse_structured(last, schema)
            if parsed is not None:
                problems = check(parsed) if check else []
                if not problems:
                    return parsed
                feedback = "That reply is valid JSON but breaks these rules:\n- " + "\n- ".join(
                    problems
                )
            else:
                feedback = (
                    "That reply does not validate against the schema:\n"
                    f"{validation_error(last, schema)}"
                )
            messages = [
                *messages,
                AIMessage(content=last),
                HumanMessage(content=f"{feedback}\nReply again with only the corrected JSON."),
            ]
        raise StructuredOutputError(
            f"{getattr(model, 'model', type(model).__name__)}: no valid {schema.__name__} "
            f"after {max_retries + 1} attempts; "
            f"last output: {last[:200]!r}"
        )

    return RunnableLambda(answer, name=schema.__name__)


def step[I, O](name: str, fn: Callable[[I, RunnableConfig], O]) -> Runnable[I, O]:
    """Name a function as a pipeline step.

    The run name shows the step by name in LangSmith or Phoenix when tracing is switched on
    (``LANGSMITH_TRACING=true``), and the step is logged as it starts. ``fn`` receives the run's
    config, to pass callbacks on to the runs it starts.

    Args:
        name: Step name, e.g. ``"route"``.
        fn: The step: ``(input, config) -> output``.

    Returns:
        The named Runnable.
    """

    def run(value: I, config: RunnableConfig) -> O:
        log.info("step %s", name)
        return fn(value, config)

    return RunnableLambda(run, name=name)


def batch_map[I, O](
    fn: Callable[[I, RunnableConfig], O],
    items: Sequence[I],
    config: RunnableConfig | None,
    workers: int,
) -> list[O]:
    """Apply ``fn`` to every item as one LangChain batch, at most ``workers`` at a time.

    The *parallelisation* pattern: results keep the input order, and each call's runs are
    children of the current step, even on worker threads.

    Args:
        fn: ``(item, config) -> result``.
        items: Inputs.
        config: The calling step's config (carries the callbacks).
        workers: Maximum concurrency. Local models serve few requests at once, so this
            stays small.

    Returns:
        One result per item, in order.
    """

    def call(item: I, config: RunnableConfig) -> O:  # LangChain passes a param named config
        return fn(item, config)

    runnable: Runnable[I, O] = RunnableLambda(call, name="item")
    return runnable.batch(list(items), {**(config or {}), "max_concurrency": workers})
