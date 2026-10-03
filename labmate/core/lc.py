"""LangChain building blocks shared by the chains and graphs.

- :func:`prompt` makes a chat prompt from a template.
- :func:`structured` turns a prompt into a validated Pydantic object. It puts the JSON
  schema in the system prompt (the MLX model builds ignore Ollama's ``format``), then
  validates the reply and retries with the error shown to the model.
- :func:`step` names a function as a pipeline step, so it shows up by name in the trace.
- :func:`batch_map` runs a function over items as one parallel LangChain batch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel

from labmate.core.structured import (
    StructuredOutputError,
    parse_structured,
    schema_instruction,
    validation_error,
)

log = logging.getLogger(__name__)


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
        :class:`~labmate.core.structured.StructuredOutputError` when no attempt validates.
    """
    json_schema = schema.model_json_schema()
    constrained: Runnable[Any, AIMessage] = model.bind(format=json_schema)
    system = SystemMessage(content=schema_instruction(schema))

    def answer(value: PromptValue, config: RunnableConfig) -> M:
        messages: list[BaseMessage] = [system, *value.to_messages()]
        last = ""
        for attempt in range(1, max_retries + 2):
            last = str(constrained.invoke(messages, config).content)
            parsed = parse_structured(last, schema)
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
            log.info(
                "%s rejected (attempt %d): %s",
                schema.__name__,
                attempt,
                " ".join(feedback.split())[:500],
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
