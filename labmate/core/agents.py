"""Running a tool-calling agent until its job is done."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

MAX_NUDGES = 2
"""How many times a stopped agent is asked to carry on."""


def run_agent(
    agent: Any,
    task: str,
    config: RunnableConfig,
    *,
    done: Callable[[list[AnyMessage]], bool],
    nudge: str,
) -> list[AnyMessage]:
    """Run an agent and, if it stops before its job is done, ask it to carry on.

    A ReAct loop ends when the model replies without a tool call. Some models, some of the time,
    reply with nothing at all after a tool result (``gemma4:26b-mlx`` with thinking off does), so
    the loop would end with no answer. The agent is then given the conversation so far plus a
    short reminder, up to :data:`MAX_NUDGES` times.

    Args:
        agent: A compiled agent (``create_agent``).
        task: The first user message.
        config: Run config (recursion limit, callbacks).
        done: Whether the messages so far show the job done.
        nudge: The reminder sent when it is not.

    Returns:
        The whole conversation.
    """
    messages: list[AnyMessage] = agent.invoke({"messages": [HumanMessage(task)]}, config)[
        "messages"
    ]
    for _ in range(MAX_NUDGES):
        if done(messages):
            break
        messages = agent.invoke({"messages": [*messages, HumanMessage(nudge)]}, config)["messages"]
    return messages
