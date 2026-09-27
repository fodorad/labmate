"""Graph factories for LangGraph Studio (``langgraph dev`` reads ``langgraph.json``).

Studio runs the graphs with its own checkpointer and shows every node, the state and
the interrupts; the models still go through labmate's recorded backend.
"""

from __future__ import annotations

from typing import Any

from labmate.ask.agent import build_agent
from labmate.ask.graph import build_graph
from labmate.ask.session import open_ask
from labmate.config import load_config


def make_graph() -> Any:
    """The ask agent, compiled without a checkpointer (Studio brings its own).

    Returns:
        The compiled graph.
    """
    return build_graph(open_ask(load_config(), label="ask-studio")).compile()


def make_prebuilt() -> Any:
    """The prebuilt baseline agent.

    Returns:
        The compiled graph.
    """
    return build_agent(open_ask(load_config(), label="ask-studio"))
