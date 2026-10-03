"""Graph factories for LangGraph Studio (``langgraph dev`` reads ``langgraph.json``).

Studio runs the graphs with its own checkpointer and shows every node, the state and the
interrupts. Set ``LANGSMITH_TRACING=false`` to keep traces on this machine.
"""

from __future__ import annotations

from typing import Any

from labmate.ask.agent import build_agent
from labmate.ask.graph import build_graph
from labmate.ask.session import open_ask
from labmate.config import load_config


def make_graph() -> Any:
    """The ask graph, compiled without a checkpointer (Studio brings its own).

    Returns:
        The compiled graph.
    """
    return build_graph(open_ask(load_config())).compile()


def make_agent() -> Any:
    """The tool-calling agent.

    Returns:
        The compiled graph.
    """
    return build_agent(open_ask(load_config()))
