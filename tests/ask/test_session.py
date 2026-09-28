import sqlite3

import httpx
import pytest

from labmate.ask.graph import compile_graph
from labmate.ask.session import open_ask
from labmate.core.llm.client import OllamaClient


def test_closing_releases_the_databases_even_when_ollama_is_gone(config):
    def refuse(request):
        raise httpx.ConnectError("refused")

    s = open_ask(config, client=OllamaClient(transport=httpx.MockTransport(refuse)))
    compile_graph(s)  # opens the conversation checkpoints
    s.switcher.use(config.models.text)  # a model to unload at the end
    s.close()
    with pytest.raises(sqlite3.ProgrammingError):
        s.index.db.execute("select 1")
    with pytest.raises(sqlite3.ProgrammingError):
        s.connections[0].execute("select 1")
