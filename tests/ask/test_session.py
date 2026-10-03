import sqlite3

import pytest

from labmate.ask.session import open_ask


def test_closing_a_session_closes_its_index(config, model):
    s = open_ask(config, ollama=model.transport())

    s.close()

    with pytest.raises(sqlite3.ProgrammingError):
        s.index.db.execute("select 1")
