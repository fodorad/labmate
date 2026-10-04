from langchain_core.messages import AIMessage, HumanMessage

from labmate.core.agents import MAX_NUDGES, run_agent


class StopsEarly:
    """An agent that ends without an answer until it has been reminded once."""

    def __init__(self) -> None:
        self.runs = 0

    def invoke(self, state: dict, config: dict) -> dict:
        self.runs += 1
        reply = "" if self.runs == 1 else "the answer"
        return {"messages": [*state["messages"], AIMessage(reply)]}


def answered(messages: list) -> bool:
    return bool(messages[-1].content)


def test_an_agent_that_stops_without_an_answer_is_reminded_once_and_then_left_alone():
    agent = StopsEarly()

    messages = run_agent(agent, "question", {}, done=answered, nudge="answer now")

    assert agent.runs == 2 and messages[-1].content == "the answer"
    assert [m.content for m in messages if isinstance(m, HumanMessage)] == [
        "question",
        "answer now",
    ]


def test_the_reminders_are_limited():
    class Silent:
        runs = 0

        def invoke(self, state: dict, config: dict) -> dict:
            self.runs += 1
            return {"messages": [*state["messages"], AIMessage("")]}

    agent = Silent()

    run_agent(agent, "question", {}, done=answered, nudge="answer now")

    assert agent.runs == 1 + MAX_NUDGES
