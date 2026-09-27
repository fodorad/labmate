import json

import pytest

from labmate.ask.graph import (
    NOT_FOUND,
    OFF_TOPIC,
    ask,
    build_research,
    check_understanding,
    collect,
    compile_graph,
    compose,
    evidence_ids,
    find_conflicts,
    widen,
)
from labmate.ask.schemas import Finding, Sentence, Understanding
from labmate.ask.session import open_ask
from labmate.config import ReplayMode
from labmate.core.tracing import read_trace


@pytest.fixture
def graph(indexed):
    return compile_graph(indexed)


def test_answer_with_citations_from_the_dissertation_first(indexed, graph):
    answer = ask(indexed, graph, "Which datasets are used for training?")
    assert not answer.abstained and answer.citations and answer.sentences
    assert answer.citations[0].tier == 1  # dissertation first
    assert answer.citations[0].label.startswith("Dissertation §")
    assert "[1]" in answer.text and answer.dropped == 0
    names = [s["name"] for s in read_trace(indexed.tracer.path)]
    for step in ("understand", "plan", "research", "retrieve", "grade", "conflicts",
                 "answer", "verify"):  # fmt: skip
        assert f"step.ask.{step}" in names, step


def test_research_splits_questions_and_runs_branches_in_parallel(indexed, graph):
    ask(indexed, graph, "What does BlinkLinMulT fuse and which datasets are used?")
    spans = [s for s in read_trace(indexed.tracer.path) if s["name"] == "step.ask.research"]
    assert {s["query"] for s in spans} == {"What does BlinkLinMulT fuse", "which datasets are used"}


def test_insufficient_evidence_rewrites_and_widens_to_the_papers(indexed):
    research = build_research(indexed)
    out = research.invoke(
        {
            "planned": "rarely eye datasets",
            "query": "rarely eye datasets",
            "intent": "own_work",
            "tiers": [1],
        }  # fmt: skip
    )
    finding = out["finding"]
    assert finding.loops == 2 and finding.tiers == [1, 2]
    assert finding.queries == ["rarely eye datasets", "eye datasets details"]
    assert finding.chunk_ids and finding.sufficient
    assert widen([1, 2, 3]) == [1, 2, 3]


def test_research_stops_at_the_loop_budget(indexed):
    indexed.config.ask.max_loops = 1
    out = build_research(indexed).invoke(
        {"planned": "rarely x", "query": "rarely x", "intent": "own_work", "tiers": [1]}
    )
    assert out["finding"].loops == 1 and not out["finding"].sufficient


def test_follow_up_questions_use_the_conversation(indexed, graph):
    ask(indexed, graph, "Which datasets are used?", thread="t1")
    ask(indexed, graph, "and how long does training take?", thread="t1")
    understood = [s for s in read_trace(indexed.tracer.path) if s["name"] == "step.ask.understand"]
    assert len(understood) == 2
    state = graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert [t.question for t in state["history"]] == [
        "Which datasets are used?", "and how long does training take?",
    ]  # fmt: skip
    assert (
        state["understanding"].standalone == "Which datasets are used how long does training take?"
    )
    # a new thread starts fresh
    ask(indexed, graph, "Which datasets are used?", thread="t2")
    other = graph.get_state({"configurable": {"thread_id": "t2"}}).values
    assert len(other["history"]) == 1


def test_off_topic_questions_are_declined_without_searching(indexed, graph):
    answer = ask(indexed, graph, "What will the weather be tomorrow?")
    assert answer.abstained and answer.text == OFF_TOPIC and answer.reason == "off topic"
    assert not any(s["name"] == "step.ask.retrieve" for s in read_trace(indexed.tracer.path))


def test_ambiguous_questions_interrupt_for_a_choice(indexed, graph):
    seen = {}

    def choose(question, options):
        seen["options"] = options
        return options[0]

    answer = ask(
        indexed, graph, "How does the transformer fuse landmarks?", thread="c", choose=choose
    )
    assert seen["options"] == ["BlinkLinMulT", "the outside transformer"]
    state = graph.get_state({"configurable": {"thread_id": "c"}}).values
    assert state["understanding"].standalone.endswith("(BlinkLinMulT)")
    assert not answer.abstained
    root = [s for s in read_trace(indexed.tracer.path) if s["name"] == "run"][-1]
    assert root["clarified"] == "BlinkLinMulT"
    # without a chooser the first reading is taken
    assert not ask(indexed, graph, "How does the transformer fuse landmarks?", thread="d").abstained


def test_conflicts_resolve_in_favour_of_the_dissertation(indexed):
    diss = next(c.id for c in indexed.index.chunks("dissertation") if "0.912" in c.text)
    paper = next(c.id for c in indexed.index.chunks("blinklinmult") if "0.905" in c.text)
    conflicts = find_conflicts(indexed, [diss, paper])
    assert [(c.dissertation_value, c.other_value) for c in conflicts] == [("0.912", "0.905")]
    assert find_conflicts(indexed, [diss]) == []  # dissertation only: nothing to compare
    assert find_conflicts(indexed, [paper]) == []  # no dissertation evidence


def test_conflict_values_must_be_in_their_chunks(indexed, model):
    diss = next(c.id for c in indexed.index.chunks("dissertation") if "0.912" in c.text)
    paper = next(c.id for c in indexed.index.chunks("blinklinmult") if "0.905" in c.text)
    inner = model.chat_handler
    calls = {"n": 0}

    def wrong_first(body):
        reply = inner(body)
        if "Find values" in body["messages"][-1]["content"] and calls["n"] == 0:
            calls["n"] += 1
            reply["message"]["content"] = json.dumps({"items": [{
                "topic": "F1", "dissertation_value": "0.999", "dissertation_chunk": paper,
                "other_value": "0.905", "other_chunk": "nope:0001"}]})  # fmt: skip
        return reply

    model.chat_handler = wrong_first
    assert len(find_conflicts(indexed, [diss, paper])) == 1
    retry = [b for p, b in model.requests if p == "/api/chat"][-1]["messages"][-1]["content"]
    assert "'0.999' is not written in" in retry and "unknown chunk id nope:0001" in retry
    assert "dissertation_chunk must be a tier-1 chunk" in retry


def test_unsupported_sentences_are_dropped_by_the_verify_subgraph(indexed, graph):
    answer = ask(indexed, graph, "Which WRONG datasets are used for training?")
    assert answer.dropped == 1 and not any("WRONG" in s.text for s in answer.sentences)
    names = [s["name"] for s in read_trace(indexed.tracer.path)]
    assert names.count("llm.chat") > 5


def test_no_evidence_means_no_answer(indexed, graph, monkeypatch):
    monkeypatch.setattr("labmate.ask.graph.retrieve", lambda *a, **k: [])
    answer = ask(indexed, graph, "Which datasets are used?", thread="empty")
    assert answer.abstained and answer.text == NOT_FOUND


def test_replay_reproduces_the_answer_without_a_model(indexed, graph, config):
    first = ask(indexed, graph, "Which datasets are used for training?", thread="r")
    again_session = open_ask(config, ReplayMode.REPLAY)
    again = ask(again_session, compile_graph(again_session, config.ask.library / "t2.sqlite"),
                "Which datasets are used for training?", thread="r")  # fmt: skip
    assert again == first
    again_session.close()


def test_helpers(indexed):
    assert collect([Finding(query="a")], None) == []
    assert [f.query for f in collect(None, [Finding(query="b")])] == ["b"]
    assert check_understanding(Understanding(standalone=" ", intent="own_work", options=["a"]))
    assert check_understanding(Understanding(standalone="q", intent="own_work")) == []
    paper = next(c.id for c in indexed.index.chunks("blinklinmult"))
    diss = next(c.id for c in indexed.index.chunks("dissertation"))
    assert evidence_ids(indexed, [Finding(query="q", chunk_ids=[paper, diss])]) == [diss, paper]
    empty = compose(indexed, "q", [], [], dropped=2)
    assert empty.abstained and empty.dropped == 2
    answer = compose(indexed, "q", [Sentence(text="A.", chunk_ids=[diss, paper]),
                                    Sentence(text="B.", chunk_ids=[diss])], [])  # fmt: skip
    assert answer.text == "A. [1][2] B. [1]" and [c.n for c in answer.citations] == [1, 2]
