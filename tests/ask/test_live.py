from labmate.ask.live import LiveView, model_stats
from labmate.diagrams import ASK_OVERVIEW


def test_live_view_follows_a_real_run(indexed):
    from labmate.ask.graph import compile_graph

    graph = compile_graph(indexed)
    view = LiveView()
    config = {"configurable": {"thread_id": "live"}}
    for item in graph.stream({"question": "How does the transformer fuse landmarks?"}, config,
                             stream_mode=["updates", "custom"], subgraphs=True):  # fmt: skip
        view.feed(item)
    assert view.interrupt and view.interrupt["options"][0] == "BlinkLinMulT"
    assert view.done == ["understand"]
    from langgraph.types import Command

    for item in graph.stream(Command(resume="BlinkLinMulT"), config,
                             stream_mode=["updates", "custom"], subgraphs=True):  # fmt: skip
        view.feed(item)
    assert {"clarify", "plan", "research_retrieve", "research_grade", "verify_judge",
            "finalize"} <= set(view.done)  # fmt: skip
    assert view.answer is not None and view.hits
    assert any(line.startswith("retrieved") for line in view.log)
    diagram = view.mermaid()
    assert diagram.startswith(ASK_OVERVIEW.rstrip()) and "class finalize active" in diagram
    assert "class understand," in diagram
    assert LiveView().mermaid().count("class ") == 0


def test_model_stats():
    spans = [
        {"name": "llm.chat", "model": "a", "tokens_in": 10, "tokens_out": 5, "latency_ms": 1000},
        {"name": "llm.chat", "model": "b", "cached": True, "tokens_in": 1, "tokens_out": 1},
        {"name": "llm.embed", "model": "e", "tokens_in": 3, "latency_ms": 500},
        {"name": "step.x"},
    ]
    stats = model_stats(spans)
    assert stats.calls == {"a": 1, "b": 1} and stats.embeds == 1 and stats.cached == 1
    assert (stats.tokens_in, stats.tokens_out, stats.last_model) == (14, 6, "e")
    assert stats.seconds == 1.5
