import json
from types import SimpleNamespace

import pytest

from labmate.ask.evals import (
    AnswerCase,
    GeneratedQuestion,
    GeneratedQuestions,
    GoldenQuestion,
    RetrievalCase,
    answer_markdown,
    check_questions,
    evaluate_retrieval,
    gold_chunks,
    load_golden,
    make_cases,
    retrieval_markdown,
    save_retrieval,
    score_answer,
    score_method,
)
from labmate.ask.live import LiveView, describe, model_stats, node_id
from labmate.ask.schemas import Answer
from labmate.cli import main
from labmate.core.schemas import ClaimCard
from labmate.diagrams import ASK_OVERVIEW, diagrams_markdown

# --- retrieval evaluation -----------------------------------------------------------------


def card(cid, quote):
    return ClaimCard(id=cid, claim=quote, evidence_quote=quote, kind="result", section="s",
                     page=1, match=100)  # fmt: skip


def test_question_rules():
    batch = [card("d:c01", "BlinkLinMulT reaches 0.912 F1 on the RT-BENE benchmark dataset today")]
    ok = GeneratedQuestions(items=[GeneratedQuestion(id="d:c01", question="How well does it do?")])
    assert check_questions(batch, ok) == []
    bad = GeneratedQuestions(items=[
        GeneratedQuestion(id="x", question="?"),
        GeneratedQuestion(
            id="d:c01", question="Is it true BlinkLinMulT reaches 0.912 F1 on the RT-BENE benchmark"
        ),
    ])  # fmt: skip
    problems = check_questions(batch, bad)
    assert "unknown claim id x" in problems
    assert any("ending with '?'" in p for p in problems)
    assert any("don't copy the quote" in p for p in problems)
    assert "no question for claim d:c01" in check_questions(batch, GeneratedQuestions(items=[]))


def test_gold_chunks(indexed):
    chunks = indexed.index.chunks("dissertation")
    full = card("dissertation:c01", "Heavy augmentation with lighting changes improves robustness")
    assert gold_chunks(full, chunks) == [c.id for c in chunks if "Heavy augmentation" in c.text]
    assert gold_chunks(card("x", "completely unrelated words about cooking pasta"), chunks) == []


def test_score_method():
    cases = [RetrievalCase(claim_id="a", source_id="s", question="q", gold=["g"]),
             RetrievalCase(claim_id="b", source_id="s", question="q", gold=["h"])]  # fmt: skip
    ranking = {"a": ["x", "g"], "b": ["y"]}
    score = score_method("m", cases, lambda c: ranking[c.claim_id])
    assert score.recall == {1: 0.0, 3: 0.5, 5: 0.5, 10: 0.5} and score.mrr == 0.25
    assert score_method("m", [], lambda c: []).mrr == 0.0


def test_retrieval_evaluation_end_to_end(indexed, tmp_path):
    cases = make_cases(indexed, n=6)
    assert 0 < len(cases) <= 6 and all(c.gold for c in cases)
    assert {c.source_id for c in cases} == {"dissertation", "blinklinmult"}  # balanced
    scores = evaluate_retrieval(indexed, cases)
    assert [s.method for s in scores] == ["bm25", "dense", "hybrid", "hybrid + rewrite"]
    assert all(0 <= s.recall[10] <= 1 for s in scores)
    save_retrieval(tmp_path, cases, scores)
    table = (tmp_path / "retrieval.md").read_text()
    assert "| hybrid |" in table and f"{len(cases)} questions" in table
    assert json.loads((tmp_path / "retrieval_cases.json").read_text())[0]["gold"]
    assert retrieval_markdown([]).startswith("| Method")
    assert len(evaluate_retrieval(indexed, cases, rewrite=False)) == 3


# --- answer evaluation --------------------------------------------------------------------


def test_golden_answers_are_scored_the_same_for_both_agents(indexed, tmp_path):
    golden_file = tmp_path / "golden.yaml"
    golden_file.write_text(
        "- question: Which datasets are used for training?\n  sources: [dissertation]\n"
        "- question: What will the weather be?\n  abstain: true\n"
    )
    golden = load_golden(golden_file)
    assert golden[1].abstain and golden[0].sources == ["dissertation"]
    diss = indexed.index.chunks("dissertation")[6].id
    from labmate.ask.graph import compose
    from labmate.ask.schemas import Sentence

    answer = compose(indexed, golden[0].question,
                     [Sentence(text="Training uses the union of CEW, ZJU and MRL Eye datasets.",
                               chunk_ids=[diss])], [])  # fmt: skip
    spans = [{"name": "run", "latency_ms": 2500}, {"name": "llm.chat"}, {"name": "llm.chat"}]
    case = score_answer(indexed, golden[0], answer, spans)
    assert case.abstain_ok and case.source_hit and case.supported == 1 and case.sentences == 1
    assert case.calls == 2 and case.seconds == 2.5
    declined = Answer(question="w", text="no", abstained=True, agent="prebuilt")
    other = score_answer(indexed, golden[1], declined, [])
    assert other.abstain_ok and other.source_hit is None and other.supported == 0
    table = answer_markdown([case, other])
    assert "| graph | 1/1 | 1/1 | 1/1 | 2.0 | 2 s |" in table
    assert "| prebuilt | 1/1 | 0/0 | 0/0 | 0.0 | 0 s |" in table
    assert GoldenQuestion(question="q").sources == []
    assert AnswerCase.model_fields["supported"]


# --- CLI ------------------------------------------------------------------------------------


@pytest.fixture
def workdir(tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    toml = (
        f'[replay]\ndir = "{config.replay.dir}"\nlock_file = "{config.replay.lock_file}"\n'
        f'[ask]\nlibrary = "{config.ask.library}"\nindex = "{config.ask.index}"\n'
        f'threads = "{config.ask.threads}"\nchunk_words = 30\ntop_k = 4\n'
    )
    (tmp_path / "config.toml").write_text(toml)
    return tmp_path


def test_cli_index_query_and_evals(workdir, model, config, capsys):
    client = model.client()
    assert main(["ask", "query", "anything"], client=client) == 1
    assert "index is empty" in capsys.readouterr().err
    assert main(["ask", "index"], client=client) == 0
    assert "Indexed 3 source(s)" in capsys.readouterr().out
    assert main(["ask", "query", "Which datasets are used for training?"], client=client) == 0
    out = capsys.readouterr().out
    assert "[1] Dissertation §" in out
    argv = ["ask", "query", "How does the transformer fuse landmarks?", "--choice", "2"]
    assert main(argv, client=client) == 0
    assert main(["ask", "query", "CEW ZJU datasets", "--agent", "prebuilt"], client=client) == 0
    assert "(dissertation:" in capsys.readouterr().out
    assert main(["ask", "query", "Which WRONG datasets are used?"], client=client) == 0
    assert "removed by the fact-check" in capsys.readouterr().out
    assert main(["ask", "eval-retrieval", "-n", "4", "--no-rewrite"], client=client) == 0
    assert (workdir / "evals" / "ask" / "retrieval.md").exists()
    assert main(["ask", "eval-answers"], client=client) == 1
    assert "golden.yaml" in capsys.readouterr().err
    (config.ask.library / "golden.yaml").write_text(
        "- question: Which datasets are used?\n  sources: [dissertation]\n"
    )
    assert main(["ask", "eval-answers"], client=client) == 0
    assert "| graph |" in (workdir / "evals" / "ask" / "answers.md").read_text()


def test_cli_conflict_note_and_missing_library(workdir, model, config, capsys, tmp_path):
    client = model.client()
    main(["ask", "index"], client=client)
    capsys.readouterr()
    question = "What F1 does BlinkLinMulT rarely reach on RT-BENE?"  # widens to the papers
    assert main(["ask", "query", question], client=client) == 0
    assert "note: F1 on RT-BENE: dissertation 0.912, other 0.905" in capsys.readouterr().out
    (config.ask.library / "library.toml").unlink()
    assert main(["ask", "index"], client=client) == 1
    assert "library.toml" in capsys.readouterr().err


def test_cli_graphs(workdir, capsys):
    assert main(["graphs", "--out", "docs/graphs.md"]) == 0
    text = (workdir / "docs" / "graphs.md").read_text()
    assert text.count("```mermaid") == 7 and "research_retrieve" in text


def test_cli_unreachable_ollama(workdir, capsys):
    import httpx

    from labmate.core.llm.client import OllamaClient

    def refuse(request):
        raise httpx.ConnectError("refused")

    client = OllamaClient(transport=httpx.MockTransport(refuse))
    assert main(["ask", "index", "--fresh", "--mode", "live"], client=client) == 2
    assert "Cannot reach Ollama" in capsys.readouterr().err


# --- live view and diagrams ---------------------------------------------------------------


def test_node_ids_and_descriptions():
    assert node_id((), "plan") == "plan"
    assert node_id(("research:abc",), "grade") == "research_grade"
    assert node_id(("research:abc",), "finish") is None
    assert node_id(("verify:1",), "judge") == "verify_judge"
    assert node_id(("other:1",), "x") is None
    events = [
        {"event": "understood", "intent": "thesis", "standalone": "q", "options": ["a", "b"]},
        {"event": "planned", "queries": ["x", "y"]},
        {"event": "retrieved", "query": "x", "tiers": [1], "hits": [{}]},
        {"event": "graded", "relevant": ["a"], "sufficient": False, "missing": ""},
        {"event": "rewritten", "query": "z", "tiers": [1, 2]},
        {"event": "conflicts", "items": []},
        {"event": "drafted", "sentences": ["a"]},
        {"event": "verified", "kept": 1, "dropped": 0},
        {"event": "unknown"},
    ]
    lines = [describe(e) for e in events]
    assert lines[0] == "understood as thesis: q; ambiguous: a, b"
    assert lines[3] == "graded 1 relevant, missing: ?"
    assert lines[4] == "rewrote the query (tiers [1, 2]): z"


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


def test_diagrams_markdown():
    text = diagrams_markdown()
    assert text.startswith("# labmate graphs") and "create_agent" in text


def test_studio_factories(indexed, config, monkeypatch):
    from labmate.ask import studio

    monkeypatch.setattr(studio, "load_config", lambda: config)
    assert "understand" in studio.make_graph().get_graph().nodes
    assert "tools" in studio.make_prebuilt().get_graph().nodes


def test_dashboard_questions_come_from_the_golden_set(config):
    from labmate.ask.dashboard import SAMPLE_QUESTIONS, _golden_questions

    assert _golden_questions(config) == SAMPLE_QUESTIONS
    (config.ask.library / "golden.yaml").write_text(
        "- question: A?\n- question: B?\n  abstain: true\n"
    )
    assert _golden_questions(config) == ["A?"]
    assert SimpleNamespace  # keeps the import used
