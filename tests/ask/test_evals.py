import json

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
from labmate.ask.schemas import Answer
from labmate.core.schemas import ClaimCard


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
