import time

from labmate.ask.evals import load_golden, score_answer
from labmate.ask.schemas import Answer, Citation, Sentence


def answered(agent="graph", source="dissertation", dropped=0):
    return Answer(
        question="q",
        text="A. [1]",
        sentences=[Sentence(text="A.", chunk_ids=["d:0001"])],
        citations=[Citation(n=1, chunk_id="d:0001", source_id=source, label="D")],
        dropped=dropped,
        agent=agent,
    )


def test_an_answer_citing_an_expected_source_scores_a_hit(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text("- question: q\n  sources: [dissertation]\n")
    (golden,) = load_golden(path)

    hit = score_answer(golden, answered(), time.perf_counter())
    miss = score_answer(golden, answered(source="outside"), time.perf_counter())

    assert hit.abstain_ok and hit.source_hit and hit.sentences == 1
    assert miss.source_hit is False


def test_declining_is_right_exactly_when_the_library_cannot_answer(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text("- question: q\n  abstain: true\n")
    (golden,) = load_golden(path)
    declined = Answer(question="q", text="no", abstained=True)

    assert score_answer(golden, declined, time.perf_counter()).abstain_ok
    assert score_answer(golden, answered(), time.perf_counter()).abstain_ok is False
    assert score_answer(golden, declined, time.perf_counter()).source_hit is None
