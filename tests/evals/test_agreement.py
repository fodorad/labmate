import pytest

from paper2carousel.evals.agreement import (
    agreement,
    agreement_markdown,
    cohen_kappa,
    judge_labels,
)
from paper2carousel.evals.labels import LabelledBullet
from paper2carousel.steps.llm import LLM
from tests.conftest import agentic_chat


def test_cohen_kappa_known_values():
    assert cohen_kappa("aabb", "aabb") == 1.0
    assert cohen_kappa("aaaa", "aaaa") == 1.0  # single class, perfect agreement
    assert cohen_kappa("aaaa", "bbbb") == 0.0
    assert cohen_kappa("ab", "ba") == -1.0
    # textbook example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa = 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa("ab", "a")


def test_agreement_and_markdown():
    human = ["supported", "supported", "partial", "unsupported"]
    predicted = ["supported", "partial", "unsupported", "unsupported"]
    result = agreement(human, predicted, "judge")
    assert result.n == 4 and result.accuracy == 0.5
    assert result.accuracy_binary == 0.75
    assert result.confusion["partial"]["unsupported"] == 1
    assert result.confusion["supported"]["supported"] == 1
    text = agreement_markdown([result])
    assert "| `judge` | 4 | 0.50 |" in text
    assert "| partial | 0 | 0 | 1 |" in text


def test_judge_labels_uses_the_pipeline_judge(fake):
    fake.chat_handler = agentic_chat
    labels = [
        LabelledBullet(id="a", paper="p", text="Accuracy is 84.6%.", evidence=["84.6%"]),
        LabelledBullet(id="b", paper="p", text="WRONG claim", evidence=["84.6%"]),
        LabelledBullet(id="c", paper="p", text="No evidence at all", evidence=[]),
    ]
    judge = LLM(fake.client(), "gemma4:26b-mlx")
    assert judge_labels(labels, judge, workers=2) == ["supported", "unsupported", "supported"]
    prompts = [b["messages"][-1]["content"] for p, b in fake.requests if p == "/api/chat"]
    assert any('evidence (e1): "84.6%"' in p for p in prompts)
