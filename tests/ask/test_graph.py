import pytest

from labmate.ask.answer import NOT_FOUND, OFF_TOPIC
from labmate.ask.graph import ask, compile_graph


@pytest.fixture
def graph(indexed):
    return compile_graph(indexed)


def embeds(model):
    return [b for p, b in model.requests if p == "/api/embed"]


def test_a_question_is_answered_with_numbered_citations(indexed, graph):
    answer = ask(graph, "Which datasets are used for training?")

    assert not answer.abstained and answer.sentences and answer.dropped == 0
    assert answer.citations[0].label.startswith("Dissertation §")
    assert "[1]" in answer.text
    assert {c.n for c in answer.citations} == set(range(1, len(answer.citations) + 1))


def test_an_off_topic_question_is_searched_once_and_declined_when_nothing_is_found(
    indexed, graph, model
):
    searches = len(embeds(model))

    answer = ask(graph, "What will the weather be tomorrow?")

    assert answer.abstained and answer.text == OFF_TOPIC and answer.reason == "off topic"
    assert len(embeds(model)) - searches == 1  # one look, no rewriting rounds


def test_a_question_wrongly_called_off_topic_is_still_answered_from_the_library(indexed, graph):
    # a real run refused "How fast is LinMulT at inference?" this way: the model called it off topic
    answer = ask(graph, "Which OFFBEAT datasets are used for training?")

    assert not answer.abstained and answer.citations


def test_thin_evidence_makes_the_query_be_rewritten_and_searched_again(indexed, graph, model):
    answer = ask(graph, "rarely eye datasets")

    queries = [b["input"][0] for b in embeds(model)[-2:]]  # the last two searches
    assert queries[0].endswith("rarely eye datasets")
    assert queries[1].endswith("eye datasets details")  # "rarely" is gone after the rewrite
    assert not answer.abstained


def test_the_loop_budget_stops_the_rewriting(indexed, graph, model):
    indexed.config.ask.max_loops = 1
    searches = len(embeds(model))

    answer = ask(compile_graph(indexed), "rarely eye datasets")

    assert len(embeds(model)) - searches == 1  # no second search
    assert not answer.abstained  # the first search found relevant chunks


def test_ambiguous_questions_pause_for_a_choice_and_search_with_it(indexed, graph, model):
    seen = {}

    def choose(question, options):
        seen["options"] = options
        return options[0]

    answer = ask(graph, "How does the transformer fuse landmarks?", choose)

    assert seen["options"] == ["BlinkLinMulT", "the outside transformer"]
    assert embeds(model)[-1]["input"][0].endswith("(BlinkLinMulT)")
    assert not answer.abstained


def test_sentences_with_numbers_the_evidence_lacks_are_dropped(indexed, graph):
    answer = ask(graph, "Which WRONG datasets are used for training?")

    assert answer.dropped == 1
    assert not any("99.9" in s.text for s in answer.sentences)


def test_sentences_the_judge_finds_unsupported_are_dropped(indexed, graph, model):
    answer = ask(graph, "Which BOGUS datasets are used for training?")

    assert answer.dropped == 1
    assert not any("BOGUS" in s.text for s in answer.sentences)
    judged = [
        b for p, b in model.requests if p == "/api/chat" and "verdicts" in str(b.get("format"))
    ]
    assert judged and judged[-1]["model"] == "gemma4:26b-mlx"


def test_an_empty_index_means_no_answer(indexed, graph):
    for source in indexed.index.sources():
        indexed.index.remove_source(source.id)

    answer = ask(graph, "Which datasets are used?")

    assert answer.abstained and answer.text == NOT_FOUND
