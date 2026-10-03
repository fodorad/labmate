from labmate.ask.answer import NOT_FOUND, compose, is_supported, structure_problems
from labmate.ask.schemas import Sentence


def chunk_with(index, text):
    return next(c.id for c in index.chunks() if text in c.text)


def test_citations_are_numbered_in_order_of_first_use(indexed):
    arch = chunk_with(indexed.index, "0.912 F1")
    data = chunk_with(indexed.index, "CEW, ZJU")
    sentences = [
        Sentence(text="It reaches 0.912 F1.", chunk_ids=[arch, data]),
        Sentence(text="It trains on CEW.", chunk_ids=[data]),
    ]

    answer = compose(indexed.index, "q", sentences)

    assert answer.text == "It reaches 0.912 F1. [1][2] It trains on CEW. [2]"
    assert [c.chunk_id for c in answer.citations] == [arch, data]
    assert answer.citations[0].label.startswith("Dissertation §")


def test_a_sentence_needs_every_number_it_states_in_its_evidence(indexed):
    arch = chunk_with(indexed.index, "0.912 F1")

    assert is_supported(indexed.index, Sentence(text="It reaches 0.912 F1.", chunk_ids=[arch]))
    assert not is_supported(indexed.index, Sentence(text="It reaches 0.999 F1.", chunk_ids=[arch]))
    assert not is_supported(indexed.index, Sentence(text="A claim.", chunk_ids=["nope:0001"]))


def test_unsupported_sentences_are_dropped_and_counted(indexed):
    arch = chunk_with(indexed.index, "0.912 F1")
    sentences = [
        Sentence(text="It reaches 0.912 F1.", chunk_ids=[arch]),
        Sentence(text="It reaches 0.999 F1.", chunk_ids=[arch]),
    ]

    answer = compose(indexed.index, "q", sentences, dropped=1)

    assert [s.text for s in answer.sentences] == ["It reaches 0.912 F1."]
    assert answer.dropped == 2 and not answer.abstained


def test_an_answer_with_nothing_supported_is_an_abstention(indexed):
    answer = compose(indexed.index, "q", [Sentence(text="It is 42.", chunk_ids=["nope:0001"])])

    assert answer.abstained and answer.text == NOT_FOUND and answer.dropped == 1


def test_a_draft_must_cite_known_ids_in_short_sentences_without_ids_in_the_text():
    sentences = [
        Sentence(text="Fine.", chunk_ids=["a:0001"]),
        Sentence(text="Cites a stranger.", chunk_ids=["z:0009"]),
        Sentence(text="word " * 40, chunk_ids=["a:0001"]),
        Sentence(text="Inline [a:0001] id.", chunk_ids=["a:0001"]),
    ]

    problems = structure_problems(sentences, {"a:0001"})

    assert len(problems) == 3
    assert (
        "sentence 2" in problems[0] and "sentence 3" in problems[1] and "sentence 4" in problems[2]
    )
