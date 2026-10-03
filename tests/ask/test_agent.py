from labmate.ask.agent import ask_agent, build_agent, make_tools, parse_answer


def test_the_agent_answers_with_cited_sentences_only(indexed, model):
    answer = ask_agent(indexed, build_agent(indexed), "CEW ZJU datasets")

    assert answer.agent == "agent" and not answer.abstained
    assert len(answer.sentences) == 1  # the uncited sentence is dropped
    assert answer.dropped == 1
    assert "CEW" in answer.citations[0].text
    assert any(b.get("tools") for p, b in model.requests if p == "/api/chat")  # tools were offered


def test_tools_search_read_and_list(indexed):
    search, read, sources = make_tools(indexed)

    assert "Dissertation" in search.invoke({"query": "CEW datasets"})
    first = indexed.index.chunks("dissertation")[0].id
    assert read.invoke({"chunk_id": first}).startswith(f"[{first}] Dissertation")
    assert "unknown chunk id" in read.invoke({"chunk_id": "nope:0000"})
    assert "dissertation: Dissertation." in sources.invoke({})


def test_citations_of_unknown_chunks_and_unsupported_numbers_do_not_count(indexed):
    arch = next(c.id for c in indexed.index.chunks() if "0.912 F1" in c.text)
    text = f"It reaches 0.912 F1 [{arch}]. Made up [nope:0001]. It reaches 0.5 F1 [{arch}]."

    answer = parse_answer(indexed, "q", text)

    assert [s.chunk_ids for s in answer.sentences] == [[arch]]
    assert answer.dropped == 2


def test_an_answer_without_citations_is_an_abstention_carrying_the_agents_words(indexed):
    answer = parse_answer(indexed, "q", "I could not find it.")

    assert answer.abstained and answer.text == "I could not find it."


def test_the_judge_also_checks_what_the_agent_wrote(indexed):
    arch = next(c.id for c in indexed.index.chunks() if "0.912 F1" in c.text)
    text = f"BOGUS: it reaches 0.912 F1 [{arch}]. It reaches 0.912 F1 [{arch}]."

    answer = parse_answer(indexed, "q", text)

    assert [s.text for s in answer.sentences] == ["It reaches 0.912 F1."]
    assert answer.dropped == 1
