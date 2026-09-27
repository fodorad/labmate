import json

from paper2flow.schemas import Paper, Section
from paper2flow.steps.extract import (
    MIN_SECTION_CHARS,
    chunk_section,
    extract_claims,
    normalize,
    numeric_tokens,
    quote_score,
)
from paper2flow.steps.llm import LLM

TEXT = (
    "The Transformer allows for signif-\nicantly more parallelization and reaches a new state "
    "of the art after training for twelve hours on eight P100 GPUs."
)


def test_normalize_joins_hyphenation_and_whitespace():
    assert normalize("signif-\n  icantly   MORE\nwork") == "significantly more work"


def test_quote_score_exact_fuzzy_and_rejected():
    assert quote_score("allows for significantly more parallelization", TEXT) == 100
    assert quote_score("twelve hours on eight V100 GPUs", TEXT) == 0  # altered number
    assert quote_score("training for twelve hours on eigth P100 GPUs", TEXT) > 90  # typo ok
    assert quote_score("state of the art after training for 12 hours", TEXT) == 0
    assert quote_score("P100 GPUs", TEXT) == 0  # too short to be evidence


def test_ligatures_and_ellipsis_shortened_quotes_are_accepted():
    text = "Most approaches depend on face recognition and frame-wise eye state classiﬁcation."
    assert quote_score("frame-wise eye state classification", text) == 100  # "ﬁ" ligature
    shortened = "Most approaches depend on ... frame-wise eye state classiﬁcation"
    assert quote_score(shortened, text) == 100
    assert quote_score("Most approaches depend on [...] eye state classification", text) == 100
    # every piece has to match: a made-up second half fails
    assert quote_score("Most approaches depend on … perfect accuracy everywhere", text) < 90
    assert quote_score("Most … state", text) == 0  # no piece long enough to be evidence


def test_numeric_tokens():
    assert numeric_tokens("reaches 84.6% (up from 82.1%), n=6 on p100.") == [
        "84.6%",
        "82.1%",
        "n=6",
        "p100",
    ]


def test_chunking_keeps_title_and_page_and_respects_size():
    section = Section(title="Model", page=3, text="".join(f"line {i}\n" for i in range(400)))
    chunks = chunk_section(section, max_chars=500)
    assert len(chunks) > 1 and all(len(c.text) <= 500 for c in chunks)
    assert "".join(c.text for c in chunks) == section.text
    assert {(c.title, c.page) for c in chunks} == {("Model", 3)}
    assert chunk_section(section, max_chars=10_000) == [section]


def test_unverifiable_quotes_are_rejected_and_ids_are_sequential(fake):
    def model(body):
        props = body["format"]["properties"]
        assert "claims" in props
        claims = [
            {"claim": "Good", "evidence_quote": "reaches a new state of the art", "kind": "result"},
            {
                "claim": "Made up",
                "evidence_quote": "beats GPT-9 on every benchmark",
                "kind": "result",
            },
        ]
        return {"model": body["model"], "message": {"content": json.dumps({"claims": claims})}}

    fake.chat_handler = model
    paper = Paper(
        paper_id="x",
        title="T",
        sections=[
            Section(title="Intro", page=1, text=TEXT + " " * MIN_SECTION_CHARS),
            Section(title="Tiny", page=2, text="Too short."),
            Section(title="Results", page=5, text=TEXT + " " * MIN_SECTION_CHARS),
        ],
    )
    claims = extract_claims(paper, LLM(fake.client(), "qwen3.6:35b-mlx"), workers=2)
    assert [c.id for c in claims.cards] == ["c01", "c02"]
    assert [(c.section, c.page) for c in claims.cards] == [("Intro", 1), ("Results", 5)]
    assert [r.claim for r in claims.rejected] == ["Made up", "Made up"]
    assert fake.paths().count("/api/chat") == 2  # the tiny section is skipped
