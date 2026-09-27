import json

from paper2flow.schemas import Paper, PublicationDraft
from paper2flow.steps.llm import LLM
from paper2flow.steps.publication import check_publication, read_publication, with_publication

FIRST_PAGE = (
    "Journal of Imaging\nArticle\nBlinkLinMulT: Transformer-Based Eye Blink Detection\n"
    "Received: 1 August 2023\nAccepted: 15 September 2023\nPublished: 21 September 2023\n"
    "Citation: Fodor, Á. J. Imaging 2023, 9, 196.\n"
    "arXiv:2309.00001v2 [cs.CV] 3 Oct 2023"
)
PAPER = Paper(paper_id="x", title="BlinkLinMulT", year=2022)


def draft(venue="", date=""):
    return PublicationDraft(venue=venue, date=date)


def test_answers_must_be_copied_from_the_source():
    ok = draft("J. Imaging 2023, 9, 196", "21 September 2023")
    assert check_publication(ok, FIRST_PAGE) == []
    assert check_publication(draft("j.  imaging 2023, 9,\n196"), FIRST_PAGE) == []  # spacing/case
    invented = check_publication(draft("CVPR 2023", "September 2023 edition"), FIRST_PAGE)
    assert "the venue 'CVPR 2023' is not written in the source; copy it exactly" in invented
    assert any("the date" in p for p in invented)
    assert "the date 'Published' has no year" in check_publication(
        draft(date="Published"), FIRST_PAGE
    )
    assert any("preprint server" in p for p in check_publication(draft("arXiv"), FIRST_PAGE))
    assert check_publication(draft(), FIRST_PAGE) == []


def test_year_comes_from_the_date_then_the_venue_then_ingestion():
    paper = with_publication(PAPER, draft("J. Imaging 2023, 9, 196", "21 September 2023"))
    assert (paper.venue, paper.date, paper.year) == (
        "J. Imaging 2023, 9, 196",
        "21 September 2023",
        2023,
    )
    assert with_publication(PAPER, draft("NIPS 2017")).year == 2017
    kept = with_publication(PAPER.model_copy(update={"date": "2022-03-01"}), draft())
    assert (kept.venue, kept.date, kept.year) == ("", "2022-03-01", 2022)


def reply(content):
    return lambda body: {"model": body["model"], "message": {"content": json.dumps(content)}}


def test_reads_venue_and_date_with_feedback(fake):
    answers = iter([{"venue": "Imaging Journal", "date": ""},
                    {"venue": "J. Imaging 2023, 9, 196", "date": "21 September 2023"}])  # fmt: skip
    prompts = []

    def handler(body):
        prompts.append(body["messages"][-1]["content"])
        return reply(next(answers))(body)

    fake.chat_handler = handler
    llm = LLM(fake.client(), "qwen3.6:35b-mlx")
    notes = PAPER.model_copy(update={"notes": "Comments: 12 pages"})
    result = read_publication(notes, FIRST_PAGE, llm)
    assert result == draft("J. Imaging 2023, 9, 196", "21 September 2023")
    assert "Comments: 12 pages" in prompts[0] and "Ignore the arXiv stamp" in prompts[0]
    assert "is not written in the source" in prompts[1]


def test_gives_up_quietly_and_skips_empty_sources(fake):
    fake.chat_handler = reply({"venue": "Made Up Conference", "date": ""})
    llm = LLM(fake.client(), "qwen3.6:35b-mlx")
    assert read_publication(PAPER, FIRST_PAGE, llm) == draft()
    calls = len(fake.requests)
    assert read_publication(PAPER, "", llm) == draft() and len(fake.requests) == calls
