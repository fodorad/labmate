from paper2carousel.config import ReplayMode
from paper2carousel.llm.replay import CassetteStore, ReplayClient
from paper2carousel.schemas import Paper, Section
from paper2carousel.steps.draft import SECTION_CHARS, build_prompt, draft_deck
from tests.conftest import DECK, deck_chat

PAPER = Paper(
    paper_id="x",
    title="A Test Paper",
    abstract="We study attention.",
    sections=[
        Section(title="Intro", page=1, text="a" * 5000),
        Section(title="Results", page=3, text="84.6%"),
    ],
)


def test_prompt_truncates_sections_and_keeps_facts():
    prompt = build_prompt(PAPER)
    assert "## Intro (page 1)" in prompt and "## Results (page 3)" in prompt
    assert "a" * SECTION_CHARS in prompt and "a" * (SECTION_CHARS + 1) not in prompt
    assert "84.6%" in prompt and "6 to 8 slides" in prompt


def test_draft_deck_uses_schema_prompt_and_context_window(fake, tmp_path):
    fake.chat_handler = deck_chat
    deck = draft_deck(PAPER, fake.client(), "qwen3.6:35b-mlx", num_ctx=8192)
    assert deck == DECK
    body = next(b for p, b in fake.requests if p == "/api/chat")
    assert body["messages"][0]["role"] == "system"
    assert body["options"]["num_ctx"] == 8192 and body["think"] is False


def test_draft_is_replayable(fake, tmp_path):
    fake.chat_handler = deck_chat
    store = CassetteStore(tmp_path)
    draft_deck(PAPER, ReplayClient(fake.client(), store, ReplayMode.RECORD), "qwen3.6:35b-mlx")
    offline = ReplayClient(None, store, ReplayMode.REPLAY)
    assert draft_deck(PAPER, offline, "qwen3.6:35b-mlx") == DECK
