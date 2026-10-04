import json

import httpx
import pytest

from labmate.ask.evals import AnswerCase
from labmate.bench.cells import Cell, cells_markdown, load, save
from labmate.bench.guard import NotEnoughMemory, parse_vm_stat, require_room
from labmate.bench.micro import ModelResult, judge_test, unload_all
from labmate.bench.quality import ask_quality
from labmate.bench.report import micro_markdown, profile_rows
from labmate.config import Config, ProfileConfig
from tests.conftest import chat, reply


def verdicts(kind: str):
    """A judge that answers every bullet of a card with the same verdict."""

    def handler(body):
        bullets = body["messages"][-1]["content"].count("Bullet ")
        items = [{"bullet": n, "verdict": kind, "reason": "x"} for n in range(1, bullets + 1)]
        return reply(body, json.dumps({"verdicts": items}))

    return handler


def test_a_judge_that_trusts_everything_misses_every_mistake(fake):
    fake.chat_handler = verdicts("supported")

    trusting = judge_test(chat(fake, "gemma4:26b-mlx"))

    assert (trusting.caught, trusting.mistakes) == (0, 15)
    assert trusting.false_alarms == 0 and "number on the wrong thing" in trusting.missed


def test_a_judge_that_doubts_everything_catches_all_mistakes_and_raises_false_alarms(fake):
    fake.chat_handler = verdicts("unsupported")

    doubting = judge_test(chat(fake, "gemma4:26b-mlx"))

    assert (doubting.caught, doubting.false_alarms, doubting.clean) == (15, 9, 9)


def test_the_tables_say_which_profiles_fit_in_memory_together():
    config = Config(
        profiles={
            "current": ProfileConfig(text="big", critic="medium"),
            "mixed": ProfileConfig(text="medium", critic="small"),
            "single": ProfileConfig(text="medium", critic="medium"),
        }
    )
    sizes = {"big": 21.0, "medium": 16.0, "small": 6.6}

    rows = {row.profile: row for row in profile_rows(config, sizes, ram_gb=32)}

    assert not rows["current"].fits and rows["current"].gb == 37.0  # swaps on a 32 GB Mac
    assert not rows["mixed"].fits and rows["mixed"].gb == 22.6  # the GPU may use 24 GB, minus 5
    assert rows["single"].gb == 16.0 and rows["single"].fits  # one model counts once
    results = [ModelResult(tag, size_gb=gb, load_s=4, decode_tps=30) for tag, gb in sizes.items()]
    text = micro_markdown(results, config, 32)
    assert "**no, a model is evicted**" in text and "`small`" in text


def test_unloading_asks_ollama_to_drop_every_loaded_model():
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "a"}, {"name": "b"}]})
        asked.append(request.content)
        return httpx.Response(200, json={})

    unload_all("http://ollama", httpx.MockTransport(handler))

    assert len(asked) == 2 and all(b'"keep_alive":0' in a.replace(b" ", b"") for a in asked)


def test_ask_quality_averages_answer_source_and_kept_sentences():
    def case(ok, hit, kept, dropped):
        return AnswerCase(
            question="q", agent="graph", abstained=False, abstain_ok=ok, source_hit=hit,
            sentences=kept, dropped=dropped, seconds=1.0, answer="a",
        )  # fmt: skip

    quality = ask_quality([case(True, True, 3, 1), case(True, False, 1, 3)])

    assert quality.parts == {
        "right answer or refusal": 1.0,
        "expected source cited": 0.5,
        "sentences kept": 0.5,
    }
    assert quality.score == 66.7


def test_the_table_marks_runs_over_the_target_and_cells_survive_a_round_trip(tmp_path):
    fast = Cell("cv2job", "mixed", "a", "b", seconds=40, quality=90, parts={"x": 0.9})
    slow = Cell("cv2job", "current", "a", "c", seconds=83, quality=95, parts={"x": 0.95})
    save(fast, tmp_path)
    save(slow, tmp_path)

    cells = load(tmp_path)
    table = cells_markdown(cells)

    assert sorted(c.profile for c in cells) == ["current", "mixed"]
    assert table.index("| mixed |") < table.index("| current |")  # fastest first
    assert "| 40 s | yes |" in table and "| 83 s | **no** |" in table


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  100000.
Pages active:                                500000.
Pages inactive:                               50000.
Pages speculative:                            10000.
Pages purgeable:                               5000.
"""


def test_the_guard_reads_free_memory_and_refuses_models_that_do_not_fit():
    free = parse_vm_stat(VM_STAT)

    assert round(free, 1) == 2.7  # free + inactive + speculative + purgeable pages
    require_room(models_gb=10.0, ram_gb=32, free_gb=20.0)
    with pytest.raises(NotEnoughMemory, match="do not fit in 19.0 GB"):
        require_room(models_gb=23.3, ram_gb=32, free_gb=30.0)
    with pytest.raises(NotEnoughMemory, match="only 8.0 GB are free"):
        require_room(models_gb=10.0, ram_gb=32, free_gb=8.0)


def test_a_decider_is_scored_against_the_labels(fake):
    from labmate.bench.triage import evaluate, triage_markdown

    papers = [
        {"title": "Alpha", "abstract": "a", "expected": {"A": "deep"}},
        {"title": "Bravo", "abstract": "b", "expected": {"A": "skip"}},
        {"title": "Charlie", "abstract": "c", "expected": {"A": "post"}},
    ]
    golden = {"interests": {"A": "video"}, "papers": papers}
    scores = {"Alpha": 3.5, "Bravo": 2.5, "Charlie": 2.0}
    fake.systemone_handler = lambda body: {
        "answers": {
            "relevance": {"score": scores[body["state"].split("Paper title: ")[1].split("\n")[0]]}
        }
    }

    result = evaluate(Config(), "clef-flash", golden, fake.transport())

    assert result.kind == "decision" and result.cases == 3
    assert (result.exact, result.read, result.deep) == (2, 2, 3)  # Bravo is rated post, not skip
    assert "2/3" in triage_markdown([result], {"clef-flash": 10.0})
