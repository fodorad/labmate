import base64
import struct
import zlib

from labmate.core.probe import (
    PROBE_MAX_TOKENS,
    VISION_SIZE,
    CheckResult,
    ProbeReport,
    declared_capabilities,
    probe_chat_model,
    probe_image_model,
    run_probe,
    solid_png,
)

TEXT, CRITIC = "qwen3.6:35b-mlx", "gemma4:26b-mlx"


def by_check(results):
    return {r.check: r for r in results}


def test_solid_png_is_a_valid_png():
    png = solid_png((1, 2, 3), size=4)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (4, 4)
    idat_len = struct.unpack(">I", png[33:37])[0]
    pixels = zlib.decompress(png[41 : 41 + idat_len])
    assert pixels == (b"\x00" + bytes([1, 2, 3]) * 4) * 4


def test_well_behaved_models_pass_every_check(fake, tmp_path):
    report = run_probe(
        fake.client(), "0.24.0", [(TEXT, False), (CRITIC, True)], ["x/flux2-klein:latest"], tmp_path
    )
    failures = [r for r in report.results if r.passed is False]
    assert failures == []
    checks = by_check([r for r in report.results if r.model == CRITIC])
    assert checks["vision"].passed is True
    assert checks["declared capabilities"].detail == "completion, tools, vision"
    assert checks["cold load + unload"].metrics["cold_load_s"] == 2.0
    assert checks["structured output"].metrics["gen_tok_per_s"] == 50.0
    assert (tmp_path / "probe_report.json").exists()
    assert "| `gemma4:26b-mlx` | vision | ✅ |" in (tmp_path / "probe_report.md").read_text()
    assert len(list((tmp_path / "images").glob("*.png"))) == 2
    assert fake.loaded == set()  # everything unloaded at the end


def test_models_are_probed_one_at_a_time(fake, tmp_path):
    fake.loaded.add("no-tools:latest")  # something left over from before
    loaded_during_calls = []
    inner = fake.chat_handler

    def spy(body):
        loaded_during_calls.append(set(fake.loaded))
        return inner(body)

    fake.chat_handler = spy
    run_probe(fake.client(), "0.24.0", [(TEXT, False), (CRITIC, False)], [], tmp_path)
    assert all(len(loaded) == 1 for loaded in loaded_during_calls)


def test_think_flag_only_sent_to_thinking_models(fake):
    probe_chat_model(fake.client(), TEXT, vision=False)
    probe_chat_model(fake.client(), CRITIC, vision=False)
    chats = [body for path, body in fake.requests if path == "/api/chat"]
    assert all(b["think"] is False for b in chats if b["model"] == TEXT)
    assert all("think" not in b for b in chats if b["model"] == CRITIC)


def test_invalid_json_fails_structured_output(fake):
    fake.chat_handler = lambda body: {"model": body["model"], "message": {"content": "{bad"}}
    checks = by_check(probe_chat_model(fake.client(), TEXT, False))
    result = checks["structured output"]
    assert result.passed is None  # diagnostic only
    assert result.detail.startswith("(would fail) strict 0/10, lenient 0/10; first non-strict")
    assert checks["structured output (schema in prompt)"].passed is False


def test_fenced_json_counts_as_lenient_but_fails_strict(fake):
    inner = fake.chat_handler

    def fenced(body):
        response = inner(body)
        if "format" in body:
            content = response["message"]["content"]
            response["message"]["content"] = f"```json\n{content}\n```"
        return response

    fake.chat_handler = fenced
    result = by_check(probe_chat_model(fake.client(), TEXT, False))["structured output"]
    assert result.passed is None
    assert result.detail.startswith("(would fail) strict 0/10, lenient 10/10")


def test_slow_unload_is_waited_for(fake):
    fake.unload_delay_polls = 2
    result = by_check(probe_chat_model(fake.client(), TEXT, False))["cold load + unload"]
    assert result.passed is True and "unload_s" in result.metrics


def test_tools_skipped_when_not_declared_and_vision_run_when_declared(fake):
    no_tools = by_check(probe_chat_model(fake.client(), "no-tools:latest", vision=False))
    assert (
        no_tools["tool calling"].passed is None
        and no_tools["tool calling"].detail == "not declared"
    )
    assert not any(p == "/api/chat" and "tools" in b for p, b in fake.requests)
    gemma = by_check(probe_chat_model(fake.client(), CRITIC, vision=False))
    assert gemma["vision"].passed is None  # declared, but not this model's role
    assert gemma["vision"].detail.startswith("(would pass)")


def test_missing_or_wrong_tool_call_fails(fake):
    fake.chat_handler = lambda body: {"model": body["model"], "message": {"content": "Sure!"}}
    assert by_check(probe_chat_model(fake.client(), TEXT, False))["tool calling"].passed is False

    def wrong(body):
        call = {"function": {"name": "use_paper_figure", "arguments": {"figure_id": "fig1"}}}
        return {"model": body["model"], "message": {"content": "", "tool_calls": [call]}}

    fake.chat_handler = wrong
    assert by_check(probe_chat_model(fake.client(), TEXT, False))["tool calling"].passed is False


def test_nondeterminism_is_detected(fake):
    counter = iter(range(100))
    fake.chat_handler = lambda b: {"model": b["model"], "message": {"content": str(next(counter))}}
    result = by_check(probe_chat_model(fake.client(), TEXT, False))["determinism"]
    assert result.passed is False and result.detail.startswith("3 distinct")


def test_wrong_colour_fails_vision(fake):
    fake.chat_handler = lambda b: {"model": b["model"], "message": {"content": "Blue"}}
    assert by_check(probe_chat_model(fake.client(), CRITIC, True))["vision"].passed is False
    info = by_check(probe_chat_model(fake.client(), CRITIC, False))["vision"]
    assert info.passed is None and info.detail.startswith("(would fail)")


def test_model_that_stays_resident_fails_unload(fake):
    fake.sticky.add(TEXT)
    result = by_check(probe_chat_model(fake.client(), TEXT, False))["cold load + unload"]
    assert result.passed is False


def test_exceptions_become_failed_checks_not_crashes(fake):
    def explode(body):
        raise RuntimeError("server melted")

    fake.chat_handler = explode
    results = probe_chat_model(fake.client(), TEXT, False)
    failed = [r for r in results if r.passed is False]
    assert len(failed) == 4  # the format-only diagnostic is informational
    assert all("server melted" in r.detail or "RuntimeError" in r.detail for r in failed)


def test_capabilities_lookup_failure_is_tolerated(fake):
    assert declared_capabilities(fake.client(), "not-installed") == []


def test_image_nondeterminism_is_detected(fake, tmp_path):
    counter = iter(range(100))
    inner = fake.image_handler
    fake.image_handler = lambda b: inner({**b, "options": {"seed": next(counter)}})
    results = by_check(probe_image_model(fake.client(), "x/flux2-klein:latest", tmp_path))
    assert results["image generation 1080x1350"].passed is True
    assert results["image determinism"].passed is False


def test_failing_image_model_skips_determinism(fake, tmp_path):
    fake.image_handler = lambda b: {"model": b["model"], "response": "no image here"}
    results = probe_image_model(fake.client(), "x/flux2-klein:latest", tmp_path)
    assert [r.passed for r in results] == [False]


def test_report_markdown_escapes_pipes_and_marks_info():
    report = ProbeReport(ollama_version="x")
    report.results.append(CheckResult(model="m", check="c", passed=None, detail="a|b"))
    markdown = report.to_markdown()
    assert "a\\|b" in markdown
    assert "ℹ️" in markdown


def test_unload_failure_does_not_abort_probe(fake):
    client = fake.client()

    def broken_unload(model):
        raise RuntimeError("unload endpoint gone")

    client.unload = broken_unload
    results = probe_chat_model(client, TEXT, False)
    assert by_check(results)["structured output (schema in prompt)"].passed is True
    assert by_check(results)["cold load + unload"].passed is False


def test_every_probe_chat_call_is_token_capped(fake):
    probe_chat_model(fake.client(), CRITIC, vision=True)
    chats = [body for path, body in fake.requests if path == "/api/chat"]
    assert chats and all(b["options"]["num_predict"] == PROBE_MAX_TOKENS for b in chats)


def test_progress_is_logged_per_check(fake, caplog):
    caplog.set_level("INFO", logger="labmate.core.probe")
    probe_chat_model(fake.client(), TEXT, vision=False)
    messages = [r.getMessage() for r in caplog.records]
    assert any("think=False" in m for m in messages)
    assert any("structured output" in m and " ok " in m for m in messages)


def test_schema_in_prompt_rescues_backends_that_ignore_format(fake):
    inner = fake.chat_handler

    def mlx_like(body):  # ignores format=, but follows a system instruction
        response = inner(body)
        if "format" in body and body["messages"][0]["role"] != "system":
            response["message"]["content"] = "**Main Result Claim:** accuracy improves"
        return response

    fake.chat_handler = mlx_like
    checks = by_check(probe_chat_model(fake.client(), TEXT, False))
    assert checks["structured output"].detail.startswith("(would fail)")
    assert checks["structured output (schema in prompt)"].passed is True
    prompted = [
        b for p, b in fake.requests if p == "/api/chat" and b["messages"][0]["role"] == "system"
    ]
    assert len(prompted) == 10
    assert '"evidence_quote"' in prompted[0]["messages"][0]["content"]


def test_vision_image_is_large_enough(fake):
    probe_chat_model(fake.client(), CRITIC, vision=True)
    image_body = next(
        b for p, b in fake.requests if p == "/api/chat" and b["messages"][-1].get("images")
    )
    png = base64.b64decode(image_body["messages"][-1]["images"][0])
    assert struct.unpack(">II", png[16:24]) == (VISION_SIZE, VISION_SIZE)
