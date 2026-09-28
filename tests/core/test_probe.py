from labmate.core.probe import PROBE_MAX_TOKENS, UnloadWait, probe_chat_model, run_probe

TEXT, CRITIC = "qwen3.6:35b-mlx", "gemma4:26b-mlx"


def by_check(results):
    return {r.check: r for r in results}


def test_well_behaved_models_pass_every_check(fake, tmp_path):
    report = run_probe(fake.client(), "0.24.0", [TEXT, CRITIC], tmp_path)
    assert [r for r in report.results if r.passed is False] == []
    checks = by_check([r for r in report.results if r.model == CRITIC])
    assert checks["cold load + unload"].metrics["cold_load_s"] == 2.0
    assert checks["structured output"].metrics["gen_tok_per_s"] == 50.0
    assert (tmp_path / "probe_report.json").exists()
    assert "| `gemma4:26b-mlx` | determinism | ✅ |" in (tmp_path / "probe_report.md").read_text()
    assert fake.loaded == set()  # everything unloaded at the end


def test_models_are_probed_one_at_a_time(fake, tmp_path):
    fake.loaded.add("no-tools:latest")  # something left over from before
    loaded_during_calls = []
    inner = fake.chat_handler

    def spy(body):
        loaded_during_calls.append(set(fake.loaded))
        return inner(body)

    fake.chat_handler = spy
    run_probe(fake.client(), "0.24.0", [TEXT, CRITIC], tmp_path)
    assert all(len(loaded) == 1 for loaded in loaded_during_calls)


def test_invalid_json_fails_structured_output(fake):
    fake.chat_handler = lambda body: {"model": body["model"], "message": {"content": "{bad"}}
    checks = by_check(probe_chat_model(fake.client(), TEXT))
    assert checks["structured output"].passed is None  # format= alone is diagnostic only
    assert checks["structured output (schema in prompt)"].passed is False


def test_schema_in_prompt_rescues_backends_that_ignore_format(fake):
    inner = fake.chat_handler

    def mlx_like(body):  # ignores format=, but follows a system instruction
        response = inner(body)
        if "format" in body and body["messages"][0]["role"] != "system":
            response["message"]["content"] = "**Main Result Claim:** accuracy improves"
        return response

    fake.chat_handler = mlx_like
    checks = by_check(probe_chat_model(fake.client(), TEXT))
    assert checks["structured output"].detail.startswith("(would fail)")
    assert checks["structured output (schema in prompt)"].passed is True


def test_missing_or_wrong_tool_call_fails(fake):
    fake.chat_handler = lambda body: {"model": body["model"], "message": {"content": "Sure!"}}
    assert by_check(probe_chat_model(fake.client(), TEXT))["tool calling"].passed is False

    def wrong(body):
        call = {"function": {"name": "use_paper_figure", "arguments": {"figure_id": "fig1"}}}
        return {"model": body["model"], "message": {"content": "", "tool_calls": [call]}}

    fake.chat_handler = wrong
    assert by_check(probe_chat_model(fake.client(), TEXT))["tool calling"].passed is False


def test_tool_calling_is_skipped_for_models_without_tools(fake):
    checks = by_check(probe_chat_model(fake.client(), "no-tools:latest"))
    assert checks["tool calling"].passed is None
    assert not any(p == "/api/chat" and "tools" in b for p, b in fake.requests)


def test_nondeterminism_is_detected(fake):
    counter = iter(range(100))
    fake.chat_handler = lambda b: {"model": b["model"], "message": {"content": str(next(counter))}}
    assert by_check(probe_chat_model(fake.client(), TEXT))["determinism"].passed is False


def test_model_that_stays_resident_fails_unload(fake):
    fake.sticky.add(TEXT)
    wait = UnloadWait(timeout_s=0.05, poll_s=0.0)
    result = by_check(probe_chat_model(fake.client(), TEXT, wait))["cold load + unload"]
    assert result.passed is False and "still resident 0.05s" in result.detail


def test_exceptions_become_failed_checks_not_crashes(fake):
    def explode(body):
        raise RuntimeError("server melted")

    fake.chat_handler = explode
    failed = [r for r in probe_chat_model(fake.client(), TEXT) if r.passed is False]
    assert len(failed) == 4  # the format-only diagnostic is informational
    assert all("server melted" in r.detail or "RuntimeError" in r.detail for r in failed)


def test_every_probe_call_is_token_capped(fake):
    probe_chat_model(fake.client(), CRITIC)
    chats = [body for path, body in fake.requests if path == "/api/chat"]
    assert chats and all(b["options"]["num_predict"] == PROBE_MAX_TOKENS for b in chats)
