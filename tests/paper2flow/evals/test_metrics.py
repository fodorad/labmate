import json

from labmate.core.tracing import latest_completed
from labmate.paper2flow.evals.metrics import results_markdown, run_metrics


def _root(trace_id, status, latency=1000.0, cached=False):
    return [
        {"trace_id": trace_id, "span_id": f"{trace_id}-l", "parent_id": f"{trace_id}-r",
         "name": "llm.chat", "status": "ok", "latency_ms": 1.0, "cached": cached,
         "tokens_in": 10, "tokens_out": 5},
        {"trace_id": trace_id, "span_id": f"{trace_id}-r", "parent_id": None, "name": "run",
         "status": status, "latency_ms": latency},
    ]  # fmt: skip


def _write(path, *roots):
    path.write_text("".join(json.dumps(s) + "\n" for r in roots for s in r))
    return path


def test_run_metrics_on_a_finished_run(finished):
    m = run_metrics(finished)
    assert m.paper_id == "2401.00001" and m.title
    assert m.claims_verified == 6 and m.claims_rejected == 0
    assert m.bullets_first == m.bullets_final > 0
    assert m.unsupported_first == 0 and m.dropped == 0 and m.rounds == 1
    assert 0 <= m.block_fit <= 1
    assert m.slides == 4 and m.flow_nodes == 4 and m.flow_details == 2
    # the paused run (route, extract, outline) and the approved run both count
    assert m.llm_calls > 10 and m.wall_s >= 0 and m.swaps >= 1
    assert m.unsupported_first_pct == 0


def test_run_metrics_without_flows(finished):
    (finished / "09_flows.json").unlink()
    m = run_metrics(finished)
    assert m.flow_nodes == m.flow_details == 0


def test_latest_completed_joins_paused_and_approved_runs(tmp_path):
    trace = _write(
        tmp_path / "t.jsonl",
        _root("old", "ok"),
        _root("paused", "awaiting_approval"),
        _root("done", "ok"),
    )
    assert {s["trace_id"] for s in latest_completed(trace)} == {"paused", "done"}


def test_latest_completed_prefers_live_runs_over_replays(tmp_path):
    trace = _write(tmp_path / "t.jsonl", _root("live", "ok"), _root("replay", "ok", cached=True))
    assert {s["trace_id"] for s in latest_completed(trace)} == {"live"}
    only_replay = _write(tmp_path / "r.jsonl", _root("replay", "ok", cached=True))
    assert {s["trace_id"] for s in latest_completed(only_replay)} == {"replay"}


def test_latest_completed_without_finished_runs(tmp_path):
    assert latest_completed(tmp_path / "missing.jsonl") == []
    paused = _write(tmp_path / "t.jsonl", _root("p", "awaiting_approval"), _root("e", "error"))
    assert latest_completed(paused) == []


def test_results_markdown(finished):
    m = run_metrics(finished)
    failing = m.model_copy(update={"bullets_first": 10, "unsupported_first": 3, "dropped": 1})
    text = results_markdown([m, failing])
    assert sum(line.startswith("| ") for line in text.splitlines()) == 3  # header + 2 rows
    assert "3/10 (30%)" in text and "1 were still unsupported" in text


def test_results_markdown_without_bullets(finished):
    m = run_metrics(finished).model_copy(update={"bullets_first": 0, "unsupported_first": 0})
    assert m.unsupported_first_pct == 0
    assert "Across" not in results_markdown([m])
