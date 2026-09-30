import json

from labmate.core.tracing import latest_completed
from labmate.paper2flow.chain import ARTIFACTS
from labmate.paper2flow.evals.metrics import model_swaps, results_markdown, run_metrics


def _root(trace_id, status, latency=1000.0, cached=False):
    return [
        {"trace_id": trace_id, "span_id": f"{trace_id}-l", "parent_id": f"{trace_id}-r",
         "name": "llm.chat", "status": "ok", "latency_ms": 1.0, "cached": cached,
         "tokens_in": 10, "tokens_out": 5, "start_ts": "2026-09-29T10:00:00+00:00"},
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
    assert 0 <= m.card_fit <= 1
    assert m.cards == 4 and m.flow_nodes == 4 and m.flow_details == 2
    assert m.llm_calls == 17 and m.wall_s >= 0
    assert m.swaps == 2  # writer -> judge for the fact-check -> writer for the flows
    assert m.unsupported_first_pct == 0


def test_run_metrics_without_flows(finished):
    (finished / ARTIFACTS["flows"]).unlink()
    m = run_metrics(finished)
    assert m.flow_nodes == m.flow_details == 0


def test_the_latest_finished_run_is_the_one_measured(tmp_path):
    trace = _write(tmp_path / "t.jsonl", _root("old", "ok"), _root("done", "ok"),
                   _root("failed", "error"))  # fmt: skip
    assert {s["trace_id"] for s in latest_completed(trace)} == {"done"}


def test_model_swaps_follow_the_call_order():
    calls = [{"name": "llm.chat", "model": m, "start_ts": f"t{i}"}
             for i, m in enumerate(["w", "w", "j", "j", "w"])]  # fmt: skip
    assert model_swaps(list(reversed(calls))) == 2


def test_latest_completed_prefers_live_runs_over_replays(tmp_path):
    trace = _write(tmp_path / "t.jsonl", _root("live", "ok"), _root("replay", "ok", cached=True))
    assert {s["trace_id"] for s in latest_completed(trace)} == {"live"}
    only_replay = _write(tmp_path / "r.jsonl", _root("replay", "ok", cached=True))
    assert {s["trace_id"] for s in latest_completed(only_replay)} == {"replay"}


def test_latest_completed_without_finished_runs(tmp_path):
    assert latest_completed(tmp_path / "missing.jsonl") == []
    failed = _write(tmp_path / "t.jsonl", _root("e", "error"))
    assert latest_completed(failed) == []


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
