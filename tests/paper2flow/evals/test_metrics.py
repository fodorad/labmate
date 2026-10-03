from labmate.paper2flow.evals.metrics import results_markdown, run_metrics


def test_run_metrics_on_a_finished_run(finished):
    m = run_metrics(finished)
    assert m.paper_id == "2401.00001" and m.title
    assert m.claims_verified == 6 and m.claims_rejected == 0
    assert m.bullets_first == m.bullets_final > 0
    assert m.unsupported_first == 0 and m.dropped == 0 and m.rounds == 1
    assert m.cards == 4 and m.flow_nodes == 4 and m.flow_details == 2
    assert m.unsupported_first_pct == 0


def test_results_markdown(finished):
    m = run_metrics(finished)
    failing = m.model_copy(update={"bullets_first": 10, "unsupported_first": 3, "dropped": 1})
    text = results_markdown([m, failing])
    assert sum(line.startswith("| ") for line in text.splitlines()) == 3  # header + 2 rows
    assert "3/10 (30%)" in text and "1 were still unsupported" in text
