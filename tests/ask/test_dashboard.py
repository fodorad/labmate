def test_dashboard_questions_come_from_the_golden_set(config):
    from labmate.ask.dashboard import SAMPLE_QUESTIONS, _golden_questions

    assert _golden_questions(config) == SAMPLE_QUESTIONS
    (config.ask.library / "golden.yaml").write_text(
        "- question: A?\n- question: B?\n  abstain: true\n"
    )
    assert _golden_questions(config) == ["A?"]
