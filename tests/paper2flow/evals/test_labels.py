import csv

import pytest

from labmate.paper2flow.evals.labels import (
    bullets_for_labelling,
    export_labels,
    parse_label,
    read_labels,
    sample,
)
from tests.paper2flow.evals.conftest import write_audit


def test_bullets_are_distinct_with_evidence_and_pipeline_verdict(tmp_path):
    bullets = bullets_for_labelling(write_audit(tmp_path / "p1"))
    texts = [b.text for b in bullets]
    assert len(texts) == len(set(texts)) == 11  # 6 good + 4 wrong + 1 fixed
    wrong = next(b for b in bullets if b.text == "WRONG bullet 0")
    assert wrong.id == "p1/r0s2b1" and wrong.pipeline == "unsupported"
    assert wrong.evidence == ["Accuracy is 84.6%.", "Memory drops by 38%."]
    assert next(b for b in bullets if b.text == "Fixed bullet").id == "p1/r1s2b1"


def test_sample_is_stratified_and_seeded(tmp_path):
    bullets = bullets_for_labelling(write_audit(tmp_path / "p1"))
    picked = sample(bullets, 6, seed=1)
    assert len(picked) == 6
    assert sum(b.pipeline != "supported" for b in picked) == 3
    assert [b.id for b in sample(bullets, 6, seed=1)] == [b.id for b in picked]
    # fewer rejected than half: fill up with accepted ones
    assert sum(b.pipeline != "supported" for b in sample(bullets, 10)) == 4
    # few accepted: take more rejected
    few = write_audit(tmp_path / "p2", n_supported=1, n_rejected=6)
    assert len(sample(bullets_for_labelling(few), 6)) == 6


@pytest.mark.parametrize(
    ("cell", "label"),
    [("", None), (" S ", "supported"), ("p", "partial"), ("unsupported", "unsupported")],
)
def test_parse_label(cell, label):
    assert parse_label(cell) == label


def test_parse_label_rejects_typos():
    with pytest.raises(ValueError, match="unknown label"):
        parse_label("yes")


def test_export_is_blind_and_keeps_existing_labels(tmp_path):
    runs = [write_audit(tmp_path / "p1")]
    out = tmp_path / "evals" / "labels.csv"
    assert export_labels(runs, out, n=4) == 4
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["id", "paper", "text", "evidence", "human"]
    assert "unsupported" not in out.read_text()  # the pipeline verdict is not leaked

    rows[0]["human"] = "u"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert export_labels(runs, out, n=8) == 4
    labels = read_labels(out)
    assert len(labels) == 8 and len({b.text for b in labels}) == 8
    assert labels[0].human == "unsupported" and labels[0].id == rows[0]["id"]
    assert all(b.human is None for b in labels[1:])
    assert all(b.evidence for b in labels)
    assert export_labels(runs, out, n=8) == 0
