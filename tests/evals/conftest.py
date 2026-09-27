"""Fixtures for the evaluation tests: a finished run and a synthetic fact-check audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from paper2flow.config import Config
from paper2flow.engines.plain import run
from paper2flow.schemas import (
    Bullet,
    BulletCheck,
    ClaimCard,
    Claims,
    FactChecked,
    FactCheckReport,
    SlideText,
    WrittenSlides,
)
from tests.conftest import agentic_chat


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.replay.dir = tmp_path / "cassettes"
    cfg.replay.lock_file = tmp_path / "models.lock"
    cfg.tracing.runs_dir = tmp_path / "runs"
    return cfg


@pytest.fixture
def finished(fake, arxiv, config) -> Path:
    """A run that paused at the gate and was then approved."""
    fake.chat_handler = agentic_chat
    run(config, ref="2401.00001", client=fake.client(), http=arxiv.client())
    result = run(config, ref="2401.00001", client=fake.client(), http=arxiv.client(), approve=True)
    assert result.status == "done"
    return result.run_dir


def card(i: int, quote: str) -> ClaimCard:
    return ClaimCard(
        id=f"c{i:02d}",
        claim=quote,
        evidence_quote=quote,
        kind="result",
        section="Results",
        page=3,
        match=100.0,
    )


def check(slide: int, bullet: int, text: str, ids: list[str], verdict: str) -> BulletCheck:
    return BulletCheck(
        slide=slide, bullet=bullet, text=text, claim_ids=ids, verdict=verdict, reason="r"
    )


def write_audit(run_dir: Path, n_supported: int = 6, n_rejected: int = 4) -> Path:
    """A run directory with claims and a two-round fact-check audit, no model needed."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cards = [card(1, "Accuracy is 84.6%."), card(2, "Memory drops by 38%.")]
    (run_dir / "02_claims.json").write_text(Claims(cards=cards, rejected=[]).model_dump_json())
    round0 = [check(1, i + 1, f"Good bullet {i}", ["c01"], "supported") for i in range(n_supported)]
    round0 += [
        check(2, i + 1, f"WRONG bullet {i}", ["c01", "c02"], "unsupported")
        for i in range(n_rejected)
    ]
    round1 = [*round0[:n_supported], check(2, 1, "Fixed bullet", ["c02"], "supported")]
    report = FactCheckReport(rounds=[round0, round1], dropped=round0[n_supported:])
    slides = WrittenSlides(
        hook="h", slides=[SlideText(title="t", bullets=[Bullet(text="x", claim_ids=["c01"])])]
    )
    (run_dir / "05_factcheck.json").write_text(
        FactChecked(slides=slides, report=report).model_dump_json()
    )
    return run_dir
